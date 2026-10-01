import io
from pathlib import Path

import pytest
from PIL import Image

import dvr_timestamp_indexer as indexer
import dvr_timestamp_renamer as renamer
from dvr_timestamp_indexer import (
    CSV_FIELDS,
    OcrResult,
    ProgressReporter,
    crop_overlay,
    index_videos,
    is_camera_photo,
    media_paths,
    parse_timestamp,
    preprocessing_variants,
    read_index,
    recovery_preprocessing_variants,
    reprocess_rows,
    resolution_from_frame,
    resolve_index_path,
    video_paths,
    write_index,
    write_index_atomically,
)


def test_parse_timestamp_accepts_xiaomi_overlay_text():
    assert parse_timestamp("MI 2025/10/31 15:04:05") == "2025-10-31 15:04:05"
    assert parse_timestamp("2025 / 1 / 2 3 : 4 : 5") == "2025-01-02 03:04:05"
    assert parse_timestamp("2025/10/3115:04:05") == "2025-10-31 15:04:05"


def test_parse_timestamp_rejects_invalid_or_ambiguous_text():
    assert parse_timestamp("2025/99/31 15:04:05") is None
    assert parse_timestamp("31/10/2025 15:04:05") is None
    assert parse_timestamp("2025/10/31 25:04:05") is None


def test_crop_and_preprocessing_scale_with_resolution():
    for dimensions in ((640, 360), (2560, 1440)):
        crop = crop_overlay(Image.new("RGB", dimensions), 0.40, 0.125)
        assert crop.size == (round(dimensions[0] * 0.40), round(dimensions[1] * 0.125))
        variants = preprocessing_variants(crop)
        assert set(variants) == {"grayscale", "threshold_120", "threshold_170", "threshold_210"}
        assert variants["grayscale"].size == (crop.width * 3, crop.height * 3)
        recovery = recovery_preprocessing_variants(Image.new("RGB", dimensions))
        assert len(recovery) == 9
        assert recovery["tight_4x_grayscale"].size == (round(dimensions[0] * 0.27) * 4, round(dimensions[1] * 0.09) * 4)


def test_extract_timestamp_records_first_usable_frame_resolution(monkeypatch, tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.touch()
    monkeypatch.setattr(indexer.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(indexer, "first_usable_frame", lambda path: Image.new("RGB", (1920, 1080)))
    monkeypatch.setattr(indexer, "_run_tesseract", lambda image, executable: ("2025/10/31 15:04:05", "98.0"))

    result = indexer.extract_timestamp(video)

    assert result.status == "ok"
    assert result.resolution == "1920x1080"
    assert resolution_from_frame(Image.new("RGB", (640, 360))) == "640x360"


def test_media_paths_include_camera_photos_but_exclude_unrelated_pngs(tmp_path: Path):
    root = tmp_path / "archive"
    nested = root / "nested"
    nested.mkdir(parents=True)
    for name in ("z.mp4", "IMG_123.PNG", "img_9.png", "logo.png", "IMG_x.PNG"):
        (nested / name).touch()

    paths = media_paths(root)

    assert [path.relative_to(root).as_posix() for path in paths] == [
        "nested/IMG_123.PNG",
        "nested/img_9.png",
        "nested/z.mp4",
    ]
    assert is_camera_photo(nested / "IMG_123.PNG")
    assert not is_camera_photo(nested / "IMG_x.PNG")


def test_extract_timestamp_loads_camera_png_with_its_native_resolution(monkeypatch, tmp_path: Path):
    photo = tmp_path / "IMG_123.PNG"
    Image.new("RGB", (300, 200)).save(photo)
    monkeypatch.setattr(indexer.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(
        indexer,
        "first_usable_frame",
        lambda path: pytest.fail("camera PNGs must not use video decoding"),
    )
    monkeypatch.setattr(
        indexer,
        "_run_tesseract",
        lambda image, executable: ("2025/10/31 15:04:05", "98.0"),
    )

    result = indexer.extract_timestamp(photo)
    recovery = indexer.extract_recovery_timestamp(photo)

    assert result.status == "ok"
    assert result.resolution == "300x200"
    assert recovery.status == "ok"
    assert recovery.resolution == "300x200"


def test_extract_timestamp_reports_unreadable_camera_png_as_decode_error(monkeypatch, tmp_path: Path):
    photo = tmp_path / "IMG_123.PNG"
    photo.write_bytes(b"not a PNG")
    monkeypatch.setattr(indexer.shutil, "which", lambda executable: executable)

    result = indexer.extract_timestamp(photo)

    assert result.status == "decode_error"


def test_csv_is_sorted_and_retains_failures(tmp_path: Path):
    root = tmp_path / "archive"
    (root / "nested").mkdir(parents=True)
    first = root / "nested" / "a.mp4"
    second = root / "z.mp4"
    first.touch()
    second.touch()

    def fake_extractor(path: Path) -> OcrResult:
        if path == first:
            return OcrResult(
                "2025-10-31 15:04:05",
                "ok",
                "2025/10/31 15:04:05",
                "98.0",
                resolution="1920x1080",
            )
        return OcrResult(status="decode_error", error="invalid data")

    rows = index_videos(root, video_paths(root), fake_extractor)
    output = tmp_path / "index.csv"
    write_index(output, rows)
    assert output.read_text(encoding="utf-8").splitlines() == [
        "file_path,actual_datetime,resolution,status,ocr_text,confidence,error",
        "nested/a.mp4,2025-10-31 15:04:05,1920x1080,ok,2025/10/31 15:04:05,98.0,",
        "z.mp4,,,decode_error,,,invalid data",
    ]


def test_parallel_indexing_retains_path_order_and_reports_progress(tmp_path: Path):
    root = tmp_path / "archive"
    root.mkdir()
    first = root / "a.mp4"
    second = root / "b.mp4"
    first.touch()
    second.touch()
    stream = io.StringIO()

    rows = index_videos(
        root,
        [first, second],
        lambda path: OcrResult(status="ok" if path == first else "no_timestamp"),
        workers=2,
        progress=ProgressReporter(2, stream),
    )

    assert [row["file_path"] for row in rows] == ["a.mp4", "b.mp4"]
    updates = stream.getvalue().splitlines()
    assert {line.rsplit(": ", 1)[-1] for line in updates if line.startswith("completed")} == {
        "ok",
        "no_timestamp",
    }
    assert {line.split("] ", 1)[-1] for line in updates if line.startswith("processing")} == {
        "a.mp4",
        "b.mp4",
    }


def test_reprocess_preserves_successes_and_updates_only_blank_rows(tmp_path: Path):
    root = tmp_path / "archive"
    root.mkdir()
    (root / "retry.mp4").touch()
    rows = [
        {"file_path": "ok.mp4", "actual_datetime": "2025-10-31 15:04:05", "resolution": "1920x1080", "status": "ok", "ocr_text": "original", "confidence": "99.0", "error": ""},
        {"file_path": "retry.mp4", "actual_datetime": "", "resolution": "", "status": "no_timestamp", "ocr_text": "old", "confidence": "", "error": ""},
        {"file_path": "gone.mp4", "actual_datetime": "", "resolution": "", "status": "no_timestamp", "ocr_text": "old", "confidence": "", "error": ""},
    ]

    updated = reprocess_rows(
        root,
        rows,
        lambda path: OcrResult(
            "2025-10-31 12:00:00", "ok", "new", "90.0", resolution="640x360"
        ),
    )
    assert updated[0] == rows[0]
    assert updated[1]["actual_datetime"] == "2025-10-31 12:00:00"
    assert updated[1]["resolution"] == "640x360"
    assert updated[1]["ocr_text"] == "new"
    assert updated[2]["status"] == "missing_file"


def test_reprocess_retries_blank_camera_png_rows(tmp_path: Path):
    root = tmp_path / "archive"
    root.mkdir()
    (root / "IMG_123.PNG").touch()
    rows = [renamer_row("IMG_123.PNG", actual_datetime="", resolution="", status="no_timestamp")]

    updated = reprocess_rows(
        root,
        rows,
        lambda path: OcrResult("2025-10-31 12:00:00", "ok", resolution="1920x1080"),
    )

    assert updated[0]["actual_datetime"] == "2025-10-31 12:00:00"
    assert updated[0]["status"] == "ok"


def test_parallel_reprocess_only_queues_existing_blank_media(tmp_path: Path):
    root = tmp_path / "archive"
    root.mkdir()
    (root / "retry.mp4").touch()
    rows = [
        renamer_row("already-indexed.mp4"),
        renamer_row("retry.mp4", actual_datetime="", resolution="", status="no_timestamp"),
        renamer_row("missing.mp4", actual_datetime="", resolution="", status="no_timestamp"),
        renamer_row("../unsafe.mp4", actual_datetime="", resolution="", status="no_timestamp"),
    ]
    seen: list[Path] = []
    stream = io.StringIO()

    updated = reprocess_rows(
        root,
        rows,
        lambda path: seen.append(path) or OcrResult("2025-10-31 12:00:00", "ok"),
        workers=2,
        progress=ProgressReporter(1, stream),
    )

    assert seen == [root / "retry.mp4"]
    assert [row["status"] for row in updated] == [
        "ok",
        "ok",
        "missing_file",
        "invalid_path",
    ]
    assert "processing [1/1] retry.mp4" in stream.getvalue()
    assert "completed [1/1] retry.mp4: ok" in stream.getvalue()


def test_workers_argument_defaults_to_eight_and_rejects_non_positive_values(capsys):
    parser = indexer.build_parser()

    assert parser.parse_args(["archive", "index.csv"]).workers == 8
    assert parser.parse_args(["archive", "index.csv", "--workers", "3"]).workers == 3
    with pytest.raises(SystemExit):
        parser.parse_args(["archive", "index.csv", "--workers", "0"])
    assert "workers must be a positive integer" in capsys.readouterr().err


def test_main_reports_index_progress_on_stderr_and_summary_on_stdout(
    monkeypatch, tmp_path: Path, capsys
):
    root = tmp_path / "archive"
    root.mkdir()
    (root / "video.mp4").touch()
    output = tmp_path / "index.csv"
    monkeypatch.setattr(
        indexer,
        "extract_timestamp",
        lambda path, tesseract: OcrResult("2025-10-31 15:04:05", "ok"),
    )

    assert indexer.main([str(root), str(output), "--workers", "1"]) == 0

    captured = capsys.readouterr()
    assert captured.out == "indexed 1 media file(s): ok=1\n"
    assert "indexing 1 media file(s) with 1 worker(s)" in captured.err
    assert "processing [1/1] video.mp4" in captured.err
    assert "completed [1/1] video.mp4: ok" in captured.err


def test_reprocess_rejects_unsafe_paths_and_replaces_csv_atomically(tmp_path: Path):
    root = tmp_path / "archive"
    root.mkdir()
    with pytest.raises(ValueError, match="outside"):
        resolve_index_path(root, "../elsewhere.mp4")
    with pytest.raises(ValueError, match="relative"):
        resolve_index_path(root, "/tmp/elsewhere.mp4")

    output = tmp_path / "index.csv"
    original_rows = [{field: "" for field in CSV_FIELDS}]
    original_rows[0]["file_path"] = "retry.mp4"
    write_index(output, original_rows)
    replacement_rows = [{**original_rows[0], "status": "no_timestamp"}]
    write_index_atomically(output, replacement_rows)
    assert read_index(output) == replacement_rows

    output.write_text("file_path,actual_datetime\nretry.mp4,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        read_index(output)


def renamer_row(
    file_path: str,
    actual_datetime: str = "2025-11-20 14:23:05",
    resolution: str = "1920x1080",
    status: str = "ok",
) -> dict[str, str]:
    return {
        "file_path": file_path,
        "actual_datetime": actual_datetime,
        "resolution": resolution,
        "status": status,
        "ocr_text": "",
        "confidence": "",
        "error": "",
    }


def test_renamer_builds_chronological_paths_and_prefers_resolution(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    files = {
        "high-a.mp4": b"high-a",
        "high-b.mp4": b"high-b",
        "low.mp4": b"low",
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    rows = [
        renamer_row("high-a.mp4"),
        renamer_row("high-b.mp4"),
        renamer_row("low.mp4", resolution="640x360"),
    ]
    digests = {name: renamer.sha256_file(root / name) for name in files}

    plan = renamer.build_plan(root, tmp_path / "output", rows, digests)
    by_source = {move.source_path: move for move in plan}
    winners = sorted(
        move.destination_path
        for move in plan
        if move.classification == "canonical"
    )

    assert winners == [
        "2025/11/20/2025-11-20_14-23-05__01.mp4",
        "2025/11/20/2025-11-20_14-23-05__02.mp4",
    ]
    assert by_source["low.mp4"].classification == "lower-resolution"
    assert by_source["low.mp4"].destination_path.startswith(
        "_review/lower-resolution/"
    )


def test_renamer_classifies_exact_duplicates_and_metadata_conflicts(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4", "d.mp4"):
        (root / name).write_bytes(b"same" if name < "c.mp4" else b"conflict")
    rows = [
        renamer_row("a.mp4"),
        renamer_row("b.mp4"),
        renamer_row("c.mp4"),
        renamer_row("d.mp4", actual_datetime="2025-11-20 14:23:06"),
    ]
    digests = {row["file_path"]: renamer.sha256_file(root / row["file_path"]) for row in rows}

    plan = renamer.build_plan(root, tmp_path / "output", rows, digests)
    by_source = {move.source_path: move for move in plan}

    assert by_source["a.mp4"].classification == "canonical"
    assert by_source["b.mp4"].classification == "exact-duplicate"
    assert by_source["c.mp4"].classification == "metadata-conflict"
    assert by_source["d.mp4"].classification == "metadata-conflict"


def test_renamer_routes_unresolved_and_invalid_ok_rows_to_review(tmp_path: Path):
    rows = [
        renamer_row("blank.mp4", actual_datetime="", resolution="", status="no_timestamp"),
        renamer_row("invalid.mp4", resolution="1920-by-1080"),
    ]
    digests = {"blank.mp4": "a" * 64, "invalid.mp4": "b" * 64}

    plan = renamer.build_plan(tmp_path / "source", tmp_path / "output", rows, digests)
    by_source = {move.source_path: move for move in plan}

    assert by_source["blank.mp4"].classification == "unresolved"
    assert by_source["invalid.mp4"].classification == "metadata-conflict"


def test_renamer_requires_exact_index_coverage_and_safe_paths(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "indexed.mp4").write_bytes(b"indexed")
    (root / "extra.mp4").write_bytes(b"extra")

    with pytest.raises(ValueError, match="unindexed media"):
        renamer.validate_initial_archive(root, [renamer_row("indexed.mp4")])
    with pytest.raises(ValueError, match="duplicate CSV path"):
        renamer.validate_initial_archive(
            root,
            [renamer_row("indexed.mp4"), renamer_row("indexed.mp4")],
        )
    with pytest.raises(ValueError, match="escapes input root"):
        renamer.validate_initial_archive(root, [renamer_row("../outside.mp4")])


def test_renamer_keeps_png_and_mp4_timestamps_independent(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    files = {
        "video.mp4": b"video",
        "IMG_123.PNG": b"high photo",
        "IMG_124.PNG": b"low photo",
        "unrelated.png": b"ignored",
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    rows = [
        renamer_row("video.mp4", resolution="1920x1080"),
        renamer_row("IMG_123.PNG", resolution="3000x2000"),
        renamer_row("IMG_124.PNG", resolution="640x360"),
    ]
    paths = renamer.validate_initial_archive(root, rows)
    digests = {name: renamer.sha256_file(path) for name, path in paths.items()}

    plan = renamer.build_plan(root, tmp_path / "output", rows, digests)
    by_source = {move.source_path: move for move in plan}

    assert by_source["video.mp4"].destination_path.endswith(".mp4")
    assert by_source["video.mp4"].classification == "canonical"
    assert by_source["IMG_123.PNG"].destination_path.endswith(".png")
    assert by_source["IMG_123.PNG"].classification == "canonical"
    assert by_source["IMG_124.PNG"].destination_path.endswith(".png")
    assert by_source["IMG_124.PNG"].destination_path.startswith(
        "_review/lower-resolution/"
    )

    output = tmp_path / "output"
    renamer.apply_plan(root, output, rows, plan)
    assert (output / by_source["video.mp4"].destination_path).is_file()
    assert (output / by_source["IMG_123.PNG"].destination_path).is_file()
    assert (output / by_source["IMG_124.PNG"].destination_path).is_file()
    assert (root / "unrelated.png").read_bytes() == b"ignored"
    assert {
        row["file_path"] for row in read_index(output / "index.csv")
    } == {move.destination_path for move in plan}


def test_renamer_rejects_symlinked_sources(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    (root / "linked.mp4").symlink_to(target)

    with pytest.raises(ValueError, match="symlinked source path"):
        renamer.validate_initial_archive(root, [renamer_row("linked.mp4")])


def test_renamer_rejects_nested_roots_and_nonempty_destination(tmp_path: Path, capsys):
    root = tmp_path / "source"
    root.mkdir()
    (root / "video.mp4").write_bytes(b"video")
    index = tmp_path / "index.csv"
    write_index(index, [renamer_row("video.mp4")])

    with pytest.raises(ValueError, match="inside input"):
        renamer.validate_roots(root, root / "organized")

    output = tmp_path / "output"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep", encoding="utf-8")
    assert renamer.main([str(root), str(index), str(output)]) == 2
    assert "absent or empty" in capsys.readouterr().err
    assert (root / "video.mp4").is_file()


def test_renamer_dry_run_does_not_create_destination(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "video.mp4").write_bytes(b"video")
    index = tmp_path / "source-index.csv"
    write_index(index, [renamer_row("video.mp4")])
    output = tmp_path / "output"

    assert renamer.main([str(root), str(index), str(output)]) == 0
    assert not output.exists()
    assert (root / "video.mp4").read_bytes() == b"video"


def test_renamer_apply_writes_manifest_and_updated_index(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "video.mp4").write_bytes(b"video")
    index = tmp_path / "source-index.csv"
    write_index(index, [renamer_row("video.mp4")])
    output = tmp_path / "output"

    assert renamer.main([str(root), str(index), str(output), "--apply"]) == 0

    destination = output / "2025/11/20/2025-11-20_14-23-05.mp4"
    assert destination.read_bytes() == b"video"
    assert not (root / "video.mp4").exists()
    manifest = renamer.read_manifest(output / renamer.MANIFEST_NAME)
    assert manifest[0].move_status == "completed"
    assert manifest[0].destination_path == destination.relative_to(output).as_posix()
    assert read_index(output / "index.csv")[0]["file_path"] == manifest[0].destination_path
    assert read_index(index)[0]["file_path"] == "video.mp4"


def test_renamer_safe_move_never_overwrites_different_content(tmp_path: Path):
    source = tmp_path / "source.mp4"
    destination = tmp_path / "destination.mp4"
    source.write_bytes(b"source")
    destination.write_bytes(b"unrelated")

    with pytest.raises(RuntimeError, match="different content"):
        renamer.safe_move(source, destination, renamer.sha256_file(source))

    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"unrelated"


def test_renamer_safe_move_refuses_a_destination_created_during_publication(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source.mp4"
    destination = tmp_path / "destination.mp4"
    source.write_bytes(b"source")
    original_link = renamer.os.link

    def publish_competing_file(source_name: Path, destination_name: Path) -> None:
        destination.write_bytes(b"competing")
        original_link(source_name, destination_name)

    monkeypatch.setattr(renamer.os, "link", publish_competing_file)

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        renamer.safe_move(source, destination, renamer.sha256_file(source))

    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"competing"


def test_renamer_resumes_after_a_failed_move(tmp_path: Path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "a.mp4").write_bytes(b"a")
    (root / "b.mp4").write_bytes(b"b")
    index = tmp_path / "source-index.csv"
    write_index(
        index,
        [renamer_row("a.mp4"), renamer_row("b.mp4", actual_datetime="2025-11-20 14:23:06")],
    )
    output = tmp_path / "output"
    original_safe_move = renamer.safe_move
    calls = 0

    def fail_second_move(source: Path, destination: Path, expected_sha256: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure")
        original_safe_move(source, destination, expected_sha256)

    monkeypatch.setattr(renamer, "safe_move", fail_second_move)
    assert renamer.main([str(root), str(index), str(output), "--apply"]) == 2
    move_statuses = [
        move.move_status
        for move in renamer.read_manifest(output / renamer.MANIFEST_NAME)
    ]
    assert move_statuses == [
        "completed",
        "failed",
    ]

    monkeypatch.setattr(renamer, "safe_move", original_safe_move)
    assert renamer.main([str(root), str(index), str(output), "--apply"]) == 0
    assert all(
        move.move_status == "completed"
        for move in renamer.read_manifest(output / renamer.MANIFEST_NAME)
    )
    assert len(read_index(output / "index.csv")) == 2
