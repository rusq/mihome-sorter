from pathlib import Path

import pytest
from PIL import Image

import dvr_timestamp_indexer as indexer
from dvr_timestamp_indexer import (
    CSV_FIELDS,
    OcrResult,
    crop_overlay,
    index_videos,
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
