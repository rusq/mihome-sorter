#!/usr/bin/env python3
"""Reorganize an indexed DVR archive into a chronological directory tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dvr_timestamp_indexer import (
    is_supported_media_path,
    read_index,
    write_index_atomically,
)

MANIFEST_FIELDS = (
    "input_root",
    "output_root",
    "source_path",
    "destination_path",
    "classification",
    "reason",
    "actual_datetime",
    "resolution",
    "sha256",
    "move_status",
    "error",
)
MANIFEST_NAME = "reorganization-manifest.csv"
OUTPUT_INDEX_NAME = "index.csv"
RESOLUTION_RE = re.compile(r"(?P<width>[1-9]\d*)x(?P<height>[1-9]\d*)")


@dataclass
class PlannedMove:
    """One source video and its deterministic destination."""

    input_root: str
    output_root: str
    source_path: str
    destination_path: str
    classification: str
    reason: str
    actual_datetime: str
    resolution: str
    sha256: str
    move_status: str = "planned"
    error: str = ""

    def as_dict(self) -> dict[str, str]:
        return {field: str(getattr(self, field)) for field in MANIFEST_FIELDS}


@dataclass(frozen=True)
class Candidate:
    row: dict[str, str]
    sha256: str
    timestamp: datetime | None
    dimensions: tuple[int, int] | None

    @property
    def eligible(self) -> bool:
        return (
            self.row["status"] == "ok"
            and self.timestamp is not None
            and self.dimensions is not None
        )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_index_timestamp(value: str) -> datetime | None:
    """Parse an exact camera-local index timestamp."""
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return parsed if parsed.strftime("%Y-%m-%d %H:%M:%S") == value else None


def parse_resolution(value: str) -> tuple[int, int] | None:
    """Parse a positive WIDTHxHEIGHT resolution."""
    match = RESOLUTION_RE.fullmatch(value)
    if not match:
        return None
    return int(match["width"]), int(match["height"])


def _relative_media_paths(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and is_supported_media_path(path)
    }


def _contains_symlink(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _validated_source_paths(
    input_root: Path, rows: list[dict[str, str]], *, require_all_present: bool
) -> dict[str, Path]:
    errors: list[str] = []
    paths: dict[str, Path] = {}
    counts = Counter(row["file_path"] for row in rows)
    for name, count in sorted(counts.items()):
        if count != 1:
            errors.append(f"duplicate CSV path ({count} rows): {name}")

    for row in rows:
        name = row["file_path"]
        relative = Path(name)
        if not name:
            errors.append("CSV contains an empty file_path")
            continue
        if relative.is_absolute():
            errors.append(f"absolute CSV path: {name}")
            continue
        unresolved = input_root / relative
        if _contains_symlink(input_root, relative):
            errors.append(f"symlinked source path: {name}")
            continue
        resolved = unresolved.resolve()
        try:
            resolved.relative_to(input_root)
        except ValueError:
            errors.append(f"CSV path escapes input root: {name}")
            continue
        if not is_supported_media_path(relative):
            errors.append(f"CSV path is not supported camera media: {name}")
            continue
        if require_all_present and not resolved.is_file():
            errors.append(f"indexed media file is missing: {name}")
            continue
        paths[name] = resolved

    if errors:
        raise ValueError("\n".join(errors))
    return paths


def validate_roots(input_root: Path, output_root: Path) -> tuple[Path, Path]:
    """Resolve and validate disjoint input and output roots."""
    source = input_root.resolve()
    destination = output_root.resolve()
    if not source.is_dir():
        raise ValueError(f"input root is not a directory: {source}")
    if source == destination:
        raise ValueError("output root must be different from input root")
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("output root must not be inside input root")
    try:
        source.relative_to(destination)
    except ValueError:
        pass
    else:
        raise ValueError("input root must not be inside output root")
    return source, destination


def validate_initial_archive(
    input_root: Path, rows: list[dict[str, str]]
) -> dict[str, Path]:
    """Require an exact one-row-per-live-camera-media source inventory."""
    paths = _validated_source_paths(input_root, rows, require_all_present=True)
    indexed = set(paths)
    live = _relative_media_paths(input_root)
    missing = sorted(indexed - live)
    extra = sorted(live - indexed)
    errors = [*(f"indexed media file is missing: {path}" for path in missing)]
    errors.extend(f"unindexed media file exists: {path}" for path in extra)
    if errors:
        raise ValueError("\n".join(errors))
    return paths


def _canonical_destination(
    timestamp: datetime, ordinal: int | None, media_suffix: str
) -> str:
    stem = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    ordinal_suffix = f"__{ordinal:02d}" if ordinal is not None else ""
    return (
        f"{timestamp:%Y}/{timestamp:%m}/{timestamp:%d}/"
        f"{stem}{ordinal_suffix}{media_suffix}"
    )


def _review_destinations(candidates: list[Candidate], classification: str) -> dict[str, str]:
    by_digest: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_digest[candidate.sha256].append(candidate)
    destinations: dict[str, str] = {}
    for digest, members in sorted(by_digest.items()):
        ordered = sorted(members, key=lambda item: item.row["file_path"])
        for number, member in enumerate(ordered, start=1):
            suffix = f"__{number:02d}" if len(ordered) > 1 else ""
            media_suffix = Path(member.row["file_path"]).suffix.lower()
            destinations[member.row["file_path"]] = (
                f"_review/{classification}/{digest[:2]}/{digest}{suffix}{media_suffix}"
            )
    return destinations


def build_plan(
    input_root: Path,
    output_root: Path,
    rows: list[dict[str, str]],
    digests: dict[str, str],
) -> list[PlannedMove]:
    """Classify rows and return a deterministic move plan."""
    candidates = [
        Candidate(
            row=row,
            sha256=digests[row["file_path"]],
            timestamp=parse_index_timestamp(row["actual_datetime"]),
            dimensions=parse_resolution(row["resolution"]),
        )
        for row in rows
    ]
    by_digest: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_digest[candidate.sha256].append(candidate)

    classifications: dict[str, tuple[str, str]] = {}
    representatives: list[Candidate] = []
    for digest, members in sorted(by_digest.items()):
        eligible = [candidate for candidate in members if candidate.eligible]
        metadata = {
            (candidate.timestamp, candidate.dimensions) for candidate in eligible
        }
        if len(metadata) > 1:
            reason = "byte-identical files have conflicting valid timestamp or resolution metadata"
            for candidate in members:
                classifications[candidate.row["file_path"]] = (
                    "metadata-conflict",
                    reason,
                )
            continue
        if eligible:
            representative = min(eligible, key=lambda item: item.row["file_path"])
            representatives.append(representative)
            for candidate in members:
                if candidate is representative:
                    continue
                classifications[candidate.row["file_path"]] = (
                    "exact-duplicate",
                    f"byte-identical to {representative.row['file_path']}",
                )
            continue
        for candidate in members:
            if candidate.row["status"] == "ok":
                classification = "metadata-conflict"
                reason = "status is ok but timestamp or resolution is invalid"
            else:
                classification = "unresolved"
                reason = f"index status is {candidate.row['status'] or 'blank'}"
            classifications[candidate.row["file_path"]] = (classification, reason)

    by_timestamp: dict[tuple[datetime, str], list[Candidate]] = defaultdict(list)
    for candidate in representatives:
        assert candidate.timestamp is not None
        media_suffix = Path(candidate.row["file_path"]).suffix.lower()
        by_timestamp[(candidate.timestamp, media_suffix)].append(candidate)

    destinations: dict[str, str] = {}
    for (timestamp, media_suffix), members in sorted(by_timestamp.items()):
        max_pixels = max(
            candidate.dimensions[0] * candidate.dimensions[1]
            for candidate in members
            if candidate.dimensions is not None
        )
        winners = []
        for candidate in members:
            assert candidate.dimensions is not None
            pixels = candidate.dimensions[0] * candidate.dimensions[1]
            if pixels < max_pixels:
                classifications[candidate.row["file_path"]] = (
                    "lower-resolution",
                    f"lower resolution than another file at {candidate.row['actual_datetime']}",
                )
            else:
                winners.append(candidate)
        winners.sort(key=lambda item: (item.sha256, item.row["file_path"]))
        for number, winner in enumerate(winners, start=1):
            ordinal = number if len(winners) > 1 else None
            name = winner.row["file_path"]
            destinations[name] = _canonical_destination(timestamp, ordinal, media_suffix)
            classifications[name] = (
                "canonical",
                "highest-resolution file for timestamp"
                if len(winners) == 1
                else "tied highest-resolution file for timestamp",
            )

    for classification in (
        "exact-duplicate",
        "lower-resolution",
        "unresolved",
        "metadata-conflict",
    ):
        members = [
            candidate
            for candidate in candidates
            if classifications.get(candidate.row["file_path"], (None,))[0]
            == classification
        ]
        destinations.update(_review_destinations(members, classification))

    planned: list[PlannedMove] = []
    for candidate in candidates:
        name = candidate.row["file_path"]
        classification, reason = classifications[name]
        planned.append(
            PlannedMove(
                input_root=str(input_root),
                output_root=str(output_root),
                source_path=name,
                destination_path=destinations[name],
                classification=classification,
                reason=reason,
                actual_datetime=candidate.row["actual_datetime"],
                resolution=candidate.row["resolution"],
                sha256=candidate.sha256,
            )
        )
    return planned


def write_manifest(path: Path, moves: list[PlannedMove]) -> None:
    """Atomically write the complete recovery manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        temporary = Path(file.name)
        try:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(move.as_dict() for move in moves)
            file.flush()
            os.fsync(file.fileno())
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def read_manifest(path: Path) -> list[PlannedMove]:
    """Read and strictly validate a recovery manifest's columns."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(
                f"manifest must have exactly these columns: {', '.join(MANIFEST_FIELDS)}"
            )
        return [PlannedMove(**row) for row in reader]


def _static_move_data(move: PlannedMove) -> tuple[str, ...]:
    return tuple(
        getattr(move, field)
        for field in MANIFEST_FIELDS
        if field not in {"move_status", "error"}
    )


def load_resume_plan(
    input_root: Path,
    output_root: Path,
    rows: list[dict[str, str]],
    manifest_path: Path,
) -> list[PlannedMove]:
    """Validate and return an interrupted operation's manifest."""
    saved = read_manifest(manifest_path)
    if not saved:
        raise ValueError("resume manifest is empty")
    if any(
        move.input_root != str(input_root) or move.output_root != str(output_root)
        for move in saved
    ):
        raise ValueError("resume manifest belongs to different input or output roots")
    if len(saved) != len(rows) or {move.source_path for move in saved} != {
        row["file_path"] for row in rows
    }:
        raise ValueError("resume manifest does not match the input index")

    _validated_source_paths(input_root, rows, require_all_present=False)
    extra = _relative_media_paths(input_root) - {move.source_path for move in saved}
    if extra:
        raise ValueError(
            "unindexed media file exists during resume: " + ", ".join(sorted(extra))
        )
    digests = {move.source_path: move.sha256 for move in saved}
    expected = build_plan(input_root, output_root, rows, digests)
    saved_by_source = {move.source_path: move for move in saved}
    for move in expected:
        existing = saved_by_source[move.source_path]
        if _static_move_data(move) != _static_move_data(existing):
            raise ValueError(
                f"resume manifest plan does not match current index: {move.source_path}"
            )
        if existing.move_status not in {"planned", "completed", "failed"}:
            raise ValueError(
                f"invalid move_status for {existing.source_path}: {existing.move_status}"
            )
    return saved


def _safe_destination(root: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute():
        raise ValueError(f"absolute destination path: {relative_name}")
    destination = (root / relative).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"destination escapes output root: {relative_name}") from exc
    return destination


def safe_move(source: Path, destination: Path, expected_sha256: str) -> None:
    """Move one verified file without replacing an existing destination."""
    source_exists = source.is_file()
    destination_exists = destination.is_file()
    if destination_exists:
        if sha256_file(destination) != expected_sha256:
            raise RuntimeError(f"destination exists with different content: {destination}")
        if source_exists:
            if sha256_file(source) != expected_sha256:
                raise RuntimeError(f"source content changed: {source}")
            source.unlink()
        return
    if not source_exists:
        raise RuntimeError(f"source and destination are both missing: {source}")
    if sha256_file(source) != expected_sha256:
        raise RuntimeError(f"source content changed: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev == destination.parent.stat().st_dev:
        try:
            # link(2) atomically fails when another process has already
            # published this name.  rename(2) would replace that file.
            os.link(source, destination)
        except FileExistsError as exc:
            raise RuntimeError(f"refusing to overwrite destination: {destination}")
        source.unlink()
        return

    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            with source.open("rb") as source_file:
                shutil.copyfileobj(source_file, file, length=1024 * 1024)
            file.flush()
            os.fsync(file.fileno())
            shutil.copystat(source, temporary)
            if sha256_file(temporary) != expected_sha256:
                raise RuntimeError(f"copied content verification failed: {source}")
            if destination.exists():
                raise RuntimeError(f"refusing to overwrite destination: {destination}")
            os.link(temporary, destination)
            temporary.unlink()
            source.unlink()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def apply_plan(
    input_root: Path,
    output_root: Path,
    index_rows: list[dict[str, str]],
    moves: list[PlannedMove],
) -> None:
    """Apply or resume a plan and write the destination index when complete."""
    manifest_path = output_root / MANIFEST_NAME
    write_manifest(manifest_path, moves)
    for move in moves:
        source = input_root / move.source_path
        destination = _safe_destination(output_root, move.destination_path)
        try:
            safe_move(source, destination, move.sha256)
        except Exception as exc:
            move.move_status = "failed"
            move.error = str(exc)
            write_manifest(manifest_path, moves)
            raise
        move.move_status = "completed"
        move.error = ""
        write_manifest(manifest_path, moves)

    destinations = {move.source_path: move.destination_path for move in moves}
    updated_rows = [
        {**row, "file_path": destinations[row["file_path"]]} for row in index_rows
    ]
    write_index_atomically(output_root / OUTPUT_INDEX_NAME, updated_rows)


def _print_plan(moves: list[PlannedMove], *, resume: bool = False) -> None:
    counts = Counter(move.classification for move in moves)
    prefix = "resume plan" if resume else "dry-run plan"
    print(
        f"{prefix}: {len(moves)} media file(s): "
        + ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    )
    for move in moves:
        print(
            f"{move.classification}: {move.source_path} -> "
            f"{move.destination_path} ({move.reason})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely reorganize an indexed DVR archive by OCR timestamp."
    )
    parser.add_argument("input_root", type=Path, help="archive root used by the index")
    parser.add_argument("index_csv", type=Path, help="timestamp index CSV")
    parser.add_argument("output_root", type=Path, help="separate chronological archive")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="move files; without this flag, only print the complete plan",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_root, output_root = validate_roots(args.input_root, args.output_root)
        rows = read_index(args.index_csv)
        manifest_path = output_root / MANIFEST_NAME
        resume = manifest_path.is_file()
        if resume:
            moves = load_resume_plan(
                input_root, output_root, rows, manifest_path
            )
        else:
            if output_root.exists() and any(output_root.iterdir()):
                raise ValueError(
                    "output root must be absent or empty unless it contains "
                    "a valid resume manifest"
                )
            paths = validate_initial_archive(input_root, rows)
            digests = {
                name: sha256_file(path) for name, path in sorted(paths.items())
            }
            moves = build_plan(input_root, output_root, rows, digests)
        _print_plan(moves, resume=resume)
        if not args.apply:
            return 0
        apply_plan(input_root, output_root, rows, moves)
    except (OSError, ValueError, csv.Error, RuntimeError) as exc:
        print(f"could not reorganize archive: {exc}", file=sys.stderr)
        return 2

    print(f"moved {len(moves)} media file(s) into {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
