#!/usr/bin/env python3
"""Build a read-only CSV index from Xiaomi camera timestamp overlays."""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
from PIL import Image, ImageFilter, ImageOps

TIMESTAMP_RE = re.compile(
    r"(?P<year>\d{4})\s*/\s*(?P<month>\d{1,2})\s*/\s*(?P<day>\d{1,2})"
    + r"\s*(?P<hour>\d{1,2})\s*:\s*(?P<minute>\d{1,2})\s*:\s*(?P<second>\d{1,2})"
)
CAMERA_PHOTO_RE = re.compile(r"IMG_\d+\.PNG\Z", re.IGNORECASE)
CSV_FIELDS = (
    "file_path",
    "actual_datetime",
    "resolution",
    "status",
    "ocr_text",
    "confidence",
    "error",
)


@dataclass(frozen=True)
class OcrResult:
    actual_datetime: str = ""
    status: str = "no_timestamp"
    ocr_text: str = ""
    confidence: str = ""
    error: str = ""
    resolution: str = ""


def parse_timestamp(text: str) -> str | None:
    """Return an ISO-like camera-local timestamp when *text* contains one."""
    match = TIMESTAMP_RE.search(text)
    if not match:
        return None
    try:
        value = datetime(
            **{key: int(number) for key, number in match.groupdict().items()}
        )
    except ValueError:
        return None
    return value.strftime("%Y-%m-%d %H:%M:%S")


def crop_overlay(
    image: Image.Image, width_ratio: float = 0.40, height_ratio: float = 0.125
) -> Image.Image:
    """Crop the scalable upper-left region containing Xiaomi's timestamp overlay."""
    if not 0 < width_ratio <= 1 or not 0 < height_ratio <= 1:
        raise ValueError("crop ratios must be greater than zero and at most one")
    width, height = image.size
    return image.crop(
        (0, 0, max(1, round(width * width_ratio)), max(1, round(height * height_ratio)))
    )


def preprocessing_variants(image: Image.Image) -> dict[str, Image.Image]:
    """Produce high-contrast images for white timestamp text on a dark backing."""
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    enlarged = gray.resize((gray.width * 3, gray.height * 3), Image.Resampling.LANCZOS)
    variants = {"grayscale": enlarged}
    for threshold in (120, 170, 210):
        variants[f"threshold_{threshold}"] = enlarged.point(
            lambda pixel, t=threshold: 255 if pixel >= t else 0
        )
    return variants


def recovery_preprocessing_variants(frame: Image.Image) -> dict[str, Image.Image]:
    """Return tighter, enlarged OCR candidates for low-resolution timestamp overlays."""
    variants: dict[str, Image.Image] = {}
    presets = (
        ("tight_4x", 0.27, 0.09, 4),
        ("tight_6x", 0.27, 0.09, 6),
        ("medium_4x", 0.33, 0.11, 4),
    )
    for name, width_ratio, height_ratio, scale in presets:
        crop = crop_overlay(frame, width_ratio, height_ratio)
        enlarged = ImageOps.autocontrast(ImageOps.grayscale(crop)).resize(
            (crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS
        )
        variants[f"{name}_grayscale"] = enlarged
        variants[f"{name}_sharpen"] = enlarged.filter(
            ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2)
        )
        variants[f"{name}_threshold_150"] = enlarged.point(
            lambda pixel: 255 if pixel >= 150 else 0
        )
    return variants


def _run_tesseract(image: Image.Image, executable: str) -> tuple[str, str]:
    """Run Tesseract TSV output and return recognized text plus mean word confidence."""
    png = io.BytesIO()
    image.save(png, format="PNG", dpi=(300, 300))
    command = [
        executable,
        "stdin",
        "stdout",
        "--psm",
        "6",
        "-c",
        "tessedit_char_whitelist=0123456789/:",
        "tsv",
    ]
    completed = subprocess.run(
        command, input=png.getvalue(), capture_output=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(
            completed.stderr.decode("utf-8", errors="replace").strip()
            or "tesseract failed"
        )

    rows = list(
        csv.DictReader(
            completed.stdout.decode("utf-8", errors="replace").splitlines(),
            delimiter="\t",
        )
    )
    words = [row["text"].strip() for row in rows if row.get("text", "").strip()]
    confidences = [
        float(row["conf"]) for row in rows if row.get("conf", "") not in {"", "-1"}
    ]
    confidence = f"{sum(confidences) / len(confidences):.1f}" if confidences else ""
    return " ".join(words), confidence


def first_usable_frame(path: Path, max_frames: int = 10) -> Image.Image:
    """Decode the first image that can be converted to Pillow, without altering the video."""
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            for number, frame in enumerate(container.decode(stream), start=1):
                try:
                    return frame.to_image()
                except Exception:  # pragma: no cover - decoder-specific failures
                    if number >= max_frames:
                        break
    except (av.FFmpegError, OSError, IndexError) as exc:
        raise RuntimeError(str(exc)) from exc
    raise RuntimeError(f"no usable video frame in first {max_frames} decoded frames")


def is_camera_photo(path: Path) -> bool:
    """Return whether *path* is a Xiaomi camera PNG photo."""
    return bool(CAMERA_PHOTO_RE.fullmatch(path.name))


def is_supported_media_path(path: Path) -> bool:
    """Return whether *path* is an indexable Xiaomi video or camera photo."""
    return path.suffix.lower() == ".mp4" or is_camera_photo(path)


def first_usable_image(path: Path) -> Image.Image:
    """Load a camera photo or decode the first usable video frame."""
    if is_camera_photo(path):
        try:
            with Image.open(path) as image:
                image.load()
                return image.copy()
        except OSError as exc:
            raise RuntimeError(str(exc)) from exc
    return first_usable_frame(path)


def resolution_from_frame(frame: Image.Image) -> str:
    """Return a decoded frame's pixel dimensions in CSV-friendly form."""
    return f"{frame.width}x{frame.height}"


def extract_timestamp(path: Path, tesseract: str = "tesseract") -> OcrResult:
    """OCR a camera media overlay, returning reviewable failures rather than guesses."""
    if shutil.which(tesseract) is None:
        return OcrResult(
            status="ocr_error", error=f"Tesseract executable not found: {tesseract}"
        )
    try:
        frame = first_usable_image(path)
    except Exception as exc:
        return OcrResult(status="decode_error", error=str(exc))
    resolution = resolution_from_frame(frame)

    candidates: dict[str, tuple[str, str]] = {}
    texts: list[str] = []
    errors: list[str] = []
    for variant_name, image in preprocessing_variants(crop_overlay(frame)).items():
        try:
            text, confidence = _run_tesseract(image, tesseract)
        except Exception as exc:
            errors.append(f"{variant_name}: {exc}")
            continue
        texts.append(f"{variant_name}={text}")
        timestamp = parse_timestamp(text)
        if timestamp:
            candidates[timestamp] = (variant_name, confidence)

    ocr_text = " | ".join(texts)
    if len(candidates) == 1:
        timestamp, (_, confidence) = next(iter(candidates.items()))
        return OcrResult(timestamp, "ok", ocr_text, confidence, resolution=resolution)
    if len(candidates) > 1:
        return OcrResult(
            status="ambiguous",
            ocr_text=ocr_text,
            error="conflicting timestamps: " + ", ".join(sorted(candidates)),
            resolution=resolution,
        )
    if errors and not texts:
        return OcrResult(status="ocr_error", error="; ".join(errors), resolution=resolution)
    return OcrResult(
        status="no_timestamp",
        ocr_text=ocr_text,
        error="; ".join(errors),
        resolution=resolution,
    )


def extract_recovery_timestamp(path: Path, tesseract: str = "tesseract") -> OcrResult:
    """Retry OCR with low-resolution-focused crops and accept only a supported winner."""
    if shutil.which(tesseract) is None:
        return OcrResult(
            status="ocr_error", error=f"Tesseract executable not found: {tesseract}"
        )
    try:
        frame = first_usable_image(path)
    except Exception as exc:
        return OcrResult(status="decode_error", error=str(exc))
    resolution = resolution_from_frame(frame)

    candidates: dict[str, list[tuple[str, str]]] = {}
    texts: list[str] = []
    errors: list[str] = []
    for variant_name, image in recovery_preprocessing_variants(frame).items():
        try:
            text, confidence = _run_tesseract(image, tesseract)
        except Exception as exc:
            errors.append(f"{variant_name}: {exc}")
            continue
        texts.append(f"{variant_name}={text}")
        timestamp = parse_timestamp(text)
        if timestamp:
            candidates.setdefault(timestamp, []).append((variant_name, confidence))

    ocr_text = " | ".join(texts)
    if candidates:
        ranked = sorted(candidates.items(), key=lambda item: (-len(item[1]), item[0]))
        timestamp, votes = ranked[0]
        runner_up_count = len(ranked[1][1]) if len(ranked) > 1 else 0
        if len(votes) >= 2 and len(votes) > runner_up_count:
            confidences = [float(confidence) for _, confidence in votes if confidence]
            confidence = (
                f"{sum(confidences) / len(confidences):.1f}" if confidences else ""
            )
            return OcrResult(
                timestamp, "ok", ocr_text, confidence, resolution=resolution
            )
        candidate_summary = ", ".join(
            f"{value} ({len(value_votes)} votes)" for value, value_votes in ranked
        )
        return OcrResult(
            status="ambiguous",
            ocr_text=ocr_text,
            error="unsupported or tied timestamps: " + candidate_summary,
            resolution=resolution,
        )
    if errors and not texts:
        return OcrResult(status="ocr_error", error="; ".join(errors), resolution=resolution)
    return OcrResult(
        status="no_timestamp",
        ocr_text=ocr_text,
        error="; ".join(errors),
        resolution=resolution,
    )


def video_paths(root: Path) -> list[Path]:
    """Return MP4 videos below *root* for compatibility with existing callers."""
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".mp4"
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def media_paths(root: Path) -> list[Path]:
    """Return indexable videos and Xiaomi camera photos below *root*."""
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and is_supported_media_path(path)
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def index_videos(
    root: Path,
    paths: Iterable[Path],
    extractor: Callable[[Path], OcrResult] = extract_timestamp,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        result = extractor(path)
        rows.append({"file_path": path.relative_to(root).as_posix(), **result.__dict__})
    return rows


def write_index(output: Path, rows: Iterable[dict[str, str]]) -> None:
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_index_atomically(output: Path, rows: Iterable[dict[str, str]]) -> None:
    """Replace an existing index only after its complete replacement has been written."""
    output = output.resolve()
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=output.parent, delete=False
    ) as file:
        temporary = Path(file.name)
        try:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
            os.replace(temporary, output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def read_index(index_csv: Path) -> list[dict[str, str]]:
    with index_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(
                f"index must have exactly these columns: {', '.join(CSV_FIELDS)}"
            )
        return list(reader)


def resolve_index_path(root: Path, file_path: str) -> Path:
    """Resolve a CSV path while preventing it from escaping the declared archive root."""
    relative = Path(file_path)
    if relative.is_absolute():
        raise ValueError("file_path must be relative")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("file_path resolves outside the input root") from exc
    return candidate


def reprocess_rows(
    root: Path,
    rows: Iterable[dict[str, str]],
    extractor: Callable[[Path], OcrResult] = extract_recovery_timestamp,
) -> list[dict[str, str]]:
    """Retry only index rows that still have no authoritative timestamp."""
    updated_rows: list[dict[str, str]] = []
    for row in rows:
        updated = dict(row)
        if updated["actual_datetime"]:
            updated_rows.append(updated)
            continue
        try:
            path = resolve_index_path(root, updated["file_path"])
        except ValueError as exc:
            result = OcrResult(status="invalid_path", error=str(exc))
        else:
            if not path.is_file():
                result = OcrResult(
                    status="missing_file", error="media file does not exist"
                )
            else:
                result = extractor(path)
        updated.update(result.__dict__)
        updated_rows.append(updated)
    return updated_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR Xiaomi camera media overlays into a read-only CSV index."
    )
    parser.add_argument(
        "input_root", type=Path, help="archive directory to scan recursively"
    )
    parser.add_argument("output_csv", type=Path, help="CSV file to create or replace")
    parser.add_argument(
        "--tesseract",
        default="tesseract",
        help="Tesseract executable (default: tesseract)",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="retry blank timestamps from the existing output CSV in place",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.input_root.resolve()
    if not root.is_dir():
        print(f"input root is not a directory: {root}", file=sys.stderr)
        return 2
    if args.reprocess:
        if not args.output_csv.is_file():
            print(
                f"reprocess requires an existing CSV: {args.output_csv}",
                file=sys.stderr,
            )
            return 2
        try:
            existing_rows = read_index(args.output_csv)
            rows = reprocess_rows(
                root,
                existing_rows,
                lambda path: extract_recovery_timestamp(path, args.tesseract),
            )
            write_index_atomically(args.output_csv, rows)
        except (OSError, ValueError, csv.Error) as exc:
            print(f"could not reprocess {args.output_csv}: {exc}", file=sys.stderr)
            return 2
    else:
        paths = media_paths(root)
        rows = index_videos(
            root, paths, lambda path: extract_timestamp(path, args.tesseract)
        )
        write_index(args.output_csv, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    action = "reprocessed" if args.reprocess else "indexed"
    print(
        f"{action} {len(rows)} media file(s): "
        + ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
