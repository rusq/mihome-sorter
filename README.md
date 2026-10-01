# DVR timestamp indexer

This read-only utility scans Xiaomi Home video downloads and saved camera photos, then creates a CSV mapping each media file to the timestamp displayed in its first usable image. It does not rename, move, modify, or delete source media.

It indexes MP4 videos plus camera photos named `IMG_<integer>.PNG` (case-insensitive); unrelated PNG files are ignored.

The expected overlay is the Xiaomi camera form `MI 2025/10/31 15:04:05` in the upper-left corner. The reported time is camera-local, with no timezone conversion. The indexer uses a tight, proportionally scaled upper-left crop so the date line is not diluted by the scene below it.

## Requirements

- [`uv`](https://docs.astral.sh/uv/)
- [Tesseract OCR](https://tesseract-ocr.github.io/) on `PATH` (for example, `brew install tesseract` on macOS)

## Create an index

From this directory:

```sh
uv run dvr-timestamp-index . video-timestamps.csv
```

The output begins with exactly `file_path,actual_datetime,resolution,status`; paths are relative to the scanned directory, times use `YYYY-MM-DD HH:MM:SS`, and resolution is the first usable decoded frame's pixel dimensions (for example, `1920x1080`). Extra diagnostic columns keep OCR evidence and failures reviewable.

The indexer uses eight OCR workers by default. It writes a startup message plus
line-oriented `processing` and `completed` updates to stderr, so its CSV output
and final summary remain easy to capture separately. Lower the concurrency for
a slow disk, or raise it when the machine has capacity:

```sh
uv run dvr-timestamp-index . video-timestamps.csv --workers 4
```

Before using it to rename files, review rows whose `status` is not `ok`:

```sh
awk -F, 'NR == 1 || $3 != "ok"' video-timestamps.csv
```

Re-run after correcting a problem or changing the Tesseract executable:

```sh
uv run dvr-timestamp-index . video-timestamps.csv --tesseract /opt/homebrew/bin/tesseract
```

## Recover blank timestamps

After an initial scan, retry only rows whose `actual_datetime` is blank. This uses tighter, high-scale OCR crops intended for low-resolution (for example, 640×360) media, leaves already successful rows untouched, and atomically replaces the existing CSV only after the complete update has been written:

```sh
uv run dvr-timestamp-index . video-timestamps.csv --reprocess
```

The retry still does not infer dates from paths or filenames. Rows without a unique supported OCR result remain blank with an `ambiguous` or failure status for manual review.
Recovery uses the same default worker pool and stderr progress messages, but
only queues existing media whose timestamp is still blank.

## Reorganize an indexed archive

After reviewing the completed index, preview a chronological reorganization into
a separate destination root:

```sh
uv run dvr-timestamp-rename . video-timestamps.csv ../DVR-chronological
```

The preview is read-only. It validates that the CSV contains exactly one row for
every source MP4 and supported camera PNG, calculates SHA-256 checksums, and prints every proposed move.
Apply the reviewed plan explicitly:

```sh
uv run dvr-timestamp-rename . video-timestamps.csv ../DVR-chronological --apply
```

Canonical media use `YYYY/MM/DD/YYYY-MM-DD_HH-MM-SS.<extension>`. When multiple
non-identical files of the same media type have the same timestamp and highest
resolution, all are kept with deterministic `__01`, `__02`, and later suffixes.
MP4s and PNGs are compared independently, so a photo cannot displace a video
(or vice versa). Lower-resolution same-second files, exact duplicate copies,
unresolved rows, and conflicting metadata are retained below `_review/`; the
command never deletes them.

The destination receives `reorganization-manifest.csv`, which records every
original path, destination, checksum, classification, and move state. It is
updated after every move so the same command can resume an interrupted run.
Once all moves finish, destination `index.csv` contains the updated relative
paths. Keep the original CSV until the operation has completed and the new
archive has been verified. Existing source directories and unsupported files are
left untouched.

## Development checks

```sh
uv run --group dev pytest
```
