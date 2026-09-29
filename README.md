# DVR timestamp indexer

This read-only utility scans Xiaomi Home video downloads and creates a CSV mapping each video to the timestamp displayed in its first usable frame. It does not rename, move, modify, or delete videos.

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

Before using it to rename files, review rows whose `status` is not `ok`:

```sh
awk -F, 'NR == 1 || $3 != "ok"' video-timestamps.csv
```

Re-run after correcting a problem or changing the Tesseract executable:

```sh
uv run dvr-timestamp-index . video-timestamps.csv --tesseract /opt/homebrew/bin/tesseract
```

## Recover blank timestamps

After an initial scan, retry only rows whose `actual_datetime` is blank. This uses tighter, high-scale OCR crops intended for low-resolution (for example, 640×360) videos, leaves already successful rows untouched, and atomically replaces the existing CSV only after the complete update has been written:

```sh
uv run dvr-timestamp-index . video-timestamps.csv --reprocess
```

The retry still does not infer dates from paths or filenames. Rows without a unique supported OCR result remain blank with an `ambiguous` or failure status for manual review.

## Development checks

```sh
uv run --group dev pytest
```
