# Repository Guidelines

## Project Structure & Module Organization

This repository is a read-only Xiaomi DVR timestamp indexer plus its source-video archive. Keep implementation files at the repository root:

- `dvr_timestamp_indexer.py` contains the CLI, video decoding, image preprocessing, OCR, CSV handling, and recovery logic.
- `tests/test_dvr_timestamp_indexer.py` contains Pytest coverage.
- `pyproject.toml` and `uv.lock` define the locked Python environment.
- `README.md` documents normal indexing and `--reprocess` recovery use.

Video directories and `*.mp4` files are input data. Never rename, move, modify, or delete them from this tool. Treat generated CSV indexes as user data, not test fixtures.

## Build, Test, and Development Commands

Use `uv`; Python 3.11 or later is required.

```sh
uv run --group dev pytest -q
uv run dvr-timestamp-index . index.csv
uv run dvr-timestamp-index . index.csv --reprocess
```

The first command runs all tests. The second scans every MP4 below the input root and creates a CSV. The recovery command updates only CSV rows with blank timestamps and atomically replaces the completed CSV.

Tesseract must be installed and available on `PATH`; use `--tesseract /path/to/tesseract` when necessary. Prefer `uv run --offline` only when the locked dependencies are already cached.

## Coding Style & Naming Conventions

Use Python with four-space indentation, type annotations for public helpers, concise docstrings, and standard-library-first dependencies. Keep functions focused: decoding, OCR variants, timestamp parsing, CSV validation, and file writes should remain separately testable.

Use `snake_case` for functions and variables, `PascalCase` for classes, and descriptive status values such as `ok`, `no_timestamp`, `ambiguous`, and `missing_file`. Never derive a timestamp from filenames or directory names; only accept a validated OCR timestamp.

## Testing Guidelines

Add tests beside related behavior in `tests/test_dvr_timestamp_indexer.py`, named `test_<behavior>`. Cover success and failure paths, especially malformed OCR text, unsafe CSV paths, ambiguous candidate votes, and atomic CSV replacement. Avoid tests that alter real videos or overwrite the archive’s `index.csv`; use `tmp_path`, fakes, and small synthetic images.

## Commit & Pull Request Guidelines

The current history contains only `Initial commit (OCR)`, so there is no established convention beyond short, descriptive subjects. Use imperative, scoped messages such as `ocr: improve low-resolution recovery`. Keep commits limited to source, tests, documentation, and lockfile changes. PRs should describe OCR behavior changes, list validation commands, and call out any effect on CSV statuses or timestamp acceptance rules.
