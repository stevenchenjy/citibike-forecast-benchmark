# Execution Log

## 2026-07-28 — Session start

- Read the complete 675-line governing specification before repository changes.
- Created the independent local Git repository `citibike-forecast-benchmark`.
- Began Milestone 0 scaffolding. Source parser assumptions are intentionally deferred until the official source schema is inspected.

## 2026-07-28 — Milestone 0 complete

- Added the specified repository structure, Git ignore rules and tracked placeholders, Python 3.11 pin, `uv.lock`, dependency definition, Makefile, base/smoke/core/model/horizon configuration, CLI skeleton, state files, source downloader, utilities, and initial contract tests.
- Replaced Hatchling with setuptools because the generated editable `.pth` was not processed by the installed Python runtime; setuptools produced a working editable Python 3.11 installation. This is an environment-packaging fix only, not a change to experiment design.
- Verified `uv run --extra dev pytest`: 13 passed. Verified `uv run python -m citibike_benchmark.cli --help`.
- Next milestone is source acquisition and schema audit. No source field mapping has been made yet.

## 2026-07-28 — Milestone 1 complete

- Shallow-cloned the official `DanieleGammelli/variational-poisson-rnn` source to `data/external/variational-poisson-rnn/` and pinned commit `abf77f79fc64be75ae9102ec8d537f77ed9c5f8f`.
- Inspected the schema before implementing mappings. The source has 30 60-minute demand-rate CSVs, each with 8,760 complete rows for 2018 and separate `count_pickup` / `count_return` targets. The raw `inventory_change` events validate the target signs and hourly totals within the 2018 demand scope.
- Wrote `data/manifests/source_download.json`, generated `data/manifests/source_manifest.json` (64 input hashes), and wrote `reports/audit/data_audit.md`.
- Found and documented two non-silent source limitations: every calendar date has 24 unzoned local hour labels across DST transitions; and three raw positive events are timestamped in 2019 outside the named 2018 demand scope. The canonical panel will retain an actual America/New_York grid while explicitly flagging affected DST rows incomplete and excluding those three out-of-scope events from exact replay.
- Verified `make data`, `make inspect`, and `uv run --extra dev pytest`: 14 passed.
