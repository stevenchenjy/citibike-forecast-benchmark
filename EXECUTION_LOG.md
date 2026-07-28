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

## 2026-07-28 — Milestone 2 complete

- Implemented the source-specific canonical panel adapter after the source audit. It maps source pickup/return count fields separately, joins validated station capacity, creates profile-specific partitioned Parquet datasets, records cache manifests, and writes data-quality tables.
- Implemented and tested the explicit DST policy: source spring-forward 02:00 zero rows are removed because that local hour does not exist; source fall-back 01:00 rows are treated as ambiguous and paired with a synthetic second occurrence. Both fall rows are `data_complete=false`, so later model fitting and scoring can exclude them rather than turn an ambiguity into a purported observation.
- Built `configs/smoke.yaml`: 5 stations × 8,760 actual hours = 43,800 rows (10 incomplete DST rows). Built `configs/core.yaml`: 30 stations × 8,760 actual hours = 262,800 rows (60 incomplete DST rows). Both have zero duplicate station timestamps.
- Verified `uv run --extra dev pytest`: 16 passed.

## 2026-07-28 — Milestone 3 complete: smoke pipeline

- Implemented strict chronological forecast examples for the two-hour (`t+1`, `t+2`) and day-ahead (24-hour next-day path) tracks. Every demand-history feature is read at or before the forecast origin; target rows and DST-ambiguous rows are excluded from training and scoring.
- Ran the full no-weather smoke profile on five deterministic source stations and exactly 60 consecutive days: 46 training days, 7 validation days, and 7 test days. The run produced 25,200 held-out pickup/return predictions across seasonal naive, expanding historical average, recent four-week average, Poisson GLM, and tiny LightGBM Poisson.
- Added day-ahead inventory evaluation. It searches all feasible starting inventories, replays aggregate hourly demand, reports pickup and return failures, service level, oracle regret, starting-inventory error, and the reversed within-hour ordering sensitivity. The smoke evaluation contains 350 station-days (five models × five stations × seven days × two orderings).
- Generated the interim smoke report, forecast/station/runtime/decision tables, figures, saved predictions, serialized models, and run manifest. The report explicitly labels itself as a smoke-scale integration check rather than a core conclusion; weather remains disabled.
- Installed Homebrew `libomp` because the macOS LightGBM wheel could not load without it. Updated bootstrap to install it when needed. Updated `uv` setup to use a non-editable installation because the target runtime did not load source paths from editable `.pth` files reliably.
- Verified `make smoke` and `uv run --no-editable --extra dev pytest`: 16 passed.
