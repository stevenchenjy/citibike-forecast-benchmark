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

## 2026-07-28 — Milestone 4 complete: no-weather core forecast comparison

- Replaced the smoke-scale row-by-row feature construction with a vectorized origin-keyed feature panel. It keeps lag/rolling features at or before each origin, precomputes shifted historical/recent/seasonal baselines, and is covered by an explicit no-target-leakage test.
- Corrected rolling-origin boundaries after detecting that the initial constructor left the final 28 usable 2018 days unused. The accepted folds test `2018-10-08`–`2018-11-05`, `2018-11-06`–`2018-12-03`, and `2018-12-04`–`2018-12-31`, with expanding training windows and 28-day validation windows.
- Fixed global Poisson GLM regularization at `alpha=0.1` after feature standardization; the validation parameter search remains intentionally limited to LightGBM. This is documented in `references/METHOD_NOTES.md` and avoids multiplying the large CPU-only global fit by an unrequired GLM sweep.
- Discovered a required-support mismatch caused by seasonal-naive’s seven-day lag reaching an explicitly ambiguous DST observation for 60 fold-1 target rows. Preserved exact comparability by excluding those same rows from every model’s score, rather than imputing or pretending the seasonal value exists. The accepted run has 1,812,600 nonnegative, non-null predictions across all five models, 30 stations, three folds, two tracks, and separate pickup/return targets.
- Wrote complete forecast, station, runtime, and paired day-level bootstrap tables. The current core run manifest is `artifacts/run_manifests/core_no_weather_e474ce35b5c7.json`.
- Verified `uv run --no-editable --extra dev pytest`: 18 passed. Manual output checks confirmed identical model support by fold/track/target, zero negative predictions, zero null targets/predictions, and full required 2018 test coverage.

## 2026-07-28 — Milestones 5 and 8 complete for no-weather core

- Ran exact station-day inventory evaluation for every core day-ahead forecast under both aggregate within-hour orderings. Every model has 2,490 station-days per ordering (the 30 affected DST paths are excluded uniformly), and `decision_metrics.csv` reports failed pickups/returns, total failures, service level, oracle regret, within-one-oracle percentage, and starting-inventory error.
- Generated the core no-weather final report, ten required no-weather figures, complete forecast/station/bootstrap/decision/runtime/data-quality tables, and `RESULTS_SUMMARY.md`. The report records that LightGBM improved forecast MAE but did not yield the fewest aggregate simulated failures under the primary ordering.
- Added `reports/audit/leakage_audit.md` with concrete origin/target timestamp checks and `reports/audit/reproducibility_audit.md` with reproduction evidence. Enriched the run manifest with config content/hash, source and output hashes, packages, hardware, seeds, fold/station lists, runtime, warnings, and fallbacks.
- Added the declared `tabulate` dependency for reproducible Markdown table generation. Verified `make report` and `uv run --no-editable --extra dev pytest`: 18 passed.

## 2026-07-28 — Milestone 6 complete: observed-weather hindsight sensitivity

- Built the separate `hourly_panel_observed_weather_hindsight_upper_bound` profile and ran historical average, Poisson GLM, and LightGBM over the unchanged 30 stations, three expanding folds, separate pickup/return targets, and both two-hour/day-ahead tracks. Realized target-time fields from the audited VP-RNN weather source are the only added inputs.
- The accepted weather run is `observed_weather_hindsight_upper_bound_a4abb742757a`. It has 1,087,560 predictions; all are nonnegative and non-null, and the three models have identical target/origin support. Its manifest records the exact label, source, output hashes, and feature-vintage limitation.
- Added profile-local table materialization under `reports/runs/<run_id>/` so an auxiliary run cannot overwrite the canonical no-weather `reports/tables/` evidence. Re-materialized the cached core outputs without refitting and verified the canonical forecast table retains all five no-weather models.
- Added the required weather-sensitivity table and figure to the final report. Averaging pickups and returns, observed weather improves LightGBM MAE by 0.0731 at two hours and 0.4334 day-ahead; it improves the weaker GLM by 0.6656 and 1.6817 respectively. These values are explicitly presented only as a hindsight upper bound, not a live weather-forecast result.
- Verified `uv sync --extra dev --no-editable --reinstall-package citibike-forecast-benchmark`, `uv run --no-editable pytest -q` (19 passed), cached core/table regeneration, cached weather/table regeneration, and core report generation. The optional GRU gate is now open.

## 2026-07-28 — Milestone 7 complete: compact Poisson GRU

- Added a modern PyTorch compact global Poisson GRU with 24-hour origin-ending pickup/return sequences, a completeness channel for source-DST gaps, station embeddings, calendar/horizon context, Poisson negative log-likelihood, gradient clipping, fixed seeds, validation early stopping, CPU support, and MPS auto-selection when safe.
- The accepted separate no-weather run is `poisson_gru_no_weather_0eab625edb6e`: two models (historical average and GRU), 725,040 predictions, 30 stations, three folds, both forecast tracks, both targets, identical support, and no negative/null/duplicate predictions. The complete separate inventory replay contains 9,960 station-day/order rows.
- MPS completed a tiny smoke fit but exited natively (status 139) under the full workload. CPU GRU batches above 16 did the same. A direct full held-out CPU batch-16 probe succeeded, so the accepted run uses that deterministic safe fallback, five epochs maximum, and explicit 10,000/5,000 training/validation caps. This resource/runtime limitation is recorded rather than hidden or worked around by changing the core experiment.
- The GRU reaches 3.4463 two-hour MAE and 4.4192 day-ahead MAE. It does not beat LightGBM and has 71,172 primary-ordering inventory failures versus historical average’s 64,546. The final report retains this limited/negative result without post-hoc tuning.
- Added `make gru`, profile-isolated report/table routing, GRU tests, and a single-fold-safe smoke report path. Verified direct MPS/CPU probes, `uv run --no-editable pytest -q` (21 passed), GRU forecast integrity, GRU decision evaluation, core table restoration, and report generation.

## 2026-07-28 — Milestone 8 complete: final report and acceptance audit

- Exercised both documented clean-checkout workflows: `make reproduce-smoke` and `make reproduce-core`. Each re-ran source inspection and the full 21-test suite. The smoke workflow generated its isolated `reports/smoke_0c68da351c78_report.md`; the core workflow regenerated its canonical five-model tables, decision table, figures, and `reports/final_report.md` from saved artifacts without refitting models.
- Fixed the smoke report’s former fold-2 assumption and made report/table paths profile-local outside the explicitly named `core_no_weather` primary run. Fixed the source-audit generator so its canonical-panel validation section survives a repeat `make inspect`.
- Performed a final local-link audit of `reports/final_report.md`: no missing links. Recomputed every recorded output hash for `core_no_weather_e474ce35b5c7`, `observed_weather_hindsight_upper_bound_a4abb742757a`, and `poisson_gru_no_weather_0eab625edb6e`: no mismatches. Confirmed canonical forecast/station/bootstrap/runtime/decision tables contain only the five required no-weather models.
- Acceptance criteria are satisfied. `PROJECT_STATE.md` is `COMPLETE` and `NEXT_ACTION.md` records that no required action remains.

## 2026-07-28 — GitHub research presentation package

- Expanded the root README into a reader-facing research summary with headline results, methods, model definitions, decision evaluation, observed-weather caveat, and reproducibility commands.
- Added `docs/RESEARCH_REPORT.md`, a GitHub-readable paper-style report with data provenance, DST policy, leakage controls, chronological folds, model definitions, forecast and decision results, weather/GRU sections, limitations, and a visual appendix.
- Promoted the accepted core, weather, and GRU figures plus the report-linked CSV tables into a small versioned publication package. Smoke integration figures remain generated-only so the public narrative stays focused on the accepted core experiment.
- Regenerated figures with human-readable model and horizon labels; visual inspection, `pytest` (21 passed), Markdown-link checks, and `git diff --check` passed before publication.
