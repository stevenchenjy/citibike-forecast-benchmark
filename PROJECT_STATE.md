# Project State

Status: `COMPLETE`

Current phase: Complete

Completed milestones:

- Milestone 0 — repository and environment: complete and verified on 2026-07-28.
- Milestone 1 — data acquisition and audit: complete and verified on 2026-07-28.
- Milestone 2 — canonical panel: complete and verified on 2026-07-28.
- Milestone 3 — baseline backtest and smoke report: complete and verified on 2026-07-28.
- Milestone 4 — statistical and tree-model core backtest: complete and verified on 2026-07-28.
- Milestone 5 — core decision-aware evaluation: complete and verified on 2026-07-28.
- Milestone 6 — observed-weather hindsight sensitivity: complete and verified on 2026-07-28.
- Milestone 7 — optional compact Poisson GRU: complete and verified on 2026-07-28.
- Milestone 8 — final report and audits: complete and verified on 2026-07-28.

Current constraints and decisions:

- The authoritative contract is `../citibike_model_comparison_codex_spec.md`, read in full on 2026-07-28.
- Python 3.11 is required; `uv` is the primary environment manager with `venv + pip` documented as a fallback.
- The official VP-RNN repository remains the required default source. No field mapping has been assumed before source-schema inspection.
- The official source is pinned at `abf77f79fc64be75ae9102ec8d537f77ed9c5f8f`; its 30 60-minute station demand files, matching raw event data, station capacities, weather file, schemas, hashes, and licenses are audited in `reports/audit/data_audit.md` and `data/manifests/source_manifest.json`.
- The source has 24 unzoned civil-hour labels on both 2018 DST transitions. Panel construction must use the documented explicit policy and mark affected rows incomplete; it must not silently treat those labels as unambiguous local instants.
- The primary comparison excludes weather. The separately saved `observed_weather_hindsight_upper_bound` run uses realized target-time source weather solely as a hindsight upper bound; it is not forecast-vintage or operationally available weather.
- Weather result integrity is verified: 1,087,560 nonnegative, non-null predictions cover the same 30 stations, folds, targets, and horizon tracks for historical average, Poisson GLM, and LightGBM. Its run-specific tables live under `reports/runs/observed_weather_hindsight_upper_bound_a4abb742757a/`; canonical `reports/tables/` remains the no-weather primary comparison.
- The gated `poisson_gru_no_weather_0eab625edb6e` run is separate and no-weather. It has 725,040 nonnegative, non-null predictions with exact historical-average support and a complete separate decision replay. Modern PyTorch MPS and large CPU GRU batches exited natively on this Mac, so the accepted result records the safe deterministic CPU batch-size-16 fallback and its explicit 10,000/5,000 train/validation caps.

Last verified action: both documented `make reproduce-*` workflows completed successfully. The smoke profile writes `reports/smoke_0c68da351c78_report.md`; the core profile restores the five-model canonical tables and final report. All local report links resolve, and every recorded output hash for the core, weather, and GRU manifests matches its file. The test suite passed 21 tests.
