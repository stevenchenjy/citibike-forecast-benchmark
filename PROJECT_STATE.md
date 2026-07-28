# Project State

Status: `IN_PROGRESS`

Current phase: Milestone 7 — optional compact Poisson GRU (gates passed)

Completed milestones:

- Milestone 0 — repository and environment: complete and verified on 2026-07-28.
- Milestone 1 — data acquisition and audit: complete and verified on 2026-07-28.
- Milestone 2 — canonical panel: complete and verified on 2026-07-28.
- Milestone 3 — baseline backtest and smoke report: complete and verified on 2026-07-28.
- Milestone 4 — statistical and tree-model core backtest: complete and verified on 2026-07-28.
- Milestone 5 — core decision-aware evaluation: complete and verified on 2026-07-28.
- Milestone 6 — observed-weather hindsight sensitivity: complete and verified on 2026-07-28.
- Milestone 8 — final report and audits: complete and verified on 2026-07-28, including the separate weather-sensitivity additions.

Current constraints and decisions:

- The authoritative contract is `../citibike_model_comparison_codex_spec.md`, read in full on 2026-07-28.
- Python 3.11 is required; `uv` is the primary environment manager with `venv + pip` documented as a fallback.
- The official VP-RNN repository remains the required default source. No field mapping has been assumed before source-schema inspection.
- The official source is pinned at `abf77f79fc64be75ae9102ec8d537f77ed9c5f8f`; its 30 60-minute station demand files, matching raw event data, station capacities, weather file, schemas, hashes, and licenses are audited in `reports/audit/data_audit.md` and `data/manifests/source_manifest.json`.
- The source has 24 unzoned civil-hour labels on both 2018 DST transitions. Panel construction must use the documented explicit policy and mark affected rows incomplete; it must not silently treat those labels as unambiguous local instants.
- The primary comparison excludes weather. The separately saved `observed_weather_hindsight_upper_bound` run uses realized target-time source weather solely as a hindsight upper bound; it is not forecast-vintage or operationally available weather.
- Weather result integrity is verified: 1,087,560 nonnegative, non-null predictions cover the same 30 stations, folds, targets, and horizon tracks for historical average, Poisson GLM, and LightGBM. Its run-specific tables live under `reports/runs/observed_weather_hindsight_upper_bound_a4abb742757a/`; canonical `reports/tables/` remains the no-weather primary comparison.

Last verified action: the weather sensitivity was materialized and the core report regenerated without refitting the accepted primary models. The report contains eleven figures including the weather hindsight plot, complete machine-readable core tables, separate weather tables, leakage and reproducibility audits, and enriched run manifests. The test suite passed 19 tests.
