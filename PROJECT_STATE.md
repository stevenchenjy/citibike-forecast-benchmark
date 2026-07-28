# Project State

Status: `IN_PROGRESS`

Current phase: Milestone 4 — statistical and tree-model core backtest

Completed milestones:

- Milestone 0 — repository and environment: complete and verified on 2026-07-28.
- Milestone 1 — data acquisition and audit: complete and verified on 2026-07-28.
- Milestone 2 — canonical panel: complete and verified on 2026-07-28.
- Milestone 3 — baseline backtest and smoke report: complete and verified on 2026-07-28.

Current constraints and decisions:

- The authoritative contract is `../citibike_model_comparison_codex_spec.md`, read in full on 2026-07-28.
- Python 3.11 is required; `uv` is the primary environment manager with `venv + pip` documented as a fallback.
- The official VP-RNN repository remains the required default source. No field mapping has been assumed before source-schema inspection.
- The official source is pinned at `abf77f79fc64be75ae9102ec8d537f77ed9c5f8f`; its 30 60-minute station demand files, matching raw event data, station capacities, weather file, schemas, hashes, and licenses are audited in `reports/audit/data_audit.md` and `data/manifests/source_manifest.json`.
- The source has 24 unzoned civil-hour labels on both 2018 DST transitions. Panel construction must use the documented explicit policy and mark affected rows incomplete; it must not silently treat those labels as unambiguous local instants.
- The primary comparison will exclude weather. Any later weather run will be named `observed_weather_hindsight_upper_bound`.

Last verified action: `make smoke` completed the no-weather five-station pipeline from cached panel through report generation. It produced 25,200 held-out forecast rows and 350 station-day inventory evaluations. The test suite passed 16 tests.
