# Project State

Status: `IN_PROGRESS`

Current phase: Milestone 1 — data acquisition and audit

Completed milestones:

- Milestone 0 — repository and environment: complete and verified on 2026-07-28.

Current constraints and decisions:

- The authoritative contract is `../citibike_model_comparison_codex_spec.md`, read in full on 2026-07-28.
- Python 3.11 is required; `uv` is the primary environment manager with `venv + pip` documented as a fallback.
- The official VP-RNN repository remains the required default source. No field mapping has been assumed before source-schema inspection.
- The primary comparison will exclude weather. Any later weather run will be named `observed_weather_hindsight_upper_bound`.

Last verified action: `uv run --extra dev pytest` passed 13 tests and `python -m citibike_benchmark.cli --help` passed. The Python 3.11 lockfile and local environment were created successfully.
