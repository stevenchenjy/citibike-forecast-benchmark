# Reproducibility Audit

Accepted run: `core_no_weather_e474ce35b5c7`.

- Source: official VP-RNN repository, commit `abf77f79fc64be75ae9102ec8d537f77ed9c5f8f`.
- Primary config: `configs/core.yaml`; weather is disabled.
- Input hashes, resolved package versions, operating system/hardware information, station list, fold ranges, random seeds, output hashes, runtime, warnings, and fallbacks are in `artifacts/run_manifests/core_no_weather_e474ce35b5c7.json`.
- Data/source schema and input-file hashes are recorded in `data/manifests/source_manifest.json` and [data audit](data_audit.md).
- Reproduction commands are `make setup`, `make data`, `make inspect`, `make test`, `make core`, `make decision`, and `make report`.

The source checkout, raw/interim/processed data, models, predictions, figures, tables, and manifests are intentionally gitignored because they are generated or large. The commands and content hashes make them auditable from a clean checkout.
