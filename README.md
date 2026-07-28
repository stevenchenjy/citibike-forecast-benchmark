# Citi Bike Forecast Benchmark

A CPU-first, reproducible benchmark for hourly station-level Citi Bike pickup and return forecasts and simulated inventory decisions.

The required source is the official open-source [variational-poisson-rnn repository](https://github.com/DanieleGammelli/variational-poisson-rnn). The legacy repository is used only as a data source and method reference; this project uses modern Python 3.11 dependencies.

## Quick start

```bash
uv sync --extra dev --no-editable
make data
make inspect
make test
make smoke
```

The `--no-editable` setting is deliberate: it avoids a local CPython `.pth` issue observed on the target Mac and ensures the documented direct `python -m citibike_benchmark.cli ...` commands resolve consistently. If `uv` is unavailable, create a Python 3.11 virtual environment and install with `python -m pip install '.[dev]'`.

On macOS the bootstrap script installs Homebrew `libomp` when necessary because the LightGBM macOS wheel requires it. Linux runners normally supply an OpenMP runtime through their system package manager.

The primary comparison excludes weather and evaluates seasonal naive, historical average, recent average, global Poisson GLM, and global LightGBM Poisson under strict chronological splits. Pickups and returns are separate targets. The weather sensitivity is explicitly named `observed_weather_hindsight_upper_bound`; observed future weather is not available in live operation.

The gated optional compact Poisson GRU can be reproduced after the core run with `uv sync --extra dev --extra deep --no-editable` followed by `make gru`. Its artifacts are kept separate from the primary five-model tables.

See [PROJECT_STATE.md](PROJECT_STATE.md) for the resumable work state and [NEXT_ACTION.md](NEXT_ACTION.md) for the one next executable action.
