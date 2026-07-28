# Citi Bike Forecast Benchmark

A CPU-first, reproducible benchmark for hourly station-level Citi Bike pickup and return forecasts and simulated inventory decisions.

The required source is the official open-source [variational-poisson-rnn repository](https://github.com/DanieleGammelli/variational-poisson-rnn). The legacy repository is used only as a data source and method reference; this project uses modern Python 3.11 dependencies.

## Quick start

```bash
uv sync --extra dev
make data
make inspect
make test
make smoke
```

If `uv` is unavailable, create a Python 3.11 virtual environment and install the package with `python -m pip install -e '.[dev]'`.

The primary comparison excludes weather and evaluates seasonal naive, historical average, recent average, global Poisson GLM, and global LightGBM Poisson under strict chronological splits. Pickups and returns are separate targets. The weather sensitivity is explicitly named `observed_weather_hindsight_upper_bound`; observed future weather is not available in live operation.

See [PROJECT_STATE.md](PROJECT_STATE.md) for the resumable work state and [NEXT_ACTION.md](NEXT_ACTION.md) for the one next executable action.
