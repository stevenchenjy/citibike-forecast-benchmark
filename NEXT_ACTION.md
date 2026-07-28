# Next Action

```bash
uv run --no-editable python -m citibike_benchmark.cli backtest --config configs/observed_weather_hindsight_upper_bound.yaml
```

Implement and run the strictly separate observed-weather sensitivity for historical/calendar where meaningful, Poisson GLM, and LightGBM. It must retain the core folds/stations, use observed weather only, and be labeled `observed_weather_hindsight_upper_bound` in every output and report statement.
