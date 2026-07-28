# Citi Bike Forecast Benchmark Report

## smoke run

This is a smoke-scale, no-weather verification run, not a core finding. It uses five source-defined stations, 60 consecutive 2018 days, one strict chronological split (46 training, 7 validation, 7 test days), separate pickup and return targets, and both required horizon tracks. Weather is disabled.

The lowest average smoke MAE is **recent_average** (2.598); this result is only an integration check and must not be generalized to the 30-station core experiment.

The inventory table uses day-ahead prediction paths, searches every feasible starting inventory, replays realized aggregate hourly demand, and compares failures with an oracle. The reversed within-hour ordering is reported as a sensitivity result, not as an operational claim.

## Outputs

- [Forecast metrics](tables/forecast_metrics.csv)
- [Station metrics](tables/station_metrics.csv)
- [Runtime metrics](tables/runtime_metrics.csv)
- [Decision metrics](tables/decision_metrics.csv)
- [MAE figure](figures/smoke_0c68da351c78_mae.png)
- [Decision figure](figures/smoke_0c68da351c78_decisions.png)

The future weather sensitivity must be named `observed_weather_hindsight_upper_bound` and is not run or interpreted here. Observed future weather would not represent forecast-vintage live-operational input.
