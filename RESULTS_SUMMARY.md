# Results Summary

## Core no-weather result

The accepted 30-station, three-fold no-weather run is `core_no_weather_e474ce35b5c7`.

- LightGBM Poisson has the lowest aggregate MAE: 2.9036 at two hours and 3.5136 day-ahead, versus historical average at 4.0896 for both tracks.
- Recent four-week average captures a small improvement over historical average (3.9816 MAE), but not the LightGBM gain.
- The fixed-regularization global Poisson GLM is worse than historical average in this run.
- Lower forecast MAE does not automatically lower simulated inventory failures: recent average has 64,360 aggregate failures under the primary ordering, while LightGBM has 67,902. This is an evaluation result, not a causal claim.
- These findings exclude weather. The required `observed_weather_hindsight_upper_bound` sensitivity has not yet run and must not be interpreted as operational weather-forecast value.

See [reports/final_report.md](reports/final_report.md) for tables, figures, decision metrics, limits, and uncertainty output.
