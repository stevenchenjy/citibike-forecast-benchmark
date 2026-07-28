# Results Summary

## Core no-weather result

The accepted 30-station, three-fold no-weather run is `core_no_weather_e474ce35b5c7`.

- LightGBM Poisson has the lowest aggregate MAE: 2.9036 at two hours and 3.5136 day-ahead, versus historical average at 4.0896 for both tracks.
- Recent four-week average captures a small improvement over historical average (3.9816 MAE), but not the LightGBM gain.
- The fixed-regularization global Poisson GLM is worse than historical average in this run.
- Lower forecast MAE does not automatically lower simulated inventory failures: recent average has 64,360 aggregate failures under the primary ordering, while LightGBM has 67,902. This is an evaluation result, not a causal claim.
- The separate `observed_weather_hindsight_upper_bound` sensitivity used realized target-time source weather, never forecast-vintage weather. It improves mean LightGBM MAE by 0.0731 at two hours and 0.4334 day-ahead; it improves the weaker Poisson GLM more substantially. This is a hindsight upper bound, not an operational weather-forecast result. Archived forecast-vintage weather is needed before making a live-operation claim.

See [reports/final_report.md](reports/final_report.md) for tables, figures, decision metrics, limits, and uncertainty output.
