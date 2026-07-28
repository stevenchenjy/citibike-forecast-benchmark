# Citi Bike Forecast Benchmark — Core No-Weather Report

## Design

This primary comparison excludes weather. It uses the official VP-RNN 30-station source at 60-minute resolution, separate pickup and return counts, and three strict expanding chronological folds. The test windows are 2018-10-08–11-05, 2018-11-06–12-03, and 2018-12-04–12-31. The 60 source-DST-ambiguous rows are retained in the panel but excluded whenever their use would make a model support unequal.

## Forecast accuracy

| model              | track     |    mae |   rmse |   wape |   mae_improvement_vs_historical |
|:-------------------|:----------|-------:|-------:|-------:|--------------------------------:|
| lightgbm_poisson   | day_ahead | 3.5136 | 5.8788 | 0.4382 |                          0.5760 |
| recent_average     | day_ahead | 3.9816 | 6.9570 | 0.4966 |                          0.1080 |
| historical_average | day_ahead | 4.0896 | 6.9317 | 0.5101 |                          0.0000 |
| seasonal_naive     | day_ahead | 4.3287 | 7.4303 | 0.5399 |                         -0.2391 |
| poisson_glm        | day_ahead | 6.4753 | 9.7489 | 0.8077 |                         -2.3857 |
| lightgbm_poisson   | two_hour  | 2.9036 | 4.7288 | 0.3622 |                          1.1860 |
| recent_average     | two_hour  | 3.9816 | 6.9570 | 0.4966 |                          0.1080 |
| historical_average | two_hour  | 4.0896 | 6.9317 | 0.5101 |                          0.0000 |
| seasonal_naive     | two_hour  | 4.3287 | 7.4303 | 0.5399 |                         -0.2391 |
| poisson_glm        | two_hour  | 4.8428 | 8.2673 | 0.6040 |                         -0.7532 |

Negative MAE improvement means worse than historical average; positive means better. Recent average is the direct test of whether near-term history captures most obtainable improvement. Peak-period and station-level results are in the linked figures/tables; no causal interpretation is made.

## Paired uncertainty

| model            |   mean_mae_difference |   min_ci_low |   max_ci_high |
|:-----------------|----------------------:|-------------:|--------------:|
| lightgbm_poisson |               -0.8892 |     -15.8151 |        6.3117 |
| poisson_glm      |                3.0510 |     -16.5471 |       29.2837 |
| recent_average   |               -0.1816 |     -12.6340 |       15.7436 |
| seasonal_naive   |                0.2378 |     -14.7179 |       12.7052 |

Intervals are paired day-block bootstrap comparisons against historical average; inspect `bootstrap_comparisons.csv` for each target, fold, horizon, and track.

## Inventory decisions

| model              |   total_failures |   average_regret |   service_level |
|:-------------------|-----------------:|-----------------:|----------------:|
| historical_average |            64546 |           4.3069 |          0.9540 |
| lightgbm_poisson   |            67902 |           5.6639 |          0.9512 |
| poisson_glm        |            73899 |           8.0644 |          0.9466 |
| recent_average     |            64360 |           4.2362 |          0.9540 |
| seasonal_naive     |            65092 |           4.5361 |          0.9533 |

Each station-day searches every feasible starting inventory using the model’s day-ahead path, then replays realized aggregate hourly demand. Regret is relative to a realized-path oracle. Pickup-then-return is the main aggregate convention; reversed ordering is a reported sensitivity, not an operational assertion. Truck routing is outside scope.

## Limits and next work

Observed trips are treated as realized demand, so stockouts/capacity constraints can censor latent demand. The source’s repeated/nonexistent DST labels and three raw events outside the 2018 scope are documented in the data audit. Archived GBFS availability and forecast-vintage weather are future work.

## Observed-weather hindsight upper bound

`observed_weather_hindsight_upper_bound` joins the realized target-hour weather fields to each forecast target. It is a sensitivity analysis only: observed future weather was unavailable in live operation, is **not** forecast-vintage weather, and cannot be interpreted as an operational forecast-input result. Its gains are larger day-ahead than two-hour in this source; archived forecast-vintage weather remains required before making an operational weather claim.

| model              | track     |   observed_weather_hindsight_upper_bound_mae |   no_weather_mae |   mae_improvement_vs_no_weather |
|:-------------------|:----------|---------------------------------------------:|-----------------:|--------------------------------:|
| historical_average | day_ahead |                                       4.0896 |           4.0896 |                          0.0000 |
| lightgbm_poisson   | day_ahead |                                       3.0802 |           3.5136 |                          0.4334 |
| poisson_glm        | day_ahead |                                       4.7936 |           6.4753 |                          1.6817 |
| historical_average | two_hour  |                                       4.0896 |           4.0896 |                          0.0000 |
| lightgbm_poisson   | two_hour  |                                       2.8305 |           2.9036 |                          0.0731 |
| poisson_glm        | two_hour  |                                       4.1771 |           4.8428 |                          0.6657 |

## Optional compact Poisson GRU

This no-weather model was added only after the baseline, decision, report, and audit gates passed. It uses 24-hour origin-ending pickup/return histories, station embeddings, Poisson negative log-likelihood, fixed seeds, and validation early stopping. The capped run used at most 10,000 training and 5,000 validation examples per fit, 5 epochs, and 95.8 total fitting seconds.

| model              | track     |    mae |   rmse |   wape |   total_failures |   service_level |   average_regret |
|:-------------------|:----------|-------:|-------:|-------:|-----------------:|----------------:|-----------------:|
| historical_average | day_ahead | 4.0896 | 6.9317 | 0.5101 |            64546 |          0.9540 |           4.3069 |
| poisson_gru        | day_ahead | 4.4192 | 7.8537 | 0.5512 |            71172 |          0.9487 |           6.9728 |
| poisson_gru        | two_hour  | 3.4463 | 6.3320 | 0.4299 |            71172 |          0.9487 |           6.9728 |
| historical_average | two_hour  | 4.0896 | 6.9317 | 0.5101 |            64546 |          0.9540 |           4.3069 |

The compact GRU improves the two-hour MAE over historical average but does not beat LightGBM; it is worse day-ahead and has more primary-ordering simulated failures than historical average. It is retained as a negative/limited-capacity result rather than tuned until it wins. The installed PyTorch MPS backend, and CPU batches above 16, exited natively during the full workload; the accepted run therefore uses the tested deterministic CPU batch-size-16 fallback.

- [GRU forecast metrics](runs/poisson_gru_no_weather_0eab625edb6e/forecast_metrics.csv)
- [GRU bootstrap comparisons](runs/poisson_gru_no_weather_0eab625edb6e/bootstrap_comparisons.csv)
- [GRU decision metrics](runs/poisson_gru_no_weather_0eab625edb6e/decision_metrics.csv)
- [GRU runtime metrics](runs/poisson_gru_no_weather_0eab625edb6e/runtime_metrics.csv)
- [GRU compact summary](runs/poisson_gru_no_weather_0eab625edb6e/gru_summary.csv)

## Machine-readable outputs

- [Forecast metrics](tables/forecast_metrics.csv)
- [Station metrics](tables/station_metrics.csv)
- [Bootstrap comparisons](tables/bootstrap_comparisons.csv)
- [Decision metrics](tables/decision_metrics.csv)
- [Runtime metrics](tables/runtime_metrics.csv)
- [Data quality](tables/data_quality.csv)
- [Observed-weather hindsight sensitivity](tables/weather_sensitivity.csv)

## Figures

- [Figure: core_no_weather_e474ce35b5c7_overall_mae](figures/core_no_weather_e474ce35b5c7_overall_mae.png)
- [Figure: core_no_weather_e474ce35b5c7_targets](figures/core_no_weather_e474ce35b5c7_targets.png)
- [Figure: core_no_weather_e474ce35b5c7_peak_mae](figures/core_no_weather_e474ce35b5c7_peak_mae.png)
- [Figure: core_no_weather_e474ce35b5c7_station_heatmap](figures/core_no_weather_e474ce35b5c7_station_heatmap.png)
- [Figure: core_no_weather_e474ce35b5c7_decisions](figures/core_no_weather_e474ce35b5c7_decisions.png)
- [Figure: core_no_weather_e474ce35b5c7_runtime](figures/core_no_weather_e474ce35b5c7_runtime.png)
- [Figure: core_no_weather_e474ce35b5c7_station_128_forecast](figures/core_no_weather_e474ce35b5c7_station_128_forecast.png)
- [Figure: core_no_weather_e474ce35b5c7_station_402_forecast](figures/core_no_weather_e474ce35b5c7_station_402_forecast.png)
- [Figure: core_no_weather_e474ce35b5c7_station_519_forecast](figures/core_no_weather_e474ce35b5c7_station_519_forecast.png)
- [Figure: core_no_weather_e474ce35b5c7_lightgbm_importance](figures/core_no_weather_e474ce35b5c7_lightgbm_importance.png)
- [Figure: observed_weather_hindsight_upper_bound_weather_sensitivity](figures/observed_weather_hindsight_upper_bound_weather_sensitivity.png)
- [Figure: poisson_gru_no_weather_0eab625edb6e_comparison](figures/poisson_gru_no_weather_0eab625edb6e_comparison.png)
