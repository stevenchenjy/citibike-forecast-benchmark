# Citi Bike Forecast Benchmark

> **Short answer:** among the primary no-weather models, **LightGBM with a Poisson objective produces the most accurate pickup/return forecasts**. Better forecast MAE did **not**, however, automatically produce the fewest simulated station-level inventory failures.

A reproducible, CPU-first benchmark for hourly Citi Bike station demand forecasting and transparent inventory-decision evaluation. It uses modern Python 3.11 tooling and the official open-source [variational-poisson-rnn](https://github.com/DanieleGammelli/variational-poisson-rnn) dataset as a data source—not its legacy runtime environment.

## What question does this answer?

For a manageable network of 30 Citi Bike stations, how much do recent-demand and machine-learning models improve on a leakage-safe historical baseline—and do those gains reduce simulated pickup and return failures?

The benchmark predicts **pickups** and **returns** separately at two operational horizons:

1. **Two-hour ahead:** forecast the next one and two hourly counts at every origin.
2. **Day ahead:** forecast all 24 hourly counts for the following local calendar day from the prior day's final hour.

## Main result

The primary comparison excludes weather and uses the same 30 stations, chronological folds, target rows, and evaluation support for all five required models.

| Model | Two-hour MAE | Day-ahead MAE | Primary inventory failures* |
| --- | ---: | ---: | ---: |
| **LightGBM Poisson** | **2.9036** | **3.5136** | 67,902 |
| Recent four-week average | 3.9816 | 3.9816 | **64,360** |
| Historical average | 4.0896 | 4.0896 | 64,546 |
| Seasonal naive | 4.3287 | 4.3287 | 65,092 |
| Global Poisson GLM | 4.8428 | 6.4753 | 73,899 |

\*Total failed pickups plus failed returns under the documented `pickups_then_returns` aggregate hourly convention. Each model is evaluated over 2,490 station-days; reversed within-hour ordering is also reported as a sensitivity check.

### Interpretation

- **Best forecast model:** LightGBM Poisson wins both horizons, especially day-ahead forecasting.
- **Recent demand matters:** the simple recent four-week average improves modestly on the historical baseline, but does not reach LightGBM accuracy.
- **Accuracy is not the whole operational objective:** Recent Average has fewer simulated inventory failures than LightGBM under this transparent, daily starting-inventory rule. This is an evaluation finding, **not a causal claim** about deployments.
- **GRU does not win here:** the gated compact Poisson GRU improves two-hour MAE versus historical average (3.4463), but is worse day-ahead (4.4192) and does not beat LightGBM.

Read the complete result, uncertainty intervals, station-level outputs, and figures in the [final report](reports/final_report.md).

For a paper-style walkthrough of the research question, methods, results, visual appendix, and limitations, read [**Research Report**](docs/RESEARCH_REPORT.md).

| Forecast accuracy | Inventory outcome |
| --- | --- |
| ![Overall MAE by model and horizon](reports/figures/core_no_weather_e474ce35b5c7_overall_mae.png) | ![Decision failures and oracle regret](reports/figures/core_no_weather_e474ce35b5c7_decisions.png) |

Observed-weather sensitivity is displayed separately because it is a hindsight upper bound, not an operational model input:

![Observed-weather hindsight sensitivity](reports/figures/observed_weather_hindsight_upper_bound_weather_sensitivity.png)

## Experiment design

| Component | Design |
| --- | --- |
| Source data | Official VP-RNN repository, pinned at `abf77f79fc64be75ae9102ec8d537f77ed9c5f8f` |
| Network | 30 source-defined Citi Bike stations |
| Period and resolution | Complete usable 2018 period, 60-minute intervals, `America/New_York` |
| Targets | Separate hourly pickup and return counts |
| Splits | 3 strict expanding chronological folds; 28-day validation and 28-day test windows |
| Test windows | 2018-10-08–11-05, 2018-11-06–12-03, 2018-12-04–12-31 |
| Leakage policy | Every lag/rolling feature ends at the forecast origin; validation/test values never fit preprocessing or tune models |
| Uncertainty | Paired day-block bootstrap confidence intervals against Historical Average |
| Decision test | Choose a feasible beginning-of-day inventory from forecasts, replay realized hourly demand, and compare with a realized-path oracle |

The source labels DST civil hours without timezone offsets. The pipeline explicitly removes the nonexistent spring-forward hour, represents both fall-back occurrences, and marks the resulting 60 core rows incomplete rather than pretending they are unambiguous observations. Any scoring row that would make seasonal-naive support unequal is excluded uniformly for every model.

## What exactly is “Historical Average”?

It is **not** a same-calendar-date or past-year lookup. For a station, target type, weekday, and hour, it is the expanding mean of all matching observations available **before the forecast origin**.

For example, to forecast pickups at a station on Tuesday at 08:00, the model averages prior observed Tuesday-08:00 pickups at that station. It never uses the forecast day's value or future data. This is the principal leakage-safe seasonal baseline.

## Models compared

| Model | Prediction rule / inputs |
| --- | --- |
| Seasonal naive | Same station and hour, exactly seven days earlier |
| Historical average | Expanding station × weekday × hour historical mean |
| Recent average | Same station × weekday × hour over the latest four available weeks |
| Global Poisson GLM | Station identity, calendar, horizon, demand lags/rollups, balance, and capacity; train-only preprocessing |
| LightGBM Poisson | Global gradient-boosted Poisson model with restrained validation search |
| Optional Poisson GRU | Separate, no-weather compact PyTorch sequence model; added only after all core gates passed |

## Inventory-decision evaluation

Forecast accuracy alone is not treated as operational value. For each station-day, the benchmark:

1. uses the day-ahead pickup/return forecast to choose a feasible starting inventory;
2. replays realized hourly aggregate demand against station capacity;
3. reports failed pickups, failed returns, total failures, service level, and regret versus an oracle that knows the realized path;
4. repeats the test with reversed within-hour ordering as a sensitivity analysis.

This version deliberately excludes truck-routing optimization and does not infer latent demand hidden by real-world stockouts or full docks.

## Observed-weather sensitivity: an upper bound, not an operational result

Weather is absent from the primary comparison. Afterwards, the separate experiment named **`observed_weather_hindsight_upper_bound`** joins weather that was actually observed at the target time.

For LightGBM, observed weather lowers MAE from 2.9036 to 2.8305 at two hours (improvement 0.0731), and from 3.5136 to 3.0802 day-ahead (improvement 0.4334).

These numbers are **not** a live-weather-forecast result: actual future weather was unavailable at forecast time. Archived forecast-vintage weather would be required before making an operational claim.

## Reproduce

Requirements: Python 3.11 and [`uv`](https://docs.astral.sh/uv/). On macOS, the bootstrap setup can install `libomp`, which the LightGBM wheel needs.

```bash
uv sync --extra dev --no-editable
make data      # clone or verify the pinned source
make inspect   # inspect schema and write the source audit
make test
make smoke     # 5 stations, one split, full end-to-end integration check
make core      # 30 stations, three rolling folds
make decision
make report
```

For the documented clean-checkout workflows:

```bash
make reproduce-smoke
make reproduce-core
```

To reproduce the optional GRU after the core run:

```bash
uv sync --extra dev --extra deep --no-editable
make gru
```

`--no-editable` is deliberate: it avoids a local CPython `.pth` issue on the target Mac and makes the documented module commands resolve consistently. The GRU result is isolated from the primary five-model tables. On this Mac, PyTorch MPS and CPU GRU batches above 16 exited natively under the full workload, so the accepted GRU run uses a deterministic CPU batch-size-16 fallback with explicit training caps.

## Outputs and audit trail

- [Final report](reports/final_report.md): results, decision outcomes, uncertainty, limits, figures, and links to machine-readable outputs.
- [Results summary](RESULTS_SUMMARY.md): concise findings.
- [Source data audit](reports/audit/data_audit.md): inspected schemas, mappings, quality checks, DST policy, and source pin.
- [Leakage audit](reports/audit/leakage_audit.md): origin/target timestamp checks and feature-availability controls.
- [Reproducibility audit](reports/audit/reproducibility_audit.md): commands, versions, hashes, manifests, and run evidence.
- [Research report](docs/RESEARCH_REPORT.md): GitHub-readable methods, results, visual appendix, and limitations.
- `reports/figures/` and `reports/tables/`: the versioned publication package for the accepted core, observed-weather, and GRU results.
- `artifacts/predictions/`, `artifacts/models/`, and `artifacts/run_manifests/`: saved predictions, serialized models, and audit manifests (generated locally and intentionally gitignored).

The repository is complete and resumable; see [PROJECT_STATE.md](PROJECT_STATE.md) for the verified milestone state.
