# Leakage Audit

The accepted primary run is `core_no_weather_e474ce35b5c7`. It uses only the no-weather source panel and has three expanding chronological folds.

## Timestamp checks

For a two-hour target at `2018-12-04 14:00 America/New_York`, the origin is `2018-12-04 12:00`; lag 1 is the observed origin-hour demand and every rolling window ends at that timestamp. The target timestamp is never included in the feature frame.

For a day-ahead target at `2018-12-04 14:00 America/New_York`, the origin is `2018-12-03 23:00`. Its hourly target calendar fields are known at origin, but every demand lag/rolling feature is computed at or before 23:00 on the preceding local day. No same-day observed demand enters that forecast.

## Fold checks

The accepted tests are 2018-10-08–11-05, 2018-11-06–12-03, and 2018-12-04–12-31. Training ends before validation begins and validation ends before testing begins in every fold. Poisson GLM and LightGBM refit on training plus validation only after LightGBM selection; no test target is used for fitting or tuning.

Historical-average values are cumulative station/weekday/hour means shifted by one matching calendar occurrence. Recent and seasonal values are shifted by 168-hour weekly observations. This makes every baseline value earlier than both one/two-hour and day-ahead origins.

## DST support rule

The source's fall DST hour is explicitly incomplete. If seasonal naive's required weekly observation is ambiguous, that target is excluded from **every** model in the affected fold. This preserves identical support instead of comparing models on different rows. The accepted core support check confirms equality for every fold/track/target/model combination.

## Automated evidence

- `tests/test_lag_leakage.py` checks lag and rolling shifts.
- `tests/test_backtest_feature_availability.py` verifies origin precedes target and target fields are not model inputs.
- `tests/test_split_boundaries.py` verifies strict boundaries and final-period inclusion.
- `tests/test_baselines.py` verifies historical and seasonal baseline timing.
