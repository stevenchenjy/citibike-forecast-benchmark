import pandas as pd

from citibike_benchmark.features.lags import add_lags
from citibike_benchmark.features.rolling import add_rolling_features


def test_lag_uses_prior_observation_only() -> None:
    frame = pd.DataFrame({"station_id": ["a"] * 3, "timestamp": pd.date_range("2018-01-01", periods=3, freq="h"), "pickups": [1, 2, 100]})
    result = add_lags(frame, "pickups", [1])
    assert result.loc[2, "pickups_lag_1"] == 2


def test_rolling_excludes_target_timestamp() -> None:
    frame = pd.DataFrame({"station_id": ["a"] * 3, "timestamp": pd.date_range("2018-01-01", periods=3, freq="h"), "pickups": [1, 3, 100]})
    result = add_rolling_features(frame, "pickups", [2])
    assert result.loc[2, "pickups_rolling_mean_2"] == 2
