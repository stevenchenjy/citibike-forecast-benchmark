import pandas as pd

from citibike_benchmark.models.historical_average import expanding_historical_average
from citibike_benchmark.models.seasonal_naive import seasonal_naive


def test_historical_average_never_uses_current_row() -> None:
    frame = pd.DataFrame({"station_id": ["a", "a"], "timestamp": pd.date_range("2018-01-01", periods=2, freq="7D"), "day_of_week": [0, 0], "hour": [0, 0], "pickups": [4, 10]})
    prediction = expanding_historical_average(frame, "pickups")
    assert pd.isna(prediction.iloc[0])
    assert prediction.iloc[1] == 4


def test_seasonal_naive_uses_seven_day_lag() -> None:
    frame = pd.DataFrame({"station_id": ["a"] * 169, "timestamp": pd.date_range("2018-01-01", periods=169, freq="h"), "pickups": range(169)})
    prediction = seasonal_naive(frame, "pickups")
    assert prediction.iloc[168] == 0
