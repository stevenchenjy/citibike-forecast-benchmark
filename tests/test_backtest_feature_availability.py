import pandas as pd

from citibike_benchmark.evaluation.backtest import _examples_for_days, _feature_columns, _prepare_feature_panel


def test_forecast_examples_use_only_origin_available_demand() -> None:
    timestamps = pd.date_range("2018-01-01", periods=72, freq="h", tz="America/New_York")
    panel = pd.DataFrame({
        "station_id": ["a"] * len(timestamps), "timestamp": timestamps, "date": timestamps.date,
        "hour": timestamps.hour, "day_of_week": timestamps.dayofweek, "is_weekend": timestamps.dayofweek >= 5,
        "pickups": range(len(timestamps)), "returns": range(100, 100 + len(timestamps)),
        "net_flow": [100] * len(timestamps), "station_capacity": [25] * len(timestamps), "data_complete": [True] * len(timestamps),
    })
    features = _prepare_feature_panel(panel)
    examples = _examples_for_days(features, (timestamps[-1].date(),), "two_hour")
    assert (examples["origin_timestamp"] < examples["timestamp"]).all()
    assert "actual_pickups" not in _feature_columns(examples)
    assert "actual_returns" not in _feature_columns(examples)
    # The final target count is deliberately larger than its origin count.
    final = examples.loc[examples["timestamp"] == timestamps[-1]]
    assert final["pickups_lag_1"].max() < panel.loc[panel["timestamp"] == timestamps[-1], "pickups"].iloc[0]
