"""Lag features which never read the target timestamp."""
from __future__ import annotations

import pandas as pd


def add_lags(frame: pd.DataFrame, target: str, lags: list[int]) -> pd.DataFrame:
    result = frame.sort_values(["station_id", "timestamp"]).copy()
    group = result.groupby("station_id", sort=False)[target]
    for lag in lags:
        result[f"{target}_lag_{lag}"] = group.shift(lag)
    return result
