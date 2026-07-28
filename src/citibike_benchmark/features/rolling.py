"""Rolling features shifted before aggregation to prevent target leakage."""
from __future__ import annotations

import pandas as pd


def add_rolling_features(frame: pd.DataFrame, target: str, windows: list[int]) -> pd.DataFrame:
    result = frame.sort_values(["station_id", "timestamp"]).copy()
    shifted = result.groupby("station_id", sort=False)[target].shift(1)
    for window in windows:
        roll = shifted.groupby(result["station_id"], sort=False).rolling(window, min_periods=1)
        result[f"{target}_rolling_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        result[f"{target}_rolling_std_{window}"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
    return result
