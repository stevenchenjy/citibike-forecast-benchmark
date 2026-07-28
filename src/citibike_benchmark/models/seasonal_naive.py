"""Seven-day same-station, same-hour baseline."""
from __future__ import annotations

import pandas as pd


def seasonal_naive(frame: pd.DataFrame, target: str, seasonal_lag_hours: int = 168) -> pd.Series:
    ordered = frame.sort_values(["station_id", "timestamp"])
    values = ordered.groupby("station_id", sort=False)[target].shift(seasonal_lag_hours)
    return values.reindex(frame.index)
