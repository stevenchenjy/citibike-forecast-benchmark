"""Four-week recent same-weekday/hour baseline."""
from __future__ import annotations

import pandas as pd


def recent_four_week_average(frame: pd.DataFrame, target: str, weeks: int = 4) -> pd.Series:
    ordered = frame.sort_values(["station_id", "timestamp"]).copy()
    lags = [ordered.groupby("station_id", sort=False)[target].shift(168 * week) for week in range(1, weeks + 1)]
    return pd.concat(lags, axis=1).mean(axis=1).reindex(frame.index)
