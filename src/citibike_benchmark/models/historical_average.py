"""Leakage-safe expanding historical average baseline."""
from __future__ import annotations

import pandas as pd


def expanding_historical_average(frame: pd.DataFrame, target: str) -> pd.Series:
    """Mean of prior rows in each station/weekday/hour group only."""
    ordered = frame.sort_values(["timestamp", "station_id"]).copy()
    keys = ["station_id", "day_of_week", "hour"]
    prior_sum = ordered.groupby(keys, sort=False)[target].cumsum() - ordered[target]
    prior_count = ordered.groupby(keys, sort=False).cumcount()
    prediction = prior_sum / prior_count.replace(0, float("nan"))
    return prediction.reindex(frame.index)
