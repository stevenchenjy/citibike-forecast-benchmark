"""Chronological split helpers."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def assert_chronological(fold: Fold) -> None:
    if not (fold.train_start <= fold.train_end < fold.validation_start <= fold.validation_end < fold.test_start <= fold.test_end):
        raise ValueError(f"Fold {fold.fold} boundaries are not strictly chronological")
