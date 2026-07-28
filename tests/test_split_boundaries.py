import pandas as pd
import pytest

from citibike_benchmark.data.splits import Fold, assert_chronological
from citibike_benchmark.evaluation.backtest import make_day_folds


def test_chronological_fold_is_accepted() -> None:
    stamp = pd.Timestamp("2018-01-01", tz="America/New_York")
    assert_chronological(Fold(0, stamp, stamp + pd.Timedelta("1D"), stamp + pd.Timedelta("2D"), stamp + pd.Timedelta("3D"), stamp + pd.Timedelta("4D"), stamp + pd.Timedelta("5D")))


def test_overlapping_fold_is_rejected() -> None:
    stamp = pd.Timestamp("2018-01-01", tz="America/New_York")
    with pytest.raises(ValueError, match="not strictly chronological"):
        assert_chronological(Fold(0, stamp, stamp + pd.Timedelta("2D"), stamp + pd.Timedelta("2D"), stamp + pd.Timedelta("3D"), stamp + pd.Timedelta("4D"), stamp + pd.Timedelta("5D")))


def test_rolling_folds_use_the_final_available_test_day() -> None:
    days = pd.date_range("2018-01-01", periods=364, freq="D")
    panel = pd.DataFrame({"date": days.date, "data_complete": True})
    config = {"data": {}, "splits": {"folds": 3, "min_train_days": 120, "validation_days": 28, "test_days": 28}}
    folds = make_day_folds(panel, config)
    assert len(folds) == 3
    assert folds[-1].test_days[-1] == days[-1].date()
    assert len(folds[0].train_days) == 252
