import pandas as pd
import pytest

from citibike_benchmark.data.splits import Fold, assert_chronological


def test_chronological_fold_is_accepted() -> None:
    stamp = pd.Timestamp("2018-01-01", tz="America/New_York")
    assert_chronological(Fold(0, stamp, stamp + pd.Timedelta("1D"), stamp + pd.Timedelta("2D"), stamp + pd.Timedelta("3D"), stamp + pd.Timedelta("4D"), stamp + pd.Timedelta("5D")))


def test_overlapping_fold_is_rejected() -> None:
    stamp = pd.Timestamp("2018-01-01", tz="America/New_York")
    with pytest.raises(ValueError, match="not strictly chronological"):
        assert_chronological(Fold(0, stamp, stamp + pd.Timedelta("2D"), stamp + pd.Timedelta("2D"), stamp + pd.Timedelta("3D"), stamp + pd.Timedelta("4D"), stamp + pd.Timedelta("5D")))
