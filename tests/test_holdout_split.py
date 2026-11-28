"""validation.holdout_split: the holdout path's outer split follows the fold strategy."""
import numpy as np
import pandas as pd
import pytest

from maestra.engine import split
from maestra.validation import holdout_split


def _frame(n=120):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "g": np.repeat(np.arange(12), n // 12),
        "date": pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
    })


def test_random_is_engine_split_unchanged():
    df = _frame()
    tr, ho, rec = holdout_split(df, 0.25, 7)
    etr, eho = split(df, 0.25, 7)
    assert tr.index.equals(etr.index) and ho.index.equals(eho.index)
    assert rec["strategy"] == "random" and rec["n_holdout"] == len(eho)


def test_group_holdout_is_group_disjoint_and_seeded():
    df = _frame()
    tr, ho, rec = holdout_split(df, 0.25, 1, group_column="g")
    assert not set(tr["g"]) & set(ho["g"])
    assert len(tr) + len(ho) == len(df)
    assert rec["strategy"] == "group" and "groups held out" in rec["note"]
    tr2, ho2, _ = holdout_split(df, 0.25, 1, group_column="g")
    assert ho.index.equals(ho2.index)                              # deterministic per seed


def test_time_holdout_is_strictly_later_and_keeps_ties_together():
    df = _frame()
    df = pd.concat([df, df.iloc[[-1, -2]]], ignore_index=True)     # duplicated late timestamps
    tr, ho, rec = holdout_split(df, 0.2, 0, time_column="date")
    assert pd.to_datetime(tr["date"]).max() < pd.to_datetime(ho["date"]).min()
    assert abs(len(ho) / len(df) - 0.2) < 0.05
    assert rec["strategy"] == "time"


def test_time_local_holds_out_the_tail_of_every_period():
    df = _frame(n=360)
    tr, ho, rec = holdout_split(df, 0.25, 0, time_column="date", period_column="month_of:date")
    d_tr, d_ho = pd.to_datetime(tr["date"]), pd.to_datetime(ho["date"])
    for month in sorted(d_tr.dt.month.unique()):
        assert d_tr[d_tr.dt.month == month].max() < d_ho[d_ho.dt.month == month].min()
    assert rec["strategy"] == "time_local"
    assert "month_of:date" not in tr.columns                       # period token never written


def test_degenerate_time_split_raises_instead_of_an_empty_side():
    df = pd.DataFrame({"date": ["2020-01-01"] * 10, "y": range(10)})
    with pytest.raises(ValueError, match="empty side"):
        holdout_split(df, 0.2, 0, time_column="date")
