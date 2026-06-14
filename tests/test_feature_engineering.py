import numpy as np
import pandas as pd
import pytest

from maestra import feature_engineering as fe
from maestra.feature_engineering import fit_feature_plan, propose_feature_plan


@pytest.fixture
def train():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 0.0, 40.0],
            "d": ["2021-01-05", "2021-06-15", "2022-03-01", "2022-12-25"],
            "city": ["X", "Y", "X", "Z"],
            "target": [0, 1, 0, 1],
        }
    )


def _ft(df, features, target="target"):
    t = fit_feature_plan(df, {"features": features}, target)
    return t.transform(df), t.log


def test_log_transform_adds_column(train):
    out, _ = _ft(train, [{"op": "log_transform", "column": "a", "reason": "skew"}])
    assert np.allclose(out["a_log"], np.log1p([1.0, 2.0, 3.0, 4.0]))


def test_difference_and_ratio(train):
    out, _ = _ft(train, [
        {"op": "difference", "left": "b", "right": "a", "reason": "x"},
        {"op": "ratio", "numerator": "a", "denominator": "b", "reason": "x"},
    ])
    assert out["b_minus_a"].tolist() == [9.0, 18.0, -3.0, 36.0]
    assert out.loc[0, "a_per_b"] == 0.1
    assert pd.isna(out.loc[2, "a_per_b"])  # division by zero -> NaN, not a crash


def test_date_parts_extracts_and_drops_original(train):
    out, _ = _ft(train, [{"op": "date_parts", "column": "d",
                          "parts": ["year", "month", "weekday"], "reason": "x"}])
    assert "d" not in out.columns
    assert out["d_year"].tolist() == [2021, 2021, 2022, 2022]
    assert out["d_month"].tolist() == [1, 6, 3, 12]


def test_bin_edges_fitted_on_train_only_no_leakage(train):
    """Regression: bin edges come from train; a holdout is binned with those edges, not
    re-binned on its own distribution."""
    t = fit_feature_plan(train, {"features": [{"op": "bin", "column": "a", "n_bins": 2,
                                               "reason": "x"}]}, "target")
    # train 'a' = [1,2,3,4] -> median edge 2.5. Holdout values 3 and 4 are both above it.
    holdout = pd.DataFrame({"a": [3.0, 4.0], "b": [1.0, 2.0], "d": ["2021-01-01", "2021-01-02"],
                            "city": ["X", "Y"], "target": [0, 1]})
    out = t.transform(holdout)
    assert out["a_bin"].tolist() == [1, 1]  # train edge used; holdout-refit would give [0, 1]


def test_target_is_protected(train):
    out, log = _ft(train, [{"op": "log_transform", "column": "target", "reason": "x"}])
    assert "target_log" not in out.columns
    assert any("Zielspalte" in line for line in log)


def test_non_numeric_column_skipped(train):
    out, log = _ft(train, [{"op": "bin", "column": "city", "n_bins": 2, "reason": "x"}])
    assert "city_bin" not in out.columns
    assert any("nicht numerisch" in line for line in log)


def test_missing_column_skipped(train):
    out, log = _ft(train, [{"op": "log_transform", "column": "ghost", "reason": "x"}])
    assert list(out.columns) == list(train.columns)
    assert any("nicht vorhanden" in line for line in log)


def test_unknown_op_skipped(train):
    out, log = _ft(train, [{"op": "fourier", "column": "a", "reason": "x"}])
    assert list(out.columns) == list(train.columns)
    assert any("unbekannte Op" in line for line in log)


def test_new_column_collision_skipped(train):
    df = train.assign(a_log=99.0)
    out, log = _ft(df, [{"op": "log_transform", "column": "a", "reason": "x"}])
    assert (out["a_log"] == 99.0).all()  # existing column untouched
    assert any("existiert schon" in line for line in log)


def test_input_not_mutated(train):
    before = train.copy()
    fit_feature_plan(train, {"features": [{"op": "log_transform", "column": "a",
                                          "reason": "x"}]}, "target").transform(train)
    pd.testing.assert_frame_equal(train, before)


def test_propose_feature_plan_delegates_with_schema(monkeypatch):
    captured = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        return {"features": [], "overall_rationale": "ok"}

    monkeypatch.setattr(fe, "call_structured", fake_call)
    out = propose_feature_plan("gpt-4o", {"n_rows": 1, "columns": []}, "target")

    assert out["overall_rationale"] == "ok"
    assert captured["tool_name"] == "feature_plan"
    assert captured["parameters_schema"] is fe.FE_SCHEMA
