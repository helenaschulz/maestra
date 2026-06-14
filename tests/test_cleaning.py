import numpy as np
import pandas as pd
import pytest

from maestra import cleaning
from maestra.cleaning import fit_cleaning_plan, propose_cleaning_plan


@pytest.fixture
def train():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [10.0, np.nan, 30.0, np.nan],
            "city": ["B", "B", None, "H"],
            "target": [0, 1, 0, 1],
        }
    )


def _fit_transform(df, plan, target="target"):
    t = fit_cleaning_plan(df, plan, target)
    return t.transform(df), t.log


def test_drop_applies_and_logs(train):
    plan = {"columns_to_drop": [{"column": "id", "reason": "ID"}], "imputations": []}
    clean, log = _fit_transform(train, plan)
    assert "id" not in clean.columns
    assert any(line.startswith("DROP 'id'") for line in log)


def test_target_is_protected_from_drop_and_impute(train):
    plan = {
        "columns_to_drop": [{"column": "target", "reason": "oops"}],
        "imputations": [{"column": "target", "strategy": "median", "reason": "oops"}],
    }
    clean, log = _fit_transform(train, plan)
    assert "target" in clean.columns
    assert sum("ist Zielspalte" in line for line in log) == 2


def test_hallucinated_column_is_skipped_not_crashed(train):
    plan = {"columns_to_drop": [{"column": "ghost", "reason": "x"}], "imputations": []}
    clean, log = _fit_transform(train, plan)
    assert list(clean.columns) == list(train.columns)
    assert any("existiert nicht" in line for line in log)


def test_median_imputation_fills_all_missing(train):
    plan = {
        "columns_to_drop": [],
        "imputations": [{"column": "age", "strategy": "median", "reason": "num"}],
    }
    clean, _ = _fit_transform(train, plan)
    assert clean["age"].isna().sum() == 0
    assert clean["age"].tolist() == [10.0, 20.0, 30.0, 20.0]  # median of [10, 30] = 20


def test_imputation_is_fitted_on_train_only_no_leakage(train):
    """Regression: fill value is a train statistic, applied unchanged to the holdout.

    The holdout's own values must NOT influence the fill — otherwise test information
    leaks into the features.
    """
    plan = {
        "columns_to_drop": [],
        "imputations": [{"column": "age", "strategy": "median", "reason": "num"}],
    }
    transform = fit_cleaning_plan(train, plan, "target")  # train age median = 20.0

    holdout = pd.DataFrame({"id": [9], "age": [np.nan], "city": ["B"], "target": [1]})
    out = transform.transform(holdout)
    assert out["age"].tolist() == [20.0]  # train median, not derived from holdout


def test_most_frequent_imputation(train):
    plan = {
        "columns_to_drop": [],
        "imputations": [{"column": "city", "strategy": "most_frequent", "reason": "cat"}],
    }
    clean, _ = _fit_transform(train, plan)
    assert clean["city"].isna().sum() == 0
    assert clean.loc[2, "city"] == "B"


def test_constant_imputation_uses_fill_value(train):
    plan = {
        "columns_to_drop": [],
        "imputations": [
            {"column": "city", "strategy": "constant", "fill_value": "UNK", "reason": "c"}
        ],
    }
    clean, _ = _fit_transform(train, plan)
    assert clean.loc[2, "city"] == "UNK"


def test_unknown_strategy_is_skipped(train):
    plan = {
        "columns_to_drop": [],
        "imputations": [{"column": "age", "strategy": "bogus", "reason": "x"}],
    }
    clean, log = _fit_transform(train, plan)
    assert clean["age"].isna().sum() == 2  # untouched
    assert any("unbekannte Strategie" in line for line in log)


def test_numeric_strategy_on_text_column_is_skipped(train):
    plan = {
        "columns_to_drop": [],
        "imputations": [{"column": "city", "strategy": "mean", "reason": "x"}],
    }
    clean, log = _fit_transform(train, plan)
    assert clean["city"].isna().sum() == 1  # untouched
    assert any("passt nicht zum dtype" in line for line in log)


def test_transform_does_not_mutate_input(train):
    before = train.copy()
    plan = {
        "columns_to_drop": [{"column": "id", "reason": "x"}],
        "imputations": [{"column": "age", "strategy": "median", "reason": "x"}],
    }
    fit_cleaning_plan(train, plan, "target").transform(train)
    pd.testing.assert_frame_equal(train, before)


def test_propose_cleaning_plan_delegates_with_schema(monkeypatch):
    captured = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        return {"columns_to_drop": [], "imputations": [], "overall_rationale": "ok"}

    monkeypatch.setattr(cleaning, "call_structured", fake_call)
    out = propose_cleaning_plan("gpt-4o", {"n_rows": 1, "columns": []}, "target")

    assert out["overall_rationale"] == "ok"
    assert captured["model"] == "gpt-4o"
    assert captured["parameters_schema"] is cleaning.PLAN_SCHEMA
    assert "target" in captured["user_prompt"]
