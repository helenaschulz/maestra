import numpy as np
import pandas as pd
import pytest

from automl_agent import cleaning
from automl_agent.cleaning import apply_cleaning_plan, propose_cleaning_plan


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [10.0, np.nan, 30.0, np.nan],
            "city": ["B", "B", None, "H"],
            "target": [0, 1, 0, 1],
        }
    )


def test_drop_applies_and_logs(df):
    plan = {"columns_to_drop": [{"column": "id", "reason": "ID"}], "imputations": []}
    clean, log = apply_cleaning_plan(df, plan, "target")
    assert "id" not in clean.columns
    assert any(line.startswith("DROP 'id'") for line in log)


def test_target_is_protected_from_drop_and_impute(df):
    plan = {
        "columns_to_drop": [{"column": "target", "reason": "oops"}],
        "imputations": [{"column": "target", "strategy": "median", "reason": "oops"}],
    }
    clean, log = apply_cleaning_plan(df, plan, "target")
    assert "target" in clean.columns
    assert sum("ist Zielspalte" in line for line in log) == 2


def test_hallucinated_column_is_skipped_not_crashed(df):
    plan = {"columns_to_drop": [{"column": "ghost", "reason": "x"}], "imputations": []}
    clean, log = apply_cleaning_plan(df, plan, "target")
    assert list(clean.columns) == list(df.columns)
    assert any("existiert nicht" in line for line in log)


def test_median_imputation_fills_all_missing(df):
    plan = {
        "columns_to_drop": [],
        "imputations": [{"column": "age", "strategy": "median", "reason": "num"}],
    }
    clean, _ = apply_cleaning_plan(df, plan, "target")
    assert clean["age"].isna().sum() == 0
    assert clean["age"].tolist() == [10.0, 20.0, 30.0, 20.0]  # median of [10,30] = 20


def test_most_frequent_imputation(df):
    plan = {
        "columns_to_drop": [],
        "imputations": [{"column": "city", "strategy": "most_frequent", "reason": "cat"}],
    }
    clean, _ = apply_cleaning_plan(df, plan, "target")
    assert clean["city"].isna().sum() == 0
    assert clean.loc[2, "city"] == "B"


def test_constant_imputation_uses_fill_value(df):
    plan = {
        "columns_to_drop": [],
        "imputations": [
            {"column": "city", "strategy": "constant", "fill_value": "UNK", "reason": "c"}
        ],
    }
    clean, _ = apply_cleaning_plan(df, plan, "target")
    assert clean.loc[2, "city"] == "UNK"


def test_unknown_strategy_is_skipped(df):
    plan = {
        "columns_to_drop": [],
        "imputations": [{"column": "age", "strategy": "bogus", "reason": "x"}],
    }
    clean, log = apply_cleaning_plan(df, plan, "target")
    assert clean["age"].isna().sum() == 2  # untouched
    assert any("unbekannte Strategie" in line for line in log)


def test_impute_skipped_when_no_missing(df):
    plan = {
        "columns_to_drop": [],
        "imputations": [{"column": "id", "strategy": "median", "reason": "x"}],
    }
    _, log = apply_cleaning_plan(df, plan, "target")
    assert any("keine fehlenden Werte" in line for line in log)


def test_input_dataframe_not_mutated(df):
    before = df.copy()
    apply_cleaning_plan(df, {"columns_to_drop": [{"column": "id", "reason": "x"}]}, "target")
    pd.testing.assert_frame_equal(df, before)


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
