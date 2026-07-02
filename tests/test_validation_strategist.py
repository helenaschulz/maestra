"""Tests for the Validation Strategist: the LLM proposes a fold strategy, deterministic code
verifies it, and the fold builders honour it. The LLM is never called (proposals are dicts)."""
import numpy as np
import pandas as pd

from maestra import validation
from maestra.validation_strategist import validate_fold_strategy


def _grouped_df(n_groups=10, rows_per_group=6):
    rng = np.random.default_rng(0)
    gid = np.repeat(np.arange(n_groups), rows_per_group)
    return pd.DataFrame({
        "customer_id": gid,
        "x": rng.normal(size=n_groups * rows_per_group),
        "y": (gid % 2),  # target depends on the entity — the leaky setup
    })


# --- fold builders -----------------------------------------------------------------

def test_group_folds_never_split_an_entity():
    df = _grouped_df()
    folds = validation._make_folds(df, "y", 3, seed=0, stratified=False, group_column="customer_id")
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        train_groups = set(df["customer_id"].iloc[train_idx])
        val_groups = set(df["customer_id"].iloc[val_idx])
        assert not (train_groups & val_groups)  # an entity lives in exactly one side


def test_time_folds_validate_strictly_later_than_train():
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=30, freq="D")[np.random.default_rng(1).permutation(30)],
        "x": range(30),
        "y": [0, 1] * 15,
    })
    folds = validation._make_folds(df, "y", 3, seed=0, stratified=False, time_column="ts")
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        assert df["ts"].iloc[train_idx].max() < df["ts"].iloc[val_idx].min()  # past -> future only


# --- deterministic verification of proposals ----------------------------------------

def test_group_proposal_verified(monkeypatch):
    df = _grouped_df()
    proposal = {"strategy": "group", "group_column": "customer_id",
                "rationale": "six rows per customer"}
    verified, log = validate_fold_strategy(proposal, df, "y")
    assert verified["strategy"] == "group" and verified["group_column"] == "customer_id"
    assert any("FOLDS group by 'customer_id'" in line for line in log)


def test_nonexistent_column_falls_back_to_random():
    df = _grouped_df()
    verified, log = validate_fold_strategy(
        {"strategy": "group", "group_column": "patient_id", "rationale": "r"}, df, "y")
    assert verified["strategy"] == "random" and verified["group_column"] is None
    assert any("fallback" in line for line in log)


def test_group_column_without_repeats_falls_back():
    df = pd.DataFrame({"rowid": range(10), "x": range(10), "y": [0, 1] * 5})
    verified, log = validate_fold_strategy(
        {"strategy": "group", "group_column": "rowid", "rationale": "r"}, df, "y")
    assert verified["strategy"] == "random"
    assert any("no repeated entities" in line for line in log)


def test_unparseable_time_column_falls_back():
    df = pd.DataFrame({"when": ["yesterday", "??", "soon", "later"], "y": [0, 1, 0, 1]})
    verified, log = validate_fold_strategy(
        {"strategy": "time", "time_column": "when", "rationale": "r"}, df, "y")
    assert verified["strategy"] == "random"
    assert any("not sortable" in line for line in log)


def test_leakage_warnings_are_logged_not_applied():
    df = _grouped_df()
    proposal = {"strategy": "random", "rationale": "iid",
                "leakage_warnings": [{"column": "x", "reason": "recorded after the outcome"}]}
    verified, log = validate_fold_strategy(proposal, df, "y")
    assert verified["strategy"] == "random"
    assert any("LEAKAGE WARNING 'x'" in line for line in log)
    assert "x" in df.columns  # advice only — nothing was dropped


# --- pipeline wiring (LLM + engine mocked) ------------------------------------------

def test_pipeline_threads_group_column_into_cv(monkeypatch):
    from maestra import pipeline
    from maestra.validation import CVResult

    df = _grouped_df()
    captured = {}

    def fake_cross_validate(df_, target, **kwargs):
        captured.update(kwargs)
        return CVResult("accuracy", "binary", [0.8, 0.8], 0.8, 0.0, 2, False, True)

    monkeypatch.setattr(pipeline, "propose_fold_strategy",
                        lambda *a, **k: {"strategy": "group", "group_column": "customer_id",
                                         "rationale": "entities repeat"})
    monkeypatch.setattr(pipeline, "propose_cleaning_plan",
                        lambda *a, **k: {"columns_to_drop": [], "imputations": []})
    monkeypatch.setattr(pipeline, "cross_validate", fake_cross_validate)
    monkeypatch.setattr(pipeline, "fit_predictor",
                        lambda *a, **k: __import__("maestra.engine", fromlist=["TrainingResult"]).TrainingResult(
                            "binary", "accuracy", pd.DataFrame(), {}))

    result = pipeline.run_pipeline(df, "y", model="m", test_size=0.2, time_limit=1, seed=0,
                                   model_dir="x", cv_folds=2, fold_advisor=True, use_fe=False)

    assert captured["group_column"] == "customer_id"          # the verified strategy reached the CV
    assert result.fold_strategy["strategy"] == "group"        # and is reported for the log
    assert any("FOLDS group" in line for line in result.fold_strategy["log"])


def test_fold_advisor_requires_cv():
    import pytest

    from maestra.pipeline import run_pipeline
    df = _grouped_df()
    with pytest.raises(ValueError, match="requires --cv"):
        run_pipeline(df, "y", model="m", test_size=0.2, time_limit=1, seed=0,
                     model_dir="x", fold_advisor=True)
