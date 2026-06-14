"""Tests for the agentic failure-diagnosis loop. The LLM and AutoGluon are mocked, so the
recovery logic is verified deterministically and offline — no real crash is coaxed."""
import pandas as pd
import pytest

from maestra import diagnosis, pipeline
from maestra.diagnosis import diagnose_failure
from maestra.engine import TrainingResult
from maestra.pipeline import PipelineError, run_pipeline

PARAMS = dict(model="m", test_size=0.25, time_limit=10, seed=0, model_dir="x")


@pytest.fixture
def df():
    return pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [4, 3, 2, 1], "y": [0, 1, 0, 1]})


@pytest.fixture
def fake_training():
    return TrainingResult("binary", "accuracy", pd.DataFrame({"model": ["m"]}), {"accuracy": 0.9})


def _flaky_engine(monkeypatch, fail_times, fake_training):
    """Engine that raises RuntimeError the first ``fail_times`` calls, then succeeds."""
    state = {"n": 0}

    def engine(*a, **k):
        state["n"] += 1
        if state["n"] <= fail_times:
            raise RuntimeError(f"boom #{state['n']}")
        return fake_training

    monkeypatch.setattr(pipeline, "train_and_evaluate", engine)
    return state


def test_recovers_via_increase_time_limit(df, fake_training, monkeypatch):
    _flaky_engine(monkeypatch, fail_times=1, fake_training=fake_training)
    monkeypatch.setattr(
        pipeline,
        "diagnose_failure",
        lambda *a, **k: {"action": "increase_time_limit", "new_time_limit": 300, "diagnosis": "too short"},
    )
    result = run_pipeline(df, "y", use_llm=False, max_attempts=2, **PARAMS)
    assert result.attempts == 2
    assert result.training.metrics["accuracy"] == 0.9
    assert len(result.diagnosis_log) == 1


def test_revise_plan_fixes_empty_feature_matrix(df, fake_training, monkeypatch):
    # Attempt 1: plan drops every feature -> real _validate_trainable raises.
    monkeypatch.setattr(
        pipeline,
        "propose_cleaning_plan",
        lambda *a, **k: {"columns_to_drop": [{"column": "f1", "reason": "x"},
                                             {"column": "f2", "reason": "x"}], "imputations": []},
    )
    # Diagnosis: keep the features this time.
    monkeypatch.setattr(
        pipeline,
        "diagnose_failure",
        lambda *a, **k: {"action": "revise_plan", "diagnosis": "dropped all features",
                         "new_plan": {"columns_to_drop": [], "imputations": []}},
    )
    monkeypatch.setattr(pipeline, "train_and_evaluate", lambda *a, **k: fake_training)

    result = run_pipeline(df, "y", use_llm=True, max_attempts=2, **PARAMS)
    assert result.attempts == 2
    assert result.n_cols_after == 3  # features kept after revision
    assert result.plan["columns_to_drop"] == []


def test_give_up_raises_pipeline_error(df, fake_training, monkeypatch):
    _flaky_engine(monkeypatch, fail_times=99, fake_training=fake_training)
    monkeypatch.setattr(
        pipeline, "diagnose_failure", lambda *a, **k: {"action": "give_up", "diagnosis": "hopeless"}
    )
    with pytest.raises(PipelineError, match="gave up"):
        run_pipeline(df, "y", use_llm=False, max_attempts=3, **PARAMS)


def test_exhausted_budget_surfaces_original_error(df, fake_training, monkeypatch):
    _flaky_engine(monkeypatch, fail_times=99, fake_training=fake_training)
    monkeypatch.setattr(
        pipeline,
        "diagnose_failure",
        lambda *a, **k: {"action": "increase_time_limit", "new_time_limit": 999, "diagnosis": "x"},
    )
    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(df, "y", use_llm=False, max_attempts=2, **PARAMS)


def test_empty_feature_matrix_raises_without_loop(df, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "propose_cleaning_plan",
        lambda *a, **k: {"columns_to_drop": [{"column": "f1", "reason": "x"},
                                             {"column": "f2", "reason": "x"}], "imputations": []},
    )
    with pytest.raises(PipelineError, match="No feature columns"):
        run_pipeline(df, "y", use_llm=True, max_attempts=1, **PARAMS)


def test_diagnose_failure_delegates_with_schema(monkeypatch):
    captured = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        return {"diagnosis": "d", "action": "give_up"}

    monkeypatch.setattr(diagnosis, "call_structured", fake_call)
    out = diagnose_failure("gpt-4o", "Traceback: boom", profile={"columns": []},
                           plan=None, time_limit=10, target="y")

    assert out["action"] == "give_up"
    assert captured["tool_name"] == "diagnose_failure"
    assert captured["parameters_schema"] is diagnosis.DIAGNOSIS_SCHEMA
    assert "boom" in captured["user_prompt"]
