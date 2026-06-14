"""Tests for the agentic failure-diagnosis loop. The LLM and AutoGluon are mocked, so the
recovery logic is verified deterministically and offline — no real crash is coaxed."""
import pandas as pd
import pytest

from maestra import diagnosis, pipeline
from maestra.diagnosis import diagnose_failure
from maestra.engine import TrainingResult
from maestra.pipeline import PipelineError, run_pipeline

PARAMS = dict(model="m", test_size=0.25, time_limit=10, seed=0, model_dir="x", use_fe=False)


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


# --- Point 3: quality gate (success-but-weak revision) ------------------------------

def _training(val_score):
    return TrainingResult("binary", "accuracy", pd.DataFrame({"model": ["m"]}),
                          {"accuracy": val_score}, val_score=val_score)


def _patch_llm_run(monkeypatch, val_score, diag):
    """use_llm path: cleaning plan is a no-op; engine always returns the given val_score;
    diagnose_failure returns `diag` and counts its calls."""
    monkeypatch.setattr(pipeline, "propose_cleaning_plan",
                        lambda *a, **k: {"columns_to_drop": [], "imputations": []})
    monkeypatch.setattr(pipeline, "train_and_evaluate", lambda *a, **k: _training(val_score))
    calls = {"n": 0}

    def diagnose(*a, **k):
        calls["n"] += 1
        return diag

    monkeypatch.setattr(pipeline, "diagnose_failure", diagnose)
    return calls


def test_weak_metric_triggers_exactly_one_revision(df, monkeypatch):
    calls = _patch_llm_run(monkeypatch, val_score=0.50,
                           diag={"action": "revise_plan", "diagnosis": "weak",
                                 "new_plan": {"columns_to_drop": [], "imputations": []}})
    result = run_pipeline(df, "y", use_llm=True, max_attempts=3, revise_below=0.9,
                          model="m", test_size=0.25, time_limit=1, seed=0, model_dir="x", use_fe=False)
    assert calls["n"] == 1                      # revised once, not on every attempt
    assert result.attempts == 2                 # original + one retrain
    assert result.diagnosis_log[0]["trigger"] == "weak_metric"


def test_weak_metric_needs_attempt_budget(df, monkeypatch):
    calls = _patch_llm_run(monkeypatch, val_score=0.50,
                           diag={"action": "revise_plan", "diagnosis": "weak", "new_plan": {}})
    result = run_pipeline(df, "y", use_llm=True, max_attempts=1, revise_below=0.9,
                          model="m", test_size=0.25, time_limit=1, seed=0, model_dir="x", use_fe=False)
    assert calls["n"] == 0                       # no room to revise with max_attempts=1
    assert result.attempts == 1


def test_strong_metric_does_not_trigger(df, monkeypatch):
    calls = _patch_llm_run(monkeypatch, val_score=0.95,
                           diag={"action": "revise_plan", "diagnosis": "x", "new_plan": {}})
    result = run_pipeline(df, "y", use_llm=True, max_attempts=3, revise_below=0.9,
                          model="m", test_size=0.25, time_limit=1, seed=0, model_dir="x", use_fe=False)
    assert calls["n"] == 0                       # 0.95 >= 0.9 floor -> no revision
    assert result.attempts == 1


def test_weak_metric_give_up_accepts_run(df, monkeypatch):
    calls = _patch_llm_run(monkeypatch, val_score=0.50,
                           diag={"action": "give_up", "diagnosis": "hopeless"})
    result = run_pipeline(df, "y", use_llm=True, max_attempts=3, revise_below=0.9,
                          model="m", test_size=0.25, time_limit=1, seed=0, model_dir="x", use_fe=False)
    assert calls["n"] == 1                       # gate fired once
    assert result.attempts == 1                  # run accepted, not retrained
    assert result.training.val_score == 0.50
