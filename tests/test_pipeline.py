"""Pipeline wiring tests. AutoGluon and the LLM are mocked so the suite is fast and
offline — we assert the conductor calls the right steps and threads data correctly."""
import pandas as pd
import pytest

from automl_agent import pipeline
from automl_agent.engine import TrainingResult
from automl_agent.pipeline import run_pipeline


@pytest.fixture
def df():
    return pd.DataFrame({"id": [1, 2, 3, 4], "f": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})


@pytest.fixture
def fake_training():
    return TrainingResult(
        problem_type="binary",
        eval_metric="accuracy",
        leaderboard=pd.DataFrame({"model": ["m"], "score_test": [0.9]}),
        metrics={"accuracy": 0.9},
    )


def _patch_engine(monkeypatch, fake_training):
    monkeypatch.setattr(
        pipeline, "train_and_evaluate", lambda *a, **k: fake_training
    )


class _FakePredictor:
    def predict(self, X):
        return pd.Series(["A"] * len(X), index=X.index)


def test_submission_built_from_test_set(df, monkeypatch):
    training = TrainingResult("binary", "accuracy", pd.DataFrame({"model": ["m"]}),
                              {"accuracy": 0.9}, predictor=_FakePredictor())
    monkeypatch.setattr(pipeline, "train_and_evaluate", lambda *a, **k: training)

    test_df = pd.DataFrame({"id": [101, 102, 103], "f": [9.0, 8.0, 7.0]})
    result = run_pipeline(df, "y", model="m", test_size=0.25, time_limit=1, seed=0,
                          model_dir="x", use_llm=False, test_df=test_df, id_col="id")

    sub = result.submission
    assert list(sub.columns) == ["id", "y"]
    assert sub["id"].tolist() == [101, 102, 103]   # ids preserved, in test order
    assert sub["y"].tolist() == ["A", "A", "A"]


def test_missing_id_col_raises(df, fake_training, monkeypatch):
    _patch_engine(monkeypatch, fake_training)
    test_df = pd.DataFrame({"row": [1], "f": [9.0]})  # no 'id' column
    with pytest.raises(pipeline.PipelineError, match="id column"):
        run_pipeline(df, "y", model="m", test_size=0.25, time_limit=1, seed=0,
                     model_dir="x", use_llm=False, test_df=test_df, id_col="id")


def test_unknown_target_raises(df):
    with pytest.raises(ValueError, match="not in CSV"):
        run_pipeline(df, "missing", model="m", test_size=0.25, time_limit=1, seed=0, model_dir="x")


def test_no_llm_skips_cleaning(df, fake_training, monkeypatch):
    _patch_engine(monkeypatch, fake_training)

    def boom(*a, **k):
        raise AssertionError("LLM must not be called with use_llm=False")

    monkeypatch.setattr(pipeline, "propose_cleaning_plan", boom)

    result = run_pipeline(
        df, "y", model="m", test_size=0.25, time_limit=1, seed=0, model_dir="x", use_llm=False
    )
    assert result.plan is None
    assert result.n_cols_before == result.n_cols_after == 3
    assert result.training.metrics["accuracy"] == 0.9


def test_llm_path_applies_plan_before_training(df, fake_training, monkeypatch):
    _patch_engine(monkeypatch, fake_training)
    monkeypatch.setattr(
        pipeline,
        "propose_cleaning_plan",
        lambda *a, **k: {"columns_to_drop": [{"column": "id", "reason": "ID"}], "imputations": []},
    )

    result = run_pipeline(
        df, "y", model="gpt-4o", test_size=0.25, time_limit=1, seed=0, model_dir="x"
    )
    assert result.plan is not None
    assert result.n_cols_before == 3
    assert result.n_cols_after == 2  # 'id' dropped
    assert any("DROP 'id'" in line for line in result.cleaning_log)
