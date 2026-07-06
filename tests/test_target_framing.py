"""Tests for the target framing agent (M11): verifier fallbacks, original-space scoring,
and the transformed CV path. AutoGluon is mocked throughout, so the suite stays fast/offline.
The central guarantees: a defective proposal NEVER reaches the arbiter, and a transformed
trial CV is scored in ORIGINAL units with AutoGluon's sign convention (else the paired
comparison against the untransformed base would be meaningless)."""
import numpy as np
import pandas as pd
import pytest

from maestra import validation
from maestra.engine import TrainingResult
from maestra.pipeline import run_pipeline
from maestra.target_framing import (
    TARGET_FRAMING_SCHEMA,
    target_stats,
    validate_target_framing,
)
from maestra.validation import _ag_score, cross_validate


def _skewed_df(n=40):
    rng = np.random.default_rng(0)
    x = np.arange(n, dtype=float)
    return pd.DataFrame({"x": x, "y": np.expm1(0.1 * x) + rng.uniform(0, 0.1, n)})


# ---------------------------------------------------------------- verifier ---------------

def test_validate_none_proposal_is_no_transform():
    tt, log = validate_target_framing({"transform": "none", "rationale": "symmetric"},
                                      _skewed_df(), "y", "regression", "root_mean_squared_error")
    assert tt is None
    assert any("FRAMING none (LLM)" in line for line in log)


@pytest.mark.parametrize("problem_type, metric, df, reason_part", [
    ("binary", "accuracy", _skewed_df(), "regression only"),          # classification target
    ("regression", "spearmanr", _skewed_df(), "cannot be rescored"),  # unsupported metric
    ("regression", "root_mean_squared_error",
     pd.DataFrame({"x": [1.0, 2.0], "y": [-1.0, 3.0]}), "negative"),  # log1p undefined
    ("regression", "root_mean_squared_error",
     pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]}), "not numeric"),
])
def test_validate_falls_back_on_defects(problem_type, metric, df, reason_part):
    proposal = {"transform": "log1p", "rationale": "skewed"}
    tt, log = validate_target_framing(proposal, df, "y", problem_type, metric)
    assert tt is None
    assert any("fallback" in line and reason_part in line for line in log)


def test_validate_accepts_clean_log1p_and_roundtrips():
    tt, log = validate_target_framing({"transform": "log1p", "rationale": "long tail"},
                                      _skewed_df(), "y", "regression", "root_mean_squared_error")
    assert tt is not None and tt.name == "log1p"
    assert any("FRAMING log1p proposed" in line for line in log)
    s = pd.Series([0.0, 1.0, 100.0, 5000.0])
    assert tt.inverse(tt.forward(s)).tolist() == pytest.approx(s.tolist())


def test_schema_vocabulary_is_fixed():
    assert TARGET_FRAMING_SCHEMA["properties"]["transform"]["enum"] == ["none", "log1p"]
    assert "rationale" in TARGET_FRAMING_SCHEMA["required"]


def test_target_stats_reports_skew_signal():
    stats = target_stats(_skewed_df(), "y")
    assert stats["mean"] > stats["median"]  # the signal the LLM decides from
    assert stats["skewness"] > 1


# ---------------------------------------------------------- original-space scoring -------

def test_ag_score_matches_autogluon_sign_convention():
    y = pd.Series([1.0, 2.0, 3.0])
    perfect, off = pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 3.0, 4.0])
    assert _ag_score(y, perfect, "root_mean_squared_error") == 0.0
    assert _ag_score(y, off, "root_mean_squared_error") == pytest.approx(-1.0)  # errors negated
    assert _ag_score(y, off, "mean_absolute_error") == pytest.approx(-1.0)
    assert _ag_score(y, perfect, "r2") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="not supported"):
        _ag_score(y, off, "spearmanr")


class _FakeRegressor:
    """Predicts the true log-space relationship (log1p(y) ≈ 0.1·x), so inverted predictions
    land near the original-space truth — including on val rows the fold never saw."""

    class eval_metric:
        greater_is_better = True

    def predict(self, X):
        return pd.Series(0.1 * X["x"].to_numpy(dtype=float), index=X.index)


def test_cross_validate_with_transform_scores_in_original_space(monkeypatch):
    """Folds train in log space, but fold scores must be original-space rmse (negated).
    A log-space rmse here would be tiny (<1); the original-space score is orders larger."""
    df = _skewed_df(30)

    def fake_train_and_evaluate(train, val, target, time_limit, model_dir, eval_metric=None, presets=None, sample_weight=None):
        assert train[target].max() < 10  # proof the fold actually trained on log1p(y)
        return TrainingResult("regression", "root_mean_squared_error", pd.DataFrame(),
                              {"root_mean_squared_error": -0.001},  # log-space metric: a decoy
                              predictor=_FakeRegressor())

    monkeypatch.setattr(validation, "train_and_evaluate", fake_train_and_evaluate)
    tt, _ = validate_target_framing({"transform": "log1p", "rationale": "t"},
                                    df, "y", "regression", "root_mean_squared_error")
    cv = cross_validate(df, "y", cleaning_plan=None, feature_plan=None, model_dir="x",
                        time_limit=1, n_folds=3, seed=0, target_transform=tt)

    assert all(s <= 0 for s in cv.fold_scores)          # negated-error convention kept
    assert all(s != -0.001 for s in cv.fold_scores)      # NOT the decoy log-space metric
    # OOF predictions are stored in original space: same scale as y, far above log space.
    oof = cv.oof_pred.dropna().astype(float)
    assert oof.max() > 10


def test_cross_validate_transform_requires_predictor(monkeypatch):
    def fake_train_and_evaluate(train, val, target, time_limit, model_dir, eval_metric=None, presets=None, sample_weight=None):
        return TrainingResult("regression", "root_mean_squared_error", pd.DataFrame(), {})

    monkeypatch.setattr(validation, "train_and_evaluate", fake_train_and_evaluate)
    tt, _ = validate_target_framing({"transform": "log1p", "rationale": "t"},
                                    _skewed_df(), "y", "regression", "root_mean_squared_error")
    with pytest.raises(ValueError, match="fitted predictor"):
        cross_validate(_skewed_df(), "y", cleaning_plan=None, feature_plan=None,
                       model_dir="x", time_limit=1, n_folds=3, seed=0, target_transform=tt)


# ---------------------------------------------------------------- pipeline guard ---------

def test_target_framing_requires_cv():
    with pytest.raises(ValueError, match="requires --cv"):
        run_pipeline(_skewed_df(), "y", model="m", test_size=0.2, time_limit=1, seed=0,
                     model_dir="x", target_framing=True)
