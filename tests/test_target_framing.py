"""Tests for the target framing agent (M11): verifier fallbacks, original-space scoring,
and the transformed CV path. AutoGluon is mocked throughout, so the suite stays fast/offline.
The central guarantees: a defective proposal NEVER reaches the arbiter, and a transformed
trial CV is scored in ORIGINAL units with AutoGluon's sign convention (else the paired
comparison against the untransformed base would be meaningless)."""
import numpy as np
import pandas as pd
import pytest

from maestra import pipeline, validation
from maestra.engine import TrainingResult
from maestra.pipeline import run_pipeline
from maestra.target_framing import (
    FRAMING_METRICS,
    SUPPORTED_METRICS,
    TARGET_FRAMING_SCHEMA,
    target_stats,
    validate_target_framing,
)
from maestra.validation import (
    CVResult,
    _ag_score,
    cross_validate,
    improves_beyond_noise,
    rescore_cv,
)


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


# ------------------------------------------ arbiter metric (F2 Messung 4, Finding 2) -----

def test_ag_score_rmsle_value_and_clip():
    y = pd.Series([0.0, np.e - 1])                       # log1p(y) = [0, 1]
    assert _ag_score(y, y, "root_mean_squared_log_error") == pytest.approx(0.0)
    assert _ag_score(y, pd.Series([0.0, 0.0]), "root_mean_squared_log_error") == pytest.approx(
        -np.sqrt(0.5))
    # a negative prediction is clipped at 0 (as a non-negative submission would be), not NaN
    assert _ag_score(y, pd.Series([-5.0, np.e - 1]), "root_mean_squared_log_error") == 0.0


def _y_and_folds():
    """3 folds of a heavy-tailed target, rows labelled 0..11 in fold order."""
    y = pd.Series([1.0, 10.0, 100.0, 1000.0] * 3) * pd.Series([1.0] * 4 + [1.5] * 4 + [2.0] * 4)
    folds = [y.index[i * 4:(i + 1) * 4] for i in range(3)]
    return y, folds


def _cv_from_oof(y, folds, oof, metric="root_mean_squared_error"):
    """A CVResult whose fold scores are the engine's own (here: raw RMSE) from ``oof``."""
    scores = [_ag_score(y.loc[i], oof.loc[i], metric) for i in folds]
    return CVResult(metric, "regression", scores, float(np.mean(scores)), float(np.std(scores)),
                    len(folds), False, True, oof_pred=oof.astype(object), fold_val_index=folds)


def _absolute_vs_relative_arms():
    """Base: small ABSOLUTE errors, huge relative ones on small rows (good RMSE, bad RMSLE).
    Trial: a uniform +10% RELATIVE error (worse RMSE on the tail, far better RMSLE) -- the
    Finding-2 shape: which arm wins depends only on the metric."""
    y, folds = _y_and_folds()
    base = _cv_from_oof(y, folds, y + pd.Series([20.0, 20.0, 10.0, 0.0] * 3))
    trial = _cv_from_oof(y, folds, y * 1.1)
    return y, base, trial


def test_rescore_cv_recomputes_per_fold_in_the_new_metric():
    y, base, _ = _absolute_vs_relative_arms()
    r = rescore_cv(base, y, "root_mean_squared_log_error")
    assert r.eval_metric == "root_mean_squared_log_error" and r.greater_is_better
    assert r.n_folds == base.n_folds and len(r.fold_scores) == 3        # still paired per fold
    expected = [_ag_score(y.loc[i], base.oof_pred.loc[i].astype(float),
                          "root_mean_squared_log_error") for i in base.fold_val_index]
    assert r.fold_scores == pytest.approx(expected)
    assert base.eval_metric == "root_mean_squared_error"               # input left untouched


def test_rescore_cv_requires_oof_and_fold_indices():
    cv = CVResult("root_mean_squared_error", "regression", [-1.0, -1.0], -1.0, 0.0, 2, False)
    with pytest.raises(ValueError, match="out-of-fold"):
        rescore_cv(cv, pd.Series([1.0, 2.0]), "root_mean_squared_log_error")


def test_the_arbiter_verdict_flips_with_the_metric():
    """The mechanism of Finding 2 in miniature: same folds, same predictions, only the metric
    differs, and the accept rule's verdict flips."""
    y, base, trial = _absolute_vs_relative_arms()
    accepted_rmse, _ = improves_beyond_noise(base=base, trial=trial)
    rmsle = "root_mean_squared_log_error"
    accepted_rmsle, _ = improves_beyond_noise(base=rescore_cv(base, y, rmsle),
                                              trial=rescore_cv(trial, y, rmsle))
    assert not accepted_rmse and accepted_rmsle


def test_validate_rejects_an_unknown_framing_metric():
    tt, log = validate_target_framing({"transform": "log1p", "rationale": "t"}, _skewed_df(), "y",
                                      "regression", "root_mean_squared_error", "spearmanr")
    assert tt is None and "framing metric" in log[-1]
    tt, _ = validate_target_framing({"transform": "log1p", "rationale": "t"}, _skewed_df(), "y",
                                    "regression", "root_mean_squared_error",
                                    "root_mean_squared_log_error")
    assert tt is not None
    assert FRAMING_METRICS >= SUPPORTED_METRICS | {"root_mean_squared_log_error"}


def test_cross_validate_records_fold_val_index(monkeypatch):
    df = _skewed_df(30)

    def fake_train_and_evaluate(train, val, target, time_limit, model_dir, eval_metric=None, presets=None, sample_weight=None):
        return TrainingResult("regression", "root_mean_squared_error", pd.DataFrame(),
                              {"root_mean_squared_error": -1.0}, predictor=_FakeRegressor())

    monkeypatch.setattr(validation, "train_and_evaluate", fake_train_and_evaluate)
    cv = cross_validate(df, "y", cleaning_plan=None, feature_plan=None, model_dir="x",
                        time_limit=1, n_folds=3, seed=0)
    assert len(cv.fold_val_index) == 3
    covered = pd.Index([]).append(list(cv.fold_val_index))
    assert sorted(covered) == sorted(df.index)                          # a partition of the rows


def _run_framing(monkeypatch, framing_metric):
    """run_pipeline's framing arbiter with the CV mocked to the two Finding-2 arms."""
    y, base, trial = _absolute_vs_relative_arms()
    df = pd.DataFrame({"x": np.arange(len(y), dtype=float), "y": y})
    monkeypatch.setattr(pipeline, "propose_cleaning_plan",
                        lambda *a, **k: {"columns_to_drop": [], "imputations": []})
    monkeypatch.setattr(pipeline, "propose_target_framing",
                        lambda *a, **k: {"transform": "log1p", "rationale": "long tail"})
    monkeypatch.setattr(pipeline, "cross_validate",
                        lambda *a, **k: trial if k.get("target_transform") else base)
    monkeypatch.setattr(pipeline, "fit_predictor", lambda *a, **k: TrainingResult(
        "regression", "root_mean_squared_error", pd.DataFrame({"model": ["m"]}), {}))
    result = run_pipeline(df, "y", model="m", test_size=0.2, time_limit=1, seed=0, model_dir="x",
                          cv_folds=3, use_fe=False, target_framing=True,
                          framing_metric=framing_metric)
    return result, base, trial


def test_pipeline_default_arbiter_keeps_the_engine_metric(monkeypatch):
    result, base, _ = _run_framing(monkeypatch, None)
    assert result.target_framing["accepted"] is False                   # raw RMSE: rejected
    assert result.target_framing["arbiter_metric"] == "root_mean_squared_error"
    assert result.cv is base


def test_pipeline_framing_metric_decides_in_rmsle_but_reports_engine_cv(monkeypatch):
    result, _, trial = _run_framing(monkeypatch, "root_mean_squared_log_error")
    assert result.target_framing["accepted"] is True                    # RMSLE: accepted
    assert result.target_framing["arbiter_metric"] == "root_mean_squared_log_error"
    assert "arbiter metric root_mean_squared_log_error" in result.target_framing["log"][-1]
    # the reported CV is the adopted arm in the ENGINE metric, comparable to a --no-llm run
    assert result.cv is trial and result.cv.eval_metric == "root_mean_squared_error"


def test_framing_metric_requires_target_framing():
    with pytest.raises(ValueError, match="--framing-metric"):
        run_pipeline(_skewed_df(), "y", model="m", test_size=0.2, time_limit=1, seed=0,
                     model_dir="x", cv_folds=3, framing_metric="root_mean_squared_log_error")
