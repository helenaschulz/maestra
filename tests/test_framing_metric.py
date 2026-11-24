"""F3-0: the target-framing arbiter scores in the run's DEPLOYMENT metric when that metric is
log-scale (RMSLE), recomputed from out-of-fold predictions; every other metric stays bit-identical.
No AutoGluon, no LLM: CVs are built by hand from synthetic OOF predictions."""
import numpy as np
import pandas as pd
import pytest

from maestra import pipeline
from maestra.intervention import run_counterfactual
from maestra.validation import CVResult, _ag_score, improves_beyond_noise, rescore_cv

# 3 folds x 12 rows: 10 small counts + 2 large ones per fold -- a right-skewed count target.
_Y = pd.Series(np.concatenate([[1, 2, 3, 4, 5, 2, 3, 4, 1, 2, 900 + 10 * f, 1100 - 10 * f]
                               for f in range(3)]).astype(float))
_FOLDS = [list(range(12 * f, 12 * f + 12)) for f in range(3)]


def _cv(preds: np.ndarray) -> CVResult:
    """A raw-space CV in AutoGluon's convention (negated RMSE) with OOF preds + fold rows."""
    oof = pd.Series(preds, index=_Y.index)
    scores = [_ag_score(_Y.loc[i], oof.loc[i], "root_mean_squared_error") for i in _FOLDS]
    return CVResult("root_mean_squared_error", "regression", scores, float(np.mean(scores)),
                    float(np.std(scores)), 3, False, True, oof_pred=oof, fold_val_index=_FOLDS)


def _raw_model():
    """Raw-space fit: nails the expensive tail, ~+18 absolute error on every small count."""
    p = _Y.to_numpy().copy()
    small = p < 100
    p[small] = 20 + np.arange(small.sum()) % 3
    return _cv(p)


def _log1p_model():
    """log1p fit: relative errors -- small counts near-exact, the tail off by ~5%."""
    p = _Y.to_numpy().copy()
    small = p < 100
    p[small] = p[small] + 0.1
    p[~small] = p[~small] * (0.95 - 0.01 * (np.arange((~small).sum()) % 2))
    return _cv(p)


def test_rmsle_score_matches_the_formula():
    t, p = pd.Series([0.0, 9.0, 99.0]), pd.Series([1.0, 9.0, 120.0])
    expected = -np.sqrt(np.mean((np.log1p(p) - np.log1p(t)) ** 2))
    assert _ag_score(t, p, "root_mean_squared_log_error") == pytest.approx(expected)


def test_rmsle_floors_negative_predictions_at_zero():
    t = pd.Series([0.0, 4.0])
    assert _ag_score(t, pd.Series([-3.0, 4.0]), "root_mean_squared_log_error") == pytest.approx(0.0)


def test_rescore_cv_recomputes_per_fold_without_refitting():
    cv = _raw_model()
    r = rescore_cv(cv, _Y, "root_mean_squared_log_error")
    assert r.eval_metric == "root_mean_squared_log_error" and r.n_folds == 3
    assert r.fold_scores[0] == pytest.approx(
        _ag_score(_Y.loc[_FOLDS[0]], cv.oof_pred.loc[_FOLDS[0]], "root_mean_squared_log_error"))


def test_rescore_cv_needs_oof_and_fold_rows():
    with pytest.raises(ValueError):
        rescore_cv(CVResult("r2", "regression", [0.1, 0.2], 0.15, 0.05, 2, False), _Y,
                   "root_mean_squared_log_error")


# (a) the RMSLE-scored arbiter accepts log1p where the old raw-RMSE arbiter rejected it
def test_rmsle_arbiter_accepts_log1p_the_raw_arbiter_rejects():
    base, trial = _raw_model(), _log1p_model()
    old, _ = run_counterfactual("target:log1p", "target_framing", "target_framing",
                                base=base, trial_fn=lambda: trial)
    assert old.accepted is False  # raw-count RMSE: the tail error makes log1p look worse
    new, new_base = run_counterfactual(
        "target:log1p", "target_framing", "target_framing", base=base, trial_fn=lambda: trial,
        rescore=lambda c: rescore_cv(c, _Y, "root_mean_squared_log_error"))
    assert new.accepted is True
    # the returned base stays in the run's own metric (reported CV unchanged in kind)
    assert new_base is trial and new_base.eval_metric == "root_mean_squared_error"
    # the recorded counterfactual is in the deployment metric
    assert new.base_scores == rescore_cv(base, _Y, "root_mean_squared_log_error").fold_scores


# (b) no rescore -> bit-identical to the plain accept rule
def test_without_rescore_the_gate_is_bit_identical():
    base, trial = _raw_model(), _log1p_model()
    outcome, new_base = run_counterfactual("x", "k", "p", base=base, trial_fn=lambda: trial)
    accepted, delta = improves_beyond_noise(base=base, trial=trial)
    assert (outcome.accepted, outcome.cv_delta) == (accepted, delta)
    assert outcome.base_scores == base.fold_scores
    assert new_base is base


def _pipeline_run(monkeypatch, framing_metric):
    df = pd.DataFrame({"f": np.arange(len(_Y), dtype=float), "count": _Y})
    base, trial = _raw_model(), _log1p_model()
    calls = []

    def fake_cv(*a, **k):
        calls.append(k.get("target_transform"))
        return trial if k.get("target_transform") is not None else base

    monkeypatch.setattr(pipeline, "cross_validate", fake_cv)
    monkeypatch.setattr(pipeline, "propose_target_framing",
                        lambda *a, **k: {"transform": "log1p", "rationale": "skewed counts"})
    monkeypatch.setattr(pipeline, "fit_predictor", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_validate_trainable", lambda *a, **k: None)
    return pipeline.run_pipeline(df, "count", model="m", test_size=0.2, time_limit=1, seed=0,
                                 model_dir="x", cv_folds=3, use_llm=False, target_framing=True,
                                 framing_metric=framing_metric)


def test_pipeline_rmsle_run_adopts_log1p(monkeypatch):
    res = _pipeline_run(monkeypatch, "root_mean_squared_log_error")
    assert res.target_framing["accepted"] is True
    assert res.target_framing["arbiter_metric"] == "root_mean_squared_log_error"
    assert res.cv.eval_metric == "root_mean_squared_error"  # reported CV stays the run's metric


def test_pipeline_without_framing_metric_is_unchanged(monkeypatch):
    res = _pipeline_run(monkeypatch, None)
    assert res.target_framing["accepted"] is False
    assert res.target_framing["arbiter_metric"] == "root_mean_squared_error"
