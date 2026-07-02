"""Tests for cross-validation + adversarial validation. AutoGluon is mocked (cross_validate)
or only its data prep is exercised (adversarial), so the suite stays fast and offline.
The central test proves the per-fold leakage guarantee."""
import numpy as np
import pandas as pd
import pytest

from maestra import validation
from maestra.engine import TrainingResult
from maestra.validation import CVResult, _process_fold, cross_validate


def test_process_fold_imputes_with_fold_train_only():
    """The leakage guarantee: a fold's val rows are imputed with the FOLD-train statistic,
    never a value computed over the whole dataset."""
    plan = {"columns_to_drop": [], "imputations": [{"column": "x", "strategy": "mean", "reason": "t"}]}
    fold_train = pd.DataFrame({"x": [10.0, 20.0], "y": [0, 1]})        # fold-train mean = 15
    fold_val = pd.DataFrame({"x": [100.0, np.nan], "y": [0, 1]})       # the NaN must become 15
    _, proc_val = _process_fold(fold_train, fold_val, "y", plan, None)
    # 15.0 = mean(10, 20); a leaky global fit would give mean(10, 20, 100) = 43.33
    assert proc_val["x"].tolist() == [100.0, 15.0]


def test_cross_validate_aggregates_fold_scores(monkeypatch):
    df = pd.DataFrame({"x": list(range(12)), "y": [0, 1] * 6})
    scores = iter([0.7, 0.8, 0.9])

    def fake_train_and_evaluate(train, val, target, time_limit, model_dir, eval_metric=None):
        return TrainingResult("binary", "accuracy", pd.DataFrame(), {"accuracy": next(scores)})

    monkeypatch.setattr(validation, "train_and_evaluate", fake_train_and_evaluate)
    cv = cross_validate(df, "y", cleaning_plan=None, feature_plan=None,
                        model_dir="x", time_limit=1, n_folds=3, seed=0)

    assert cv.fold_scores == [0.7, 0.8, 0.9]
    assert cv.mean == pytest.approx(0.8)
    assert cv.std == pytest.approx(np.std([0.7, 0.8, 0.9]))
    assert cv.n_folds == 3
    assert cv.eval_metric == "accuracy"
    assert cv.stratified is True  # classification target -> stratified by default


class _eval_metric:
    greater_is_better = True


class _FakeProbaPredictor:
    """Predicts a positive-class probability that ranks rows by their feature value, so the
    out-of-fold AUC is well-defined (not degenerate)."""

    positive_class = 1
    eval_metric = _eval_metric

    def predict(self, X):
        return (X["x"] >= X["x"].median()).astype(int)

    def predict_proba(self, X):
        lo, hi = X["x"].min(), X["x"].max()
        p = (X["x"] - lo) / (hi - lo) if hi > lo else X["x"] * 0 + 0.5
        return pd.DataFrame({0: 1 - p, 1: p}, index=X.index)


def test_cross_validate_collects_oof_probabilities(monkeypatch):
    """AUC/log-loss need out-of-fold PROBABILITIES: every row gets a probability from a model
    that did not see it, assembled df-indexed — and scoring on them is the comparable CV score."""
    from maestra import benchmark

    df = pd.DataFrame({"x": [float(i) for i in range(12)], "y": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]})

    def fake_train_and_evaluate(train, val, target, time_limit, model_dir, eval_metric=None):
        return TrainingResult("binary", "roc_auc", pd.DataFrame(), {"roc_auc": 0.5},
                              predictor=_FakeProbaPredictor())

    monkeypatch.setattr(validation, "train_and_evaluate", fake_train_and_evaluate)
    cv = cross_validate(df, "y", cleaning_plan=None, feature_plan=None,
                        model_dir="x", time_limit=1, n_folds=3, seed=0)

    assert cv.oof_proba is not None
    assert list(cv.oof_proba.columns) == [0, 1]                 # one column per class
    assert cv.oof_proba.index.equals(df.index)                  # df-indexed
    assert not cv.oof_proba.isna().any().any()                  # every row covered, once
    assert cv.oof_proba.sum(axis=1).round(6).eq(1.0).all()      # rows are probabilities

    # The CV score for a probability metric is computed on these pooled OOF probabilities.
    score = benchmark._roc_auc_proba(df["y"], cv.oof_proba, positive_class=1)
    from sklearn.metrics import roc_auc_score
    assert score == pytest.approx(roc_auc_score(df["y"], cv.oof_proba[1]))


class _MissingClassPredictor:
    """A multiclass predictor that only ever outputs probabilities over classes 'A' and 'B' —
    it never saw 'C'. Its predict_proba therefore lacks the 'C' column, exactly as AutoGluon's
    would when a fold's training part misses a rare class."""

    eval_metric = _eval_metric

    def predict(self, X):
        return pd.Series(["A"] * len(X), index=X.index)

    def predict_proba(self, X):
        return pd.DataFrame({"A": [0.6] * len(X), "B": [0.4] * len(X)}, index=X.index)


def test_oof_proba_reindexes_to_full_class_set(monkeypatch):
    """A fold missing a class must not leave NaNs in the pooled OOF probabilities: the matrix
    spans EVERY class in the data and each row still sums to 1 (the absent class filled with 0).
    Regression for the log_loss-blows-up bug on many-class tasks (e.g. leaf-classification)."""
    df = pd.DataFrame({"x": [float(i) for i in range(12)],
                       "y": ["A", "B", "C"] * 4})  # 'C' is a real class in the data

    def fake_train_and_evaluate(train, val, target, time_limit, model_dir, eval_metric=None):
        return TrainingResult("multiclass", "log_loss", pd.DataFrame(), {"log_loss": 0.5},
                              predictor=_MissingClassPredictor())

    monkeypatch.setattr(validation, "train_and_evaluate", fake_train_and_evaluate)
    cv = cross_validate(df, "y", cleaning_plan=None, feature_plan=None,
                        model_dir="x", time_limit=1, n_folds=3, seed=0)

    assert list(cv.oof_proba.columns) == ["A", "B", "C"]        # full class set, not just the fold's
    assert not cv.oof_proba.isna().any().any()                  # no NaN despite the missing 'C' column
    assert (cv.oof_proba["C"] == 0.0).all()                     # absent class -> 0, not NaN
    assert cv.oof_proba.sum(axis=1).round(6).eq(1.0).all()      # every row stays a probability vector


def test_integer_regression_target_is_not_classification():
    """sklearn calls any many-valued integer column 'multiclass'; a price-like target must
    NOT be stratified (StratifiedKFold would crash on singleton 'classes' — house-prices bug)."""
    prices = pd.Series(range(100_000, 100_000 + 500))          # 500 distinct int values
    assert validation._is_classification(prices) is False
    assert validation._is_classification(pd.Series([0, 1] * 50)) is True          # few int classes
    assert validation._is_classification(pd.Series(["a", "b", "c"] * 5)) is True  # strings stay classes
    # end-to-end: folds on an integer regression target must not raise
    df = pd.DataFrame({"x": range(40), "y": range(1000, 1040)})
    folds = validation._make_folds(df, "y", 3, seed=0, stratified=validation._is_classification(df["y"]))
    assert len(folds) == 3


def test_make_folds_is_deterministic():
    df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
    a = validation._make_folds(df, "y", 5, seed=1, stratified=True)
    b = validation._make_folds(df, "y", 5, seed=1, stratified=True)
    assert [v.tolist() for _, v in a] == [v.tolist() for _, v in b]


def test_build_adversarial_data_drops_id_and_labels_rows():
    plan = {"columns_to_drop": [{"column": "id", "reason": "ID"}], "imputations": []}
    train = pd.DataFrame({"id": [1, 2], "f": [1.0, 2.0], "y": [0, 1]})
    test = pd.DataFrame({"id": [9, 10], "f": [3.0, 4.0]})
    data, feats = validation._build_adversarial_data(train, test, "y", plan)

    assert "id" not in feats and feats == ["f"]            # id dropped by cleaning, target excluded
    assert validation._ADV_LABEL in data.columns
    assert sorted(data[validation._ADV_LABEL].unique().tolist()) == [0, 1]
    assert len(data) == 4


def test_improves_beyond_noise_paired_rule():
    """The arbiter's accept rule: paired per-fold deltas, majority of folds, 2-SEM threshold —
    the fix for the old too-permissive rule (1 sigma of 3 correlated fold scores)."""
    from maestra.validation import improves_beyond_noise

    def cv(mean, folds, gib=True):
        return CVResult("m", "binary", folds, mean, float(np.std(folds)), len(folds), True, gib)

    base = cv(0.80, [0.80, 0.80, 0.80])
    # consistent, clearly-above-noise improvement -> keep
    ok, delta = improves_beyond_noise(base, cv(0.85, [0.85, 0.84, 0.86]))
    assert ok and delta == pytest.approx(0.05)
    # tiny, noisy improvement (would have passed the old 1-sigma-of-means rule with std~0) -> reject
    ok, _ = improves_beyond_noise(base, cv(0.8033, [0.81, 0.81, 0.79]))
    assert not ok
    # lower-is-better metrics compare in the right direction
    base_ll = cv(0.50, [0.50, 0.50, 0.50], gib=False)
    ok, delta = improves_beyond_noise(base_ll, cv(0.41, [0.40, 0.42, 0.41], gib=False))
    assert ok and delta == pytest.approx(0.09)


def test_improves_beyond_noise_falls_back_without_fold_scores():
    from maestra.validation import improves_beyond_noise

    base = CVResult("m", "binary", [], 0.80, 0.02, 3, True, True)
    trial = CVResult("m", "binary", [], 0.85, 0.02, 3, True, True)
    ok, delta = improves_beyond_noise(base, trial, sigma_mult=2.0)
    assert ok and delta == pytest.approx(0.05)          # 0.05 > 2*0.02
    ok, _ = improves_beyond_noise(base, CVResult("m", "binary", [], 0.83, 0.02, 3, True, True))
    assert not ok                                        # 0.03 < 0.04
