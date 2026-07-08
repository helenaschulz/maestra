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

    def fake_train_and_evaluate(train, val, target, time_limit, model_dir, eval_metric=None, presets=None, sample_weight=None):
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

    def fake_train_and_evaluate(train, val, target, time_limit, model_dir, eval_metric=None, presets=None, sample_weight=None):
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


class _BoolProbaPredictor:
    """Same shape as _FakeProbaPredictor, but the class labels are native Python bools --
    the Kaggle spaceship-titanic `Transported` case."""

    positive_class = True
    eval_metric = _eval_metric

    def predict(self, X):
        return X["x"] >= X["x"].median()

    def predict_proba(self, X):
        lo, hi = X["x"].min(), X["x"].max()
        p = (X["x"] - lo) / (hi - lo) if hi > lo else X["x"] * 0 + 0.5
        return pd.DataFrame({False: 1 - p, True: p}, index=X.index)


def test_cross_validate_handles_boolean_target_classes(monkeypatch):
    """Regression: a bool target ([False, True]) made `.loc[rows, oof_classes]` ambiguous --
    pandas reads a bool list as a column MASK, not labels, silently selecting the wrong shape
    (found on real Kaggle data: spaceship-titanic's `Transported` column)."""
    df = pd.DataFrame({"x": [float(i) for i in range(12)],
                       "y": [False, False, False, True, True, True] * 2})

    def fake_train_and_evaluate(train, val, target, time_limit, model_dir, eval_metric=None, presets=None, sample_weight=None):
        return TrainingResult("binary", "roc_auc", pd.DataFrame(), {"roc_auc": 0.5},
                              predictor=_BoolProbaPredictor())

    monkeypatch.setattr(validation, "train_and_evaluate", fake_train_and_evaluate)
    cv = cross_validate(df, "y", cleaning_plan=None, feature_plan=None,
                        model_dir="x", time_limit=1, n_folds=3, seed=0)

    assert list(cv.oof_proba.columns) == [False, True]
    assert not cv.oof_proba.isna().any().any()
    assert cv.oof_proba.sum(axis=1).round(6).eq(1.0).all()
    assert not cv.oof_pred.isna().any()


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

    def fake_train_and_evaluate(train, val, target, time_limit, model_dir, eval_metric=None, presets=None, sample_weight=None):
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


# --- RollingOriginSplit (F2): a standalone, sklearn-compatible rolling-origin splitter -----

def test_rolling_origin_split_get_n_splits_matches_n_origins():
    from maestra.validation import RollingOriginSplit

    splitter = RollingOriginSplit(n_origins=3, horizon=5, gap=2)
    assert splitter.get_n_splits() == 3


def test_rolling_origin_split_yields_exactly_n_origins_folds_test_strictly_after_train():
    from maestra.validation import RollingOriginSplit

    X = np.arange(50)
    splitter = RollingOriginSplit(n_origins=4, horizon=5, gap=0)
    folds = list(splitter.split(X))
    assert len(folds) == 4
    for train_idx, test_idx in folds:
        assert len(test_idx) == 5
        assert train_idx.max() < test_idx.min()  # every test row strictly later than every train row
        assert test_idx.tolist() == sorted(test_idx.tolist())  # contiguous, time-ordered block


def test_rolling_origin_split_tiles_expanding_windows_toward_the_back():
    from maestra.validation import RollingOriginSplit

    X = np.arange(30)
    folds = list(RollingOriginSplit(n_origins=3, horizon=5, gap=0).split(X))
    # each origin's test block starts exactly where the previous one ended (contiguous tiling)
    assert [f[1].tolist() for f in folds] == [
        list(range(15, 20)), list(range(20, 25)), list(range(25, 30))]
    # training prefixes expand: origin i+1 trains on strictly more rows than origin i
    sizes = [len(train_idx) for train_idx, _ in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3


def test_rolling_origin_split_gap_embargoes_rows_between_train_and_test():
    from maestra.validation import RollingOriginSplit

    X = np.arange(30)
    train_idx, test_idx = next(iter(RollingOriginSplit(n_origins=1, horizon=5, gap=3).split(X)))
    assert train_idx.max() == test_idx.min() - 1 - 3  # exactly `gap` embargoed rows in between


def test_rolling_origin_split_raises_when_too_few_rows():
    from maestra.validation import RollingOriginSplit

    with pytest.raises(ValueError, match="RollingOriginSplit needs at least"):
        list(RollingOriginSplit(n_origins=5, horizon=10, gap=0).split(np.arange(20)))


def test_make_folds_with_a_splitter_bypasses_every_other_strategy():
    """A splitter (F2) takes priority even when group_column/time_column are also passed."""
    from maestra.validation import RollingOriginSplit

    df = pd.DataFrame({"g": [0] * 30, "x": range(30), "y": range(30)})
    folds = validation._make_folds(df, "y", n_folds=99, seed=0, stratified=False,
                                   group_column="g", splitter=RollingOriginSplit(3, 5))
    assert len(folds) == 3  # splitter's own count, not the (irrelevant) n_folds=99


def test_make_folds_with_a_splitter_and_time_column_sorts_by_time_first():
    from maestra.validation import RollingOriginSplit

    rng = np.random.default_rng(4)
    order = rng.permutation(30)
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=30, freq="D").to_numpy()[order],
                       "y": np.arange(30)[order]})
    folds = validation._make_folds(df, "y", n_folds=1, seed=0, stratified=False,
                                   time_column="ts", splitter=RollingOriginSplit(1, 5))
    train_idx, test_idx = folds[0]
    assert df["ts"].iloc[train_idx].max() < df["ts"].iloc[test_idx].min()


def test_cross_validate_with_a_splitter_reports_the_splitters_actual_fold_count():
    """cross_validate's n_folds reflects the splitter, not the (here mismatched) n_folds kwarg."""
    from sklearn.linear_model import LinearRegression

    from maestra.engine import SklearnEngine
    from maestra.validation import RollingOriginSplit

    df = pd.DataFrame({"x": list(range(40)), "y": [float(2 * i) for i in range(40)]})
    cv = cross_validate(df, "y", cleaning_plan=None, feature_plan=None, model_dir="x",
                        time_limit=1, n_folds=99, seed=0,
                        engine=SklearnEngine(LinearRegression()),
                        splitter=RollingOriginSplit(n_origins=4, horizon=5))
    assert cv.n_folds == 4 and len(cv.fold_scores) == 4


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


def test_paired_delta_test_nb_correction_inflates_the_threshold():
    """test_train_ratio > 0 (Nadeau-Bengio) only ever raises the bar -- a delta that clears
    the naive rule can fail the corrected one, never the reverse."""
    from maestra.validation import paired_delta_test

    deltas = [0.10, 0.09, 0.02]  # mean 0.07, majority positive
    assert paired_delta_test(deltas, test_train_ratio=0.0) is True   # naive rule: accept
    assert paired_delta_test(deltas, test_train_ratio=0.5) is False  # k=3-fold ratio: reject
    # default reproduces the naive rule exactly (backward compatible)
    assert paired_delta_test(deltas) is True


def test_improves_beyond_noise_applies_nb_correction_from_n_folds():
    """The paired branch derives test_train_ratio = 1/(k-1) from base.n_folds (N1, 2026-07-05)
    -- the same marginal case as the ratio test above, reached through the real CVResult path."""
    from maestra.validation import improves_beyond_noise

    def cv(mean, folds, gib=True):
        return CVResult("m", "binary", folds, mean, float(np.std(folds)), len(folds), True, gib)

    base = cv(0.80, [0.80, 0.80, 0.80])
    trial = cv(0.87, [0.90, 0.89, 0.82])  # per-fold deltas [0.10, 0.09, 0.02], same as above
    ok, delta = improves_beyond_noise(base, trial)
    assert not ok and delta == pytest.approx(0.07)


def test_paired_delta_mde_matches_the_accept_threshold():
    from maestra.validation import paired_delta_mde

    deltas = [0.10, 0.09, 0.02]
    std = float(np.std(deltas, ddof=1))
    mde = paired_delta_mde(std, n=3, test_train_ratio=0.5)
    assert mde == pytest.approx(2.0 * std * (1.0 / 3 + 0.5) ** 0.5)
    assert paired_delta_mde(std, n=1) == float("inf")


def test_improves_beyond_noise_falls_back_without_fold_scores():
    from maestra.validation import improves_beyond_noise

    base = CVResult("m", "binary", [], 0.80, 0.02, 3, True, True)
    trial = CVResult("m", "binary", [], 0.85, 0.02, 3, True, True)
    ok, delta = improves_beyond_noise(base, trial, sigma_mult=2.0)
    assert ok and delta == pytest.approx(0.05)          # 0.05 > 2*0.02
    ok, _ = improves_beyond_noise(base, CVResult("m", "binary", [], 0.83, 0.02, 3, True, True))
    assert not ok                                        # 0.03 < 0.04


def test_cross_validate_with_a_sklearn_engine_runs_real_folds_no_autogluon():
    """P3: engine=SklearnEngine(...) takes the separate, engine-agnostic path -- no AutoGluon
    call at all (no monkeypatching of train_and_evaluate needed, unlike every other test here)."""
    from sklearn.linear_model import LinearRegression

    from maestra.engine import SklearnEngine

    df = pd.DataFrame({"x": list(range(40)), "y": [float(2 * i) for i in range(40)]})
    cv = cross_validate(df, "y", cleaning_plan=None, feature_plan=None, model_dir="x",
                        time_limit=1, n_folds=4, seed=0, engine=SklearnEngine(LinearRegression()))
    assert cv.n_folds == 4 and len(cv.fold_scores) == 4
    assert cv.problem_type == "regression"
    assert cv.eval_metric == "SklearnEngine.score"
    assert cv.greater_is_better is True
    # a perfectly linear y = 2x -> every fold's held-out R^2 should be (near) perfect
    assert cv.mean == pytest.approx(1.0, abs=1e-6)


def test_cross_validate_with_engine_still_refits_cleaning_per_fold():
    """The engine path reuses `_process_fold` -- the leakage guarantee holds there too."""
    from sklearn.linear_model import LinearRegression

    from maestra.engine import SklearnEngine

    plan = {"columns_to_drop": [], "imputations": [{"column": "x", "strategy": "mean", "reason": "t"}]}
    x = [float(i) for i in range(24)]
    x[4] = np.nan
    df = pd.DataFrame({"x": x, "y": [float(2 * i) for i in range(24)]})
    cv = cross_validate(df, "y", cleaning_plan=plan, feature_plan=None, model_dir="x",
                        time_limit=1, n_folds=2, seed=0, engine=SklearnEngine(LinearRegression()))
    assert len(cv.fold_scores) == 2   # ran without raising on the NaN -> imputation applied per fold


def test_cross_validate_with_engine_uses_an_explicit_scorer():
    from sklearn.linear_model import LinearRegression

    from maestra.engine import SklearnEngine

    df = pd.DataFrame({"x": list(range(40)), "y": [float(2 * i) for i in range(40)]})
    cv = cross_validate(df, "y", cleaning_plan=None, feature_plan=None, model_dir="x",
                        time_limit=1, n_folds=4, seed=0,
                        engine=SklearnEngine(LinearRegression(), scoring="neg_mean_squared_error"))
    assert cv.mean == pytest.approx(0.0, abs=1e-6)  # perfect fit -> MSE 0 -> neg_MSE 0


def test_cross_validate_autogluon_engine_instance_takes_the_untouched_ag_path(monkeypatch):
    """Passing an AutoGluonEngine explicitly (the spec's 'existing callers set AutoGluonEngine')
    must behave identically to engine=None -- both dispatch to the untouched AutoGluon branch,
    not to _cross_validate_with_engine."""
    from maestra.engine import AutoGluonEngine

    df = pd.DataFrame({"x": list(range(12)), "y": [0, 1] * 6})
    scores = iter([0.7, 0.8, 0.9])

    def fake_train_and_evaluate(train, val, target, time_limit, model_dir, eval_metric=None,
                                presets=None, sample_weight=None):
        return TrainingResult("binary", "accuracy", pd.DataFrame(), {"accuracy": next(scores)})

    monkeypatch.setattr(validation, "train_and_evaluate", fake_train_and_evaluate)
    cv = cross_validate(df, "y", cleaning_plan=None, feature_plan=None, model_dir="x", time_limit=1,
                        n_folds=3, seed=0, engine=AutoGluonEngine("y", model_dir="x", time_limit=1))
    assert cv.fold_scores == [0.7, 0.8, 0.9]  # from the mocked AutoGluon path, not from the engine
