"""Offline tests for the public compare() API (P3). Real sklearn estimators throughout --
no mocking needed, no LLM, no AutoGluon (that's the point: see test_compare_works_without_autogluon)."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression

from maestra import CompareResult, compare


def _linear_df(n=40, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    y = 2.0 * x + (rng.normal(0, noise, n) if noise else 0.0)
    return pd.DataFrame({"x": x, "y": y})


def test_compare_returns_a_compare_result_with_the_expected_shape():
    df = _linear_df()
    result = compare(DummyRegressor(strategy="mean"), LinearRegression(), df, "y", cv=4)
    assert isinstance(result, CompareResult)
    assert result.verdict in ("improved", "no_improvement", "underpowered")
    assert result.n_folds == 4 and result.n_seeds == 1
    assert len(result.deltas) == 4


def test_compare_detects_a_real_improvement():
    """A perfectly linear target: LinearRegression must decisively beat a constant-mean dummy."""
    df = _linear_df()
    result = compare(DummyRegressor(strategy="mean"), LinearRegression(), df, "y", cv=5)
    assert result.verdict == "improved"
    assert result.mean_delta > 0
    assert "improved" in result.summary()


def test_compare_symmetric_flip_reports_no_improvement_or_underpowered():
    """Flipping the arguments (candidate now the WORSE model) must never report 'improved'."""
    df = _linear_df()
    result = compare(LinearRegression(), DummyRegressor(strategy="mean"), df, "y", cv=5)
    assert result.verdict in ("no_improvement", "underpowered")
    assert result.mean_delta < 0


def test_compare_two_identical_estimators_is_no_improvement_or_underpowered():
    df = _linear_df(noise=5.0)
    result = compare(LinearRegression(), LinearRegression(), df, "y", cv=5)
    assert result.verdict in ("no_improvement", "underpowered")


def test_compare_classification_target_uses_estimators_own_accuracy():
    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(size=n)
    y = (x > 0).astype(int)
    df = pd.DataFrame({"x": x, "y": y})
    result = compare(DummyClassifier(strategy="most_frequent"), LogisticRegression(), df, "y", cv=5)
    assert result.metric == "estimator.score"
    assert result.greater_is_better is True


def test_compare_seeds_greater_than_one_pools_paired_deltas_across_seeds():
    df = _linear_df(noise=2.0)
    result = compare(DummyRegressor(strategy="mean"), LinearRegression(), df, "y", cv=4, seeds=3)
    assert result.n_seeds == 3 and len(result.deltas) == 12  # 3 seeds x 4 folds, pooled


def test_compare_uses_an_explicit_scorer_when_given():
    df = _linear_df()
    result = compare(DummyRegressor(strategy="mean"), LinearRegression(), df, "y", cv=4,
                     metric="neg_mean_squared_error")
    assert result.metric == "neg_mean_squared_error"
    assert result.verdict == "improved"  # LinearRegression has far lower (less negative) MSE


def test_compare_summary_is_a_markdown_string():
    df = _linear_df()
    result = compare(DummyRegressor(strategy="mean"), LinearRegression(), df, "y", cv=4)
    text = result.summary()
    assert isinstance(text, str) and "compare() verdict" in text and "mean delta" in text


def test_compare_regularized_vs_plain_linear_regression_picks_the_better_fit():
    """A concrete, realistic comparison: heavy L1 regularisation should lose to plain OLS on a
    clean linear signal -- the exact scenario compare() exists for."""
    df = _linear_df(noise=0.01)
    result = compare(Lasso(alpha=50.0), LinearRegression(), df, "y", cv=5)
    assert result.verdict == "improved"


def test_compare_with_a_rolling_origin_splitter_uses_its_actual_fold_count():
    """F2: an arbitrary sklearn splitter overrides the plain k-fold split; n_folds in the
    result reflects the SPLITTER's fold count, not the (deliberately mismatched) cv= kwarg."""
    from maestra.validation import RollingOriginSplit

    df = _linear_df(n=40)
    result = compare(DummyRegressor(strategy="mean"), LinearRegression(), df, "y", cv=99,
                     splitter=RollingOriginSplit(n_origins=4, horizon=5))
    assert result.n_folds == 4
    assert len(result.deltas) == 4
    assert result.verdict == "improved"  # a perfectly linear signal still decisively wins


def test_compare_works_without_autogluon_installed(monkeypatch):
    """The Colab notebook's whole premise: compare() must not need AutoGluon at all."""
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "autogluon" or name.startswith("autogluon."):
            raise ImportError(f"simulated: {name} is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    df = _linear_df()
    result = compare(DummyRegressor(strategy="mean"), LinearRegression(), df, "y", cv=4)
    assert result.verdict == "improved"
