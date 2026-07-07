"""Offline tests for the Engine protocol (P3). SklearnEngine is exercised for real against
tiny sklearn estimators (no mocking needed -- it's plain sklearn); AutoGluonEngine mocks
`fit_predictor`, same convention as test_validation.py's `train_and_evaluate` mocks."""
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from maestra import engine as engine_mod
from maestra.engine import AutoGluonEngine, SklearnEngine, TrainingResult


def _clf_data():
    X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5, 6, 7], "b": [1, 0, 1, 0, 1, 0, 1, 0]})
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def _reg_data():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
    return X, y


def test_sklearn_engine_fits_and_predicts_a_classifier():
    X, y = _clf_data()
    eng = SklearnEngine(LogisticRegression()).fit(X, y)
    preds = eng.predict(X)
    assert list(preds.index) == list(X.index)
    assert set(preds.unique()) <= {0, 1}


def test_sklearn_engine_predict_proba_present_for_a_classifier():
    X, y = _clf_data()
    eng = SklearnEngine(LogisticRegression()).fit(X, y)
    proba = eng.predict_proba(X)
    assert proba is not None
    assert list(proba.columns) == [0, 1]
    assert (proba.sum(axis=1).round(6) == 1.0).all()


def test_sklearn_engine_predict_proba_is_none_without_the_method():
    from sklearn.base import BaseEstimator

    class NoProba(BaseEstimator):
        def fit(self, X, y):
            self.mean_ = y.mean()
            return self

        def predict(self, X):
            return [self.mean_] * len(X)

        def score(self, X, y):
            return 0.0

    eng = SklearnEngine(NoProba())
    X, y = _reg_data()
    eng.fit(X, y)
    assert eng.predict_proba(X) is None


def test_sklearn_engine_score_defaults_to_the_estimators_own_metric():
    X, y = _reg_data()  # y = 2*a, perfectly linear -> R^2 == 1.0
    eng = SklearnEngine(LinearRegression()).fit(X, y)
    assert eng.score(X, y) == pytest.approx(1.0)


def test_sklearn_engine_score_uses_an_explicit_scorer_when_given():
    X, y = _reg_data()
    eng = SklearnEngine(LinearRegression(), scoring="neg_mean_squared_error").fit(X, y)
    # a perfect fit -> MSE 0 -> neg_mean_squared_error 0 (already sign-flipped to gib=True)
    assert eng.score(X, y) == pytest.approx(0.0, abs=1e-8)


def test_sklearn_engine_clones_fresh_per_fit_call():
    """Two fit() calls on the same engine must not share fitted state across folds."""
    est = LogisticRegression()
    eng = SklearnEngine(est)
    X1, y1 = _clf_data()
    eng.fit(X1, y1)
    first_fitted = eng._fitted
    eng.fit(X1, y1)
    assert eng._fitted is not first_fitted
    assert est.coef_ is None if hasattr(est, "coef_") else True  # the template itself never fits


class _FakePredictor:
    def __init__(self, problem_type, eval_metric_name, gib=True):
        self.problem_type = problem_type
        self.eval_metric = type("M", (), {"name": eval_metric_name, "greater_is_better": gib})()

    def predict(self, X):
        return pd.Series([0] * len(X), index=X.index)

    def predict_proba(self, X):
        return pd.DataFrame({0: [1.0] * len(X), 1: [0.0] * len(X)}, index=X.index)

    def evaluate(self, df, silent=True):
        return {self.eval_metric.name: 0.75}


def test_autogluon_engine_fits_via_fit_predictor_and_scores(monkeypatch):
    fake_result = TrainingResult("binary", "accuracy", pd.DataFrame(), {},
                                 predictor=_FakePredictor("binary", "accuracy"))
    calls = []

    def fake_fit_predictor(train, target, time_limit, model_dir, eval_metric=None, presets=None,
                           sample_weight=None):
        calls.append((target, time_limit, model_dir))
        return fake_result

    monkeypatch.setattr(engine_mod, "fit_predictor", fake_fit_predictor)
    X, y = _clf_data()
    eng = AutoGluonEngine("label", model_dir="x/fold_0", time_limit=5).fit(X, y.rename("label"))
    assert calls == [("label", 5, "x/fold_0")]
    assert eng.score(X, y) == pytest.approx(0.75)
    assert eng.predict_proba(X) is not None


def test_autogluon_engine_score_flips_sign_for_a_lower_is_better_metric(monkeypatch):
    fake_result = TrainingResult("regression", "root_mean_squared_error", pd.DataFrame(), {},
                                 predictor=_FakePredictor("regression", "root_mean_squared_error",
                                                          gib=False))
    monkeypatch.setattr(engine_mod, "fit_predictor", lambda *a, **k: fake_result)
    X, y = _reg_data()
    eng = AutoGluonEngine("label", model_dir="x", time_limit=5).fit(X, y.rename("label"))
    # the fake predictor.evaluate always returns 0.75; gib=False -> normalised to -0.75
    assert eng.score(X, y) == pytest.approx(-0.75)


def test_autogluon_engine_predict_proba_is_none_for_regression(monkeypatch):
    fake_result = TrainingResult("regression", "root_mean_squared_error", pd.DataFrame(), {},
                                 predictor=_FakePredictor("regression", "root_mean_squared_error"))
    monkeypatch.setattr(engine_mod, "fit_predictor", lambda *a, **k: fake_result)
    X, y = _reg_data()
    eng = AutoGluonEngine("label", model_dir="x", time_limit=5).fit(X, y.rename("label"))
    assert eng.predict_proba(X) is None
