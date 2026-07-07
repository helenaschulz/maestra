"""AutoGluon engine — the only place numbers are crunched.

The LLM never computes. Train/holdout split, model search, hyperparameter tuning and
metric calculation all happen here, inside AutoGluon. This module returns plain data
(leaderboard, metrics) and prints nothing, so it is easy to test and reuse.

Also defines the minimal ``Engine`` fit/predict protocol (P3): ``SklearnEngine`` lets
``compare()`` (see ``maestra.__init__``) run the arbiter over ANY sklearn-compatible
estimator, not just AutoGluon. ``AutoGluonEngine`` exists for API symmetry with
``SklearnEngine`` and is usable standalone, but ``validation.py::cross_validate``'s
existing AutoGluon path (presets, sample_weight, custom Scorer objects, target
transforms) is untouched code, not routed through it — those are AutoGluon-native
concepts with no sklearn equivalent, so unifying them into one fit/predict interface
would either lose functionality or bloat the interface. ``cross_validate`` special-cases
``engine is None`` (and ``AutoGluonEngine`` instances) to keep that path byte-identical.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from autogluon.tabular import TabularPredictor


@dataclass
class TrainingResult:
    """Outcome of a single train+evaluate run on a holdout set."""

    problem_type: str
    eval_metric: str
    leaderboard: pd.DataFrame
    metrics: dict[str, float]
    predictor: object | None = None  # the fitted TabularPredictor, for downstream prediction
    val_score: float | None = None   # AutoGluon's best internal validation score (NOT the holdout)


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into ``(train, holdout)`` by random row sampling.

    A simple random split is enough for this project; AutoGluon does its own internal
    validation split on the training portion for model selection.
    """
    holdout = df.sample(frac=test_size, random_state=seed)
    train = df.drop(index=holdout.index)
    return train, holdout


def train_and_evaluate(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target: str,
    time_limit: int,
    model_dir: str,
    eval_metric: str | None = None,
    presets: str | None = None,
    sample_weight: str | None = None,
) -> TrainingResult:
    """Fit an AutoGluon predictor and evaluate it on the holdout set.

    Args:
        train: Training rows (features + target).
        holdout: Held-out rows for honest evaluation.
        target: Name of the label column.
        time_limit: Training budget in seconds.
        model_dir: Where AutoGluon persists the trained models.
        presets: AutoGluon quality preset (e.g. ``"best_quality"`` for multi-layer
            stacking + bagging). ``None`` uses AutoGluon's default (``"medium_quality"``,
            fast). Whatever is chosen must be identical on the CV folds and the final model,
            or the CV↔LB gap compares two different model configurations.
        sample_weight: Name of a per-row weight column (excluded from features by AutoGluon)
            used in BOTH training and metric evaluation — e.g. Walmart's holiday-week ×5
            weighting. Must be present in ``train``/``holdout``. ``None`` = unweighted.

    Returns:
        A :class:`TrainingResult`. The problem type and evaluation metric are inferred
        by AutoGluon from the target column.
    """
    predictor = TabularPredictor(label=target, path=model_dir, eval_metric=eval_metric,
                                 sample_weight=sample_weight).fit(
        train, time_limit=time_limit, presets=presets)
    leaderboard = predictor.leaderboard(holdout, silent=True)
    # Best internal validation score (AutoGluon's own train/val split) — used by the
    # quality gate so the holdout is never consulted for that decision.
    val_score = float(leaderboard["score_val"].max()) if "score_val" in leaderboard.columns else None
    return TrainingResult(
        problem_type=predictor.problem_type,
        eval_metric=predictor.eval_metric.name,
        leaderboard=leaderboard,
        metrics=predictor.evaluate(holdout, silent=True),
        predictor=predictor,
        val_score=val_score,
    )


def predict(predictor, X: pd.DataFrame) -> pd.Series:
    """Predict labels for ``X`` with a fitted predictor (delegates to AutoGluon)."""
    return predictor.predict(X)


def predict_proba(predictor, X: pd.DataFrame) -> pd.DataFrame:
    """Predict class probabilities for ``X`` (delegates to AutoGluon).

    Returns a DataFrame with one column per class (columns are the predictor's class
    labels, in its own order); for binary problems both classes are present. This is the
    format-neutral primitive — reshaping into a competition's submission format happens in
    the pipeline, where the sample submission's columns are known.
    """
    return predictor.predict_proba(X)


def fit_predictor(train: pd.DataFrame, target: str, time_limit: int, model_dir: str,
                  eval_metric: str | None = None, presets: str | None = None,
                  sample_weight: str | None = None) -> TrainingResult:
    """Fit a predictor on ALL labeled rows, with no holdout evaluation.

    Used by the cross-validation path, where the honest estimate is the CV score and this
    final model exists only for prediction (submission/report). ``metrics`` is therefore
    empty; ``val_score`` is AutoGluon's internal validation score. ``presets`` /
    ``sample_weight`` — see :func:`train_and_evaluate` (must match what the CV folds used).
    """
    predictor = TabularPredictor(label=target, path=model_dir, eval_metric=eval_metric,
                                 sample_weight=sample_weight).fit(
        train, time_limit=time_limit, presets=presets)
    leaderboard = predictor.leaderboard(silent=True)
    val_score = float(leaderboard["score_val"].max()) if "score_val" in leaderboard.columns else None
    return TrainingResult(
        problem_type=predictor.problem_type,
        eval_metric=predictor.eval_metric.name,
        leaderboard=leaderboard,
        metrics={},
        predictor=predictor,
        val_score=val_score,
    )


class Engine(ABC):
    """Minimal fit/predict/score protocol so a cross-validation loop can run over ANY model,
    not just AutoGluon. One instance is used per fold: ``fit`` receives already-cleaned
    feature/target frames (cleaning and feature engineering stay outside the engine, in
    ``validation.py``) and returns a fitted engine; ``score`` evaluates it on a held-out fold.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Engine":
        """Fit on training rows. Returns ``self`` (fitted), so calls can be chained."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict labels for ``X``."""

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame | None:
        """Predict class probabilities, or ``None`` if the wrapped model can't (default)."""
        return None

    @abstractmethod
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Score a fitted engine on held-out rows. Higher must always mean better — unlike
        AutoGluon's metrics (which can be true errors, e.g. RMSE), this generic path fixes
        the sklearn convention (accuracy, R², sklearn's ``neg_*`` losses) so CV means compare
        cleanly across engines without a separate ``greater_is_better`` flag.
        """


class SklearnEngine(Engine):
    """Wraps any sklearn-compatible estimator (fit/predict, optionally predict_proba). Cloned
    fresh per fold via :func:`sklearn.base.clone` so folds never share fitted state."""

    def __init__(self, estimator, *, scoring: str | None = None):
        self._template = estimator
        self._scoring = scoring
        self._fitted = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnEngine":
        from sklearn.base import clone

        self._fitted = clone(self._template).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self._fitted.predict(X), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame | None:
        if not hasattr(self._fitted, "predict_proba"):
            return None
        proba = self._fitted.predict_proba(X)
        return pd.DataFrame(proba, index=X.index, columns=self._fitted.classes_)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        if self._scoring is not None:
            from sklearn.metrics import get_scorer

            return float(get_scorer(self._scoring)(self._fitted, X, y))
        return float(self._fitted.score(X, y))  # sklearn default: accuracy / R^2, both gib=True


class AutoGluonEngine(Engine):
    """Adapter over AutoGluon, for API symmetry with :class:`SklearnEngine` and standalone use
    (e.g. directly via ``compare()``). NOT what ``validation.py::cross_validate`` calls for its
    own default AutoGluon path — that path's presets/sample_weight/custom-Scorer/target-transform
    support has no equivalent in this minimal interface, so ``cross_validate`` keeps its
    existing, untouched code for ``engine=None``/an ``AutoGluonEngine`` instance (see module
    docstring) rather than routing through ``fit``/``predict`` here.
    """

    def __init__(self, target: str, *, model_dir: str, time_limit: int,
                eval_metric: str | None = None, presets: str | None = None):
        self._target = target
        self._model_dir = model_dir
        self._time_limit = time_limit
        self._eval_metric = eval_metric
        self._presets = presets
        self._result: TrainingResult | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AutoGluonEngine":
        train = X.assign(**{self._target: y.to_numpy()})
        self._result = fit_predictor(train, self._target, self._time_limit, self._model_dir,
                                     eval_metric=self._eval_metric, presets=self._presets)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return predict(self._result.predictor, X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame | None:
        if self._result.problem_type not in ("binary", "multiclass"):
            return None
        return predict_proba(self._result.predictor, X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        gib = getattr(self._result.predictor.eval_metric, "greater_is_better", True)
        val = X.assign(**{self._target: y.to_numpy()})
        metrics = self._result.predictor.evaluate(val, silent=True)
        raw = metrics.get(self._result.eval_metric) or next(iter(metrics.values()))
        return float(raw) if gib else -float(raw)  # normalise to this Engine's gib=True contract
