"""AutoGluon engine — the only place numbers are crunched.

The LLM never computes. Train/holdout split, model search, hyperparameter tuning and
metric calculation all happen here, inside AutoGluon. This module returns plain data
(leaderboard, metrics) and prints nothing, so it is easy to test and reuse.
"""
from __future__ import annotations

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
) -> TrainingResult:
    """Fit an AutoGluon predictor and evaluate it on the holdout set.

    Args:
        train: Training rows (features + target).
        holdout: Held-out rows for honest evaluation.
        target: Name of the label column.
        time_limit: Training budget in seconds.
        model_dir: Where AutoGluon persists the trained models.

    Returns:
        A :class:`TrainingResult`. The problem type and evaluation metric are inferred
        by AutoGluon from the target column.
    """
    predictor = TabularPredictor(label=target, path=model_dir, eval_metric=eval_metric).fit(
        train, time_limit=time_limit)
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
                  eval_metric: str | None = None) -> TrainingResult:
    """Fit a predictor on ALL labeled rows, with no holdout evaluation.

    Used by the cross-validation path, where the honest estimate is the CV score and this
    final model exists only for prediction (submission/report). ``metrics`` is therefore
    empty; ``val_score`` is AutoGluon's internal validation score.
    """
    predictor = TabularPredictor(label=target, path=model_dir, eval_metric=eval_metric).fit(
        train, time_limit=time_limit)
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
