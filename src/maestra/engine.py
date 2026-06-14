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
    predictor = TabularPredictor(label=target, path=model_dir).fit(train, time_limit=time_limit)
    return TrainingResult(
        problem_type=predictor.problem_type,
        eval_metric=predictor.eval_metric.name,
        leaderboard=predictor.leaderboard(holdout, silent=True),
        metrics=predictor.evaluate(holdout, silent=True),
        predictor=predictor,
    )


def predict(predictor, X: pd.DataFrame) -> pd.Series:
    """Predict labels for ``X`` with a fitted predictor (delegates to AutoGluon)."""
    return predictor.predict(X)
