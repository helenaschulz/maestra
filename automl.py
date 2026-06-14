"""AutoGluon-Engine: Split + Training + Holdout-Metriken. Kein LLM.

Das LLM rechnet nichts -- Modellwahl, Hyperparameter und Metriken macht ausschliesslich
AutoGluon.
"""
from __future__ import annotations

import pandas as pd
from autogluon.tabular import TabularPredictor


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = df.sample(frac=test_size, random_state=seed)
    train = df.drop(index=test.index)
    return train, test


def train_and_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    time_limit: int,
    model_dir: str,
) -> dict:
    predictor = TabularPredictor(label=target, path=model_dir).fit(train, time_limit=time_limit)

    print(f"\nProblemtyp (von AutoGluon inferiert): {predictor.problem_type}")
    print(f"Eval-Metrik: {predictor.eval_metric.name}")

    print("\n=== Leaderboard auf Holdout ===")
    print(predictor.leaderboard(test))

    print("\n=== Metriken bestes Modell auf Holdout ===")
    perf = predictor.evaluate(test)
    for k, v in perf.items():
        print(f"  {k}: {v}")
    return perf
