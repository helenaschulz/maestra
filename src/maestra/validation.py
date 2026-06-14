"""Trustworthy validation: leakage-free cross-validation + adversarial validation.

Cross-validation is the foundation for making Maestra competitive: a single random holdout
gives a noisy estimate, and (worse) any preprocessing fitted on the whole dataset leaks
across the split. Here every fold re-fits cleaning and feature engineering on its OWN
training part via the existing fit/transform separation, so the score is honest.

Adversarial validation checks whether train and test are drawn from the same distribution
by training a classifier to tell them apart; an AUC near 0.5 means no detectable shift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target

from maestra.cleaning import fit_cleaning_plan
from maestra.engine import predict, train_and_evaluate
from maestra.feature_engineering import fit_feature_plan

_ADV_LABEL = "__is_test__"


@dataclass
class CVResult:
    """A cross-validation estimate: per-fold scores plus their mean and spread."""

    eval_metric: str
    problem_type: str
    fold_scores: list[float]
    mean: float
    std: float
    n_folds: int
    stratified: bool
    greater_is_better: bool = True  # metric direction, for comparing CV means
    oof_pred: "pd.Series | None" = None  # out-of-fold predictions (df-indexed), for custom metrics


def _is_classification(y: pd.Series) -> bool:
    return type_of_target(y.dropna()) in ("binary", "multiclass")


def _make_folds(df: pd.DataFrame, target: str, n_folds: int, seed: int, stratified: bool):
    """Yield ``(train_idx, val_idx)`` pairs. The single place fold strategy is chosen —
    group-/time-based splitters slot in here without touching the rest of the pipeline."""
    if stratified:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(splitter.split(df, df[target]))
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(df))


def _process_fold(fold_train, fold_val, target, cleaning_plan, feature_plan, generated_features=None):
    """Clean + engineer a fold, fitting EVERY parameter on the fold's train part only.

    This is the heart of the leakage guarantee: imputation values and bin edges come from
    ``fold_train`` and are applied unchanged to ``fold_val``.
    """
    train, val = fold_train, fold_val
    if cleaning_plan is not None:
        ct = fit_cleaning_plan(train, cleaning_plan, target)
        train, val = ct.transform(train), ct.transform(val)
    if feature_plan is not None:
        ft = fit_feature_plan(train, feature_plan, target)
        train, val = ft.transform(train), ft.transform(val)
    if generated_features:
        # Lazy import avoids a top-level cycle (hybrid_features imports this module).
        from maestra.hybrid_features import apply_generated_features
        train, val = apply_generated_features(train, val, target, generated_features)
    return train, val


def cross_validate(
    df: pd.DataFrame,
    target: str,
    *,
    cleaning_plan: dict | None,
    feature_plan: dict | None,
    model_dir: str,
    time_limit: int,
    n_folds: int = 5,
    seed: int = 42,
    stratified: bool | None = None,
    generated_features: list | None = None,
    eval_metric: str | None = None,
) -> CVResult:
    """Leakage-free k-fold cross-validation of the (cleaning, FE) plans on ``df``.

    Args:
        df: Full labeled dataset.
        target: Target column.
        cleaning_plan / feature_plan: Plans whose *parameters* are re-fitted per fold.
        model_dir: Base dir for AutoGluon artefacts (one sub-dir per fold).
        time_limit: Training budget per fold.
        n_folds: Number of folds (>= 2).
        seed: Seed for the (deterministic) fold split.
        stratified: Force stratification on/off; default = on for classification targets.

    Returns:
        A :class:`CVResult` with the per-fold scores and their mean/std.
    """
    classification = _is_classification(df[target])
    use_stratified = classification if stratified is None else (stratified and classification)
    folds = _make_folds(df, target, n_folds, seed, use_stratified)

    fold_scores: list[float] = []
    eval_metric = problem_type = None
    greater_is_better = True
    oof_pred = pd.Series([None] * len(df), index=df.index, dtype=object)
    for i, (train_idx, val_idx) in enumerate(folds):
        proc_train, proc_val = _process_fold(
            df.iloc[train_idx], df.iloc[val_idx], target, cleaning_plan, feature_plan, generated_features
        )
        result = train_and_evaluate(proc_train, proc_val, target, time_limit, f"{model_dir}/fold_{i}", eval_metric)
        eval_metric, problem_type = result.eval_metric, result.problem_type
        if result.predictor is not None:
            greater_is_better = getattr(result.predictor.eval_metric, "greater_is_better", True)
            # Out-of-fold predictions: each row predicted by a model that did not see it.
            oof_pred.loc[proc_val.index] = predict(
                result.predictor, proc_val.drop(columns=[target], errors="ignore")).to_numpy()
        score = result.metrics.get(eval_metric)
        if score is None:  # fall back to any reported value
            score = next(iter(result.metrics.values()))
        fold_scores.append(float(score))

    return CVResult(
        eval_metric=eval_metric,
        problem_type=problem_type,
        fold_scores=fold_scores,
        mean=float(np.mean(fold_scores)),
        std=float(np.std(fold_scores)),
        n_folds=n_folds,
        stratified=use_stratified,
        greater_is_better=bool(greater_is_better),
        oof_pred=oof_pred,
    )


def _build_adversarial_data(train_df, test_df, target, cleaning_plan):
    """Combine train/test feature rows with an ``is-test`` label, on the cleaned features.

    Cleaning is fitted on train and applied to both so that obvious identifier columns are
    dropped — otherwise disjoint train/test id ranges would fake a perfect shift.
    Returns ``(data, feature_columns)`` or ``(None, [])`` if there are no shared features.
    """
    if cleaning_plan is not None:
        ct = fit_cleaning_plan(train_df, cleaning_plan, target)
        train_c, test_c = ct.transform(train_df), ct.transform(test_df)
    else:
        train_c, test_c = train_df, test_df

    feats = [c for c in train_c.columns if c != target and c in test_c.columns]
    if not feats:
        return None, []
    a = train_c[feats].copy()
    a[_ADV_LABEL] = 0
    b = test_c[feats].copy()
    b[_ADV_LABEL] = 1
    return pd.concat([a, b], ignore_index=True), feats


def adversarial_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    *,
    cleaning_plan: dict | None,
    model_dir: str,
    time_limit: int = 30,
) -> float | None:
    """Return the held-out AUC of a train-vs-test classifier (≈0.5 = no detectable shift).

    Returns None if there are no shared feature columns to compare.
    """
    data, feats = _build_adversarial_data(train_df, test_df, target, cleaning_plan)
    if data is None:
        return None
    predictor = TabularPredictor(label=_ADV_LABEL, eval_metric="roc_auc", path=model_dir).fit(
        data, time_limit=time_limit
    )
    leaderboard = predictor.leaderboard(silent=True)
    return float(leaderboard["score_val"].max()) if "score_val" in leaderboard.columns else None
