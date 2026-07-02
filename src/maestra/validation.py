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
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.utils.multiclass import type_of_target

from maestra.cleaning import fit_cleaning_plan
from maestra.engine import predict, predict_proba, train_and_evaluate
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
    oof_proba: "pd.DataFrame | None" = None  # out-of-fold class probabilities (df-indexed,
    # one column per class), for probability metrics (roc_auc / log_loss)


def improves_beyond_noise(base: "CVResult", trial: "CVResult", *, sigma_mult: float = 2.0,
                          min_abs: float = 1e-4) -> tuple[bool, float]:
    """Does ``trial`` beat ``base`` beyond fold noise? Returns ``(passes, mean_delta)``.

    This is the arbiter's accept rule, shared by every gate (hybrid features, Skeptic vetoes).
    Both CVs run on IDENTICAL folds, so when per-fold scores are available the comparison is
    **paired**: the trial must (a) improve the mean paired delta beyond ``sigma_mult`` standard
    errors of the paired deltas and (b) improve in a strict majority of folds. Pairing removes
    the shared fold-difficulty variance that made the old rule (mean vs. ``sigma_mult * std`` of
    3 correlated fold scores, ~1-in-6 false pass per candidate under the null, no correction for
    greedy multiple testing) far too permissive. Without per-fold scores (older records, mocks)
    it falls back to the mean-vs-std rule with the same ``sigma_mult``.
    """
    delta = (trial.mean - base.mean) if base.greater_is_better else (base.mean - trial.mean)
    paired_ok = (base.fold_scores and trial.fold_scores
                 and len(base.fold_scores) == len(trial.fold_scores) >= 2)
    if paired_ok:
        sign = 1.0 if base.greater_is_better else -1.0
        d = [sign * (t - b) for b, t in zip(base.fold_scores, trial.fold_scores)]
        n = len(d)
        sem = float(np.std(d, ddof=1)) / (n ** 0.5)
        majority = sum(1 for x in d if x > 0) > n / 2
        return (delta > max(min_abs, sigma_mult * sem)) and majority, float(delta)
    return delta > max(min_abs, sigma_mult * base.std), float(delta)


def _is_classification(y: pd.Series) -> bool:
    """True if ``y`` should be stratified like a class label.

    sklearn's ``type_of_target`` calls ANY integer-coded column with >2 distinct values
    "multiclass" — including regression targets like a sale price — which would send a
    continuous target into StratifiedKFold (crashing on singleton "classes"). So for numeric
    targets we additionally require a small number of distinct values; string/bool targets
    stay classification as-is. AutoGluon applies its own, similar inference for training.
    """
    y = y.dropna()
    if type_of_target(y) not in ("binary", "multiclass"):
        return False
    if y.dtype.kind in "iuf":  # numeric: many distinct values = a number, not class codes
        return y.nunique() <= 20
    return True


def _make_folds(df: pd.DataFrame, target: str, n_folds: int, seed: int, stratified: bool,
                group_column: str | None = None, time_column: str | None = None):
    """Yield ``(train_idx, val_idx)`` pairs (positional). The single place fold strategy is
    chosen. ``group_column`` keeps every entity's rows in ONE fold (GroupKFold);
    ``time_column`` yields expanding-window splits where validation is strictly later than
    training (TimeSeriesSplit over the time-sorted order). Both override stratification —
    they exist precisely because a random/stratified split would lie."""
    if group_column is not None:
        # Keep stratification when the target is a class label: StratifiedGroupKFold balances the
        # class mix across folds while still never splitting an entity.
        if stratified:
            splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            splitter = GroupKFold(n_splits=n_folds)
        return list(splitter.split(df, df[target], groups=df[group_column]))
    if time_column is not None:
        values = df[time_column]
        if values.dtype.kind not in "iufM":
            values = pd.to_datetime(values, errors="coerce", format="mixed")
        order = np.argsort(values.to_numpy(), kind="stable")  # positional, past -> future
        # Note: expanding windows train on growing prefixes, so the fold scores are not
        # identically distributed and their mean is a biased (typically pessimistic) estimate.
        # That is inherent to honest temporal validation, not a defect to fix here.
        splitter = TimeSeriesSplit(n_splits=n_folds)
        return [(order[tr], order[va]) for tr, va in splitter.split(order)]
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
    group_column: str | None = None,
    time_column: str | None = None,
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
    folds = _make_folds(df, target, n_folds, seed, use_stratified,
                        group_column=group_column, time_column=time_column)

    fold_scores: list[float] = []
    eval_metric = problem_type = None
    greater_is_better = True
    oof_pred = pd.Series([None] * len(df), index=df.index, dtype=object)
    # Fixed, complete column set for OOF probabilities: EVERY class in the data, stable order.
    # A fold whose training part happens to miss a rare class returns proba without that column;
    # reindexing each fold onto this full set (absent class -> 0) keeps every pooled row a valid
    # probability vector that sums to 1. Initialising from the first fold's columns instead left
    # later rows with NaNs and silently blew up any probability metric (log_loss / roc_auc).
    oof_classes = sorted(df[target].dropna().unique().tolist()) if classification else []
    oof_proba = pd.DataFrame(index=df.index, columns=oof_classes, dtype=float) if classification else None
    for i, (train_idx, val_idx) in enumerate(folds):
        proc_train, proc_val = _process_fold(
            df.iloc[train_idx], df.iloc[val_idx], target, cleaning_plan, feature_plan, generated_features
        )
        result = train_and_evaluate(proc_train, proc_val, target, time_limit, f"{model_dir}/fold_{i}", eval_metric)
        eval_metric, problem_type = result.eval_metric, result.problem_type
        if result.predictor is not None:
            greater_is_better = getattr(result.predictor.eval_metric, "greater_is_better", True)
            val_features = proc_val.drop(columns=[target], errors="ignore")
            # Out-of-fold predictions: each row predicted by a model that did not see it.
            oof_pred.loc[proc_val.index] = predict(result.predictor, val_features).to_numpy()
            # Out-of-fold class probabilities (classification only — regression has none), so
            # probability metrics (roc_auc / log_loss) can be scored on held-out rows.
            if classification:
                fold_proba = predict_proba(result.predictor, val_features)
                aligned = fold_proba.reindex(columns=oof_classes, fill_value=0.0)
                oof_proba.loc[proc_val.index, oof_classes] = aligned.to_numpy()
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
        oof_proba=oof_proba,
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
