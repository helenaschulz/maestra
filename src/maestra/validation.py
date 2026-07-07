"""Trustworthy validation: leakage-free cross-validation + adversarial validation.

Cross-validation is the foundation for making Maestra competitive: a single random holdout
gives a noisy estimate, and (worse) any preprocessing fitted on the whole dataset leaks
across the split. Here every fold re-fits cleaning and feature engineering on its OWN
training part via the existing fit/transform separation, so the score is honest.

Adversarial validation checks whether train and test are drawn from the same distribution
by training a classifier to tell them apart; an AUC near 0.5 means no detectable shift.

``cross_validate``'s ``engine`` parameter (P3) is engine-agnostic in name only: the default
(``None``/an ``AutoGluonEngine``) is the untouched AutoGluon path above; any OTHER
:class:`~maestra.engine.Engine` (e.g. ``SklearnEngine``) takes the separate, simpler
``_cross_validate_with_engine`` path, used by ``compare()`` to run the arbiter over arbitrary
sklearn-compatible estimators.
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
from maestra.engine import AutoGluonEngine, Engine, predict, predict_proba, train_and_evaluate
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


def paired_delta_test(deltas: list[float], *, sigma_mult: float = 2.0,
                      min_abs: float = 1e-4, test_train_ratio: float = 0.0) -> bool:
    """The arbiter's core accept rule on PAIRED differences (fold-wise or seed-wise).

    Passes only if the mean delta exceeds ``sigma_mult`` standard errors of the deltas AND a
    strict majority of the pairs improved. Pairing removes shared difficulty variance; the
    majority clause guards against a single outlier pair carrying the mean.

    ``test_train_ratio`` (n_val / n_train per replication) applies the Nadeau-Bengio (2003)
    variance inflation for comparisons built from overlapping-training-set replications (k-fold
    CV, or repeated holdout carves over the same pool): SEM² = S² * (1/n + test_train_ratio)
    instead of the naive S²/n, because replications sharing training data are not independent
    and the naive i.i.d. estimate understates the true variance. This is a conservative
    heuristic, not an unbiased correction (no unbiased estimator of k-fold CV variance exists —
    Bengio & Grandvalet 2004); it only pushes the accept bar higher, i.e. further into the safe
    (harder-to-accept) direction. Default 0.0 reproduces the original naive rule exactly.
    """
    n = len(deltas)
    if n < 2:
        return False
    mean = float(np.mean(deltas))
    variance_factor = 1.0 / n + test_train_ratio
    sem = float(np.std(deltas, ddof=1)) * (variance_factor ** 0.5)
    majority = sum(1 for d in deltas if d > 0) > n / 2
    return mean > max(min_abs, sigma_mult * sem) and majority


def paired_delta_mde(std: float, n: int, *, sigma_mult: float = 2.0,
                     test_train_ratio: float = 0.0) -> float:
    """The minimum mean paired delta that would clear ``paired_delta_test`` at this spread,
    sample size and ``test_train_ratio`` — the threshold an "undecided" verdict fell short of.

    Reported alongside a verdict so "undecided" reads as "no effect at least this large",
    not an unlabeled non-result — the same ``std``/``n``/``test_train_ratio`` that produced
    the verdict determine this number, so it is exactly the boundary the mean delta missed.
    """
    if n < 2:
        return float("inf")
    variance_factor = 1.0 / n + test_train_ratio
    return sigma_mult * std * (variance_factor ** 0.5)


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

    The paired branch additionally applies the Nadeau-Bengio variance inflation
    (``test_train_ratio = 1/(k-1)`` for k folds — the standard per-fold test/train size ratio
    of a k-fold split) via :func:`paired_delta_test`: k-fold replications share overlapping
    training data, so the naive SEM across fold deltas understates the true variance (N1,
    2026-07-05). Still no correction for greedy multiple testing across a candidate sequence
    (hybrid/text-feature gates) — deferred, not fixed; those lanes are measured-null and frozen.
    """
    delta = (trial.mean - base.mean) if base.greater_is_better else (base.mean - trial.mean)
    paired_ok = (base.fold_scores and trial.fold_scores
                 and len(base.fold_scores) == len(trial.fold_scores) >= 2)
    if paired_ok:
        sign = 1.0 if base.greater_is_better else -1.0
        d = [sign * (t - b) for b, t in zip(base.fold_scores, trial.fold_scores)]
        k = base.n_folds
        ratio = 1.0 / (k - 1) if k > 1 else 0.0
        return paired_delta_test(d, sigma_mult=sigma_mult, min_abs=min_abs,
                                 test_train_ratio=ratio), float(delta)
    return delta > max(min_abs, sigma_mult * base.std), float(delta)


def _ag_score(y_true: pd.Series, y_pred: pd.Series, metric: str) -> float:
    """Score regression predictions in AutoGluon's reporting convention (higher is better,
    error metrics negated). Used when a target transform makes the predictor's own metric
    (computed in transformed space) incomparable — the trial must be scored in ORIGINAL space
    with the same sign convention as the untransformed base CV, or the paired test is invalid."""
    t = y_true.to_numpy(dtype=float)
    p = y_pred.to_numpy(dtype=float)
    if metric == "root_mean_squared_error":
        return -float(np.sqrt(np.mean((t - p) ** 2)))
    if metric == "mean_absolute_error":
        return -float(np.mean(np.abs(t - p)))
    if metric == "r2":
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - np.mean(t)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot else 0.0
    raise ValueError(f"metric {metric!r} not supported for target-transformed scoring")


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
                group_column: str | None = None, time_column: str | None = None,
                period_column: str | None = None):
    """Yield ``(train_idx, val_idx)`` pairs (positional). The single place fold strategy is
    chosen. ``group_column`` keeps every entity's rows in ONE fold (GroupKFold);
    ``time_column`` yields expanding-window splits where validation is strictly later than
    training (TimeSeriesSplit over the time-sorted order); ``time_column`` + ``period_column``
    together yield LOCAL within-period blocked folds (:func:`_time_local_folds`) for a
    deployment split that repeats every period rather than cutting the whole timeline once.
    All three override stratification — they exist precisely because a random/stratified
    split would lie."""
    if group_column is not None:
        # Keep stratification when the target is a class label: StratifiedGroupKFold balances the
        # class mix across folds while still never splitting an entity.
        if stratified:
            splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            splitter = GroupKFold(n_splits=n_folds)
        return list(splitter.split(df, df[target], groups=df[group_column]))
    if time_column is not None and period_column is not None:
        return _time_local_folds(df, time_column, period_column, n_folds)
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


def _time_local_folds(df: pd.DataFrame, time_column: str, period_column: str, n_folds: int):
    """Blocked expanding folds WITHIN each period, pooled across periods.

    A plain global ``time_column`` split cuts the whole timeline once — over many periods of
    strong seasonality/trend, an early fold trains on very little data and validates on a large,
    distributionally different future block (expanding-window bias, see :func:`_make_folds`'s
    ``time_column``-only branch). Real deployment is often a REPEATING local split instead — the
    last N days of every month, the last visits of every patient's history — where every period
    contributes both train and validation rows in every fold.

    Each period's rows are time-sorted and cut into ``n_folds + 1`` contiguous blocks; fold i
    trains on blocks ``0..i`` and validates on block ``i + 1``, unioned over every period. Every
    fold is exposed to every period (curing the global split's bias) while validation within
    each period still comes strictly after that period's training rows (matching a local,
    repeating deployment split). Positional indices, like every branch of :func:`_make_folds`.
    """
    values = df[time_column]
    if values.dtype.kind not in "iufM":
        values = pd.to_datetime(values, errors="coerce", format="mixed")
    values = values.to_numpy()
    train_parts: list[list] = [[] for _ in range(n_folds)]
    val_parts: list[list] = [[] for _ in range(n_folds)]
    for _, idx in df.groupby(period_column, sort=False).indices.items():
        idx = idx[np.argsort(values[idx], kind="stable")]  # time-sorted, within this period only
        blocks = np.array_split(idx, n_folds + 1)
        for i in range(n_folds):
            train_parts[i].append(np.concatenate(blocks[: i + 1]))
            val_parts[i].append(blocks[i + 1])
    return [(np.concatenate(train_parts[i]), np.concatenate(val_parts[i])) for i in range(n_folds)]


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


def _cross_validate_with_engine(df, target, cleaning_plan, feature_plan, folds, n_folds,
                                stratified, engine: Engine, generated_features, classification):
    """The P3 engine-agnostic CV path (see :func:`cross_validate`'s ``engine`` doc): one
    ``engine.fit``/``.score`` per fold, reusing the same fold-independent cleaning/FE fitting
    as the AutoGluon path. ``engine.score``'s contract (see :class:`~maestra.engine.Engine`)
    fixes greater-is-better, so no metric-direction bookkeeping is needed here."""
    fold_scores = []
    for train_idx, val_idx in folds:
        proc_train, proc_val = _process_fold(
            df.iloc[train_idx], df.iloc[val_idx], target, cleaning_plan, feature_plan,
            generated_features)
        X_train, y_train = proc_train.drop(columns=[target]), proc_train[target]
        X_val, y_val = proc_val.drop(columns=[target]), proc_val[target]
        fitted = engine.fit(X_train, y_train)
        fold_scores.append(float(fitted.score(X_val, y_val)))
    return CVResult(
        eval_metric=f"{type(engine).__name__}.score",
        problem_type="classification" if classification else "regression",
        fold_scores=fold_scores,
        mean=float(np.mean(fold_scores)),
        std=float(np.std(fold_scores)),
        n_folds=n_folds,
        stratified=stratified,
        greater_is_better=True,
    )


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
    period_column: str | None = None,
    presets: str | None = None,
    sample_weight: str | None = None,
    target_transform=None,
    engine: "Engine | None" = None,
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
        target_transform: Optional :class:`~maestra.target_framing.TargetTransform`
            (regression only). Each fold trains on ``forward(target)``; predictions are
            inverted back and the fold is scored in ORIGINAL space via :func:`_ag_score`,
            so the result pairs cleanly against an untransformed base CV on the same folds.
        engine: ``None`` (default) or an :class:`~maestra.engine.AutoGluonEngine` instance ->
            the AutoGluon path below, unchanged (``time_limit``/``model_dir``/``presets``/
            ``sample_weight``/``target_transform``/a custom AutoGluon ``Scorer`` all apply).
            Any OTHER :class:`~maestra.engine.Engine` (e.g. :class:`~maestra.engine.SklearnEngine`)
            -> a separate, simpler path (P3): per-fold ``engine.fit``/``.score``, no
            presets/sample_weight/target_transform/custom-Scorer support (AutoGluon-native
            concepts with no equivalent here), ``eval_metric`` unused, OOF predictions/probs
            not collected. Used by ``compare()`` to run the arbiter over arbitrary
            sklearn-compatible estimators.

    Returns:
        A :class:`CVResult` with the per-fold scores and their mean/std.
    """
    classification = _is_classification(df[target])
    use_stratified = classification if stratified is None else (stratified and classification)
    folds = _make_folds(df, target, n_folds, seed, use_stratified,
                        group_column=group_column, time_column=time_column,
                        period_column=period_column)

    if engine is not None and not isinstance(engine, AutoGluonEngine):
        return _cross_validate_with_engine(df, target, cleaning_plan, feature_plan, folds,
                                           n_folds, use_stratified, engine, generated_features,
                                           classification)

    fold_scores: list[float] = []
    # ``eval_metric`` (a metric string OR a custom AutoGluon Scorer object) is passed UNCHANGED
    # to every fold, so the CV estimate uses the same metric as the final model — required for
    # the CV↔LB gap to compare like with like. ``metric_name`` is AutoGluon's resolved name,
    # captured from the first fold, for scoring the folds and labelling the CVResult. (Before
    # 2026-07-06 this argument was silently overwritten with None, so the CV always used
    # AutoGluon's default metric — harmless while every task's metric equalled the default, wrong
    # for a task that asks for mae/roc_auc/a custom scorer.)
    metric_arg = eval_metric
    metric_name = problem_type = None
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
        val_true = None
        if target_transform is not None:
            # Train (and let AutoGluon internally validate) in transformed space; keep the
            # original-space truth aside for honest scoring below.
            val_true = proc_val[target]
            proc_train = proc_train.assign(**{target: target_transform.forward(proc_train[target])})
            proc_val = proc_val.assign(**{target: target_transform.forward(proc_val[target])})
        result = train_and_evaluate(proc_train, proc_val, target, time_limit,
                                    f"{model_dir}/fold_{i}", metric_arg, presets=presets,
                                    sample_weight=sample_weight)
        metric_name, problem_type = result.eval_metric, result.problem_type
        preds_orig = None
        if result.predictor is not None:
            greater_is_better = getattr(result.predictor.eval_metric, "greater_is_better", True)
            val_features = proc_val.drop(columns=[target], errors="ignore")
            # Out-of-fold predictions: each row predicted by a model that did not see it.
            # With a target transform, stored in ORIGINAL space (inverted).
            preds = predict(result.predictor, val_features)
            preds_orig = target_transform.inverse(preds) if target_transform is not None else preds
            oof_pred.loc[proc_val.index] = preds_orig.to_numpy()
            # Out-of-fold class probabilities (classification only — regression has none), so
            # probability metrics (roc_auc / log_loss) can be scored on held-out rows.
            if classification:
                fold_proba = predict_proba(result.predictor, val_features)
                aligned = fold_proba.reindex(columns=oof_classes, fill_value=0.0)
                # Positional assignment: a BOOLEAN target (oof_classes == [False, True]) makes
                # `.loc[rows, oof_classes]` ambiguous -- pandas reads a bool list as a boolean
                # MASK over the columns, not as labels, silently selecting the wrong shape.
                # get_indexer sidesteps that regardless of the class label dtype.
                row_pos = oof_proba.index.get_indexer(proc_val.index)
                col_pos = oof_proba.columns.get_indexer(oof_classes)
                oof_proba.iloc[row_pos, col_pos] = aligned.to_numpy()
        if target_transform is not None:
            # The predictor's own metric lives in transformed space and is NOT comparable to an
            # untransformed base CV — rescore in original space, same sign convention.
            if preds_orig is None:
                raise ValueError("target_transform requires a fitted predictor to score folds")
            score = _ag_score(val_true, preds_orig, metric_name)
        else:
            score = result.metrics.get(metric_name)
            if score is None:  # fall back to any reported value
                score = next(iter(result.metrics.values()))
        fold_scores.append(float(score))

    return CVResult(
        eval_metric=metric_name,
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
