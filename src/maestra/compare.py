"""Public API (P3): honestly compare two sklearn-compatible estimators via the same paired,
Nadeau-Bengio-corrected arbiter (`validation.py::paired_delta_test`) every internal gate uses —
no LLM call, no AutoGluon required (see `engine.py`'s docstring on lazy AutoGluon imports).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CompareResult:
    """The arbiter's verdict on ``estimator_b`` vs. ``estimator_a`` (the baseline)."""

    verdict: str  # "improved" | "no_improvement" | "underpowered"
    mean_delta: float  # signed so positive always means "b is better", any metric direction
    deltas: list[float] = field(default_factory=list)  # pooled per (seed, fold) paired deltas
    mde: float = float("inf")  # minimum |mean_delta| that would have cleared the accept bar
    metric: str = ""
    greater_is_better: bool = True
    n_folds: int = 0
    n_seeds: int = 0
    a_mean: float = 0.0
    b_mean: float = 0.0

    def summary(self) -> str:
        """A short Markdown block, ready to paste into a PR description."""
        direction = "higher is better" if self.greater_is_better else "lower is better"
        lines = [
            f"**compare() verdict: {self.verdict}** ({self.metric}, {direction})",
            f"- a (baseline): {self.a_mean:.4f} · b: {self.b_mean:.4f}",
            f"- mean delta (b − a, signed so +ve = b better): {self.mean_delta:+.4f}",
            f"- minimum detectable effect: {self.mde:.4f}",
            f"- {self.n_seeds} seed(s) × {self.n_folds} fold(s), "
            f"{len(self.deltas)} paired observation(s)",
        ]
        return "\n".join(lines)


def compare(estimator_a, estimator_b, df: pd.DataFrame, target: str, *, cv: int = 5,
           seeds: int = 1, metric: str | None = None, splitter=None) -> CompareResult:
    """Does ``estimator_b`` measurably beat ``estimator_a`` on ``df``/``target``?

    Both estimators run k-fold CV on IDENTICAL folds (same seed, same splits) via
    :class:`~maestra.engine.SklearnEngine`, so the per-fold deltas are paired — the same
    accept rule as every internal gate (:func:`~maestra.validation.paired_delta_test`, with
    the Nadeau-Bengio variance correction for the k-fold train/val overlap). No cleaning/FE
    plan (raw ``df`` columns only) and no LLM call — this is a pure statistical comparison.

    ``metric``: a scorer name understood by :func:`sklearn.metrics.get_scorer` (e.g.
    ``"neg_mean_squared_error"``, ``"roc_auc"``); ``None`` uses each estimator's own default
    ``.score()`` (accuracy for classifiers, R² for regressors — both higher-is-better).

    ``seeds > 1`` repeats the whole k-fold comparison with different fold splits (seeds
    ``0..seeds-1``); every (seed, fold) paired delta pools into one set for the accept test —
    more paired observations, more statistical power, without changing what a "fold" means.

    ``splitter`` (F2): an arbitrary sklearn-compatible splitter (e.g.
    :class:`~maestra.validation.RollingOriginSplit`) overriding the plain k-fold split above —
    passed identically to both estimators, so the paired deltas stay valid. Its ACTUAL fold
    count (not ``cv``) then drives both the Nadeau-Bengio ratio and the reported ``n_folds``.

    Returns a :class:`CompareResult` — never a fitted model.
    """
    from maestra.engine import SklearnEngine
    from maestra.validation import cross_validate, paired_delta_mde, paired_delta_test

    all_deltas: list[float] = []
    a_means, b_means = [], []
    greater_is_better, metric_name = True, metric or "estimator.score"
    n_folds_used = cv
    for seed in range(seeds):
        cv_a = cross_validate(df, target, cleaning_plan=None, feature_plan=None,
                              model_dir="compare/a", time_limit=1, n_folds=cv, seed=seed,
                              engine=SklearnEngine(estimator_a, scoring=metric), splitter=splitter)
        cv_b = cross_validate(df, target, cleaning_plan=None, feature_plan=None,
                              model_dir="compare/b", time_limit=1, n_folds=cv, seed=seed,
                              engine=SklearnEngine(estimator_b, scoring=metric), splitter=splitter)
        greater_is_better = cv_a.greater_is_better  # both engines share this fixed contract
        n_folds_used = cv_a.n_folds  # the ACTUAL fold count -- a splitter may not match cv
        a_means.append(cv_a.mean)
        b_means.append(cv_b.mean)
        all_deltas.extend(b - a for a, b in zip(cv_a.fold_scores, cv_b.fold_scores))

    # test_train_ratio = 1/(n_folds_used-1): the same per-fold train/val overlap ratio
    # `improves_beyond_noise` uses for a k-fold paired comparison (N1, 2026-07-05).
    ratio = 1.0 / (n_folds_used - 1) if n_folds_used > 1 else 0.0
    if len(all_deltas) < 2:
        verdict = "underpowered"
    elif paired_delta_test(all_deltas, test_train_ratio=ratio):
        verdict = "improved"
    else:
        verdict = "no_improvement"
    std = float(np.std(all_deltas, ddof=1)) if len(all_deltas) >= 2 else 0.0
    mde = paired_delta_mde(std, len(all_deltas), test_train_ratio=ratio)

    return CompareResult(
        verdict=verdict, mean_delta=float(np.mean(all_deltas)), deltas=all_deltas, mde=mde,
        metric=metric_name, greater_is_better=greater_is_better, n_folds=n_folds_used, n_seeds=seeds,
        a_mean=float(np.mean(a_means)), b_mean=float(np.mean(b_means)),
    )
