"""F2 Messung 4 — bike-sharing fold×metric ablation, MODEL HELD CONSTANT.

Turns the F2 ledger's honestly-cautious "whole-pipeline" caveat into a clean causal
attribution, and answers the open K1 question (was log1p's acceptance a fold-structure
artifact?) — without a fresh, non-deterministic Opus run.

The trick (advisor design, 2026-07-08): the exact cleaning + feature plan that produced the
deployed 0.43660-RMSLE submission is recoverable from `runs.jsonl` (the `count`-target entry at
`_RUN_TIMESTAMP`). Reusing that ONE plan across every cell holds the model constant — a fresh
Opus run per cell would vary the plan non-deterministically, reintroducing the very confound this
isolation removes. So this script makes ZERO LLM calls; it only re-runs AutoGluon CV.

2×2 matrix, everything but the two axes identical (same plan, seed=42, n_folds=3,
presets=best_quality, time_limit=1200, eval_metric=root_mean_squared_error):

    fold structure ∈ {random KFold, time_local (month_of:datetime)}
    target space   ∈ {raw counts, log1p}

For every cell we compute RMSLE from the out-of-fold predictions (`CVResult.oof_pred`, always in
COUNT space — so RMSLE is well-defined whether the model trained on raw or log1p). RMSLE is the
competition's own metric, so an OOF-RMSLE is directly comparable to the live LB (0.43660), unlike
the pipeline's raw-count-RMSE CV number.

What this measures:
  * CV↔LB honesty of the fold structure: (time_local, raw) OOF-RMSLE vs the deployed 0.43660
    (small gap = honest) against (random, raw) OOF-RMSLE (a large optimistic gap = "random lies").
    Model constant, so the fold structure is the isolated cause.
  * The K1 log1p question: run the SAME product arbiter (`improves_beyond_noise`, NB-corrected)
    in BOTH the raw-count-RMSE space (what the product arbiter actually uses) AND the RMSLE space,
    per fold structure. If log1p's accept/reject flips purely by changing the fold structure
    (model constant), fold structure is SUFFICIENT to flip the verdict. And if log1p loses in
    raw-count RMSE but wins in RMSLE, the arbiter is scoring the wrong metric for an RMSLE comp.

Sanity anchor (STOP condition): the (time_local, raw) cell must reproduce the logged deployed
fold_scores [-36.438, -128.074, -55.723] (mean -73.4117) within AutoGluon noise. If it doesn't,
the plan/config reconstruction is broken and no RMSLE from this run may be trusted.

    ./.venv/bin/python scripts/bike_sharing_fold_metric_ablation.py

Long AutoGluon job (12 best_quality fold-fits). Each cell's model dir is deleted after its scores
are extracted (F2 disk-hygiene: the deployed run alone hit ~20GB). Per-cell results stream to
`bike_metric_ablation.jsonl` so partial progress survives an interrupt.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd

from maestra.target_framing import _log1p_transform
from maestra.validation import (
    CVResult,
    _make_folds,
    cross_validate,
    improves_beyond_noise,
    paired_delta_mde,
)

_RUN_TIMESTAMP = "2026-07-08T05:47:43"   # the deployed 0.43660 submission run in runs.jsonl
_DEPLOYED_LB = 0.43660                   # public = private RMSLE, Helena-submitted 2026-07-08
_ANCHOR_FOLD_SCORES = [-36.438125983055784, -128.07368546302712, -55.72328129865119]
_SEED = 42
_N_FOLDS = 3
_PRESETS = "best_quality"
_TIME_LIMIT = 1200
_MODEL_ROOT = "AutogluonModels/ablation"
_OUT = "bike_metric_ablation.jsonl"


def _load_deployed_plan() -> tuple[dict, dict]:
    """The cleaning + feature plan of the 0.43660 run — reused verbatim so the model is constant."""
    rec = None
    with open("runs.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("timestamp") == _RUN_TIMESTAMP and d.get("target") == "count":
                rec = d
    if rec is None:
        raise SystemExit(f"could not find the deployed run ({_RUN_TIMESTAMP}) in runs.jsonl")
    return rec["plan"], rec["feature_plan"]


def _load_data() -> pd.DataFrame:
    """The exact battery loader shape: drop the two known leaks, keep raw `datetime`, RangeIndex."""
    df = pd.read_csv("data/kaggle_bike/train.csv")
    df = df.drop(columns=[c for c in ("casual", "registered") if c in df.columns])
    return df.reset_index(drop=True)


def _rmsle_from_oof(oof: pd.Series, y_true: pd.Series, idx=None) -> tuple[float, int]:
    """RMSLE over the rows an OOF prediction exists for (optionally restricted to ``idx``).

    Count-space throughout. ``np.maximum(pred, 0)`` mirrors the submission path's
    ``clip_nonneg=True`` (pipeline.py) — a negative count prediction would make log1p NaN.
    Returns ``(rmsle, n_rows_scored)``.
    """
    mask = oof.notna()
    if idx is not None:
        keep = pd.Index(idx).intersection(oof.index[mask])
    else:
        keep = oof.index[mask]
    pred = np.maximum(oof.loc[keep].to_numpy(dtype=float), 0.0)
    true = y_true.loc[keep].to_numpy(dtype=float)
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2))), len(keep)


def _per_fold_rmsle(df: pd.DataFrame, target: str, oof: pd.Series, *, time_column=None,
                    period_column=None) -> list[float]:
    """RMSLE within each fold's validation block — parallels how ``fold_scores`` are per-fold,
    so the RMSLE-space arbiter test is apples-to-apples with the raw-count-RMSE one."""
    folds = _make_folds(df, target, _N_FOLDS, _SEED, stratified=False,
                        time_column=time_column, period_column=period_column)
    out = []
    for _, val_idx in folds:
        val_labels = df.index[val_idx]
        rmsle, _ = _rmsle_from_oof(oof, df[target], idx=val_labels)
        out.append(rmsle)
    return out


def _covered_index(oof: pd.Series) -> pd.Index:
    return oof.index[oof.notna()]


def run_cell(df, target, plan, feat, *, name, target_transform, time_column, period_column):
    """One 2×2 cell → CVResult + derived RMSLE numbers; model dir deleted after extraction."""
    model_dir = f"{_MODEL_ROOT}/{name}"
    cv = cross_validate(
        df, target, cleaning_plan=plan, feature_plan=feat,
        model_dir=model_dir, time_limit=_TIME_LIMIT, n_folds=_N_FOLDS, seed=_SEED,
        eval_metric="root_mean_squared_error", presets=_PRESETS,
        time_column=time_column, period_column=period_column,
        target_transform=target_transform,
    )
    oof = cv.oof_pred
    rmsle_native, n_native = _rmsle_from_oof(oof, df[target])
    per_fold = _per_fold_rmsle(df, target, oof, time_column=time_column, period_column=period_column)
    result = {
        "cell": name,
        "raw_fold_scores": list(cv.fold_scores),   # negated RMSE (AG higher-is-better convention)
        "raw_rmse_mean": cv.mean,
        "raw_rmse_std": cv.std,
        "greater_is_better": cv.greater_is_better,
        "rmsle_native": rmsle_native,
        "rmsle_native_n": n_native,
        "rmsle_per_fold": per_fold,
        "covered_index": _covered_index(oof).tolist(),
    }
    with open(_OUT, "a") as fh:
        fh.write(json.dumps({k: v for k, v in result.items() if k != "covered_index"}) + "\n")
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)   # disk hygiene: scores already extracted
    return result, oof


def _rmsle_cvresult(per_fold_rmsle: list[float]) -> CVResult:
    """Wrap per-fold RMSLE as a CVResult in the arbiter's higher-is-better convention (score =
    -RMSLE), so `improves_beyond_noise` (NB-corrected, k=3) runs identically to the raw path."""
    scores = [-r for r in per_fold_rmsle]
    return CVResult(eval_metric="rmsle", problem_type="regression", fold_scores=scores,
                    mean=float(np.mean(scores)), std=float(np.std(scores)), n_folds=_N_FOLDS,
                    stratified=False, greater_is_better=True)


def main():
    if os.path.exists(_OUT):
        os.remove(_OUT)
    df = _load_data()
    plan, feat = _load_deployed_plan()
    log1p = _log1p_transform()

    # Order matters: the sanity-anchor cell runs FIRST so a broken reconstruction is caught before
    # the other three fits are spent.
    cells = [
        dict(name="time_local__raw", target_transform=None,
             time_column="datetime", period_column="month_of:datetime"),
        dict(name="random__raw", target_transform=None, time_column=None, period_column=None),
        dict(name="time_local__log1p", target_transform=log1p,
             time_column="datetime", period_column="month_of:datetime"),
        dict(name="random__log1p", target_transform=log1p, time_column=None, period_column=None),
    ]

    results, oofs = {}, {}
    for spec in cells:
        print(f"\n=== cell {spec['name']} (best_quality, tl={_TIME_LIMIT}, folds={_N_FOLDS}) ===",
              flush=True)
        res, oof = run_cell(df, "count", plan, feat, **spec)
        results[spec["name"]], oofs[spec["name"]] = res, oof
        print(f"  raw-count RMSE fold_scores: {[round(s, 3) for s in res['raw_fold_scores']]}"
              f"  mean {res['raw_rmse_mean']:.4f}", flush=True)
        print(f"  OOF-RMSLE (native, n={res['rmsle_native_n']}): {res['rmsle_native']:.5f}",
              flush=True)

        if spec["name"] == "time_local__raw":
            anchor_ok = np.allclose(sorted(res["raw_fold_scores"]), sorted(_ANCHOR_FOLD_SCORES),
                                    rtol=0.15, atol=8.0)
            print(f"\n  SANITY ANCHOR vs deployed {[round(s,3) for s in _ANCHOR_FOLD_SCORES]}: "
                  f"{'OK' if anchor_ok else 'MISMATCH'}", flush=True)
            if not anchor_ok:
                print("  ** anchor mismatch — reconstruction is off; RMSLE from this run is NOT "
                      "trustworthy. Investigate before continuing. **", flush=True)

    # --- common-support pooled RMSLE (decouples fold effect from time_local's partial coverage) ---
    tl_cov = _covered_index(oofs["time_local__raw"])
    rnd_cov = _covered_index(oofs["random__raw"])
    common = tl_cov.intersection(rnd_cov)
    cs = {}
    for name in ("time_local__raw", "random__raw"):
        cs[name], _ = _rmsle_from_oof(oofs[name], df["count"], idx=common)

    # --- arbiter verdicts, both metric spaces, per fold structure ---
    print("\n\n================  SUMMARY  ================", flush=True)
    print(f"deployed LB (RMSLE, public=private): {_DEPLOYED_LB}", flush=True)
    print("\nCV↔LB honesty (raw-count models, OOF-RMSLE vs the deployed LB):", flush=True)
    for name in ("time_local__raw", "random__raw"):
        gap = results[name]["rmsle_native"] - _DEPLOYED_LB
        print(f"  {name:20s} OOF-RMSLE {results[name]['rmsle_native']:.5f}  "
              f"gap vs LB {gap:+.5f}", flush=True)
    print(f"\n  common-support (n={len(common)}) OOF-RMSLE:", flush=True)
    for name in ("time_local__raw", "random__raw"):
        print(f"    {name:20s} {cs[name]:.5f}  gap vs LB {cs[name]-_DEPLOYED_LB:+.5f}", flush=True)

    print("\nlog1p arbiter verdict, per fold structure (base=raw, trial=log1p):", flush=True)
    verdicts = {}
    for fold_name, raw_cell, log_cell in [
        ("random", "random__raw", "random__log1p"),
        ("time_local", "time_local__raw", "time_local__log1p"),
    ]:
        base_raw = CVResult("rmse", "regression", results[raw_cell]["raw_fold_scores"],
                            results[raw_cell]["raw_rmse_mean"], results[raw_cell]["raw_rmse_std"],
                            _N_FOLDS, False, greater_is_better=True)
        trial_raw = CVResult("rmse", "regression", results[log_cell]["raw_fold_scores"],
                             results[log_cell]["raw_rmse_mean"], results[log_cell]["raw_rmse_std"],
                             _N_FOLDS, False, greater_is_better=True)
        raw_ok, raw_delta = improves_beyond_noise(base_raw, trial_raw)
        raw_mde = paired_delta_mde(float(np.std(
            [t - b for b, t in zip(base_raw.fold_scores, trial_raw.fold_scores)], ddof=1)),
            _N_FOLDS, test_train_ratio=1.0 / (_N_FOLDS - 1))

        base_ll = _rmsle_cvresult(results[raw_cell]["rmsle_per_fold"])
        trial_ll = _rmsle_cvresult(results[log_cell]["rmsle_per_fold"])
        ll_ok, ll_delta = improves_beyond_noise(base_ll, trial_ll)
        ll_mde = paired_delta_mde(float(np.std(
            [t - b for b, t in zip(base_ll.fold_scores, trial_ll.fold_scores)], ddof=1)),
            _N_FOLDS, test_train_ratio=1.0 / (_N_FOLDS - 1))

        verdicts[fold_name] = dict(raw_accept=raw_ok, raw_delta=raw_delta, raw_mde=raw_mde,
                                   rmsle_accept=ll_ok, rmsle_delta=ll_delta, rmsle_mde=ll_mde)
        print(f"  {fold_name}:", flush=True)
        print(f"    raw-count RMSE space  : log1p {'ACCEPTED' if raw_ok else 'rejected'} "
              f"(delta {raw_delta:+.4f}, mde {raw_mde:.4f})  <- the product arbiter's space",
              flush=True)
        print(f"    RMSLE space           : log1p {'ACCEPTED' if ll_ok else 'rejected'} "
              f"(delta {ll_delta:+.5f}, mde {ll_mde:.5f})  <- the competition's real metric",
              flush=True)

    with open(_OUT, "a") as fh:
        fh.write(json.dumps({"summary": True, "deployed_lb": _DEPLOYED_LB,
                             "common_support_n": len(common),
                             "common_support_rmsle": cs, "verdicts": verdicts}) + "\n")
    print(f"\nfull results appended to {_OUT}", flush=True)


if __name__ == "__main__":
    main()
