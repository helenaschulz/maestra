"""N6 Teil C: does handing the outer fold structure to AutoGluon's INTERNAL validation help?

One variable: ``cross_validate(..., inner_structure=False)`` (AutoGluon splits internally at random,
the pre-N6 behaviour) vs ``inner_structure=True`` (``validation_structure`` = ``group_on`` /
``time_on`` + forward-only, N6). Identical outer folds, seeds, preset (AutoGluon default),
``time_limit`` and data per pair; fold strategy set EXPLICITLY (no LLM, zero LLM cost).

Hypotheses (fixed in docs/RESULTS.md before the run):
  H1  the optimism of AutoGluon's own ``score_val`` (internal best minus outer fold score, both in
      AutoGluon's higher-is-better convention) shrinks with the structure on.
  H2  the outer CV score does not get worse (paired, Nadeau-Bengio-corrected).

Replication unit: for datasets whose outer folds do NOT depend on the seed (GroupKFold for a
regression target, every time split) the 3 seeds re-run IDENTICAL folds -- they only measure
AutoGluon run-to-run noise, so per fold the seeds are averaged first and the paired test runs on
the k fold pairs. Where the seed changes the folds (StratifiedGroupKFold on the synthetic
classification task) all seed x fold pairs are used.

    ./.venv16/bin/python scripts/n6_inner_validation_experiment.py --out n6_inner_validation.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from maestra import validation
from maestra.engine import autogluon_version
from maestra.validation import cross_validate, paired_delta_mde, paired_delta_test

_HERE = Path(__file__).resolve().parent


def _synthetic():
    spec = importlib.util.spec_from_file_location("m1", _HERE / "group_leakage_experiment.py")
    m1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m1)
    return m1.make_grouped_dataset()  # 150 customers x 8 rows, seed 7 (the M1 dataset)


def _datasets():
    bike = pd.read_csv("data/kaggle_bike/train.csv").drop(columns=["casual", "registered"])
    return [
        # name, frame, target, cv kwargs, seed changes the folds?
        ("m1-synthetic-grouped", _synthetic(), "churned", {"group_column": "customer_id"}, True),
        ("grunfeld", pd.read_csv("data/grunfeld.csv")[["firm", "year", "value", "capital", "inv"]],
         "inv", {"group_column": "firm"}, False),
        ("mathachieve", pd.read_csv("data/mathachieve.csv")[["School", "Minority", "Sex", "SES",
                                                             "MathAch"]],
         "MathAch", {"group_column": "School"}, False),
        ("economics", pd.read_csv("data/economics.csv")[["date", "pce", "pop", "psavert",
                                                         "uempmed", "unemploy"]],
         "unemploy", {"time_column": "date"}, False),
        ("bike-sharing", bike, "count",
         {"time_column": "datetime", "period_column": "month_of:datetime"}, False),
    ]


def _spy():
    """Wrap the fold fit to capture each fold's leaderboard top model (descriptive only)."""
    real = validation.train_and_evaluate
    tops: list = []

    def wrapped(*args, **kwargs):
        res = real(*args, **kwargs)
        lb = res.leaderboard
        best = lb.sort_values("score_val", ascending=False).iloc[0]["model"] if len(lb) else None
        tops.append(str(best))
        return res

    validation.train_and_evaluate = wrapped
    return tops, lambda: setattr(validation, "train_and_evaluate", real)


def _verdict(deltas: list[float], k: int) -> dict:
    """Three-way paired verdict on 'positive = on is better', NB-corrected like every gate."""
    ratio = 1.0 / (k - 1) if k > 1 else 0.0
    n = len(deltas)
    better = paired_delta_test(deltas, test_train_ratio=ratio)
    worse = paired_delta_test([-d for d in deltas], test_train_ratio=ratio)
    std = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
    return {"mean": float(np.mean(deltas)), "n_pairs": n,
            "verdict": "on better" if better else ("on worse" if worse else "undecided"),
            "mde": paired_delta_mde(std, n, test_train_ratio=ratio)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="n6_inner_validation.jsonl")
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 7, 1])
    p.add_argument("--time-limit", type=int, default=30)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--only", nargs="*")
    args = p.parse_args()
    version = autogluon_version()

    for name, df, target, cv_kwargs, seed_changes_folds in _datasets():
        if args.only and name not in args.only:
            continue
        runs = {}
        for seed in args.seeds:
            for arm in (False, True):
                tops, restore = _spy()
                try:
                    cv = cross_validate(
                        df, target, cleaning_plan=None, feature_plan=None,
                        model_dir=f"AutogluonModels/n6/{name}/s{seed}_{'on' if arm else 'off'}",
                        time_limit=args.time_limit, n_folds=args.folds, seed=seed,
                        inner_structure=arm, **cv_kwargs)
                finally:
                    restore()
                runs[(seed, arm)] = {
                    "fold_scores": cv.fold_scores, "val_scores": cv.val_scores,
                    "top_models": tops, "inner_validation": cv.inner_validation,
                    "eval_metric": cv.eval_metric,
                }
                print(f"{name:22s} seed={seed:<3d} {'on ' if arm else 'off'} "
                      f"cv={np.mean(cv.fold_scores):.4f} val={np.mean(cv.val_scores):.4f} "
                      f"tops={tops}", flush=True)

        def per_fold(arm, key):
            m = np.array([runs[(s, arm)][key] for s in args.seeds], dtype=float)  # seeds x folds
            return m.ravel().tolist() if seed_changes_folds else m.mean(axis=0).tolist()

        score_off, score_on = per_fold(False, "fold_scores"), per_fold(True, "fold_scores")
        val_off, val_on = per_fold(False, "val_scores"), per_fold(True, "val_scores")
        opt_off = [v - s for v, s in zip(val_off, score_off)]
        opt_on = [v - s for v, s in zip(val_on, score_on)]
        rec = {
            "timestamp": datetime.now().isoformat(timespec="seconds"), "kind": "n6_inner_validation",
            "autogluon_version": version, "dataset": name, "n_rows": len(df), "target": target,
            "cv_kwargs": cv_kwargs, "seeds": args.seeds, "time_limit": args.time_limit,
            "folds": args.folds, "seed_changes_folds": seed_changes_folds,
            "eval_metric": runs[(args.seeds[0], True)]["eval_metric"],
            "inner_validation_on": runs[(args.seeds[0], True)]["inner_validation"],
            "outer_mean_off": float(np.mean(score_off)), "outer_mean_on": float(np.mean(score_on)),
            "optimism_mean_off": float(np.mean(opt_off)), "optimism_mean_on": float(np.mean(opt_on)),
            # H2: positive = structure-on scores better on the outer folds
            "h2_outer": _verdict([b - a for a, b in zip(score_off, score_on)], args.folds),
            # H1: positive = optimism shrank with the structure on
            "h1_optimism": _verdict([a - b for a, b in zip(opt_off, opt_on)], args.folds),
            "runs": {f"{s}_{'on' if a else 'off'}": v for (s, a), v in runs.items()},
        }
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        print(f"==> {name}: outer off {rec['outer_mean_off']:.4f} on {rec['outer_mean_on']:.4f} "
              f"[{rec['h2_outer']['verdict']}]  optimism off {rec['optimism_mean_off']:+.4f} "
              f"on {rec['optimism_mean_on']:+.4f} [{rec['h1_optimism']['verdict']}]", flush=True)


if __name__ == "__main__":
    main()
