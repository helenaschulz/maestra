"""N6b: does a structured holdout split (group-disjoint / forward in time) give an honest estimate?

One variable on the holdout path: ``validation.holdout_split`` random (pre-N6b) vs. structured
(the Strategist's strategy, set explicitly here: no LLM, zero LLM cost). The structured arm also
gets the inner ``validation_structure``, exactly as ``run_pipeline`` now does. A fixed answer key,
carved first with the task's own structure, is the truth both arms are graded against.

Hypotheses (fixed in docs/RESULTS.md before the run):
  H1  the gap holdout score - answer-key score (higher-is-better) shrinks with the structured split.
  H2  the answer-key score does not get worse.

    ./.venv16/bin/python scripts/n6b_holdout_split_experiment.py --out n6b_holdout_split.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from maestra.engine import autogluon_version, resolve_inner_validation, train_and_evaluate
from maestra.validation import holdout_split, paired_delta_mde, paired_delta_test

_HERE = Path(__file__).resolve().parent


def _synthetic():
    spec = importlib.util.spec_from_file_location("m1", _HERE / "group_leakage_experiment.py")
    m1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m1)
    return m1.make_grouped_dataset().drop(columns=["row_id"], errors="ignore")


def _datasets():
    bike = pd.read_csv("data/kaggle_bike/train.csv").drop(columns=["casual", "registered"])
    return [
        ("m1-synthetic-grouped", _synthetic(), "churned", {"group_column": "customer_id"}),
        ("mathachieve", pd.read_csv("data/mathachieve.csv")[["School", "Minority", "Sex", "SES",
                                                             "MathAch"]],
         "MathAch", {"group_column": "School"}),
        ("economics", pd.read_csv("data/economics.csv")[["date", "pce", "pop", "psavert",
                                                         "uempmed", "unemploy"]],
         "unemploy", {"time_column": "date"}),
        ("bike-sharing", bike, "count",
         {"time_column": "datetime", "period_column": "month_of:datetime"}),
    ]


def _score(predictor, frame):
    """AutoGluon's own higher-is-better score of the predictor on ``frame``."""
    return float(predictor.evaluate(frame, silent=True)[predictor.eval_metric.name])


def _verdict(deltas, ratio):
    better = paired_delta_test(deltas, test_train_ratio=ratio)
    worse = paired_delta_test([-d for d in deltas], test_train_ratio=ratio)
    std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    return {"mean": float(np.mean(deltas)), "n_pairs": len(deltas),
            "verdict": "structured better" if better else ("structured worse" if worse
                                                           else "undecided"),
            "mde": paired_delta_mde(std, len(deltas), test_train_ratio=ratio)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="n6b_holdout_split.jsonl")
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 7, 1, 3, 5])
    p.add_argument("--time-limit", type=int, default=30)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--only", nargs="*")
    args = p.parse_args()
    version = autogluon_version()

    for name, df, target, strat in _datasets():
        if args.only and name not in args.only:
            continue
        work, answer, key_rec = holdout_split(df, 0.2, 0, **strat)  # the truth, fixed
        runs = []
        for seed in args.seeds:
            row = {"seed": seed}
            for arm, kwargs in (("random", {}), ("structured", strat)):
                train, holdout, rec = holdout_split(work, args.test_size, seed, **kwargs)
                structure, inner = resolve_inner_validation(train.columns, **kwargs)
                res = train_and_evaluate(
                    train, holdout, target, args.time_limit,
                    f"AutogluonModels/n6b/{name}/s{seed}_{arm}", validation_structure=structure,
                    structure_source=train)
                est = _score(res.predictor, holdout)
                truth = _score(res.predictor, answer)
                row[arm] = {"estimate": est, "truth": truth, "gap": est - truth,
                            "n_train": rec["n_train"], "n_holdout": rec["n_holdout"],
                            "split_note": rec["note"], "inner": inner["note"]}
                print(f"{name:22s} seed={seed:<3d} {arm:10s} est={est:.4f} truth={truth:.4f} "
                      f"gap={est - truth:+.4f}", flush=True)
            runs.append(row)
        ratio = args.test_size / (1 - args.test_size)
        rec = {
            "timestamp": datetime.now().isoformat(timespec="seconds"), "kind": "n6b_holdout_split",
            "autogluon_version": version, "dataset": name, "n_rows": len(df), "target": target,
            "strategy": strat, "answer_key": key_rec, "seeds": args.seeds,
            "time_limit": args.time_limit, "test_size": args.test_size,
            "gap_mean_random": float(np.mean([r["random"]["gap"] for r in runs])),
            "gap_mean_structured": float(np.mean([r["structured"]["gap"] for r in runs])),
            "truth_mean_random": float(np.mean([r["random"]["truth"] for r in runs])),
            "truth_mean_structured": float(np.mean([r["structured"]["truth"] for r in runs])),
            # H1: positive = the optimism (|gap|) shrank with the structured split
            "h1_gap": _verdict([abs(r["random"]["gap"]) - abs(r["structured"]["gap"])
                                for r in runs], ratio),
            # H2: positive = the structured arm's model is better on the answer key
            "h2_truth": _verdict([r["structured"]["truth"] - r["random"]["truth"] for r in runs],
                                 ratio),
            "runs": runs,
        }
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        print(f"==> {name}: gap random {rec['gap_mean_random']:+.4f} structured "
              f"{rec['gap_mean_structured']:+.4f} [{rec['h1_gap']['verdict']}]  truth random "
              f"{rec['truth_mean_random']:.4f} structured {rec['truth_mean_structured']:.4f} "
              f"[{rec['h2_truth']['verdict']}]", flush=True)


if __name__ == "__main__":
    main()
