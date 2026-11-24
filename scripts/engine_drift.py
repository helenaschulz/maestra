"""N6 Teil A: engine-drift receipt, AutoGluon 1.5 -> 1.6.

Re-runs the E2 task battery's ``--no-llm`` BASELINE arm exactly as ``maestra-bench`` does
(``benchmark.run_task``'s baseline: same answer-key carve, CV path with ``--cv 3``, random folds,
``time_limit=60``, final fit on all work rows, graded on the held-out answer key), one seed,
no LLM at all. Run it once per engine environment; the engine version comes from the installed
package, so the SAME script under ``.venv`` (AG 1.5) and ``.venv16`` (AG 1.6) yields the paired
receipt. Also records the leaderboard's model portfolio, so a changed default preset /
portfolio is visible, not inferred.

    ./.venv/bin/python   scripts/engine_drift.py --out engine_drift.jsonl
    ./.venv16/bin/python scripts/engine_drift.py --out engine_drift.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import pandas as pd

from maestra.benchmark import _carve_answer_key, grade
from maestra.engine import autogluon_version
from maestra.pipeline import run_pipeline

# (name, target, metric) -- the E2 catalog (scripts/task_battery.py), materialised CSVs in data/.
TASKS = [
    ("insurance", "charges", "rmse"), ("loan-grade", "grade", "balanced_accuracy"),
    ("diamonds", "price", "rmse"), ("wage", "wage", "rmse"), ("credit", "Balance", "rmse"),
    ("heart", "AHD", "balanced_accuracy"), ("abalone", "rings", "rmse"),
    ("wine-quality", "quality", "rmse"), ("wine-quality-anon", "y", "rmse"),
    ("friedman-synth", "y", "rmse"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="engine_drift.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-limit", type=int, default=60)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--model-dir", default="AutogluonModels/engine_drift")
    p.add_argument("--tasks", nargs="*", help="Subset of task names (default: all).")
    args = p.parse_args()
    version = autogluon_version()
    for name, target, metric in TASKS:
        if args.tasks and name not in args.tasks:
            continue
        df = pd.read_csv(f"data/battery_{name}.csv")
        work, test_features, answer = _carve_answer_key(df, target, "rowid", 0.25, args.seed)
        t0 = time.time()
        res = run_pipeline(work, target, model="none", use_llm=False, test_size=0.2,
                           time_limit=args.time_limit, seed=args.seed,
                           model_dir=f"{args.model_dir}/{version}/{name}",
                           test_df=test_features, id_col="rowid", cv_folds=args.cv)
        score = grade(res.submission, answer, metric=metric, id_col="rowid", target=target)
        lb = res.training.leaderboard
        rec = {
            "timestamp": datetime.now().isoformat(timespec="seconds"), "kind": "engine_drift",
            "autogluon_version": version, "task": name, "metric": metric, "seed": args.seed,
            "time_limit": args.time_limit, "cv_folds": args.cv, "graded": score,
            "cv_mean": res.cv.mean, "cv_metric": res.cv.eval_metric,
            "final_best_model": str(lb.iloc[0]["model"]) if len(lb) else None,
            "final_models": sorted(str(m) for m in lb["model"]),
            "wall_seconds": round(time.time() - t0, 1),
        }
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        print(f"{version} {name:18s} {metric:18s} graded={score:.4f} "
              f"best={rec['final_best_model']} ({rec['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
