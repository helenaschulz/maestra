"""E2: the task battery — the competence map as a table instead of anecdotes.

Runs `maestra-bench`'s multi-seed comparison (M8: paired per-seed deltas, three-way verdict) over
a catalog of tabular tasks that spans the semantic spectrum: rich column semantics (where the
thesis predicts Maestra wins), mixed, and anonymous/numeric-only (the control group, where the
thesis predicts undecided-or-baseline). Every dataset downloads from a public mirror on demand.

Chunked by design: one task per invocation fits a usage budget (a task is ~5 seeds × 2 arms ×
~4 AutoGluon fits ≈ 20–30 min at the default budget). Results accumulate in `benchmark.jsonl`
(per-seed rows + one aggregate row with the verdict per task).

    ./.venv/bin/python scripts/task_battery.py --list
    ./.venv/bin/python scripts/task_battery.py --task insurance --model gpt-4o
    ./.venv/bin/python scripts/task_battery.py --task all --model gpt-4o     # everything, in one go
"""
from __future__ import annotations

import argparse
import os
import urllib.request

import pandas as pd

from maestra.benchmark import append_multi_seed, append_result, run_multi_seed
from maestra.config import load_dotenv

_RD = "https://vincentarelbundock.github.io/Rdatasets/csv"
_SEEDS = [42, 7, 1, 2, 3]

# Semantics labels: "rich" = named, meaningful columns (thesis predicts Maestra); "poor" =
# anonymous/numeric-only (control; thesis predicts undecided-or-baseline).
CATALOG = [
    dict(name="insurance", semantics="rich", target="charges", metric="rmse", id_col="rowid",
         src="https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv",
         note="medical insurance charges: age/sex/bmi/children/smoker/region"),
    dict(name="loan-grade", semantics="rich", target="grade", metric="balanced_accuracy", id_col="rowid",
         src=f"{_RD}/openintro/loans_full_schema.csv",
         keep=["emp_length", "homeownership", "annual_income", "debt_to_income", "delinq_2y",
               "total_credit_limit", "public_record_bankrupt", "term", "loan_purpose",
               "loan_amount", "grade"],
         note="loan grade (A..G) from applicant profile: income/debt/purpose/homeownership"),
    dict(name="diamonds", semantics="rich", target="price", metric="rmse", id_col="rowid",
         src=f"{_RD}/ggplot2/diamonds.csv", sample=8000,
         note="diamond price: carat/cut/color/clarity (subsampled to 8k rows for runtime)"),
    dict(name="wage", semantics="rich", target="wage", metric="rmse", id_col="rowid",
         src=f"{_RD}/ISLR/Wage.csv", drop=["logwage"],  # log of the target — a leak
         note="wages: year/age/maritl/education/jobclass/health"),
    dict(name="credit", semantics="rich", target="Balance", metric="rmse", id_col="rowid",
         src=f"{_RD}/ISLR/Credit.csv",
         note="credit card balance: Income/Limit/Rating/Student/Married"),
    dict(name="heart", semantics="rich", target="AHD", metric="balanced_accuracy", id_col="rowid",
         src="https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Heart.csv",
         note="heart disease: Age/Sex/ChestPain/Chol/MaxHR"),
    dict(name="abalone", semantics="mixed", target="rings", metric="rmse", id_col="rowid",
         src="https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
         read_kwargs={"header": None}, columns=["sex", "length", "diameter", "height",
             "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"],
         note="abalone age: physical measurements + sex (semantic names, physical domain)"),
    dict(name="wine-quality", semantics="mixed", target="quality", metric="rmse", id_col="rowid",
         src="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
         read_kwargs={"sep": ";"},
         note="wine quality from physico-chemical measurements (named columns)"),
    dict(name="wine-quality-anon", semantics="poor", target="y", metric="rmse", id_col="rowid",
         src="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
         read_kwargs={"sep": ";"}, anonymize=True,
         note="IDENTICAL data to wine-quality with column names stripped to x1..xn/y — the causal "
              "semantics control: any Maestra-vs-baseline difference between the twins is pure semantics"),
    dict(name="friedman-synth", semantics="poor", target="y", metric="rmse", id_col="rowid",
         src="synthetic:friedman", note="synthetic Friedman#1: x1..x10 anonymous — pure control"),
]


def _load(spec: dict) -> str:
    """Materialize the task as a local CSV (download / synthesize once); returns the path."""
    path = f"data/battery_{spec['name']}.csv"
    if os.path.exists(path):
        return path
    os.makedirs("data", exist_ok=True)
    if spec["src"] == "synthetic:friedman":
        from sklearn.datasets import make_friedman1
        X, y = make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=0)
        df = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(10)]).assign(y=y)
    else:
        urllib.request.urlretrieve(spec["src"], path + ".tmp")
        df = pd.read_csv(path + ".tmp", **spec.get("read_kwargs", {}))
        os.remove(path + ".tmp")
    if spec.get("columns"):
        df.columns = spec["columns"]
    if spec.get("sample") and len(df) > spec["sample"]:
        df = df.sample(spec["sample"], random_state=0).reset_index(drop=True)
    if spec.get("anonymize"):  # strip ALL semantics: features -> x1..xn, target -> y
        orig_target = df.columns[-1]  # by convention: the source file's last column is the target
        df = df.rename(columns={c: f"x{i + 1}" for i, c in enumerate(
            c for c in df.columns if c != orig_target)})
        df = df.rename(columns={orig_target: "y"})
    if spec.get("keep"):
        df = df[spec["keep"]].dropna(subset=[spec["target"]])
    for col in spec.get("drop", []):
        if col in df.columns:
            df = df.drop(columns=[col])
    if spec["id_col"] not in df.columns:
        df.insert(0, spec["id_col"], range(len(df)))
    df.to_csv(path, index=False)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", help="Task name from the catalog, or 'all'.")
    p.add_argument("--list", action="store_true", help="List catalog tasks and exit.")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--time-limit", type=int, default=60)
    p.add_argument("--cv", type=int, default=3)
    args = p.parse_args()
    load_dotenv()

    if args.list or not args.task:
        print(f"{'task':16s} {'semantics':10s} {'metric':18s} note")
        for spec in CATALOG:
            print(f"{spec['name']:16s} {spec['semantics']:10s} {spec['metric']:18s} {spec['note']}")
        return

    todo = CATALOG if args.task == "all" else [s for s in CATALOG if s["name"] == args.task]
    if not todo:
        raise SystemExit(f"unknown task {args.task!r} — see --list")

    from datetime import datetime
    for spec in todo:
        csv = _load(spec)
        print(f"\n=== {spec['name']} ({spec['semantics']} semantics) — {len(_SEEDS)} seeds ===")
        ms = run_multi_seed(csv, spec["target"], metric=spec["metric"], seeds=_SEEDS,
                            id_col=spec["id_col"], model=args.model, time_limit=args.time_limit,
                            cv_folds=args.cv, name=f"battery-{spec['name']}")
        ts = datetime.now().isoformat(timespec="seconds")
        for r in ms.per_seed:
            append_result("benchmark.jsonl", r, timestamp=ts)
        append_multi_seed("benchmark.jsonl", ms, timestamp=ts)
        print(f"  verdict: {ms.verdict}  baseline={ms.baseline_mean:.2f}  "
              f"maestra={ms.maestra_mean:.2f}  mean_delta={ms.mean_delta:+.2f}")


if __name__ == "__main__":
    main()
