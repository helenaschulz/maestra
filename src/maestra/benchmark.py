"""Mini benchmark harness: Maestra vs. the AutoGluon baseline on an honest answer key.

Shape is deliberately MLE-bench-compatible — a task produces a ``submission`` which is
*graded against a held-out answer key with the competition metric*. Here the answer key is
carved from the training CSV (stratified, fixed seed); for a real MLE-bench task it comes
from the task directory instead, but ``grade()`` and the submission flow stay identical.

Run one task:   maestra-bench --csv data/titanic.csv --target Survived --metric balanced_accuracy --id-col PassengerId
See the board:   maestra-bench --summary
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from maestra.cli import load_dotenv
from maestra.pipeline import run_pipeline

# Label-based metrics (the submission carries predicted labels).
_METRICS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
    "mcc": matthews_corrcoef,
    # quadratic weighted kappa — a common Kaggle label metric AutoGluon has no eval_metric for,
    # so it is computed on out-of-fold predictions (see mlebench_runner).
    "quadratic_weighted_kappa": lambda y, p: cohen_kappa_score(y, p, weights="quadratic"),
    "rmse": lambda y, p: mean_squared_error(y, p) ** 0.5,
    "mae": mean_absolute_error,
}
_HIGHER_IS_BETTER = {"accuracy", "balanced_accuracy", "f1_macro", "mcc", "quadratic_weighted_kappa"}


# Probability metrics — scored on class *probabilities* (one column per class), not labels.
# Computed on the pooled out-of-fold probabilities so the CV↔LB gap stays comparable.
# NOTE: log_loss is calibration-sensitive and AutoGluon's raw probabilities are not
# guaranteed calibrated — a calibration pass (e.g. CalibratedClassifierCV / temperature
# scaling on the OOF probabilities) is a possible follow-up; not done here.
def _roc_auc_proba(y_true, proba, positive_class):
    if proba.shape[1] == 2:  # binary: score the positive-class probability
        return float(roc_auc_score(y_true, proba[positive_class]))
    return float(roc_auc_score(y_true, proba, multi_class="ovr", labels=list(proba.columns)))


def _log_loss_proba(y_true, proba, positive_class):
    return float(log_loss(y_true, proba, labels=list(proba.columns)))


_PROBA_METRICS = {
    "roc_auc": _roc_auc_proba,
    "auc": _roc_auc_proba,
    "log_loss": _log_loss_proba,
}
_PROBA_HIGHER_IS_BETTER = {"roc_auc", "auc"}  # log_loss is lower-is-better


def render_table(headers: list[str], rows: list[list]) -> str:
    """Render a simple fixed-width text table (shared by maestra-bench and maestra-mlebench)."""
    cols = list(zip(headers, *rows)) if rows else [(h,) for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers)]
    lines += [fmt.format(*[str(c) for c in row]) for row in rows]
    return "\n".join(lines)


@dataclass
class BenchResult:
    name: str
    metric: str
    baseline: float
    maestra: float
    delta: float
    higher_is_better: bool
    n_train: int
    n_grade: int


def grade(submission: pd.DataFrame, answer: pd.DataFrame, *, metric: str, id_col: str, target: str) -> float:
    """Score a submission against the answer key with ``metric`` (sklearn under the hood)."""
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Known: {sorted(_METRICS)}")
    merged = answer[[id_col, target]].merge(
        submission[[id_col, target]], on=id_col, suffixes=("_true", "_pred")
    )
    if len(merged) != len(answer):
        raise ValueError(f"Submission covers {len(merged)}/{len(answer)} answer-key rows.")
    return float(_METRICS[metric](merged[f"{target}_true"], merged[f"{target}_pred"]))


def _carve_answer_key(df, target, id_col, holdout_frac, seed):
    """Hold out a stratified answer key — the 'unseen test'. Models see only ``work``."""
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = range(len(df))
    classification = type_of_target(df[target].dropna()) in ("binary", "multiclass")
    work, answer = train_test_split(
        df, test_size=holdout_frac, random_state=seed,
        stratify=df[target] if classification else None,
    )
    return work, answer.drop(columns=[target]), answer[[id_col, target]]


def _winner(delta: float, higher_is_better: bool) -> str:
    if abs(delta) < 1e-9:
        return "tie"
    return "maestra" if (delta > 0) == higher_is_better else "baseline"


def run_task(
    csv: str,
    target: str,
    *,
    metric: str,
    id_col: str = "id",
    model: str = "gpt-4o",
    time_limit: int = 120,
    seed: int = 42,
    holdout_frac: float = 0.25,
) -> BenchResult:
    """Run Maestra and the ``--no-llm`` baseline on ``csv`` and grade both on the answer key."""
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Known: {sorted(_METRICS)}")
    df = pd.read_csv(csv)
    work, test_features, answer = _carve_answer_key(df, target, id_col, holdout_frac, seed)

    common = dict(model=model, test_size=0.2, time_limit=time_limit, seed=seed,
                  test_df=test_features, id_col=id_col)
    res_m = run_pipeline(work, target, use_llm=True, model_dir="AutogluonModels/bench_maestra", **common)
    res_b = run_pipeline(work, target, use_llm=False, model_dir="AutogluonModels/bench_baseline", **common)
    maestra = grade(res_m.submission, answer, metric=metric, id_col=id_col, target=target)
    baseline = grade(res_b.submission, answer, metric=metric, id_col=id_col, target=target)

    return BenchResult(
        name=Path(csv).stem, metric=metric, baseline=baseline, maestra=maestra,
        delta=maestra - baseline, higher_is_better=metric in _HIGHER_IS_BETTER,
        n_train=len(work), n_grade=len(answer),
    )


def append_result(path: str, result: BenchResult, *, timestamp: str) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps({"timestamp": timestamp, **asdict(result)}, default=float) + "\n")


def summary(path: str) -> str:
    """Render the accumulated benchmark board (dataset × baseline/maestra/Δ/winner)."""
    try:
        rows = [json.loads(line) for line in open(path) if line.strip()]
    except FileNotFoundError:
        return f"No benchmark results at {path!r} yet."
    if not rows:
        return "No benchmark results yet."
    out = [f"{'dataset':<18}{'metric':<19}{'baseline':>10}{'maestra':>10}{'delta':>10}  winner"]
    for r in rows:
        win = _winner(r["delta"], r.get("higher_is_better", True))
        out.append(f"{r['name']:<18}{r['metric']:<19}{r['baseline']:>10.4f}"
                   f"{r['maestra']:>10.4f}{r['delta']:>+10.4f}  {win}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="maestra-bench", description="Benchmark Maestra vs. the AutoGluon baseline.")
    p.add_argument("--csv")
    p.add_argument("--target")
    p.add_argument("--metric", default="accuracy", help=f"One of: {sorted(_METRICS)}")
    p.add_argument("--id-col", default="id")
    p.add_argument("--model", default=os.environ.get("AUTOML_MODEL", "gpt-4o"))
    p.add_argument("--time-limit", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-frac", type=float, default=0.25, help="Answer-key fraction.")
    p.add_argument("--results", default="benchmark.jsonl")
    p.add_argument("--summary", action="store_true", help="Print the accumulated board and exit.")
    args = p.parse_args(argv)
    load_dotenv()

    if args.summary:
        print(summary(args.results))
        return 0
    if not (args.csv and args.target):
        print("Error: --csv and --target are required (or use --summary).")
        return 1

    result = run_task(args.csv, args.target, metric=args.metric, id_col=args.id_col,
                      model=args.model, time_limit=args.time_limit, seed=args.seed,
                      holdout_frac=args.holdout_frac)
    append_result(args.results, result, timestamp=datetime.now().isoformat(timespec="seconds"))
    print(f"\n{result.name} | {result.metric}: baseline {result.baseline:.4f}  "
          f"maestra {result.maestra:.4f}  Δ {result.delta:+.4f}  → "
          f"{_winner(result.delta, result.higher_is_better)}")
    print(f"Logged to {args.results}. Board: maestra-bench --summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
