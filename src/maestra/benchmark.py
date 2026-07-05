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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
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

from maestra.config import load_dotenv
from maestra.pipeline import run_pipeline
from maestra.validation import _is_classification

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
    hybrid: list | None = None  # generated-feature provenance (kept/delta/reason), when --hybrid ran
    seed: int | None = None  # the run's seed (carve + folds), for multi-seed provenance


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
    classification = _is_classification(df[target])  # robust to integer regression targets
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
    cv_folds: int | None = None,
    hybrid: bool = False,
    fold_advisor: bool = False,
    name: str | None = None,
    dataset_description: str | None = None,
) -> BenchResult:
    """Run Maestra and the ``--no-llm`` baseline on ``csv`` and grade both on the answer key.

    ``cv_folds`` switches both arms to the cross-validation path; ``hybrid`` additionally runs
    the generated-feature gate on the Maestra arm (requires ``cv_folds >= 2``). ``fold_advisor``
    (requires ``cv_folds``) runs the Validation Strategist on BOTH arms, so a random-vs-advised
    comparison stays paired on the same fold strategy rather than confounding it with use_llm.
    ``name`` labels the result row (default: the CSV's stem, which for files like ``train.csv``
    says nothing).
    """
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Known: {sorted(_METRICS)}")
    df = pd.read_csv(csv)
    work, test_features, answer = _carve_answer_key(df, target, id_col, holdout_frac, seed)

    common = dict(model=model, test_size=0.2, time_limit=time_limit, seed=seed,
                  test_df=test_features, id_col=id_col, cv_folds=cv_folds,
                  fold_advisor=fold_advisor, dataset_description=dataset_description)
    res_m = run_pipeline(work, target, use_llm=True, hybrid=hybrid,
                         model_dir="AutogluonModels/bench_maestra", **common)
    res_b = run_pipeline(work, target, use_llm=False, model_dir="AutogluonModels/bench_baseline", **common)
    maestra = grade(res_m.submission, answer, metric=metric, id_col=id_col, target=target)
    baseline = grade(res_b.submission, answer, metric=metric, id_col=id_col, target=target)

    label = (name or Path(csv).stem) + ("+hybrid" if hybrid else "")
    return BenchResult(
        name=label, metric=metric, baseline=baseline, maestra=maestra,
        delta=maestra - baseline, higher_is_better=metric in _HIGHER_IS_BETTER,
        n_train=len(work), n_grade=len(answer),
        hybrid=res_m.hybrid,  # provenance, so a --hybrid run is interpretable after the fact
        seed=seed,
    )


@dataclass
class MultiSeedResult:
    """Aggregate of one task run over several seeds, with a paired three-way verdict."""

    name: str
    metric: str
    seeds: list[int]
    per_seed: list[BenchResult]
    baseline_mean: float
    maestra_mean: float
    mean_delta: float          # mean(maestra - baseline), raw metric direction
    higher_is_better: bool
    verdict: str               # "maestra" | "baseline" | "undecided"
    mde: float                 # minimum |mean_delta| that would have cleared the accept bar (N1)
    failed_seeds: list[dict] = field(default_factory=list)  # [{"seed", "error"}], visibly recorded


def run_multi_seed(csv: str, target: str, *, metric: str, seeds: list[int], **kwargs) -> MultiSeedResult:
    """Run :func:`run_task` once per seed and settle the comparison with the arbiter's rule.

    Every seed re-carves the answer key and re-splits the folds, so the seeds are genuine
    replications. Baseline and Maestra share each seed's carve, so the per-seed deltas are
    PAIRED — the same machinery as the fold-wise gate (``paired_delta_test``: mean beyond
    2 standard errors AND a strict majority of seeds). The verdict is deliberately three-way:
    **undecided-within-noise is a first-class, honest outcome**, not a failure to report.

    The accept bar applies the same Nadeau-Bengio variance inflation as the fold-wise gate
    (N1, 2026-07-05): ``test_train_ratio = holdout_frac / (1 - holdout_frac)``, since every
    seed re-carves an answer key of that fraction from the same pool — the seed replications
    share overlapping training data exactly like k-fold replications do, so the naive SEM
    across seed deltas understates the true variance. ``mde`` (minimum detectable effect) is
    the mean delta this run's spread/seed-count would have needed to clear the bar — reported
    on every result so "undecided" separates "no effect" from "underpowered".

    A single seed's crash (e.g. an AutoGluon internal fragility on a rare data/fold shape) does
    NOT void the whole run: it is caught, recorded in ``failed_seeds`` with its error, and the
    remaining seeds still produce a verdict — conservatively, since ``paired_delta_test`` already
    returns False below 2 data points. If every seed fails, that is a real error, not a silent
    "undecided": it raises.
    """
    from maestra.validation import paired_delta_mde, paired_delta_test

    per_seed, failed = [], []
    for s in seeds:
        try:
            per_seed.append(run_task(csv, target, metric=metric, seed=s, **kwargs))
        except Exception as exc:  # noqa: BLE001 - AutoGluon/LLM failures are unpredictable by type
            failed.append({"seed": s, "error": f"{type(exc).__name__}: {exc}"})
    if not per_seed:
        raise RuntimeError(f"all {len(seeds)} seeds failed: {failed}")

    holdout_frac = kwargs.get("holdout_frac", 0.25)
    ratio = holdout_frac / (1.0 - holdout_frac)
    higher = per_seed[0].higher_is_better
    # signed improvement per seed: positive = Maestra better, regardless of metric direction
    improvements = [(r.maestra - r.baseline) * (1.0 if higher else -1.0) for r in per_seed]
    if paired_delta_test(improvements, test_train_ratio=ratio):
        verdict = "maestra"
    elif paired_delta_test([-d for d in improvements], test_train_ratio=ratio):
        verdict = "baseline"
    else:
        verdict = "undecided"
    mde = paired_delta_mde(float(np.std(improvements, ddof=1)) if len(improvements) >= 2 else 0.0,
                           len(improvements), test_train_ratio=ratio)
    return MultiSeedResult(
        name=per_seed[0].name, metric=metric, seeds=[r.seed for r in per_seed], per_seed=per_seed,
        baseline_mean=float(np.mean([r.baseline for r in per_seed])),
        maestra_mean=float(np.mean([r.maestra for r in per_seed])),
        mean_delta=float(np.mean([r.delta for r in per_seed])),
        higher_is_better=higher, verdict=verdict, mde=mde, failed_seeds=failed,
    )


def append_result(path: str, result: BenchResult, *, timestamp: str) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps({"timestamp": timestamp, **asdict(result)}, default=float) + "\n")


def append_multi_seed(path: str, result: MultiSeedResult, *, timestamp: str) -> None:
    """Log the aggregate as one row compatible with ``summary`` (means + explicit verdict);
    the per-seed rows are logged individually by the caller."""
    record = {
        "timestamp": timestamp, "kind": "multi_seed",
        "name": f"{result.name} (n={len(result.seeds)} seeds)", "metric": result.metric,
        "baseline": result.baseline_mean, "maestra": result.maestra_mean,
        "delta": result.mean_delta, "higher_is_better": result.higher_is_better,
        "seeds": result.seeds, "verdict": result.verdict, "mde": result.mde,
        "n_train": result.per_seed[0].n_train, "n_grade": result.per_seed[0].n_grade,
        "failed_seeds": result.failed_seeds,
    }
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=float) + "\n")


def summary(path: str) -> str:
    """Render the accumulated benchmark board (dataset × baseline/maestra/Δ/winner)."""
    try:
        rows = [json.loads(line) for line in open(path) if line.strip()]
    except FileNotFoundError:
        return f"No benchmark results at {path!r} yet."
    if not rows:
        return "No benchmark results yet."
    table = [[r["name"], r["metric"], f"{r['baseline']:.4f}", f"{r['maestra']:.4f}",
              f"{r['delta']:+.4f}",
              # multi-seed rows carry the arbiter's three-way verdict; single runs keep the raw sign
              r.get("verdict") or _winner(r["delta"], r.get("higher_is_better", True))]
             for r in rows]
    return render_table(["dataset", "metric", "baseline", "maestra", "delta", "winner"], table)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="maestra-bench", description="Benchmark Maestra vs. the AutoGluon baseline.")
    p.add_argument("--csv")
    p.add_argument("--target")
    p.add_argument("--metric", default="accuracy", help=f"One of: {sorted(_METRICS)}")
    p.add_argument("--id-col", default="id")
    p.add_argument("--model", default=os.environ.get("AUTOML_MODEL", "gpt-4o"))
    p.add_argument("--time-limit", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None, metavar="S",
                   help="Run the task once per seed and settle the comparison with the paired "
                        "arbiter rule (verdict: maestra / baseline / undecided-within-noise). "
                        "Overrides --seed.")
    p.add_argument("--holdout-frac", type=float, default=0.25, help="Answer-key fraction.")
    p.add_argument("--cv", type=int, default=None, help="Use the k-fold CV path (both arms).")
    p.add_argument("--hybrid", action="store_true", help="Generated-feature gate on the Maestra arm (needs --cv).")
    p.add_argument("--fold-advisor", action="store_true",
                   help="Validation Strategist on BOTH arms (needs --cv) — a fair, paired "
                        "comparison of the fold strategy itself, not confounded with use_llm.")
    p.add_argument("--name", default=None, help="Label for the result row (default: CSV stem).")
    p.add_argument("--description", default=None,
                   help="Path to a provider-written dataset description fed to the judgment nodes.")
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

    description = None
    if args.description:
        with open(args.description) as fh:
            description = fh.read()
    common = dict(metric=args.metric, id_col=args.id_col, model=args.model,
                  time_limit=args.time_limit, holdout_frac=args.holdout_frac,
                  cv_folds=args.cv, hybrid=args.hybrid, fold_advisor=args.fold_advisor,
                  name=args.name, dataset_description=description)
    now = lambda: datetime.now().isoformat(timespec="seconds")  # noqa: E731

    if args.seeds:
        ms = run_multi_seed(args.csv, args.target, seeds=args.seeds, **common)
        for r in ms.per_seed:
            append_result(args.results, r, timestamp=now())
            print(f"  seed {r.seed:<4} baseline {r.baseline:.4f}  maestra {r.maestra:.4f}  "
                  f"Δ {r.delta:+.4f}")
        append_multi_seed(args.results, ms, timestamp=now())
        print(f"\n{ms.name} | {ms.metric} over seeds {ms.seeds}: "
              f"baseline {ms.baseline_mean:.4f}  maestra {ms.maestra_mean:.4f}  "
              f"Δ {ms.mean_delta:+.4f}  →  verdict: {ms.verdict}"
              + ("  (within noise — an honest result, not a failure)" if ms.verdict == "undecided" else ""))
        print(f"  minimum detectable effect at n={len(ms.seeds)} seeds: {ms.mde:.4f} "
              f"(the mean delta needed to clear the accept bar)")
        print(f"Logged to {args.results}. Board: maestra-bench --summary")
        return 0

    result = run_task(args.csv, args.target, seed=args.seed, **common)
    append_result(args.results, result, timestamp=now())
    print(f"\n{result.name} | {result.metric}: baseline {result.baseline:.4f}  "
          f"maestra {result.maestra:.4f}  Δ {result.delta:+.4f}  → "
          f"{_winner(result.delta, result.higher_is_better)}")
    print(f"Logged to {args.results}. Board: maestra-bench --summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
