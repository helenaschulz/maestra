"""MLE-bench adapter: run Maestra on a real MLE-bench task and grade it against the
competition's medal thresholds.

MLE-bench is OpenAI's benchmark of completed Kaggle competitions. A prepared task directory
holds ``train.csv``, ``test.csv`` and ``sample_submission.csv``; grading against the private
answer key (with gold/silver/bronze thresholds) is done by the ``mlebench`` package — an
*optional* dependency (heavy: Docker + competition data). It is reached only through
:func:`grade_submission` and is mocked in tests.

The adapter reuses the existing run -> submission flow (``run_pipeline``); it does not
re-implement it. The value it adds is the **CV↔LB gap**: ``--metric`` (the competition metric)
is either mapped to an AutoGluon ``eval_metric`` so the CV optimises it directly, or — when no
AutoGluon equivalent exists — computed on the CV's out-of-fold predictions. The gap between
that local score and the graded score tells you whether the CV is trustworthy.

LABEL *and* PROBABILITY METRICS: label metrics (accuracy, F1, quadratic-weighted-kappa, ...)
produce a predicted-label submission; probability metrics (roc_auc, log_loss) produce a
probability submission whose shape is derived generically from the sample submission (a
single positive-class column for binary, one column per class for multiclass), with the CV
scored on the pooled out-of-fold probabilities. log_loss is calibration-sensitive and
AutoGluon's raw probabilities are not guaranteed calibrated — calibrating the OOF
probabilities is a possible follow-up (not done here).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from maestra import benchmark, calibration
from maestra.config import load_dotenv
from maestra.pipeline import run_pipeline

# Competition metric -> AutoGluon eval_metric. When a metric is here, the CV optimises it.
_METRIC_MAP = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "f1_macro": "f1_macro",
    "mcc": "mcc",
    "roc_auc": "roc_auc",
    "auc": "roc_auc",
    "log_loss": "log_loss",
    "rmse": "root_mean_squared_error",
    "mae": "mean_absolute_error",
    "r2": "r2",
}
# Probability metrics: AutoGluon optimises them and the submission carries probabilities
# (a per-class or single positive-class column derived from the sample submission). The CV
# score is computed on the pooled out-of-fold probabilities, so the CV↔LB gap is comparable.
_PROBABILITY_METRICS = {"roc_auc", "auc", "log_loss"}


class MleBenchError(RuntimeError):
    """Raised for task-reading or grading failures (incl. the optional dep being absent)."""


def _resolve_metric(metric: str | None):
    """Map a competition metric to an (autogluon_eval_metric, mode).

    mode: "aligned" (CV optimises the AG metric, label submission), "proba" (probability
    metric — probability submission, CV scored on out-of-fold probabilities), "oof" (compute
    on out-of-fold labels), "mismatch" (no way to compare), "default" (no metric requested).
    """
    if metric is None:
        return None, "default"
    if metric in _PROBABILITY_METRICS:
        return _METRIC_MAP.get(metric), "proba"
    if metric in _METRIC_MAP:
        return _METRIC_MAP[metric], "aligned"
    if metric in benchmark._METRICS:
        return None, "oof"
    return None, "mismatch"


@dataclass
class MleTask:
    name: str
    train_csv: str
    test_csv: str
    sample_submission_csv: str
    id_col: str
    target_col: str
    submission_columns: list[str]  # the sample submission's non-id columns, in order —
    # defines the output format (single column = label/binary; one per class = multiclass proba)
    description: str | None = None  # the competition's description.md, when present — fed to
    # the judgment nodes so the LLM knows what the columns mean


@dataclass
class GradeReport:
    score: float | None
    gold: float | None
    silver: float | None
    bronze: float | None
    medal: str | None  # "gold" | "silver" | "bronze" | None
    valid: bool = True


def _find(task_dir: str, names: list[str]) -> str:
    for name in names:
        path = os.path.join(task_dir, name)
        if os.path.exists(path):
            return path
    raise MleBenchError(f"none of {names} found in task dir {task_dir!r}")


def read_task(task_dir: str) -> MleTask:
    """Read an MLE-bench task directory and derive the id + target columns.

    The sample submission is the source of truth for the id (its column also present in
    ``test.csv``) and the output format. Two shapes are supported:

    * **single column** (label or binary-probability) — the one non-id column is the target,
      a column of ``train.csv`` (e.g. random-acts-of-pizza, tps-may-2022);
    * **one column per class** (multiclass probability) — the non-id columns are class
      *values*, so the target is the ``train.csv`` column missing from ``test.csv``, verified
      against those class values (e.g. leaf-classification's ``species``).

    Anything else aborts."""
    train = _find(task_dir, ["train.csv"])
    test = _find(task_dir, ["test.csv"])
    sample = _find(task_dir, ["sample_submission.csv", "sampleSubmission.csv"])
    sub_cols = pd.read_csv(sample, nrows=1).columns.tolist()
    test_cols = pd.read_csv(test, nrows=1).columns.tolist()
    train_cols = pd.read_csv(train, nrows=1).columns.tolist()
    id_in_test = [c for c in sub_cols if c in test_cols]
    id_col = id_in_test[0] if id_in_test else sub_cols[0]
    sub_targets = [c for c in sub_cols if c != id_col]

    description = None
    desc_path = os.path.join(task_dir, "description.md")
    if os.path.exists(desc_path):  # shipped with every prepared MLE-bench task
        with open(desc_path) as fh:
            description = fh.read()

    def _task(target_col):
        return MleTask(os.path.basename(os.path.normpath(task_dir)), train, test, sample,
                       id_col, target_col, sub_targets, description)

    # Single column: it is itself the target column (label or binary probability).
    if len(sub_targets) == 1 and sub_targets[0] in train_cols:
        return _task(sub_targets[0])

    # Multiclass probability: the target is the train column absent from test; the submission
    # columns are its class values. Verify the match so we never guess.
    candidates = [c for c in train_cols if c != id_col and c not in test_cols]
    if len(candidates) == 1:
        target_col = candidates[0]
        classes = pd.read_csv(train, usecols=[target_col])[target_col].dropna().unique()
        if {str(c) for c in classes} == set(sub_targets):
            return _task(target_col)

    raise MleBenchError(
        f"unsupported submission shape: columns {sub_cols} match neither a single target "
        "column (label/binary) nor a train column's class values (multiclass probability)."
    )


def grade_submission(submission_path: str, competition_id: str, *, data_dir: str | None = None) -> GradeReport:
    """Grade a submission with mlebench's real medal thresholds (optional dependency)."""
    try:
        from mlebench.grade import grade_csv
        from mlebench.registry import registry
    except ImportError as exc:
        raise MleBenchError(
            "mlebench is not installed. Install the optional group: pip install 'maestra[mlebench]' "
            "(pulls mle-bench from git; needs Docker + prepared competition data)."
        ) from exc
    reg = registry.set_data_dir(Path(data_dir)) if data_dir else registry
    competition = reg.get_competition(competition_id)
    report = grade_csv(Path(submission_path), competition)
    medal = ("gold" if getattr(report, "gold_medal", False)
             else "silver" if getattr(report, "silver_medal", False)
             else "bronze" if getattr(report, "bronze_medal", False) else None)
    return GradeReport(
        score=getattr(report, "score", None),
        gold=getattr(report, "gold_threshold", None),
        silver=getattr(report, "silver_threshold", None),
        bronze=getattr(report, "bronze_threshold", None),
        medal=medal,
        valid=getattr(report, "valid_submission", True),
    )


def _cv_score_in_metric(metric, metric_mode, result, train_df, target_col):
    """The local CV score expressed in the COMPETITION metric, so the CV↔LB gap is comparable.

    Depending on ``metric_mode`` the score comes from AutoGluon directly (aligned), is computed
    on the out-of-fold predictions (oof) or on the pooled out-of-fold probabilities (proba).
    For probability metrics a calibration temperature is fitted on the OOF probabilities and its
    effect on the CV score recorded. Returns ``(cv_score, cv_metric, temperature, cv_score_cal)``.
    """
    cv = result.cv
    if metric_mode == "oof" and cv is not None and cv.oof_pred is not None:
        oof = cv.oof_pred.dropna()
        score = float(benchmark._METRICS[metric](train_df.loc[oof.index, target_col], oof.to_numpy()))
        return score, metric, None, None
    if metric_mode == "proba" and cv is not None and cv.oof_proba is not None:
        oof = cv.oof_proba.dropna()
        y_oof = train_df.loc[oof.index, target_col]
        # positive_class only exists for binary problems — asking a multiclass predictor for it
        # just emits an AutoGluon warning and returns None (the proba scorers ignore it anyway).
        positive = (getattr(result.training.predictor, "positive_class", None)
                    if result.training and cv.problem_type == "binary" else None)
        score = benchmark._PROBA_METRICS[metric](y_oof, oof, positive)
        # Fit T on the OOF probabilities and always record what it does to the
        # (calibration-sensitive) CV score. T<1 sharpens, T>1 softens.
        temperature = calibration.fit_temperature(y_oof, oof)
        score_cal = benchmark._PROBA_METRICS[metric](
            y_oof, calibration.apply_temperature(oof, temperature), positive)
        return score, metric, temperature, score_cal
    return (cv.mean if cv else None), (cv.eval_metric if cv else None), None, None


def _calibrate_submission(submission, columns, temperature):
    """Reshape the submission's probabilities in place with the OOF-fitted ``temperature``.

    Caution (measured on leaf-classification): the fold-OOF temperature does not necessarily
    transfer to the full-data final model — this can help or hurt the LB, hence opt-in.
    """
    if len(columns) == 1:  # binary: rebuild the 2-class distribution, calibrate, keep the positive col
        p = submission[columns[0]].to_numpy()
        two = pd.DataFrame({"neg": 1 - p, "pos": p})
        submission[columns[0]] = calibration.apply_temperature(two, temperature)["pos"].to_numpy()
    else:                  # multiclass: the columns already are the full distribution
        submission[columns] = calibration.apply_temperature(submission[columns], temperature).to_numpy()


def run_mlebench_task(
    task_dir: str,
    competition_id: str,
    *,
    model: str = "gpt-4o",
    cv_folds: int = 5,
    time_limit: int = 600,
    research: bool = False,
    hybrid: bool = False,
    no_llm: bool = False,
    calibrate: bool = False,
    seed: int = 42,
    metric: str | None = None,
    data_dir: str | None = None,
    out_dir: str = "mlebench_out",
    runs_log: str = "runs.jsonl",
) -> dict:
    """Run Maestra (with --cv) on one task, write a submission, grade it, log a record.

    ``metric`` is the *competition* metric: it is mapped to an AutoGluon eval_metric (CV
    optimises it) or computed on the CV's out-of-fold predictions, so the CV↔LB gap is
    comparable. ``metric_mode`` (aligned/oof/proba/mismatch) records how.
    """
    task = read_task(task_dir)
    train_df = pd.read_csv(task.train_csv)
    test_df = pd.read_csv(task.test_csv)

    ag_metric, metric_mode = _resolve_metric(metric)
    want_proba = metric_mode == "proba"
    result = run_pipeline(
        train_df, task.target_col, model=model, test_size=0.2, time_limit=time_limit, seed=seed,
        model_dir=f"AutogluonModels/mle_{task.name}_s{seed}", use_llm=not no_llm, cv_folds=cv_folds,
        hybrid=hybrid, research=research, eval_metric=ag_metric, test_df=test_df, id_col=task.id_col,
        proba=want_proba, proba_columns=task.submission_columns if want_proba else None,
        dataset_description=task.description,
    )
    if result.submission is None:
        raise MleBenchError("pipeline produced no submission (is the test set valid?)")

    cv_score, cv_metric, temperature, cv_score_cal = _cv_score_in_metric(
        metric, metric_mode, result, train_df, task.target_col)
    if calibrate and temperature is not None:
        _calibrate_submission(result.submission, task.submission_columns, temperature)

    os.makedirs(out_dir, exist_ok=True)
    mode = "baseline" if no_llm else "maestra"
    # Name by competition (not the generic "public" dir) + the run's variant + seed, so a file says
    # exactly which challenge and which configuration produced it.
    variant = mode + ("_hybrid" if hybrid else "") + ("_research" if research else "") + ("_cal" if calibrate else "")
    submission_path = os.path.join(out_dir, f"{competition_id}_{variant}_s{seed}_submission.csv")
    result.submission.to_csv(submission_path, index=False)
    report = grade_submission(submission_path, competition_id, data_dir=data_dir)

    comparable = metric_mode in ("aligned", "oof", "proba")
    gap = (cv_score - report.score) if (comparable and cv_score is not None and report.score is not None) else None
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "kind": "mlebench",
        "task": task.name,
        "competition_id": competition_id,
        "mode": mode,
        "seed": seed,
        "cv_score": cv_score,
        "cv_metric": cv_metric,
        "mle_score": report.score,
        "mle_metric": metric,
        "medal": report.medal,
        "thresholds": {"gold": report.gold, "silver": report.silver, "bronze": report.bronze},
        "cv_lb_gap": gap,
        "metric_mode": metric_mode,
        "submission": submission_path,
    }
    if temperature is not None:  # probability metric: record the calibration effect on the CV score
        record["temperature"] = temperature
        record["cv_score_cal"] = cv_score_cal
        record["cv_lb_gap_cal"] = (cv_score_cal - report.score) if report.score is not None else None
        record["calibrated_submission"] = bool(calibrate)
    if result.hybrid is not None:  # --hybrid provenance, so a run is interpretable after the fact
        record["hybrid"] = result.hybrid
        record["hybrid_kept"] = sum(1 for c in result.hybrid if c.get("kept"))
    with open(runs_log, "a") as fh:
        fh.write(json.dumps(record, default=float) + "\n")
    return record


def _num(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def _format_table(rows: list[tuple[dict, dict | None]]) -> str:
    headers = ["task", "metric", "baseline", "maestra", "gold", "silver", "bronze", "medal", "cv↔lb"]
    table = []
    for maestra, baseline in rows:
        thr = maestra["thresholds"]
        table.append([
            maestra["task"], maestra.get("mle_metric") or "?",
            _num(baseline.get("mle_score")) if baseline else "—", _num(maestra.get("mle_score")),
            _num(thr.get("gold")), _num(thr.get("silver")), _num(thr.get("bronze")),
            maestra.get("medal") or "-", _num(maestra.get("cv_lb_gap")),
        ])
    return benchmark.render_table(headers, table)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="maestra-mlebench",
                                description="Run Maestra on MLE-bench tasks and grade them.")
    p.add_argument("--task", action="append", default=[], metavar="DIR:COMPETITION_ID",
                   help="A prepared task dir and its competition id (repeatable). Start with ONE.")
    p.add_argument("--model", default=os.environ.get("AUTOML_MODEL", "gpt-4o"))
    p.add_argument("--cv", type=int, default=5, help="CV folds (the gate; >= 2).")
    p.add_argument("--time-limit", type=int, default=600, help="AutoGluon budget per fold.")
    p.add_argument("--metric", default=None,
                   help="Competition metric (e.g. accuracy, f1_macro, quadratic_weighted_kappa, "
                        "roc_auc, log_loss). Mapped to AutoGluon or computed on OOF preds/probabilities "
                        "so the CV↔LB gap is comparable. Probability metrics emit a probability submission.")
    p.add_argument("--research", action="store_true")
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--calibrate", action="store_true",
                   help="Temperature-scale the submission probabilities (proba metrics). The CV-side "
                        "calibration effect is logged either way.")
    p.add_argument("--no-baseline", action="store_true", help="Skip the --no-llm baseline per task.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the split/folds. Re-run with different seeds to gauge run-to-run "
                        "variance (Maestra's LLM/AutoGluon path is not bit-reproducible).")
    p.add_argument("--data-dir", default=None, help="MLE-bench data dir (for grading).")
    p.add_argument("--out-dir", default="mlebench_out")
    p.add_argument("--runs-log", default="runs.jsonl")
    args = p.parse_args(argv)
    load_dotenv()

    if not args.task:
        print("Error: at least one --task DIR:COMPETITION_ID is required.")
        return 1

    common = dict(model=args.model, cv_folds=args.cv, time_limit=args.time_limit,
                  metric=args.metric, data_dir=args.data_dir, out_dir=args.out_dir,
                  runs_log=args.runs_log, calibrate=args.calibrate, seed=args.seed)
    rows = []
    for spec in args.task:
        task_dir, competition_id = spec.rsplit(":", 1)
        maestra = run_mlebench_task(task_dir, competition_id, research=args.research,
                                    hybrid=args.hybrid, **common)
        baseline = None if args.no_baseline else run_mlebench_task(task_dir, competition_id, no_llm=True, **common)
        rows.append((maestra, baseline))

    print("\n" + _format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
