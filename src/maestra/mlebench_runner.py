"""MLE-bench adapter: run Maestra on a real MLE-bench task and grade it against the
competition's medal thresholds.

MLE-bench is OpenAI's benchmark of completed Kaggle competitions. A prepared task directory
holds ``train.csv``, ``test.csv`` and ``sample_submission.csv``; grading against the private
answer key (with gold/silver/bronze thresholds) is done by the ``mlebench`` package — an
*optional* dependency (heavy: Docker + competition data). It is reached only through
:func:`grade_submission` and is mocked in tests.

The adapter reuses the existing run -> submission flow (``run_pipeline``); it does not
re-implement it. The value it adds is the **CV↔LB gap**: with ``--metric`` aligning the CV to
the competition metric, the gap between Maestra's local cross-validation and the graded
score tells you whether the CV is trustworthy.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from maestra.cli import load_dotenv
from maestra.pipeline import run_pipeline


class MleBenchError(RuntimeError):
    """Raised for task-reading or grading failures (incl. the optional dep being absent)."""


@dataclass
class MleTask:
    name: str
    train_csv: str
    test_csv: str
    sample_submission_csv: str
    id_col: str
    target_col: str


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

    The sample submission is the source of truth: its column also present in ``test.csv`` is
    the id, the other is the target (which ``train.csv`` carries). Multi-column / probability
    submissions are out of scope for now (single-target label prediction)."""
    train = _find(task_dir, ["train.csv"])
    test = _find(task_dir, ["test.csv"])
    sample = _find(task_dir, ["sample_submission.csv", "sampleSubmission.csv"])
    sub_cols = pd.read_csv(sample, nrows=1).columns.tolist()
    test_cols = pd.read_csv(test, nrows=1).columns.tolist()
    id_in_test = [c for c in sub_cols if c in test_cols]
    id_col = id_in_test[0] if id_in_test else sub_cols[0]
    targets = [c for c in sub_cols if c != id_col]
    if not targets:
        raise MleBenchError(f"could not derive a target from sample_submission columns {sub_cols}")
    return MleTask(os.path.basename(os.path.normpath(task_dir)), train, test, sample, id_col, targets[0])


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
    eval_metric: str | None = None,
    data_dir: str | None = None,
    out_dir: str = "mlebench_out",
    runs_log: str = "runs.jsonl",
) -> dict:
    """Run Maestra (with --cv) on one task, write a submission, grade it, log a record.

    Returns the logged record. ``eval_metric`` (the AutoGluon name of the competition metric)
    aligns the CV with the LB so the CV↔LB gap is comparable; without it the record is flagged
    ``metric_aligned=False``.
    """
    task = read_task(task_dir)
    train_df = pd.read_csv(task.train_csv)
    test_df = pd.read_csv(task.test_csv)

    result = run_pipeline(
        train_df, task.target_col, model=model, test_size=0.2, time_limit=time_limit, seed=42,
        model_dir=f"AutogluonModels/mle_{task.name}", use_llm=not no_llm, cv_folds=cv_folds,
        hybrid=hybrid, research=research, eval_metric=eval_metric, test_df=test_df, id_col=task.id_col,
    )
    if result.submission is None:
        raise MleBenchError("pipeline produced no submission (is the test set valid?)")

    os.makedirs(out_dir, exist_ok=True)
    mode = "baseline" if no_llm else "maestra"
    submission_path = os.path.join(out_dir, f"{task.name}_{mode}_submission.csv")
    result.submission.to_csv(submission_path, index=False)

    report = grade_submission(submission_path, competition_id, data_dir=data_dir)

    cv_score = result.cv.mean if result.cv else None
    gap = (cv_score - report.score) if (cv_score is not None and report.score is not None) else None
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "kind": "mlebench",
        "task": task.name,
        "competition_id": competition_id,
        "mode": mode,
        "cv_score": cv_score,
        "cv_metric": result.cv.eval_metric if result.cv else None,
        "mle_score": report.score,
        "medal": report.medal,
        "thresholds": {"gold": report.gold, "silver": report.silver, "bronze": report.bronze},
        "cv_lb_gap": gap,
        "metric_aligned": eval_metric is not None,
        "submission": submission_path,
    }
    with open(runs_log, "a") as fh:
        fh.write(json.dumps(record, default=float) + "\n")
    return record


def _fmt(value, width=9) -> str:
    return f"{value:>{width}.4f}" if isinstance(value, (int, float)) else f"{'—':>{width}}"


def _format_table(rows: list[tuple[dict, dict | None]]) -> str:
    head = f"{'task':<18}{'metric':<12}{'baseline':>10}{'maestra':>10}{'gold':>9}{'silver':>9}{'bronze':>9}  medal     cv↔lb"
    out = [head]
    for maestra, baseline in rows:
        thr = maestra["thresholds"]
        out.append(
            f"{maestra['task']:<18}{(maestra['cv_metric'] or '?'):<12}"
            f"{_fmt(baseline.get('mle_score') if baseline else None, 10)}"
            f"{_fmt(maestra.get('mle_score'), 10)}"
            f"{_fmt(thr.get('gold'))}{_fmt(thr.get('silver'))}{_fmt(thr.get('bronze'))}"
            f"  {str(maestra.get('medal') or '-'):<8}  {_fmt(maestra.get('cv_lb_gap'), 7)}"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="maestra-mlebench",
                                description="Run Maestra on MLE-bench tasks and grade them.")
    p.add_argument("--task", action="append", default=[], metavar="DIR:COMPETITION_ID",
                   help="A prepared task dir and its competition id (repeatable). Start with ONE.")
    p.add_argument("--model", default=os.environ.get("AUTOML_MODEL", "gpt-4o"))
    p.add_argument("--cv", type=int, default=5, help="CV folds (the gate; >= 2).")
    p.add_argument("--time-limit", type=int, default=600, help="AutoGluon budget per fold.")
    p.add_argument("--metric", default=None,
                   help="AutoGluon eval_metric to align the CV with the competition metric.")
    p.add_argument("--research", action="store_true")
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--baseline", action="store_true", help="Also run the --no-llm baseline per task.")
    p.add_argument("--data-dir", default=None, help="MLE-bench data dir (for grading).")
    p.add_argument("--out-dir", default="mlebench_out")
    p.add_argument("--runs-log", default="runs.jsonl")
    args = p.parse_args(argv)
    load_dotenv()

    if not args.task:
        print("Error: at least one --task DIR:COMPETITION_ID is required.")
        return 1

    common = dict(model=args.model, cv_folds=args.cv, time_limit=args.time_limit,
                  eval_metric=args.metric, data_dir=args.data_dir, out_dir=args.out_dir,
                  runs_log=args.runs_log)
    rows = []
    for spec in args.task:
        task_dir, competition_id = spec.rsplit(":", 1)
        maestra = run_mlebench_task(task_dir, competition_id, research=args.research,
                                    hybrid=args.hybrid, **common)
        baseline = run_mlebench_task(task_dir, competition_id, no_llm=True, **common) if args.baseline else None
        rows.append((maestra, baseline))

    print("\n" + _format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
