"""Command-line entry point. Parses args, presents results, handles user-facing errors.

Installed as the ``maestra`` console script. All real work lives in the library
modules; this file only does I/O and formatting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

from maestra.llm import LLMError
from maestra.pipeline import PipelineError, run_pipeline
from maestra.runlog import append_run, compare_runs


def load_dotenv(path: str = ".env") -> None:
    """Load ``KEY=VALUE`` lines from a local ``.env`` into ``os.environ``.

    A tiny hand-rolled loader so the project carries no extra dependency just to read
    one API key. Existing environment variables take precedence (``setdefault``).
    """
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="maestra",
        description="LLM-conducted cleaning + AutoGluon training for a tabular CSV.",
    )
    p.add_argument("--csv", required=True, help="Path to the input CSV.")
    p.add_argument("--target", required=True, help="Name of the target column.")
    p.add_argument(
        "--model",
        default=os.environ.get("AUTOML_MODEL", "gpt-4o"),
        help="LiteLLM model string (default: gpt-4o or $AUTOML_MODEL).",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction (default 0.2).")
    p.add_argument("--time-limit", type=int, default=120, help="Training budget in seconds.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    p.add_argument("--model-dir", default="AutogluonModels", help="AutoGluon artefact dir.")
    p.add_argument("--no-llm", action="store_true", help="Skip cleaning (baseline run).")
    p.add_argument("--no-fe", action="store_true", help="Skip LLM feature engineering.")
    p.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Attempts before giving up. >1 enables the LLM failure-diagnosis loop.",
    )
    p.add_argument(
        "--revise-below",
        type=float,
        default=None,
        help="Floor on AutoGluon's internal val score; below it the LLM revises the plan "
             "once and retrains (needs --max-attempts > 1). Off by default.",
    )
    p.add_argument("--test", help="Unlabeled test CSV to predict on (for a Kaggle submission).")
    p.add_argument("--submission", help="Where to write the submission CSV (requires --test).")
    p.add_argument("--id-col", default="id", help="Identifier column for the submission (default id).")
    p.add_argument("--runs-log", default="runs.jsonl", help="Append-only run log path (default runs.jsonl).")
    p.add_argument(
        "--compare",
        action="store_true",
        help="Print the latest --no-llm vs LLM diff for this csv/target from the run log, then exit.",
    )
    return p.parse_args(argv)


def _print_result(result, model: str) -> None:
    if result.diagnosis_log:
        print(f"\n=== Diagnosis / revision loop ({result.attempts} attempts) ===")
        for i, d in enumerate(result.diagnosis_log, 1):
            why = "weak val score" if d.get("trigger") == "weak_metric" else "failed"
            print(f"  {why} -> {d.get('action')}: {d.get('diagnosis')}")

    if result.plan is None:
        print("\n[--no-llm] Cleaning skipped.")
    else:
        print(f"\n=== LLM cleaning plan ({model}) ===")
        print(json.dumps(result.plan, ensure_ascii=False, indent=2))
        print("\n=== Applied ===")
        for line in result.cleaning_log:
            print(f"  {line}")
        print(f"Columns after cleaning: {result.n_cols_clean} (from {result.n_cols_before})")

    if result.feature_plan is not None:
        print(f"\n=== LLM feature engineering ({model}) ===")
        for line in result.feature_log:
            print(f"  {line}")
        print(f"Columns after feature engineering: {result.n_cols_after} (from {result.n_cols_clean})")

    t = result.training
    print(f"\nProblem type (inferred by AutoGluon): {t.problem_type}")
    print(f"Eval metric: {t.eval_metric}")
    print("\n=== Leaderboard on holdout ===")
    print(t.leaderboard)
    print("\n=== Best-model metrics on holdout ===")
    for name, value in t.metrics.items():
        print(f"  {name}: {value}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)
    load_dotenv()

    if args.compare:
        print(compare_runs(args.runs_log, args.csv, args.target))
        return 0

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
        return 1
    except pd.errors.EmptyDataError:
        print(f"Error: CSV is empty: {args.csv}", file=sys.stderr)
        return 1

    print(f"Loaded {args.csv}: rows={len(df)}, columns={len(df.columns)}")

    if args.submission and not args.test:
        print("Error: --submission requires --test.", file=sys.stderr)
        return 1
    test_df = None
    if args.test:
        try:
            test_df = pd.read_csv(args.test)
        except FileNotFoundError:
            print(f"Error: test CSV not found: {args.test}", file=sys.stderr)
            return 1
        print(f"Loaded test {args.test}: rows={len(test_df)}")

    try:
        result = run_pipeline(
            df,
            args.target,
            model=args.model,
            test_size=args.test_size,
            time_limit=args.time_limit,
            seed=args.seed,
            model_dir=args.model_dir,
            use_llm=not args.no_llm,
            use_fe=not args.no_fe,
            max_attempts=args.max_attempts,
            revise_below=args.revise_below,
            test_df=test_df,
            id_col=args.id_col,
        )
    except ValueError as exc:  # bad target column
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        print("Hint: check the model name and that the provider's API key is set.", file=sys.stderr)
        return 2
    except PipelineError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 3

    _print_result(result, args.model)

    if result.submission is not None and args.submission:
        result.submission.to_csv(args.submission, index=False)
        print(f"\nSubmission written: {args.submission} ({len(result.submission)} rows)")
        print(result.submission.head().to_string(index=False))

    append_run(
        args.runs_log,
        result,
        csv=args.csv,
        target=args.target,
        model=args.model,
        no_llm=args.no_llm,
        max_attempts=args.max_attempts,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    print(f"\nLogged to {args.runs_log}. Compare with: maestra --csv {args.csv} --target {args.target} --compare")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
