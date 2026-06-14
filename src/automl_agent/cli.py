"""Command-line entry point. Parses args, presents results, handles user-facing errors.

Installed as the ``automl-agent`` console script. All real work lives in the library
modules; this file only does I/O and formatting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from automl_agent.llm import LLMError
from automl_agent.pipeline import PipelineError, run_pipeline


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
        prog="automl-agent",
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
    p.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Attempts before giving up. >1 enables the LLM failure-diagnosis loop.",
    )
    return p.parse_args(argv)


def _print_result(result, model: str) -> None:
    if result.diagnosis_log:
        print(f"\n=== Diagnosis loop: succeeded on attempt {result.attempts} ===")
        for i, d in enumerate(result.diagnosis_log, 1):
            print(f"  attempt {i} failed -> {d.get('action')}: {d.get('diagnosis')}")

    if result.plan is None:
        print("\n[--no-llm] Cleaning skipped.")
    else:
        print(f"\n=== LLM cleaning plan ({model}) ===")
        print(json.dumps(result.plan, ensure_ascii=False, indent=2))
        print("\n=== Applied ===")
        for line in result.cleaning_log:
            print(f"  {line}")
        print(f"Columns after cleaning: {result.n_cols_after} (from {result.n_cols_before})")

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

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
        return 1
    except pd.errors.EmptyDataError:
        print(f"Error: CSV is empty: {args.csv}", file=sys.stderr)
        return 1

    print(f"Loaded {args.csv}: rows={len(df)}, columns={len(df.columns)}")

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
            max_attempts=args.max_attempts,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
