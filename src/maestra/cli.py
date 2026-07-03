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

from maestra.config import load_dotenv
from maestra.llm import LLMError
from maestra.pipeline import PipelineError, run_pipeline
from maestra.report import generate_report
from maestra.runlog import append_run, compare_runs


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
    p.add_argument(
        "--cv",
        type=int,
        default=None,
        metavar="K",
        help="Run leakage-free K-fold cross-validation instead of a single holdout (K >= 2).",
    )
    p.add_argument("--cv-time-limit", type=int, default=None, help="Training budget per CV fold.")
    p.add_argument(
        "--fold-advisor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Validation Strategist: the LLM decides how CV folds are built (random/group/time) "
             "from the column semantics; its proposal is verified deterministically. "
             "ON BY DEFAULT when --cv is set (M9: 0 false alarms across all frontier models); "
             "use --no-fold-advisor to disable, or pass --fold-advisor without --cv for an error.",
    )
    p.add_argument(
        "--ordinal",
        action="store_true",
        help="Ordinal encoding: the LLM maps ordinal categoricals (e.g. quality ratings) to a "
             "worst->best rank the trees cannot infer from unordered labels.",
    )
    p.add_argument(
        "--skeptic",
        action="store_true",
        help="Skeptic: a second LLM attacks the cleaning plan's drops; each high-risk drop is "
             "put to the CV arbiter and vetoed only if keeping the column helps (needs --cv).",
    )
    p.add_argument(
        "--hybrid",
        action="store_true",
        help="Generate feature code, sandbox-run it, and keep only what improves the CV "
             "(requires --cv). Off by default.",
    )
    p.add_argument("--hybrid-max-candidates", type=int, default=5, help="Max feature candidates.")
    p.add_argument(
        "--hybrid-threshold",
        type=float,
        default=2.0,
        help="Keep a candidate only if its paired per-fold improvement exceeds this many standard "
             "errors (and it improves in a majority of folds).",
    )
    p.add_argument(
        "--research",
        action="store_true",
        help="Run web strategy research first and feed its brief to the planners as "
             "non-binding hypotheses (off by default; needs a search provider key).",
    )
    p.add_argument(
        "--rules-mode",
        choices=["offline", "live"],
        default="offline",
        help="Competition rules mode for --research (default offline).",
    )
    p.add_argument(
        "--description",
        help="Path to a provider-written dataset description (e.g. Kaggle's data_description.txt); "
             "fed to every judgment node so the LLM knows what columns MEAN.",
    )
    p.add_argument("--test", help="Unlabeled test CSV to predict on (for a Kaggle submission).")
    p.add_argument("--submission", help="Where to write the submission CSV (requires --test).")
    p.add_argument("--id-col", default="id", help="Identifier column for the submission (default id).")
    p.add_argument("--report", help="Write an LLM-generated Markdown report of the run to this path.")
    p.add_argument("--runs-log", default="runs.jsonl", help="Append-only run log path (default runs.jsonl).")
    p.add_argument(
        "--compare",
        action="store_true",
        help="Print the latest --no-llm vs LLM diff for this csv/target from the run log, then exit.",
    )
    return p.parse_args(argv)


def _resolve_fold_advisor(flag: bool | None, cv: int | None) -> bool:
    """Resolve the tri-state --fold-advisor/--no-fold-advisor into a concrete bool.

    ``flag`` is ``None`` when the user passed neither form → "auto": the Validation Strategist
    runs by default whenever CV is active (``cv >= 2``). This promotion is gated on M9 evidence
    (0 false alarms across all four frontier models, cross-provider). An explicit
    ``--fold-advisor``/``--no-fold-advisor`` always wins; an explicit ``--fold-advisor`` without
    ``--cv`` stays ``True`` so the pipeline raises its "needs --cv" error rather than silently
    ignoring the request.
    """
    if flag is not None:
        return flag
    return bool(cv and cv >= 2)


def _print_result(result, model: str) -> None:
    if result.research is not None:
        r = result.research
        print(f"\n=== Strategy research ({model}, rules_mode={r.get('rules_mode')}) ===")
        print(f"  {len(r.get('references') or [])} references, grounded={r.get('grounded')} "
              f"-> fed to planning as non-binding hypotheses")

    if result.fold_strategy is not None:
        fs = result.fold_strategy
        print(f"\n=== Validation Strategist ({model}) ===")
        for line in fs.get("log", []):
            print(f"  {line}")

    if result.ordinal is not None:
        print(f"\n=== Ordinal encoding ({model}) ===")
        for line in result.ordinal.get("log", []):
            print(f"  {line}")

    if result.skeptic is not None:
        print(f"\n=== Skeptic review of cleaning drops ({model}) ===")
        for r in result.skeptic:
            if r.get("vetoed"):
                print(f"  VETO keep '{r['column']}' (Δcv={r['cv_delta']:+.4f}) -- {r['reason']}")
            elif r.get("risk") == "high":
                print(f"  drop '{r['column']}' upheld (Δcv={r.get('cv_delta')})  -- flagged: {r['reason']}")

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

    if result.cv is not None:
        cv = result.cv
        scores = ", ".join(f"{s:.4f}" for s in cv.fold_scores)
        kind = "stratified " if cv.stratified else ""
        print(f"\n=== {cv.n_folds}-fold {kind}cross-validation ({cv.eval_metric}) ===")
        print(f"  mean {cv.mean:.4f} ± {cv.std:.4f}   folds: [{scores}]")

    if result.adversarial_auc is not None:
        auc = result.adversarial_auc
        verdict = "no detectable shift" if auc < 0.6 else ("mild shift" if auc < 0.75 else "strong shift")
        print(f"\n=== Adversarial validation ===\n  train-vs-test AUC: {auc:.4f}  ({verdict})")

    if result.hybrid is not None:
        kept = [r for r in result.hybrid if r.get("kept")]
        print(f"\n=== Hybrid features (CV-gated): {len(kept)}/{len(result.hybrid)} kept ===")
        for r in result.hybrid:
            delta = f"{r['cv_delta']:+.4f}" if r.get("cv_delta") is not None else "  n/a"
            mark = "KEEP" if r.get("kept") else "drop"
            print(f"  [{mark}] {r.get('name')}  Δcv={delta}  ({r.get('reason')})  src={r.get('source')}")

    if t.metrics:  # holdout path; empty under --cv (CV is the estimate)
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

    dataset_description = None
    if args.description:
        try:
            with open(args.description) as fh:
                dataset_description = fh.read()
        except FileNotFoundError:
            print(f"Error: description file not found: {args.description}", file=sys.stderr)
            return 1
        print(f"Loaded description {args.description} ({len(dataset_description)} chars)")

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
            cv_folds=args.cv,
            cv_time_limit=args.cv_time_limit,
            fold_advisor=_resolve_fold_advisor(args.fold_advisor, args.cv),
            ordinal=args.ordinal,
            skeptic=args.skeptic,
            hybrid=args.hybrid,
            hybrid_max_candidates=args.hybrid_max_candidates,
            hybrid_threshold=args.hybrid_threshold,
            research=args.research,
            rules_mode=args.rules_mode,
            test_df=test_df,
            id_col=args.id_col,
            dataset_description=dataset_description,
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

    if args.report:
        try:
            report_md = generate_report(args.model, result)
        except LLMError as exc:
            print(f"Report skipped (LLM error): {exc}", file=sys.stderr)
        else:
            with open(args.report, "w") as fh:
                fh.write(report_md)
            print(f"Report written: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
