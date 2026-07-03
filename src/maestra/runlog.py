"""Append-only run log + a baseline comparison — no framework, no DB, just JSONL.

Each run is one line in ``runs.jsonl``. This lives outside ``run_pipeline`` on purpose:
the pipeline stays side-effect-free (returns data), and the CLI does the I/O here, so
the log format is testable without touching the pipeline or AutoGluon.
"""
from __future__ import annotations

import json

from maestra.pipeline import PipelineResult


def _cv_record(cv) -> dict | None:
    """Flatten a CVResult to a JSON-friendly dict (or None when CV wasn't run)."""
    if cv is None:
        return None
    return {
        "eval_metric": cv.eval_metric,
        "mean": cv.mean,
        "std": cv.std,
        "folds": cv.n_folds,
        "stratified": cv.stratified,
        "fold_scores": cv.fold_scores,
    }


def append_run(
    path: str,
    result: PipelineResult,
    *,
    csv: str,
    target: str,
    model: str,
    no_llm: bool,
    max_attempts: int,
    timestamp: str,
) -> None:
    """Append one run as a JSON line to ``path``.

    The timestamp is passed in (not read from the clock) so callers stay deterministic
    and the record is easy to assert on in tests.
    """
    record = {
        "timestamp": timestamp,
        "csv": csv,
        "target": target,
        "model": model,
        "no_llm": no_llm,
        "max_attempts": max_attempts,
        "attempts": result.attempts,
        "plan": result.plan,
        "feature_plan": result.feature_plan,
        "diagnosis_log": result.diagnosis_log,
        "metrics": result.training.metrics if result.training else None,
        "cv": _cv_record(result.cv),
        "adversarial_auc": result.adversarial_auc,
        "research": result.research,
        "hybrid": result.hybrid,
        "target_framing": result.target_framing,
        "text_features": result.text_features,
        "cv_budget": result.cv_budget,
    }
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=float) + "\n")


def _read_runs(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _latest(runs: list[dict], *, no_llm: bool) -> dict | None:
    """Most recent matching run (the file is in chronological append order)."""
    for run in reversed(runs):
        if run.get("no_llm") is no_llm:
            return run
    return None


def compare_runs(path: str, csv: str, target: str) -> str:
    """Compare the latest LLM-cleaning vs. ``--no-llm`` baseline run for ``csv``/``target``.

    Returns a human-readable block; a hint string if either side is missing.
    """
    try:
        runs = _read_runs(path)
    except FileNotFoundError:
        return f"No run log at {path!r} yet — run once with and once without --no-llm."

    relevant = [r for r in runs if r.get("csv") == csv and r.get("target") == target]
    baseline = _latest(relevant, no_llm=True)
    llm = _latest(relevant, no_llm=False)
    if not baseline or not llm:
        return (
            f"Need one run with and one without --no-llm for {csv} / {target} to compare "
            f"(have baseline={bool(baseline)}, llm={bool(llm)})."
        )

    base_m, llm_m = baseline.get("metrics") or {}, llm.get("metrics") or {}
    shared = sorted(set(base_m) & set(llm_m))
    lines = [f"Baseline (--no-llm)  vs  LLM cleaning   ({csv} / {target})"]
    for key in shared:
        base_v, llm_v = base_m[key], llm_m[key]
        lines.append(f"  {key:<18} {base_v:.4f} -> {llm_v:.4f}  ({llm_v - base_v:+.4f})")
    return "\n".join(lines)
