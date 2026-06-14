"""Hybrid feature generation with a CV gate.

The LLM proposes *real* feature code; safety comes from two independent mechanisms:

  1. **Sandbox** — each candidate's ``fit``/``transform`` runs in an isolated subprocess
     (no network, CPU/time limits); the ``transform`` never receives the target column.
  2. **CV gate** — a candidate is kept only if it improves the leakage-free cross-validation
     score beyond the fold-to-fold noise. The cross-validation is the arbiter, not the LLM.

This module owns the sandbox runner and (further down) the candidate generation + selection.
The fold-wise application lives in :mod:`maestra.validation` so generated features go through
exactly the same leakage-free fit/transform-per-fold machinery as the structured plans.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from maestra.llm import call_structured
from maestra.validation import _is_classification, _make_folds, _process_fold, cross_validate

# A candidate is kept only if it beats the baseline CV mean by more than this many fold
# standard deviations (plus a tiny absolute floor). Conservative: noise must be cleared.
_DEFAULT_SIGMA_MULT = 1.0
_MIN_ABS_DELTA = 1e-4

# Sandbox limits. The wall-clock timeout is the reliable hard stop; the CPU/mem rlimits are
# best-effort (macOS does not enforce RLIMIT_AS). Conservative by default.
_TIMEOUT_S = 8
_CPU_SECONDS = 6
_MEM_MB = 1024

# Run the worker by file path, NOT `python -m maestra._sandbox_worker`: the latter imports
# the maestra package (and AutoGluon) on every spawn, adding seconds per candidate.
_WORKER = os.path.join(os.path.dirname(__file__), "_sandbox_worker.py")


@dataclass
class SandboxResult:
    """Outcome of running one candidate in the sandbox."""

    status: str  # "ok" | "error" | "timeout"
    train: np.ndarray | None = None
    val: np.ndarray | None = None
    error: str | None = None


def run_in_sandbox(
    code: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    *,
    timeout: int = _TIMEOUT_S,
    cpu_seconds: int = _CPU_SECONDS,
    mem_mb: int = _MEM_MB,
) -> SandboxResult:
    """Run a candidate's ``fit``/``transform`` in an isolated process; never raises.

    ``fit`` sees ``train_df`` (with target); ``transform`` is applied to both frames with the
    target column removed. Returns the numeric feature values for train and val, or a clean
    error/timeout status.
    """
    tmp = tempfile.mkdtemp(prefix="maestra_sbx_")
    try:
        train_df.to_pickle(os.path.join(tmp, "train.pkl"))
        val_df.to_pickle(os.path.join(tmp, "val.pkl"))
        with open(os.path.join(tmp, "code.py"), "w") as fh:
            fh.write(code)
        with open(os.path.join(tmp, "meta.json"), "w") as fh:
            json.dump({"target": target, "cpu_seconds": cpu_seconds, "mem_mb": mem_mb}, fh)

        try:
            subprocess.run(
                [sys.executable, _WORKER, tmp],
                timeout=timeout,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult("timeout", error=f"exceeded {timeout}s")

        result_path = os.path.join(tmp, "result.json")
        if not os.path.exists(result_path):
            return SandboxResult("error", error="worker produced no result (killed?)")
        result = json.load(open(result_path))
        if result.get("status") != "ok":
            return SandboxResult("error", error=result.get("error", "unknown error"))
        return SandboxResult(
            "ok",
            train=np.load(os.path.join(tmp, "train.npy")),
            val=np.load(os.path.join(tmp, "val.npy")),
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- candidate generation ----------------------------------------------------------

@dataclass
class GeneratedFeature:
    name: str
    idea: str
    code: str
    source: str = "profile"  # "brief" | "profile"


FEATURE_CODE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "short snake_case identifier"},
                    "idea": {"type": "string", "description": "what the feature computes and why"},
                    "code": {
                        "type": "string",
                        "description": "Python defining fit(train_df)->params and transform(df, params)->Series",
                    },
                },
                "required": ["name", "code"],
            },
        }
    },
    "required": ["features"],
}

_CODEGEN_SYSTEM_PROMPT = (
    "Du generierst Python-Code fuer NEUE Tabellen-Features. Pro Feature definierst du GENAU "
    "zwei Funktionen:\n"
    "  def fit(train_df): -> params      # darf die Zielspalte nutzen (NUR train)\n"
    "  def transform(df, params): -> pandas.Series der Laenge len(df)\n"
    "REGELN: transform bekommt df OHNE die Zielspalte und darf NICHT auf sie zugreifen. "
    "Verwende nur pandas (als pd) und numpy (als np); KEINE imports ausser pandas/numpy/math. "
    "Keine Datei-/Netzwerkzugriffe. Die Ausgabe von transform muss NUMERISCH sein. Halte den "
    "Code kurz und robust (fange z.B. Division durch null ab). Schlage nur Features vor, die "
    "plausibel Signal tragen."
)


def propose_feature_code(
    model: str, profile: dict, research_context: str | None = None, max_candidates: int = 5
) -> list[GeneratedFeature]:
    """Ask the LLM to generate feature-code candidates from the profile (+ research ideas)."""
    source = "brief" if research_context else "profile"
    user_prompt = (
        f"Spalten-Profil (JSON):\n{json.dumps(profile, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Generiere hoechstens {max_candidates} Feature-Kandidaten."
    )
    if research_context:
        user_prompt += "\n\n" + research_context
    out = call_structured(
        model=model,
        system_prompt=_CODEGEN_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_name="generate_feature_code",
        tool_description="Generiere Feature-Code-Kandidaten (fit/transform) fuer ein ML-Problem.",
        parameters_schema=FEATURE_CODE_SCHEMA,
    )
    features = []
    for item in out.get("features", [])[:max_candidates]:
        if item.get("name") and item.get("code"):
            features.append(GeneratedFeature(item["name"], item.get("idea", ""), item["code"], source))
    return features


def apply_generated_features(train, other, target, features):
    """Add each feature as a column ``gen_<name>`` to ``train`` and ``other``.

    Every feature is fitted on ``train`` and transformed onto both frames in the sandbox
    (``other`` with the target removed). Features are independent — each is fitted on the
    original ``train`` (not on columns added by earlier features). A failing candidate is
    skipped (the dry-run filters persistent failures), so this never raises.
    """
    train_cols, other_cols = {}, {}
    for feat in features:
        res = run_in_sandbox(feat.code, train, other, target)
        if res.status == "ok":
            train_cols[f"gen_{feat.name}"] = res.train
            other_cols[f"gen_{feat.name}"] = res.val
    train, other = train.copy(), other.copy()
    for col, values in train_cols.items():
        train[col] = values
    for col, values in other_cols.items():
        other[col] = values
    return train, other


# --- CV gate -----------------------------------------------------------------------

@dataclass
class CandidateRecord:
    """Provenance for one candidate: where the idea came from and the gate's verdict."""

    name: str
    idea: str
    source: str
    cv_delta: float | None
    kept: bool
    reason: str  # improved | no_improvement | sandbox_error | timeout | invalid_output


def _dry_run(df, target, candidate, cleaning_plan, feature_plan, seed) -> SandboxResult:
    """Cheap check that a candidate executes + yields valid output (no model trained)."""
    folds = _make_folds(df, target, 2, seed, _is_classification(df[target]))
    tr_idx, val_idx = folds[0]
    proc_train, proc_val = _process_fold(df.iloc[tr_idx], df.iloc[val_idx], target, cleaning_plan, feature_plan)
    return run_in_sandbox(candidate.code, proc_train, proc_val, target)


def select_features(
    df,
    target,
    candidates,
    *,
    cleaning_plan,
    feature_plan,
    model_dir,
    time_limit,
    n_folds,
    seed,
    sigma_mult: float = _DEFAULT_SIGMA_MULT,
):
    """Greedy CV gate: keep a candidate only if it improves the CV mean beyond fold noise.

    Returns ``(kept_features, records, final_cv)``. Each kept feature raises the baseline for
    the next candidate; ``final_cv`` is the CV with all kept features (reusable as the run's
    reported estimate). The cross-validation is the arbiter; a plausible-but-useless feature
    is dropped.
    """
    base = cross_validate(df, target, cleaning_plan=cleaning_plan, feature_plan=feature_plan,
                          model_dir=f"{model_dir}/base", time_limit=time_limit, n_folds=n_folds, seed=seed)
    kept: list[GeneratedFeature] = []
    records: list[CandidateRecord] = []
    for i, cand in enumerate(candidates):
        dry = _dry_run(df, target, cand, cleaning_plan, feature_plan, seed)
        if dry.status != "ok":
            reason = "timeout" if dry.status == "timeout" else "sandbox_error"
            records.append(CandidateRecord(cand.name, cand.idea, cand.source, None, False, reason))
            continue
        trial = cross_validate(df, target, cleaning_plan=cleaning_plan, feature_plan=feature_plan,
                               generated_features=kept + [cand], model_dir=f"{model_dir}/cand_{i}",
                               time_limit=time_limit, n_folds=n_folds, seed=seed)
        delta = (trial.mean - base.mean) if base.greater_is_better else (base.mean - trial.mean)
        if delta > max(_MIN_ABS_DELTA, sigma_mult * base.std):
            kept.append(cand)
            base = trial  # raise the bar for the next candidate
            records.append(CandidateRecord(cand.name, cand.idea, cand.source, float(delta), True, "improved"))
        else:
            records.append(CandidateRecord(cand.name, cand.idea, cand.source, float(delta), False, "no_improvement"))
    return kept, records, base
