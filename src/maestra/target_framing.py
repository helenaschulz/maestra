"""Target framing agent (M11) — the LLM decides how the TARGET should be presented to the engine.

AutoGluon infers the problem type and optimises the requested metric, but it never reframes the
target itself: a heavily right-skewed price/income/charge target is fitted in raw space, where a
squared-error objective is dominated by the expensive tail. Training on ``log1p(target)`` and
inverting predictions turns those absolute errors into relative ones — the textbook House-Prices
move — and it is a *setup* decision the engine does not make. That makes it exactly the kind of
blind spot this project's thesis assigns to the LLM.

Same contract as every judgment node:

* **propose** — one structured call, fixed vocabulary (``none``/``log1p``), temperature 0.
* **verify** — deterministic safety checks (regression only, supported metric, non-negative
  numeric target); any defect falls back to ``none``.
* **arbiter** — the transform is adopted ONLY if a paired CV run beats the untransformed base
  beyond noise (``improves_beyond_noise``, scored in ORIGINAL target space — a log-space score
  would not be comparable). The LLM proposes; the measurement decides.

Cost: one extra CV run, and only when the LLM proposes a transform that survives verification —
targeting, not blanket search.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from maestra.llm import call_structured

# Metrics whose original-space value we can recompute after inverting the predictions.
# AutoGluon reports metrics in "higher is better" form (errors negated); _ag_score in
# validation.py mirrors that convention for the trial CV.
SUPPORTED_METRICS = {"root_mean_squared_error", "mean_absolute_error", "r2"}

TARGET_FRAMING_SCHEMA = {
    "type": "object",
    "properties": {
        "transform": {
            "type": "string",
            "enum": ["none", "log1p"],
            "description": (
                "log1p ONLY for a continuous, non-negative, clearly right-skewed regression "
                "target (mean well above median, long expensive tail). Otherwise none."
            ),
        },
        "rationale": {
            "type": "string",
            "description": "Why this framing, grounded in the target's statistics and meaning.",
        },
    },
    "required": ["transform", "rationale"],
}

_SYSTEM_PROMPT = """\
You decide whether a tabular REGRESSION target should be log-transformed before training.

The engine (AutoGluon) fits the target exactly as given. For a heavily right-skewed non-negative
target (prices, incomes, charges, durations), a squared-error objective in raw space is dominated
by the few most expensive rows; training on log1p(target) and inverting the predictions makes
errors relative instead of absolute and usually helps. Your proposal will be VERIFIED by a paired
cross-validation in original units — you do not need to be certain, but do not propose noise.

Propose log1p ONLY when ALL of these hold:
* the target is continuous (many distinct values), non-negative, and used for regression;
* the distribution is clearly right-skewed: mean noticeably above the median, maximum several
  times the median — a long right tail, not a mild lean.

CAUTION — these are NOT log1p cases:
* classification targets of any kind, including integer class codes;
* small-range counts or ratings (a handful of distinct integer values);
* targets that are already logs, rates, percentages, or roughly symmetric;
* targets with negative values (log1p is undefined there — answer none).

When in doubt, answer none: a wrong "none" costs nothing (the engine's default), a wrong "log1p"
costs a wasted verification run."""


@dataclass
class TargetTransform:
    """An invertible target transform: fit nothing, just map both ways."""

    name: str
    forward: Callable[[pd.Series], pd.Series] = field(repr=False)
    inverse: Callable[[pd.Series], pd.Series] = field(repr=False)


def _log1p_transform() -> TargetTransform:
    return TargetTransform(
        name="log1p",
        forward=lambda s: pd.Series(np.log1p(s.to_numpy(dtype=float)), index=s.index),
        inverse=lambda s: pd.Series(np.expm1(s.to_numpy(dtype=float)), index=s.index),
    )


def target_stats(df: pd.DataFrame, target: str) -> dict:
    """Deterministic distribution summary of the target — the profile carries no distribution
    statistics, and skewness is exactly what this decision needs. Non-numeric targets get a
    minimal summary (the verifier rejects them anyway)."""
    s = df[target].dropna()
    stats: dict = {"dtype": str(s.dtype), "n_unique": int(s.nunique())}
    if s.dtype.kind in "iuf" and len(s):
        x = s.to_numpy(dtype=float)
        stats.update(
            min=float(np.min(x)), p25=float(np.percentile(x, 25)),
            median=float(np.median(x)), mean=float(np.mean(x)),
            p75=float(np.percentile(x, 75)), max=float(np.max(x)),
            skewness=float(pd.Series(x).skew()),
        )
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}


def propose_target_framing(model: str, profile: dict, df: pd.DataFrame, target: str,
                           context: str | None = None) -> dict:
    """Ask the LLM whether the target should be log-transformed. Matches TARGET_FRAMING_SCHEMA."""
    user_prompt = (
        f"Target column: {target}\n"
        f"Target distribution (JSON):\n{json.dumps(target_stats(df, target), ensure_ascii=False)}\n"
        f"Column profile (JSON):\n{json.dumps(profile, ensure_ascii=False, indent=2)}"
    )
    if context:
        user_prompt += "\n\n" + context
    return call_structured(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_name="propose_target_framing",
        tool_description="Decide whether the regression target should be log1p-transformed.",
        parameters_schema=TARGET_FRAMING_SCHEMA,
    )


def validate_target_framing(proposal: dict, df: pd.DataFrame, target: str,
                            problem_type: str | None, eval_metric: str | None,
                            ) -> tuple[TargetTransform | None, list[str]]:
    """Deterministically verify the proposal; any defect falls back to no transform.

    Safety checks only — whether the transform actually HELPS is not judged here but by the CV
    arbiter. ``problem_type``/``eval_metric`` come from the base CV, so the check runs against
    what AutoGluon actually inferred, not a re-derivation.

    Returns ``(TargetTransform | None, log_lines)``.
    """
    log: list[str] = []
    rationale = proposal.get("rationale", "")

    if proposal.get("transform") != "log1p":
        log.append("FRAMING none (LLM): " + (rationale or "no transform proposed"))
        return None, log

    def fallback(reason: str) -> tuple[None, list[str]]:
        log.append(f"FRAMING none (fallback): {reason}")
        return None, log

    if problem_type != "regression":
        return fallback(f"problem type is {problem_type!r}, log1p applies to regression only")
    if eval_metric not in SUPPORTED_METRICS:
        return fallback(
            f"metric {eval_metric!r} cannot be rescored in original space "
            f"(supported: {sorted(SUPPORTED_METRICS)})")
    s = df[target].dropna()
    if s.dtype.kind not in "iuf":
        return fallback(f"target dtype {s.dtype} is not numeric")
    if not len(s):
        return fallback("target has no non-null values")
    if float(s.min()) < 0:
        return fallback("target has negative values, log1p is undefined")
    if not np.isfinite(s.to_numpy(dtype=float)).all():
        return fallback("target has non-finite values")

    log.append(f"FRAMING log1p proposed -- {rationale}")
    return _log1p_transform(), log
