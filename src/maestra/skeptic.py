"""Skeptic agent — an adversarial reviewer of the cleaning plan, ruled by measurement.

Cleaning is where LLM judgment has historically HURT: the conductor dropped real predictive
columns as "useless" (the Stellar photometric bands `u,g,r,i,z` dropped as "unique per row";
similar losses on Titanic and leaf). The Skeptic is a second LLM in a different role — it does
not clean, it *attacks* the cleaning plan, flagging drops that might remove real signal.

Crucially it is not an LLM judging an LLM: a flag is only a nomination. Each high-risk drop is
put to the empirical arbiter — a leakage-free CV with the column KEPT versus dropped — and the
drop is vetoed ONLY if keeping the column measurably improves the score beyond fold noise. The
Skeptic makes the check cheap (it targets the few suspicious drops instead of ablating every
column); the measurement, never the model's opinion, has the final word.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

from maestra.intervention import run_counterfactual
from maestra.llm import call_structured
from maestra.validation import cross_validate

_MIN_ABS_DELTA = 1e-4

SKEPTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "risk": {"type": "string", "enum": ["high", "low"]},
                    "reason": {"type": "string"},
                },
                "required": ["column", "risk", "reason"],
            },
        }
    },
    "required": ["reviews"],
}

_SYSTEM_PROMPT = (
    "You are a skeptical senior data scientist reviewing ANOTHER data scientist's cleaning plan "
    "before it runs. Focus only on the COLUMNS TO DROP. For each drop, judge the risk that it "
    "removes genuine predictive signal. Mark risk=high when a dropped column is plausibly a real "
    "feature — especially CONTINUOUS numeric measurements (sensor readings, coordinates, "
    "brightnesses, prices) dropped for being high-cardinality or 'unique per row', which is a "
    "classic and costly mistake (a measurement is not an identifier). Mark risk=low for genuine "
    "identifiers (running integer IDs), near-constant columns, or obvious leakage. Be selective: "
    "only a real threat to signal is high. Every high-risk flag will be checked against the data, "
    "so flag what deserves a check."
)


@dataclass
class SkepticRecord:
    """Provenance for one reviewed drop: the Skeptic's flag and the arbiter's verdict."""

    column: str
    risk: str
    reason: str
    cv_delta: float | None = None   # improvement from KEEPING the column (None if not measured)
    vetoed: bool = False            # True = drop overturned, column kept
    measured: bool = True           # False when the CV budget refused the counterfactual trial


def review_cleaning_plan(model: str, plan: dict, profile: dict) -> list[dict]:
    """Ask the Skeptic LLM to risk-rate each proposed drop. Returns a list of review dicts."""
    drops = [d.get("column") for d in plan.get("columns_to_drop", []) if d.get("column")]
    if not drops:
        return []
    user_prompt = (
        f"Proposed columns to drop: {drops}\n\n"
        f"Column profile (JSON):\n{json.dumps(profile, ensure_ascii=False, indent=2)}"
    )
    out = call_structured(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_name="review_cleaning_plan",
        tool_description="Risk-rate each proposed column drop for removing real signal.",
        parameters_schema=SKEPTIC_SCHEMA,
    )
    return out.get("reviews", [])


def _plan_without_drop(plan: dict, column: str) -> dict:
    """A copy of ``plan`` with ``column`` removed from columns_to_drop (i.e. the column is kept)."""
    p = copy.deepcopy(plan)
    p["columns_to_drop"] = [d for d in p.get("columns_to_drop", []) if d.get("column") != column]
    return p


def apply_skeptic_gate(df, target, *, cleaning_plan, feature_plan, reviews, model_dir, time_limit,
                       n_folds, seed, eval_metric=None, sigma_mult=2.0,
                       group_column=None, time_column=None, period_column=None, budget=None):
    """Put each high-risk drop to the arbiter and veto it if keeping the column helps.

    Returns ``(revised_plan, records)``. The revised plan keeps any column whose retention
    improved the CV beyond fold noise; every other drop stands. Each keep-vs-drop trial is
    one counterfactual through the shared intervention core; with a ``budget`` (a
    :class:`~maestra.intervention.CVBudget`) exhausted trials are skipped and recorded as
    unmeasured — the drop then stands, the conservative default.
    """
    high = [r for r in reviews if r.get("risk") == "high" and r.get("column")]
    records = [SkepticRecord(r["column"], r["risk"], r.get("reason", "")) for r in reviews
               if r.get("risk") != "high"]
    if not high:
        return cleaning_plan, records

    cv_kwargs = dict(feature_plan=feature_plan, time_limit=time_limit, n_folds=n_folds, seed=seed,
                     eval_metric=eval_metric, group_column=group_column, time_column=time_column,
                     period_column=period_column)
    base = cross_validate(df, target, cleaning_plan=cleaning_plan,
                          model_dir=f"{model_dir}/base", **cv_kwargs)
    revised = cleaning_plan
    for i, r in enumerate(high):
        col = r["column"]
        trial_plan = _plan_without_drop(revised, col)
        outcome, base = run_counterfactual(
            f"keep:{col}", "skeptic_keep", "skeptic", base=base, budget=budget,
            trial_fn=lambda p=trial_plan, i=i: cross_validate(
                df, target, cleaning_plan=p, model_dir=f"{model_dir}/keep_{i}", **cv_kwargs),
            sigma_mult=sigma_mult, min_abs=_MIN_ABS_DELTA)
        records.append(SkepticRecord(col, "high", r.get("reason", ""), outcome.cv_delta,
                                     outcome.accepted, measured=outcome.reason != "budget_exhausted"))
        if outcome.accepted:  # keeping the column helped -> overturn the drop (bar already raised)
            revised = trial_plan
    return revised, records
