"""Validation Strategist — the judgment agent for AutoML's biggest blind spot.

AutoGluon (like every AutoML engine) assumes rows are exchangeable and validates on random
splits. It *cannot know* that rows are grouped (several rows per patient/customer/device — a
random split then leaks entity information across folds and the CV lies optimistically) or
temporal (predicting the future from the past — a random split trains on the future). Choosing
the fold strategy requires understanding what the columns MEAN, which is exactly what an LLM
can judge from the profile plus a dataset description.

The agent proposes; deterministic code verifies. A proposal naming a column that does not
exist, a group column without repeats, or an unsortable time column falls back to random folds
with an explicit log line — the LLM cannot break the pipeline, only inform it.
"""
from __future__ import annotations

import pandas as pd

from maestra.llm import call_structured

FOLD_STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {
            "type": "string",
            "enum": ["random", "group", "time"],
            "description": "How validation folds must be built for an honest estimate.",
        },
        "group_column": {
            "type": ["string", "null"],
            "description": "For strategy=group: the entity column whose rows must never be "
                           "split across folds (e.g. patient_id, customer_id).",
        },
        "time_column": {
            "type": ["string", "null"],
            "description": "For strategy=time: the column ordering past before future.",
        },
        "rationale": {"type": "string", "description": "Why this strategy, in one or two sentences."},
        "leakage_warnings": {
            "type": "array",
            "description": "Columns that look like target leaks (post-outcome or proxy columns).",
            "items": {
                "type": "object",
                "properties": {"column": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["column", "reason"],
            },
        },
    },
    "required": ["strategy", "rationale"],
}

_SYSTEM_PROMPT = (
    "You are a senior data scientist deciding how VALIDATION FOLDS must be built for a tabular "
    "ML task. This is the one decision AutoML engines cannot make: they assume rows are "
    "exchangeable and split at random. Choose from the FIXED vocabulary:\n"
    "  random - rows are independent; the default when nothing below applies.\n"
    "  group  - an ENTITY appears in multiple rows (patient, customer, device, session ...). "
    "A random split would put the same entity in train and validation and the CV would lie "
    "optimistically. Name the entity column as group_column.\n"
    "  time   - the task is to predict the FUTURE from the past (a timestamp/date/period "
    "column orders the rows and the target evolves over it). Name it as time_column.\n"
    "Evidence for group: an id-like column with n_unique clearly BELOW n_rows (repeats!), or "
    "the description says several rows belong to one entity. CAUTION: a categorical with only a "
    "FEW balanced levels (control/treatment arms, A/B variants, product categories) is a design "
    "or feature column, NOT an entity — entities are typically numerous (many patients, many "
    "firms). Evidence for time: a date/period column plus a forecasting-flavoured task; this "
    "includes a NUMERIC time axis (e.g. decimal years, monotonically increasing) — a series "
    "whose only ordering column is numeric 'time'/'year' is still temporal. Be conservative: "
    "when in doubt, random — but missing a real group/time structure is the costlier error, so "
    "weigh repeats seriously. "
    "Only name columns that exist in the profile. Additionally flag columns that look like "
    "TARGET LEAKS (recorded after the outcome, or a near-proxy of the target) under "
    "leakage_warnings — flagging is advice, not an instruction to drop."
)


def propose_fold_strategy(model: str, profile: dict, target: str, context: str | None = None) -> dict:
    """Ask the LLM how folds must be built. Returns a dict matching FOLD_STRATEGY_SCHEMA."""
    import json

    user_prompt = (
        f"Target column: {target}\n"
        f"Column profile (JSON):\n{json.dumps(profile, ensure_ascii=False, indent=2)}"
    )
    if context:
        user_prompt += "\n\n" + context
    return call_structured(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_name="propose_fold_strategy",
        tool_description="Decide the validation fold strategy (random/group/time) for this task.",
        parameters_schema=FOLD_STRATEGY_SCHEMA,
    )


def validate_fold_strategy(proposal: dict, df: pd.DataFrame, target: str) -> tuple[dict, list[str]]:
    """Deterministically verify a proposal against the actual data; fall back to random on
    any defect. Returns ``(verified_strategy, log_lines)`` where verified_strategy carries
    ``strategy`` plus (when applicable) ``group_column`` / ``time_column``.
    """
    log: list[str] = []
    strategy = proposal.get("strategy", "random")
    rationale = proposal.get("rationale", "")
    warnings = proposal.get("leakage_warnings") or []
    verified = {"strategy": "random", "group_column": None, "time_column": None,
                "rationale": rationale, "leakage_warnings": warnings}

    def fallback(reason: str):
        log.append(f"FOLDS random (fallback): {reason}")
        return verified, log

    if strategy == "group":
        col = proposal.get("group_column")
        if not col or col not in df.columns:
            return fallback(f"proposed group column {col!r} does not exist")
        if col == target:
            return fallback("group column must not be the target")
        n_unique = df[col].nunique(dropna=True)
        if n_unique >= len(df):  # no repeats -> grouping is a no-op, and GroupKFold≈random
            return fallback(f"group column {col!r} has no repeated entities")
        if n_unique < 2:
            return fallback(f"group column {col!r} has fewer than 2 groups")
        verified.update(strategy="group", group_column=col)
        log.append(f"FOLDS group by {col!r} ({n_unique} entities) -- {rationale}")
        if n_unique < 5:  # small panels are legitimate, but a few balanced levels smell of a
            # treatment/design factor (the PlantGrowth trap) — warn, don't override the judgment
            log.append(f"NOTE: only {n_unique} entities — verify {col!r} is an entity, "
                       "not a treatment/design factor")
    elif strategy == "time":
        col = proposal.get("time_column")
        if not col or col not in df.columns:
            return fallback(f"proposed time column {col!r} does not exist")
        if col == target:
            return fallback("time column must not be the target")
        values = df[col]
        if values.dtype.kind not in "iufM":  # try parsing strings as dates
            values = pd.to_datetime(values, errors="coerce", format="mixed")
        if values.isna().mean() > 0.05:
            return fallback(f"time column {col!r} is not sortable (unparseable values)")
        verified.update(strategy="time", time_column=col)
        log.append(f"FOLDS time-ordered by {col!r} -- {rationale}")
    else:
        log.append(f"FOLDS random -- {rationale}" if rationale else "FOLDS random")

    for w in warnings:
        log.append(f"LEAKAGE WARNING {w.get('column')!r}: {w.get('reason')}")
    return verified, log
