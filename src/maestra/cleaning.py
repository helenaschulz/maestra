"""The cleaning step: the LLM decides, deterministic code executes.

The LLM receives the column profile and returns a plan drawn from a *fixed vocabulary*
(drop columns, impute missing values). We then apply that plan with plain pandas. No
LLM-generated code is ever executed — every mutation is auditable and the blast radius
is bounded by the schema. The applier is defensive: it protects the target, tolerates
hallucinated columns, and never crashes on a malformed plan.

The plan is applied with a **fit/transform split** (like scikit-learn) to avoid data
leakage: imputation values are *fitted on the training rows only* and then applied
unchanged to both train and holdout. Computing a median/mode over the full dataset —
including the holdout — would leak test information into the features and inflate the
reported metrics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from maestra.llm import call_structured

#: Imputation strategies the LLM may choose from. Mirrored in ``PLAN_SCHEMA``.
STRATEGIES = ["median", "mean", "most_frequent", "constant"]

#: JSON schema for the function-calling tool. Closed and minimal on purpose.
PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "columns_to_drop": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["column", "reason"],
            },
        },
        "imputations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "strategy": {"type": "string", "enum": STRATEGIES},
                    "fill_value": {"type": ["string", "null"]},
                    "reason": {"type": "string"},
                },
                "required": ["column", "strategy", "reason"],
            },
        },
        "overall_rationale": {"type": "string"},
    },
    "required": ["columns_to_drop", "imputations", "overall_rationale"],
}

_SYSTEM_PROMPT = (
    "You are an experienced data scientist planning the data cleaning for an AutoML "
    "training run (AutoGluon). You are given a column profile as JSON. Decide which columns "
    "to drop (e.g. ID-like / high-cardinality free text / leakage) and how missing values "
    "are imputed. IMPORTANT: high cardinality (many unique values) is completely NORMAL for "
    "CONTINUOUS numeric columns (float, e.g. measurements, coordinates, brightnesses) and NO "
    "reason to drop -- such columns are often the most important features. 'Unique per row' "
    "justifies a drop ONLY for ID-like columns (running integer IDs) or high-cardinality "
    "free text, NEVER for numeric measurements. In the profile the field 'id_like'=true "
    "marks genuine ID/index candidates (computed deterministically) -- use that as guidance. "
    "Be brief and conservative: drop only what is clearly useless or harmful. Impute only "
    "columns with missing values that you keep. NEVER drop or impute the target column. "
    "AutoGluon handles encoding/scaling itself -- do not plan that."
)


def propose_cleaning_plan(
    model: str, profile: dict, target: str, research_context: str | None = None
) -> dict:
    """Ask the LLM for a structured cleaning plan for the given column profile.

    Args:
        model: LiteLLM model string.
        profile: Output of :func:`maestra.profiling.profile_dataframe`.
        target: Target column name (passed to the model so it leaves it alone).
        research_context: Optional non-binding strategy hypotheses from the research node.

    Returns:
        A plan dict matching :data:`PLAN_SCHEMA`.
    """
    user_prompt = (
        f"Target column: {target}\n"
        f"Column profile (JSON):\n{json.dumps(profile, ensure_ascii=False, indent=2)}"
    )
    if research_context:
        user_prompt += "\n\n" + research_context
    return call_structured(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_name="cleaning_plan",
        tool_description="Strukturierter Cleaning-Plan aus festem Vokabular.",
        parameters_schema=PLAN_SCHEMA,
    )


def _fit_fill_value(series: pd.Series, strategy: str, fill_value: str | None) -> Any:
    """Compute the fill value for a column under the given strategy.

    Raises:
        TypeError: If the strategy is numeric (median/mean) but the column is not.
    """
    if strategy == "median":
        return series.median()
    if strategy == "mean":
        return series.mean()
    if strategy == "most_frequent":
        mode = series.mode(dropna=True)
        return mode.iloc[0] if not mode.empty else None
    return fill_value  # "constant"


@dataclass
class CleaningTransform:
    """A fitted cleaning step: which columns to drop and what to fill missing values with.

    Fitted on the training data only, then applied to any DataFrame via
    :meth:`transform`. This is what keeps imputation leakage-free — the fill values are
    frozen statistics of the train split, not recomputed per DataFrame.
    """

    drops: list[str]
    fills: dict[str, Any]  # column -> fill value, fitted on train
    log: list[str] = field(default_factory=list)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted drops and fills to ``df`` (returns a copy)."""
        df = df.drop(columns=[c for c in self.drops if c in df.columns])
        df = df.copy()
        for col, value in self.fills.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)
        return df


def fit_cleaning_plan(train: pd.DataFrame, plan: dict, target: str) -> CleaningTransform:
    """Turn an LLM plan into a fitted :class:`CleaningTransform` using train data only.

    The returned log mirrors every decision (applied or skipped) so the run stays
    auditable. This function is total: it never raises on a malformed or hallucinated
    plan — it skips the offending operation and records why. Imputation values are
    computed from ``train`` exclusively, so applying the transform to a holdout set
    cannot leak test statistics.

    Args:
        train: The training rows the imputation statistics are fitted on.
        plan: A plan dict matching :data:`PLAN_SCHEMA`.
        target: Target column, protected from both drop and imputation.

    Returns:
        A fitted :class:`CleaningTransform`.
    """
    log: list[str] = []
    drops: list[str] = []
    for item in plan.get("columns_to_drop", []):
        col = item.get("column")
        if col == target:
            log.append(f"SKIP drop '{col}': is the target column")
        elif col not in train.columns:
            log.append(f"SKIP drop '{col}': column does not exist")
        else:
            drops.append(col)
            log.append(f"DROP '{col}' -- {item.get('reason', '')}")

    fills: dict[str, Any] = {}
    for item in plan.get("imputations", []):
        col = item.get("column")
        strategy = item.get("strategy")
        if col == target:
            log.append(f"SKIP impute '{col}': is the target column")
            continue
        if col in drops or col not in train.columns:
            log.append(f"SKIP impute '{col}': column not present (possibly dropped)")
            continue
        if strategy not in STRATEGIES:
            log.append(f"SKIP impute '{col}': unknown strategy '{strategy}'")
            continue
        try:
            fill = _fit_fill_value(train[col], strategy, item.get("fill_value"))
        except TypeError:
            log.append(f"SKIP impute '{col}': strategy '{strategy}' does not match the dtype")
            continue
        fills[col] = fill
        n_missing = int(train[col].isna().sum())
        log.append(
            f"IMPUTE '{col}' [{strategy}] fit on train (missing={n_missing}) -> {fill!r} "
            f"-- {item.get('reason', '')}"
        )

    return CleaningTransform(drops=drops, fills=fills, log=log)
