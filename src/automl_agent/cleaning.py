"""The cleaning step: the LLM decides, deterministic code executes.

The LLM receives the column profile and returns a plan drawn from a *fixed vocabulary*
(drop columns, impute missing values). We then apply that plan with plain pandas. No
LLM-generated code is ever executed — every mutation is auditable and the blast radius
is bounded by the schema. The applier is defensive: it protects the target, tolerates
hallucinated columns, and never crashes on a malformed plan.
"""
from __future__ import annotations

import json

import pandas as pd

from automl_agent.llm import call_structured

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
    "Du bist ein erfahrener Data Scientist und planst die Datenbereinigung fuer ein "
    "AutoML-Training (AutoGluon). Du bekommst ein Spalten-Profil als JSON. Entscheide, "
    "welche Spalten gedroppt werden sollen (z.B. ID-artige / hochkardinale Freitexte / "
    "Leakage) und wie fehlende Werte imputiert werden. Halte dich kurz und konservativ: "
    "droppe nur, was klar nutzlos oder schaedlich ist. Imputiere nur Spalten mit "
    "fehlenden Werten, die du behaeltst. Die Zielspalte NIE droppen oder imputieren. "
    "AutoGluon uebernimmt Encoding/Skalierung selbst -- plane das nicht."
)


def propose_cleaning_plan(model: str, profile: dict, target: str) -> dict:
    """Ask the LLM for a structured cleaning plan for the given column profile.

    Args:
        model: LiteLLM model string.
        profile: Output of :func:`automl_agent.profiling.profile_dataframe`.
        target: Target column name (passed to the model so it leaves it alone).

    Returns:
        A plan dict matching :data:`PLAN_SCHEMA`.
    """
    user_prompt = (
        f"Zielspalte: {target}\n"
        f"Spalten-Profil (JSON):\n{json.dumps(profile, ensure_ascii=False, indent=2)}"
    )
    return call_structured(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_name="cleaning_plan",
        tool_description="Strukturierter Cleaning-Plan aus festem Vokabular.",
        parameters_schema=PLAN_SCHEMA,
    )


def _impute_value(series: pd.Series, strategy: str, fill_value: str | None):
    """Compute the fill value for a column under the given strategy."""
    if strategy == "median":
        return series.median()
    if strategy == "mean":
        return series.mean()
    if strategy == "most_frequent":
        mode = series.mode(dropna=True)
        return mode.iloc[0] if not mode.empty else None
    return fill_value  # "constant"


def apply_cleaning_plan(
    df: pd.DataFrame, plan: dict, target: str
) -> tuple[pd.DataFrame, list[str]]:
    """Apply a cleaning plan deterministically and return ``(clean_df, log)``.

    The log mirrors every decision (applied or skipped) so the run stays auditable.
    The function is total: it never raises on a malformed or hallucinated plan — it
    skips the offending operation and records why.

    Args:
        df: The dataset to clean (not mutated; a copy is returned).
        plan: A plan dict matching :data:`PLAN_SCHEMA`.
        target: Target column, protected from drop and imputation.

    Returns:
        The cleaned DataFrame and a human-readable list of applied/skipped operations.
    """
    df = df.copy()
    log: list[str] = []

    for item in plan.get("columns_to_drop", []):
        col = item.get("column")
        if col == target:
            log.append(f"SKIP drop '{col}': ist Zielspalte")
        elif col not in df.columns:
            log.append(f"SKIP drop '{col}': Spalte existiert nicht")
        else:
            df = df.drop(columns=[col])
            log.append(f"DROP '{col}' -- {item.get('reason', '')}")

    for item in plan.get("imputations", []):
        col = item.get("column")
        strategy = item.get("strategy")
        if col == target:
            log.append(f"SKIP impute '{col}': ist Zielspalte")
            continue
        if col not in df.columns:
            log.append(f"SKIP impute '{col}': Spalte existiert nicht (evtl. gedroppt)")
            continue
        if strategy not in STRATEGIES:
            log.append(f"SKIP impute '{col}': unbekannte Strategie '{strategy}'")
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            log.append(f"SKIP impute '{col}': keine fehlenden Werte")
            continue
        fill = _impute_value(df[col], strategy, item.get("fill_value"))
        df[col] = df[col].fillna(fill)
        log.append(
            f"IMPUTE '{col}' [{strategy}] {n_missing} Werte -> {fill!r} "
            f"-- {item.get('reason', '')}"
        )

    return df, log
