"""Cleaning-Schritt: LLM entscheidet, Code rechnet.

Das LLM bekommt das Spalten-Profil und gibt einen strukturierten Plan aus einem
FESTEN Vokabular zurueck (Spalten droppen, fehlende Werte imputieren). Der Plan wird
deterministisch angewendet -- kein vom LLM generierter Code wird ausgefuehrt.
"""
from __future__ import annotations

import json

import pandas as pd

from llm import call_structured

_STRATEGIES = ["median", "mean", "most_frequent", "constant"]

# JSON-Schema fuer das Function-Calling. Bewusst minimal und geschlossen.
PLAN_SCHEMA = {
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
                    "strategy": {"type": "string", "enum": _STRATEGIES},
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

_SYSTEM = (
    "Du bist ein erfahrener Data Scientist und planst die Datenbereinigung fuer ein "
    "AutoML-Training (AutoGluon). Du bekommst ein Spalten-Profil als JSON. Entscheide, "
    "welche Spalten gedroppt werden sollen (z.B. ID-artige / hochkardinale Freitexte / "
    "Leakage) und wie fehlende Werte imputiert werden. Halte dich kurz und konservativ: "
    "droppe nur, was klar nutzlos oder schaedlich ist. Imputiere nur Spalten mit "
    "fehlenden Werten, die du behaeltst. Die Zielspalte NIE droppen oder imputieren. "
    "AutoGluon uebernimmt Encoding/Skalierung selbst -- plane das nicht."
)


def propose_cleaning_plan(model: str, profile: dict, target: str) -> dict:
    user = (
        f"Zielspalte: {target}\n"
        f"Spalten-Profil (JSON):\n{json.dumps(profile, ensure_ascii=False, indent=2)}"
    )
    return call_structured(
        model=model,
        system_prompt=_SYSTEM,
        user_prompt=user,
        tool_name="cleaning_plan",
        tool_description="Strukturierter Cleaning-Plan aus festem Vokabular.",
        parameters_schema=PLAN_SCHEMA,
    )


def apply_cleaning_plan(df: pd.DataFrame, plan: dict, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Wendet den Plan deterministisch an. Toleriert halluzinierte Spalten (loggt sie),
    schuetzt die Zielspalte, crasht nicht am LLM-Output."""
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
        strat = item.get("strategy")
        if col == target:
            log.append(f"SKIP impute '{col}': ist Zielspalte")
            continue
        if col not in df.columns:
            log.append(f"SKIP impute '{col}': Spalte existiert nicht (evtl. gedroppt)")
            continue
        if strat not in _STRATEGIES:
            log.append(f"SKIP impute '{col}': unbekannte Strategie '{strat}'")
            continue
        before = int(df[col].isna().sum())
        if before == 0:
            log.append(f"SKIP impute '{col}': keine fehlenden Werte")
            continue
        if strat == "median":
            fill = df[col].median()
        elif strat == "mean":
            fill = df[col].mean()
        elif strat == "most_frequent":
            mode = df[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else None
        else:  # constant
            fill = item.get("fill_value")
        df[col] = df[col].fillna(fill)
        log.append(f"IMPUTE '{col}' [{strat}] {before} Werte -> {fill!r} -- {item.get('reason', '')}")

    return df, log
