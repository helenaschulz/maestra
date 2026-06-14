"""Failure diagnosis — the agentic feedback step.

When an attempt to clean+train fails, the LLM reads the (truncated) traceback plus the
run context and chooses a *recovery action* from a fixed vocabulary: revise the cleaning
plan, raise the training time budget, or give up. As everywhere in this project the LLM
decides via structured JSON; it never returns code to execute and the retry loop that
consumes its decision is bounded (see ``pipeline.run_pipeline``).
"""
from __future__ import annotations

import json

from maestra.cleaning import PLAN_SCHEMA
from maestra.llm import call_structured

#: Recovery actions the LLM may choose. Mirrored in ``DIAGNOSIS_SCHEMA``.
ACTIONS = ["revise_plan", "increase_time_limit", "give_up"]

#: JSON schema for the diagnosis tool. ``new_plan``/``new_time_limit`` are only consulted
#: for the matching action; both are optional so the model needn't fill the unused one.
DIAGNOSIS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "diagnosis": {
            "type": "string",
            "description": "Plain-language root cause of the failure.",
        },
        "action": {
            "type": "string",
            "enum": ACTIONS,
            "description": (
                "revise_plan: a different cleaning plan would fix it (e.g. you dropped "
                "all feature columns). increase_time_limit: it was a time/resource "
                "shortfall. give_up: unrecoverable by cleaning or time."
            ),
        },
        "new_plan": {
            **PLAN_SCHEMA,
            "description": "Replacement cleaning plan; required when action is revise_plan.",
        },
        "new_time_limit": {
            "type": ["integer", "null"],
            "description": "New training budget in seconds; required for increase_time_limit.",
        },
    },
    "required": ["diagnosis", "action"],
}

_SYSTEM_PROMPT = (
    "Du debuggst eine fehlgeschlagene AutoML-Pipeline (Cleaning -> AutoGluon-Training). "
    "Du bekommst die Fehlermeldung, das Spalten-Profil (nur train), den aktuellen "
    "Cleaning-Plan und das Zeitbudget. Bestimme die Ursache und waehle GENAU EINE "
    "Recovery-Aktion. Sei konservativ: revise_plan nur, wenn eine Cleaning-Aenderung den "
    "Fehler wirklich behebt (z.B. es wurden alle Feature-Spalten gedroppt -> droppe "
    "weniger). increase_time_limit bei Zeit-/Ressourcenmangel. give_up, wenn weder "
    "Cleaning noch mehr Zeit hilft. Ein revidierter Plan muss demselben Schema folgen "
    "und die Zielspalte unangetastet lassen."
)


def diagnose_failure(
    model: str,
    error_text: str,
    *,
    profile: dict,
    plan: dict | None,
    time_limit: int,
    target: str,
) -> dict:
    """Ask the LLM to diagnose a pipeline failure and pick a recovery action.

    Args:
        model: LiteLLM model string.
        error_text: Truncated traceback / error message of the failed attempt.
        profile: Train-only column profile for context.
        plan: The cleaning plan that was in effect when the failure occurred (or None).
        time_limit: The training budget that was used.
        target: Target column name.

    Returns:
        A dict matching :data:`DIAGNOSIS_SCHEMA` (``diagnosis``, ``action`` and, depending
        on the action, ``new_plan`` or ``new_time_limit``).
    """
    context = {
        "target": target,
        "time_limit": time_limit,
        "current_plan": plan,
        "profile": profile,
        "error": error_text,
    }
    user_prompt = (
        "Die Pipeline ist fehlgeschlagen. Kontext (JSON):\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}"
    )
    return call_structured(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_name="diagnose_failure",
        tool_description="Diagnose und Recovery-Aktion fuer eine fehlgeschlagene Pipeline.",
        parameters_schema=DIAGNOSIS_SCHEMA,
    )
