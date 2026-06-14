"""Report node: the LLM explains a finished run — it does not compute.

All numbers (metrics, leaderboard, column counts) are extracted deterministically from the
:class:`~maestra.pipeline.PipelineResult` and passed to the LLM as facts. The model writes
a Markdown narrative around those facts; it is instructed to reuse the provided numbers
verbatim and invent nothing. Like every other LLM step, output goes through the structured
tool interface (a single ``report_markdown`` field).
"""
from __future__ import annotations

import json

from maestra.llm import call_structured

REPORT_SCHEMA: dict = {
    "type": "object",
    "properties": {"report_markdown": {"type": "string"}},
    "required": ["report_markdown"],
}

_SYSTEM_PROMPT = (
    "Du bist Data Scientist und schreibst einen knappen, klaren Markdown-Report ueber "
    "einen AutoML-Lauf. Du bekommst die FAKTEN als JSON. WICHTIG: Verwende ausschliesslich "
    "die uebergebenen Zahlen WOERTLICH; erfinde KEINE Werte und rechne NICHTS aus. Erklaere "
    "und fasse zusammen: was bereinigt und an Features erzeugt wurde, das Ergebnis (Metriken, "
    "bestes Modell), und etwaige Revisionen. Gliedere mit Markdown-Ueberschriften. Sei "
    "sachlich und kurz."
)


def build_report_facts(result) -> dict:
    """Extract the ground-truth facts of a run from its :class:`PipelineResult`."""
    t = result.training
    leaderboard_top = []
    if t is not None and "score_test" in t.leaderboard.columns:
        cols = [c for c in ("model", "score_test") if c in t.leaderboard.columns]
        leaderboard_top = t.leaderboard.head(5)[cols].to_dict("records")
    cv = result.cv
    return {
        "problem_type": t.problem_type if t else None,
        "eval_metric": t.eval_metric if t else None,
        "holdout_metrics": (t.metrics or None) if t else None,
        "cross_validation": (
            {"eval_metric": cv.eval_metric, "mean": cv.mean, "std": cv.std,
             "folds": cv.n_folds, "fold_scores": cv.fold_scores} if cv else None
        ),
        "adversarial_auc": result.adversarial_auc,
        "internal_val_score": t.val_score if t else None,
        "leaderboard_top": leaderboard_top,
        "columns": {
            "before": result.n_cols_before,
            "after_cleaning": result.n_cols_clean,
            "after_feature_engineering": result.n_cols_after,
        },
        "cleaning_plan": result.plan,
        "feature_plan": result.feature_plan,
        "attempts": result.attempts,
        "diagnosis_log": result.diagnosis_log,
    }


def generate_report(model: str, result) -> str:
    """Ask the LLM to write a Markdown report from the run's facts. Returns the Markdown."""
    facts = build_report_facts(result)
    out = call_structured(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=f"Fakten des Laufs (JSON):\n{json.dumps(facts, ensure_ascii=False, indent=2, default=float)}",
        tool_name="write_report",
        tool_description="Markdown-Report eines AutoML-Laufs aus den gegebenen Fakten.",
        parameters_schema=REPORT_SCHEMA,
    )
    return out["report_markdown"]
