"""Tests for the report node. The numbers are checked against TrainingResult; the LLM is
mocked, and we assert the real values are handed to it as facts (so it need invent none)."""
import pandas as pd

from maestra import report
from maestra.engine import TrainingResult
from maestra.pipeline import PipelineResult
from maestra.report import build_report_facts, generate_report


def _result():
    training = TrainingResult(
        "binary", "accuracy",
        pd.DataFrame({"model": ["WeightedEnsemble_L2", "XGBoost"], "score_test": [0.83, 0.82]}),
        {"accuracy": 0.826, "roc_auc": 0.884}, val_score=0.85,
    )
    return PipelineResult(n_cols_before=12, n_cols_after=8, plan={"columns_to_drop": []},
                          n_cols_clean=8, training=training, attempts=1,
                          feature_plan={"features": []})


def test_facts_match_training_result():
    f = build_report_facts(_result())
    assert f["problem_type"] == "binary"
    assert f["eval_metric"] == "accuracy"
    assert f["holdout_metrics"] == {"accuracy": 0.826, "roc_auc": 0.884}
    assert f["internal_val_score"] == 0.85
    assert f["leaderboard_top"][0] == {"model": "WeightedEnsemble_L2", "score_test": 0.83}
    assert f["columns"] == {"before": 12, "after_cleaning": 8, "after_feature_engineering": 8}


def test_generate_report_delegates_and_passes_real_numbers(monkeypatch):
    captured = {}

    def fake_call(**kwargs):
        captured.update(kwargs)
        return {"report_markdown": "# Run report\nLooks good."}

    monkeypatch.setattr(report, "call_structured", fake_call)
    md = generate_report("gpt-4o", _result())

    assert md == "# Run report\nLooks good."
    assert captured["tool_name"] == "write_report"
    assert captured["parameters_schema"] is report.REPORT_SCHEMA
    # The real metrics are in the facts handed to the LLM — it never has to invent them.
    assert "0.826" in captured["user_prompt"]
    assert "0.884" in captured["user_prompt"]
