"""Offline tests for the HTML dossier renderer. render_dossier is PURE (no LLM, no AutoGluon):
every case builds a fixed synthetic PipelineResult and asserts on the rendered HTML."""
import pandas as pd

from maestra.dossier import collect_interventions, render_dossier, write_dossier
from maestra.pipeline import PipelineResult
from maestra.validation import CVResult
from maestra.engine import TrainingResult


def _training(metrics=None):
    return TrainingResult("regression", "root_mean_squared_error",
                          pd.DataFrame({"model": ["m"]}), metrics or {})


def _cv(mean=310.5, metric="root_mean_squared_error", gib=False):
    return CVResult(metric, "regression", [305.0, 312.0, 314.0], mean, 4.0, 3, False,
                    greater_is_better=gib)


def _result(**over):
    base = dict(
        n_cols_before=12, n_cols_after=9, plan={"columns_to_drop": []},
        training=_training(), cv=_cv(),
        fold_strategy={"strategy": "time", "time_column": "Date", "group_column": None,
                       "period_column": None, "rationale": "the task forecasts future sales"},
        hybrid=[{"name": "age_of_house", "idea": "YrSold - YearBuilt", "source": "profile",
                 "cv_delta": 0.0, "kept": False, "reason": "no_improvement"},
                {"name": "total_area", "idea": "sum areas", "source": "profile",
                 "cv_delta": 640.0, "kept": True, "reason": "improved"}],
        target_framing={"transform": "log1p", "accepted": True, "cv_delta": 2273.0, "log": []},
        cv_budget={"limit": None, "trials_spent": 3},
        adversarial_auc=0.52,
    )
    base.update(over)
    return PipelineResult(**base)


def test_render_is_a_standalone_html_doc_with_a_traffic_light():
    html = render_dossier(_result())
    assert html.startswith("<!DOCTYPE html>") and "</html>" in html
    assert "<style>" in html and "http" not in html.split("<footer>")[0]  # inline CSS, no ext assets
    assert "GREEN" in html                                                # clean CV + no shift


def test_verdict_light_is_red_on_a_strong_train_test_shift():
    html = render_dossier(_result(adversarial_auc=0.9))
    assert "RED" in html and "will not hold" in html


def test_rejected_intervention_is_listed_on_equal_footing():
    html = render_dossier(_result())
    # the rejected generated feature appears with its delta, reason, and a 'dropped' marker
    assert "age_of_house" in html and "no_improvement" in html and "✗ dropped" in html
    # the accepted one is there too
    assert "total_area" in html and "✓ kept" in html


def test_no_raw_metric_without_a_translation_note():
    # default (no LLM): a deterministic direction note accompanies the metric
    html = render_dossier(_result())
    assert "root_mean_squared_error" in html and "lower is better" in html
    # LLM-provided note overrides the fallback
    html2 = render_dossier(_result(), metric_notes={
        "root_mean_squared_error": "on average the model misses the sale price by ~$310"})
    assert "misses the sale price by ~$310" in html2


def test_llm_sentence_overrides_the_default_not_the_colour():
    html = render_dossier(_result(), verdict_sentence="You can trust this number.")
    assert "You can trust this number." in html
    assert "GREEN" in html                       # colour stays deterministic (LLM never decides it)


def test_mde_from_a_run_record_is_surfaced():
    html = render_dossier(_result(), run_record={"mde": 1243.0})
    assert "Minimum detectable effect" in html and "1243" in html


def test_collect_interventions_normalises_the_three_kinds():
    rows = collect_interventions(_result(
        skeptic=[{"column": "flux", "risk": "high", "reason": "improved",
                  "cv_delta": 0.02, "vetoed": True, "measured": True}]))
    kinds = {r["kind"] for r in rows}
    assert kinds == {"generated_feature", "skeptic_keep", "target_framing"}
    keep = next(r for r in rows if r["kind"] == "skeptic_keep")
    assert keep["name"] == "keep:flux" and keep["accepted"] is True


def test_failed_run_no_training_is_red():
    html = render_dossier(_result(training=None, cv=None))
    assert "RED" in html and "did not produce a model" in html


def test_write_dossier_writes_the_file(tmp_path):
    p = tmp_path / "dossier.html"
    write_dossier(_result(), str(p))
    assert p.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def _audit(**over):
    from maestra.audit import AuditReport
    base = dict(csv="x.csv", n_rows=1000, n_cols=8, target="y",
                fold_strategy={"strategy": "random"}, fold_log=[], leakage_warnings=[])
    base.update(over)
    return AuditReport(**base)


def test_render_audit_high_risk_leak_is_red():
    from maestra.dossier import render_audit
    html = render_audit(_audit(target_leaks=[("snapshot", 0.98)]))
    assert "RED" in html and "near-copy of the target" in html and "snapshot" in html


def test_render_audit_time_local_names_the_period_column():
    """Found via the real Grunfeld example (2026-07-06): time_local was rendered with the same
    sentence as plain time, silently dropping the repeating-period nuance the strategy exists
    for. time_local now names BOTH the time and the period column."""
    from maestra.dossier import render_audit
    html = render_audit(_audit(fold_strategy={"strategy": "time_local", "time_column": "year",
                                              "period_column": "firm",
                                              "rationale": "repeats per firm"}))
    assert "YELLOW" in html and "year" in html and "firm" in html
    assert "repeating" in html.lower() and "within each" in html.lower()


def test_render_audit_group_folds_verdict_is_elevated_yellow():
    from maestra.dossier import render_audit
    html = render_audit(_audit(fold_strategy={"strategy": "group", "group_column": "customer_id",
                                              "rationale": "rows repeat per customer"}))
    assert "YELLOW" in html and "group the folds" in html and "customer_id" in html
    assert "rows repeat per customer" in html            # strategist reasoning surfaced


def test_render_audit_clean_is_green():
    from maestra.dossier import render_audit
    html = render_audit(_audit(adversarial_auc=0.51))
    assert "GREEN" in html and "standard validation applies" in html


def test_render_audit_uses_the_shared_html_shell():
    from maestra.dossier import render_audit
    html = render_audit(_audit())
    assert html.startswith("<!DOCTYPE html>") and "<summary>Leakage risks</summary>" in html


def test_dossier_narrative_calls_the_llm_and_feeds_the_renderer(monkeypatch):
    """The LLM produces ONLY the prose; the parsed dict drops straight into render_dossier."""
    from maestra import dossier

    captured = {}

    def fake_call_structured(*, model, system_prompt, user_prompt, **kwargs):
        captured["facts"] = user_prompt
        return {"verdict_sentence": "The estimate is trustworthy for this dataset.",
                "metric_notes": {"root_mean_squared_error": "typical miss ~$310 in sale price"}}

    monkeypatch.setattr("maestra.llm.call_structured", fake_call_structured)
    narrative = dossier.dossier_narrative("gpt-4o", _result())
    assert narrative["verdict_sentence"].startswith("The estimate is trustworthy")
    assert "green" in captured["facts"]                         # the deterministic colour is a fact
    html = render_dossier(_result(), **narrative)
    assert "typical miss ~$310 in sale price" in html
