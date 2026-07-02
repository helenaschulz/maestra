"""Tests for the Skeptic gate. The LLM review is a dict and cross_validate is mocked, so the
arbiter logic (veto only when keeping the column measurably helps) is exercised deterministically."""
import pandas as pd

from maestra import skeptic as sk
from maestra.skeptic import _plan_without_drop, apply_skeptic_gate
from maestra.validation import CVResult


def _plan(*cols):
    return {"columns_to_drop": [{"column": c, "reason": "x"} for c in cols], "imputations": []}


def _cv(mean, std=0.02):
    return CVResult("accuracy", "binary", [], mean, std, 3, True, True)


def test_plan_without_drop_keeps_other_drops():
    p = _plan_without_drop(_plan("a", "b", "c"), "b")
    assert [d["column"] for d in p["columns_to_drop"]] == ["a", "c"]


def test_high_risk_drop_vetoed_when_keeping_helps(monkeypatch):
    # base CV 0.80; keeping 'flux' -> 0.86 (>+1 sigma of 0.02) -> veto the drop
    scores = iter([_cv(0.80), _cv(0.86)])
    monkeypatch.setattr(sk, "cross_validate", lambda *a, **k: next(scores))
    reviews = [{"column": "flux", "risk": "high", "reason": "continuous measurement"}]

    revised, records = apply_skeptic_gate(
        pd.DataFrame({"y": [0, 1]}), "y", cleaning_plan=_plan("flux", "id"), feature_plan=None,
        reviews=reviews, model_dir="x", time_limit=1, n_folds=3, seed=0)

    assert [d["column"] for d in revised["columns_to_drop"]] == ["id"]  # flux drop overturned
    rec = next(r for r in records if r.column == "flux")
    assert rec.vetoed and rec.cv_delta > 0


def test_high_risk_drop_upheld_when_keeping_does_not_help(monkeypatch):
    # keeping 'flux' -> 0.805, within fold noise -> drop stands
    scores = iter([_cv(0.80), _cv(0.805)])
    monkeypatch.setattr(sk, "cross_validate", lambda *a, **k: next(scores))
    reviews = [{"column": "flux", "risk": "high", "reason": "maybe signal"}]

    revised, records = apply_skeptic_gate(
        pd.DataFrame({"y": [0, 1]}), "y", cleaning_plan=_plan("flux"), feature_plan=None,
        reviews=reviews, model_dir="x", time_limit=1, n_folds=3, seed=0)

    assert [d["column"] for d in revised["columns_to_drop"]] == ["flux"]  # drop upheld
    assert not next(r for r in records if r.column == "flux").vetoed


def test_low_risk_drops_are_recorded_but_never_measured(monkeypatch):
    calls = []
    monkeypatch.setattr(sk, "cross_validate", lambda *a, **k: calls.append(1) or _cv(0.8))
    reviews = [{"column": "id", "risk": "low", "reason": "running integer id"}]

    revised, records = apply_skeptic_gate(
        pd.DataFrame({"y": [0, 1]}), "y", cleaning_plan=_plan("id"), feature_plan=None,
        reviews=reviews, model_dir="x", time_limit=1, n_folds=3, seed=0)

    assert revised == _plan("id")            # unchanged
    assert calls == []                       # no CV ran (no high-risk flags)
    assert records[0].column == "id" and records[0].cv_delta is None


def test_pipeline_wires_skeptic_and_reports(monkeypatch):
    from maestra import pipeline
    from maestra.engine import TrainingResult

    df = pd.DataFrame({"flux": [1.0, 2.0, 3.0, 4.0] * 3, "id": range(12), "y": [0, 1] * 6})
    monkeypatch.setattr(pipeline, "propose_cleaning_plan",
                        lambda *a, **k: {"columns_to_drop": [{"column": "flux", "reason": "unique"}],
                                         "imputations": []})
    monkeypatch.setattr(pipeline, "review_cleaning_plan",
                        lambda *a, **k: [{"column": "flux", "risk": "high", "reason": "measurement"}])
    # base 0.70, keep-flux 0.90 -> veto
    scores = iter([_cv(0.70), _cv(0.90)])
    monkeypatch.setattr(pipeline, "cross_validate", lambda *a, **k: _cv(0.9))   # final reported CV
    monkeypatch.setattr("maestra.skeptic.cross_validate", lambda *a, **k: next(scores))
    monkeypatch.setattr(pipeline, "fit_predictor",
                        lambda *a, **k: TrainingResult("binary", "accuracy", pd.DataFrame(), {}))

    result = pipeline.run_pipeline(df, "y", model="m", test_size=0.2, time_limit=1, seed=0,
                                   model_dir="x", cv_folds=3, skeptic=True, use_fe=False)

    vetoed = [r for r in result.skeptic if r["vetoed"]]
    assert vetoed and vetoed[0]["column"] == "flux"           # the drop was overturned by measurement
    assert result.plan["columns_to_drop"] == []              # flux kept -> no drops remain


def test_skeptic_requires_cv():
    import pytest

    from maestra.pipeline import run_pipeline
    with pytest.raises(ValueError, match="requires --cv"):
        run_pipeline(pd.DataFrame({"a": [1, 2], "y": [0, 1]}), "y", model="m", test_size=0.2,
                     time_limit=1, seed=0, model_dir="x", skeptic=True)
