"""Offline tests for the MCP tool backend. Needs the optional 'mcp' dep group (not installed by
CI's core `.[dev,research]`) purely to import FastMCP -- skip cleanly when it's absent, matching
the project's 'core install and its tests never need an MCP runtime' rule. LLM and AutoGluon calls
are mocked throughout, same as every other test in this suite."""
import time

import pandas as pd
import pytest

pytest.importorskip("mcp")

from maestra import mcp_server as srv  # noqa: E402
from maestra.audit import AuditReport  # noqa: E402
from maestra.validation import CVResult  # noqa: E402


def _df(n=100):
    return pd.DataFrame({
        "x": range(n),
        "y": [i % 2 for i in range(n)],
    })


def _audit_report(**over):
    base = dict(csv="x.csv", n_rows=100, n_cols=2, target="y",
                fold_strategy={"strategy": "random", "rationale": "no group/time structure"},
                fold_log=[], leakage_warnings=[])
    base.update(over)
    return AuditReport(**base)


# --- shared guards ---------------------------------------------------------------------

def test_missing_file_is_a_structured_rejection():
    result = srv.audit_csv("/no/such/file.csv", "y")
    assert result["verdict"] == "rejected" and "not found" in result["reason"]


def test_missing_target_is_a_structured_rejection(tmp_path):
    p = tmp_path / "data.csv"
    _df().to_csv(p, index=False)
    result = srv.audit_csv(str(p), "no_such_column")
    assert result["verdict"] == "rejected" and "no_such_column" in result["reason"]


def test_too_few_rows_is_a_structured_rejection(tmp_path):
    p = tmp_path / "tiny.csv"
    _df(10).to_csv(p, index=False)
    result = srv.audit_csv(str(p), "y")
    assert result["verdict"] == "rejected" and "too few" in result["reason"]


def test_with_budget_rejects_on_timeout():
    def slow(**kwargs):
        time.sleep(1)
        return {"verdict": "ok"}

    srv._BUDGETS["audit_csv"] = 0.01
    try:
        result = srv._with_budget(slow, tool="audit_csv")
    finally:
        srv._BUDGETS["audit_csv"] = 60.0
    assert result["verdict"] == "rejected" and "budget" in result["reason"]


# --- Tool 1: audit_csv ------------------------------------------------------------------

def test_audit_csv_happy_path(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    _df().to_csv(p, index=False)
    report = _audit_report(target_leaks=[("x", 0.99)])
    monkeypatch.setattr("maestra.audit.audit", lambda *a, **k: report)
    monkeypatch.setattr("maestra.audit.write_audit_html", lambda *a, **k: None)

    result = srv.audit_csv(str(p), "y")
    assert result["verdict"] == "ok"
    assert result["risk_level"] == "high"
    assert result["target_leaks"] == [("x", 0.99)]
    assert result["html_report"] == f"{p}.audit.html"


# --- Tool 2: check_validation ------------------------------------------------------------

def test_check_validation_reports_no_structure_without_running_cv(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    _df().to_csv(p, index=False)
    monkeypatch.setattr("maestra.profiling.profile_dataframe", lambda *a, **k: {})
    monkeypatch.setattr("maestra.validation_strategist.propose_fold_strategy",
                       lambda *a, **k: {"strategy": "random", "rationale": "no structure"})
    monkeypatch.setattr("maestra.validation_strategist.validate_fold_strategy",
                       lambda proposal, df, target: (proposal, []))

    called = []
    monkeypatch.setattr("maestra.validation.cross_validate",
                       lambda *a, **k: called.append(1))

    result = srv.check_validation(str(p), "y")
    assert result["verdict"] == "ok"
    assert result["recommended_strategy"] == {"strategy": "random", "column": None}
    assert not called                            # no CV run needed when there's no structure


def test_check_validation_measures_the_optimism_gap(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    _df().to_csv(p, index=False)
    monkeypatch.setattr("maestra.profiling.profile_dataframe", lambda *a, **k: {})
    monkeypatch.setattr("maestra.validation_strategist.propose_fold_strategy",
                       lambda *a, **k: {"strategy": "group", "group_column": "x"})
    monkeypatch.setattr(
        "maestra.validation_strategist.validate_fold_strategy",
        lambda proposal, df, target: (
            {"strategy": "group", "group_column": "x", "rationale": "rows repeat per x"}, []))
    monkeypatch.setattr("maestra.validation._is_classification", lambda y: True)

    def fake_cv(df, target, *, model_dir, **kwargs):
        # naive (random split, no group/time args) reports a HIGHER accuracy than the grouped one
        mean = 0.95 if "naive" in model_dir else 0.80
        return CVResult("accuracy", "binary", [mean] * 3, mean, 0.01, 3, True,
                        greater_is_better=True)

    monkeypatch.setattr("maestra.validation.cross_validate", fake_cv)

    result = srv.check_validation(str(p), "y")
    assert result["verdict"] == "ok"
    assert result["recommended_strategy"] == {"strategy": "group", "column": "x"}
    assert result["optimism_gap"] == pytest.approx(0.15, abs=1e-6)
    assert result["direction"] == "optimistic (dangerous)"


def test_check_validation_pessimistic_direction_for_an_error_metric(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    _df().to_csv(p, index=False)
    monkeypatch.setattr("maestra.profiling.profile_dataframe", lambda *a, **k: {})
    monkeypatch.setattr("maestra.validation_strategist.propose_fold_strategy",
                       lambda *a, **k: {"strategy": "time", "time_column": "x"})
    monkeypatch.setattr(
        "maestra.validation_strategist.validate_fold_strategy",
        lambda proposal, df, target: (
            {"strategy": "time", "time_column": "x", "rationale": "forecasts the future"}, []))
    monkeypatch.setattr("maestra.validation._is_classification", lambda y: False)

    def fake_cv(df, target, *, model_dir, **kwargs):
        # an ERROR metric: naive reports a LOWER (better-looking) error than time-ordered folds
        mean = 5.0 if "naive" in model_dir else 8.0
        return CVResult("root_mean_squared_error", "regression", [mean] * 3, mean, 0.1, 3, False,
                        greater_is_better=False)

    monkeypatch.setattr("maestra.validation.cross_validate", fake_cv)

    result = srv.check_validation(str(p), "y")
    assert result["direction"] == "optimistic (dangerous)"
    assert result["optimism_gap"] == pytest.approx(3.0, abs=1e-6)


# --- Tool 3: feasibility -----------------------------------------------------------------

class _FakeResult:
    def __init__(self, cv, fold_strategy, training):
        self.cv = cv
        self.fold_strategy = fold_strategy
        self.training = training


class _FakeTraining:
    def __init__(self, predictor=None):
        self.predictor = predictor


def test_feasibility_happy_path(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    _df().to_csv(p, index=False)
    cv = CVResult("accuracy", "binary", [0.9, 0.91, 0.89], 0.9, 0.01, 3, True,
                 greater_is_better=True)
    fake_result = _FakeResult(cv, {"strategy": "random"}, _FakeTraining(predictor=None))
    monkeypatch.setattr("maestra.pipeline.run_pipeline", lambda *a, **k: fake_result)
    monkeypatch.setattr("maestra.audit.audit", lambda *a, **k: _audit_report())

    result = srv.feasibility(str(p), "y")
    assert result["verdict"] == "ok"
    assert result["achievable_quality"] == {"metric": "accuracy", "mean": 0.9, "std": 0.01}
    assert result["fold_strategy"] == "random"
    assert result["strongest_drivers"] == []
    assert result["risk_level"] == "low"


def test_feasibility_extracts_feature_importance_when_a_predictor_exists(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    _df().to_csv(p, index=False)
    cv = CVResult("accuracy", "binary", [0.9], 0.9, 0.0, 3, True, greater_is_better=True)

    class FakePredictor:
        def feature_importance(self, *a, **k):
            return pd.DataFrame({"importance": [0.6, 0.4]}, index=["x", "y"])

    fake_result = _FakeResult(cv, {"strategy": "random"}, _FakeTraining(predictor=FakePredictor()))
    monkeypatch.setattr("maestra.pipeline.run_pipeline", lambda *a, **k: fake_result)
    monkeypatch.setattr("maestra.audit.audit", lambda *a, **k: _audit_report())

    result = srv.feasibility(str(p), "y")
    assert result["strongest_drivers"] == [
        {"feature": "x", "importance": 0.6}, {"feature": "y", "importance": 0.4}]


def test_feasibility_rejects_when_pipeline_produces_no_cv(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    _df().to_csv(p, index=False)
    fake_result = _FakeResult(None, None, _FakeTraining(predictor=None))
    monkeypatch.setattr("maestra.pipeline.run_pipeline", lambda *a, **k: fake_result)
    monkeypatch.setattr("maestra.audit.audit", lambda *a, **k: _audit_report())

    result = srv.feasibility(str(p), "y")
    assert result["verdict"] == "rejected"
    assert result["achievable_quality"] is None


# --- Tool 4: audit_backtest ---------------------------------------------------------------

def _forecast_df(n=100):
    return pd.DataFrame({"date": range(n), "x": range(n), "sales": [i % 7 for i in range(n)]})


def _backtest_report(**over):
    from maestra.backtest_audit import BacktestAuditReport
    base = dict(csv="x.csv", n_rows=100, target="sales", time_column="date", series_column=None)
    base.update(over)
    return BacktestAuditReport(**base)


def test_audit_backtest_missing_time_column_is_a_structured_rejection(tmp_path):
    p = tmp_path / "data.csv"
    _forecast_df().to_csv(p, index=False)
    result = srv.audit_backtest(str(p), "sales", "no_such_column")
    assert result["verdict"] == "rejected" and "no_such_column" in result["reason"]


def test_audit_backtest_happy_path(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    _forecast_df().to_csv(p, index=False)
    report = _backtest_report(future_leaks=[{"column": "x", "reason": "r",
                                             "correlation_with_target": 0.9}])
    monkeypatch.setattr("maestra.backtest_audit.audit_backtest", lambda *a, **k: report)
    monkeypatch.setattr("maestra.backtest_audit.write_backtest_audit_html", lambda *a, **k: None)

    result = srv.audit_backtest(str(p), "sales", "date", series_column="store_id")
    assert result["verdict"] == "ok"
    assert result["risk_level"] == "high"
    assert result["future_leaks"] == [{"column": "x", "reason": "r", "correlation_with_target": 0.9}]
    assert result["html_report"] == f"{p}.backtest_audit.html"


# --- entry point -------------------------------------------------------------------------

def test_main_loads_dotenv_before_running_the_server(monkeypatch):
    calls = []
    monkeypatch.setattr("maestra.config.load_dotenv", lambda: calls.append("dotenv"))
    monkeypatch.setattr(srv.mcp, "run", lambda: calls.append("run"))
    srv.main()
    assert calls == ["dotenv", "run"]
