"""F3-1: covariate availability as a verdict. The LLM proposes known_future / past_only /
leaking per column; deterministic evidence (future window, calendar derivability, correlation)
decides, and wins on contradiction. Offline: the LLM proposal is a plain dict, AutoGluon mocked."""
import numpy as np
import pandas as pd
import pytest

from maestra.backtest_audit import (
    BacktestAuditReport,
    _calendar_derivable,
    check_covariate_availability,
)


def _store_panel(n_days=120, n_stores=3, seed=0):
    """Daily panel, several stores. is_weekend: calendar. promo: planned, varies per store.
    temperature: observed weather. customers: a same-day actual nearly equal to the outcome."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for s in range(n_stores):
        promo = rng.integers(0, 2, n_days)
        temp = 10 + 5 * np.sin(np.arange(n_days) / 9) + rng.normal(0, 2, n_days)
        sales = 100 + 30 * promo + 20 * (dates.dayofweek >= 5) + rng.normal(0, 3, n_days)
        for i, d in enumerate(dates):
            rows.append({"date": d, "store": s, "is_weekend": int(d.dayofweek >= 5),
                         "promo": int(promo[i]), "temperature": float(temp[i]),
                         "customers": float(sales[i] * 0.5 + rng.normal(0, 0.2)),
                         "sales": float(sales[i])})
    return pd.DataFrame(rows)


def _future(df):
    """The real future window: what test.csv supplies (no customers, no weather actuals)."""
    return df[["date", "store", "is_weekend", "promo"]].head(30)


def _by_col(out):
    return {c["column"]: c for c in out}


def _proposal(**claims):
    return {"future_leaking_columns": [],
            "covariate_availability": [{"column": c, "availability": a, "reason": "r"}
                                       for c, a in claims.items()]}


def test_calendar_column_is_known_future_without_a_future_window():
    df = _store_panel()
    assert _calendar_derivable(df, "is_weekend", "date") is True
    out = _by_col(check_covariate_availability(
        df, "sales", "date", _proposal(is_weekend="known_future")))
    assert out["is_weekend"]["availability"] == "known_future"
    assert out["is_weekend"]["corroborated"] is True


def test_weather_is_not_calendar_derivable():
    df = _store_panel()
    assert _calendar_derivable(df, "temperature", "date") is False


def test_future_unknown_column_is_leaking_with_corroboration():
    df = _store_panel()
    out = _by_col(check_covariate_availability(
        df, "sales", "date", _proposal(customers="leaking"), future_df=_future(df)))
    c = out["customers"]
    assert c["availability"] == "leaking"
    assert c["evidence"]["in_future_window"] is False
    assert c["evidence"]["correlation_with_target"] > 0.9
    assert c["action"].startswith("drop")


def test_planned_promo_in_the_future_window_is_known_future():
    df = _store_panel()
    out = _by_col(check_covariate_availability(
        df, "sales", "date", _proposal(promo="known_future"), future_df=_future(df)))
    assert out["promo"]["availability"] == "known_future"
    assert out["promo"]["evidence"]["in_future_window"] is True


# contradiction cases: the evidence wins
def test_llm_known_future_absent_from_future_window_is_overridden():
    df = _store_panel()
    out = _by_col(check_covariate_availability(
        df, "sales", "date", _proposal(temperature="known_future"), future_df=_future(df)))
    t = out["temperature"]
    assert t["availability"] == "past_only"
    assert t["corroborated"] is False


def test_llm_known_future_without_any_evidence_is_downgraded():
    df = _store_panel()
    out = _by_col(check_covariate_availability(
        df, "sales", "date", _proposal(temperature="known_future")))
    assert out["temperature"]["availability"] == "past_only"
    assert "no deterministic evidence" in out["temperature"]["reason"]


def test_llm_leaking_but_present_in_future_window_is_known_future():
    df = _store_panel()
    out = _by_col(check_covariate_availability(
        df, "sales", "date", _proposal(promo="leaking"), future_df=_future(df)))
    assert out["promo"]["availability"] == "known_future"
    assert out["promo"]["corroborated"] is False


def test_llm_leaking_without_evidence_falls_back_to_past_only():
    df = _store_panel()
    out = _by_col(check_covariate_availability(
        df, "sales", "date", _proposal(temperature="leaking")))
    assert out["temperature"]["availability"] == "past_only"


def test_unclaimed_and_bogus_columns():
    df = _store_panel()
    prop = _proposal(nonexistent="leaking", sales="known_future")
    out = _by_col(check_covariate_availability(df, "sales", "date", prop))
    assert "nonexistent" not in out and "sales" not in out and "date" not in out
    assert out["is_weekend"]["llm"] is None
    assert out["is_weekend"]["availability"] == "known_future"  # evidence alone


def test_a_leaking_covariate_raises_the_risk_level():
    base = dict(csv="x", n_rows=1, target="sales", time_column="date", series_column=None)
    leak = {"column": "customers", "llm": "leaking", "availability": "leaking",
            "corroborated": True, "evidence": {}, "action": "drop", "reason": ""}
    assert BacktestAuditReport(**base).risk_level == "low"
    assert BacktestAuditReport(**base, covariate_availability=[leak]).risk_level == "high"


def test_audit_backtest_threads_the_future_window(monkeypatch):
    import maestra.backtest_audit as bt

    df = _store_panel()
    monkeypatch.setattr(bt, "propose_future_features",
                        lambda *a, **k: _proposal(customers="leaking", promo="known_future"))
    monkeypatch.setattr(bt, "split_design_check", lambda *a, **k: None)
    monkeypatch.setattr(bt, "target_framing_flag", lambda *a, **k: None)
    report = bt.audit_backtest(df, "sales", "date", model="m", future_df=_future(df))
    cov = _by_col(report.covariate_availability)
    assert cov["customers"]["availability"] == "leaking"
    assert cov["promo"]["availability"] == "known_future"
    assert report.risk_level == "high"


def test_html_shows_the_covariate_table():
    from maestra.dossier import render_backtest_audit

    df = _store_panel()
    cov = check_covariate_availability(df, "sales", "date", _proposal(customers="leaking"),
                                       future_df=_future(df))
    r = BacktestAuditReport(csv="x", n_rows=len(df), target="sales", time_column="date",
                            series_column=None, covariate_availability=cov)
    html = render_backtest_audit(r)
    assert "Covariate availability at forecast time" in html
    assert "absent from future window" in html


@pytest.mark.parametrize("col", ["promo", "temperature"])
def test_numeric_time_index_has_no_calendar(col):
    df = _store_panel().assign(date=lambda d: d["date"].rank(method="dense").astype(int))
    assert _calendar_derivable(df, col, "date") is False
