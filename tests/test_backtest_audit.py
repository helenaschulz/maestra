"""Offline tests for the F1 backtest audit. LLM and AutoGluon calls are always mocked (monkeypatch
on the source module, since backtest_audit.py imports both lazily inside the functions that use
them). Covers all three built-in-lie scenarios plus a clean false-alarm control."""
import numpy as np
import pandas as pd
import pytest

from maestra import backtest_audit as ba_mod
from maestra.backtest_audit import (
    BacktestAuditReport,
    audit_backtest,
    check_future_features,
    quantify_backtest_lie,
    rolling_origins,
    series_leak_check,
    split_design_check,
)
from maestra.engine import TrainingResult


def _forecast_df(n=200, with_future_leak=False, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    trend = np.arange(n, dtype=float)
    target = trend + rng.normal(0, 1.0, n)
    df = pd.DataFrame({"date": dates, "price": rng.normal(50, 5, n), "sales": target})
    if with_future_leak:
        df["actual_total"] = target * 1.0001 + rng.normal(0, 0.01, n)  # near-perfect proxy
    return df


# --- quantify_backtest_lie (pure, no mocking) ---------------------------------------------

def test_quantify_backtest_lie_detects_an_optimistic_naive_backtest():
    naive = [0.90, 0.91, 0.89]
    corrected = [0.70, 0.72, 0.68]  # naive reports much higher accuracy -> optimistic
    lie = quantify_backtest_lie(naive, corrected, greater_is_better=True)
    assert lie["direction"] == "optimistic (dangerous)"
    assert lie["mean_gap"] > 0
    assert lie["n_origins"] == 3


def test_quantify_backtest_lie_flips_sign_for_a_lower_is_better_metric():
    naive = [5.0, 5.2, 4.9]      # naive reports a LOWER error -> still optimistic
    corrected = [8.0, 8.3, 7.9]
    lie = quantify_backtest_lie(naive, corrected, greater_is_better=False)
    assert lie["direction"] == "optimistic (dangerous)"
    assert lie["mean_gap"] > 0


def test_quantify_backtest_lie_negligible_gap_is_undecided():
    naive = [0.80, 0.81, 0.79]
    corrected = [0.80, 0.79, 0.81]  # noise-level difference
    lie = quantify_backtest_lie(naive, corrected, greater_is_better=True)
    assert lie["direction"] == "undecided"


def test_quantify_backtest_lie_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        quantify_backtest_lie([1.0, 2.0], [1.0], greater_is_better=True)


# --- rolling_origins (pure) ----------------------------------------------------------------

def test_rolling_origins_produces_paired_expanding_and_embargoed_train_sets():
    origins = rolling_origins(200, n_origins=3, test_frac=0.1, embargo_frac=0.05)
    assert len(origins) == 3
    for naive_sl, embargo_sl, test_sl in origins:
        assert embargo_sl.stop < naive_sl.stop  # embargo train is strictly smaller
        assert naive_sl.stop <= test_sl.start   # no overlap between train and test
        assert test_sl.stop - test_sl.start > 0


def test_rolling_origins_returns_nothing_for_too_little_data():
    assert rolling_origins(5, n_origins=3, test_frac=0.1, embargo_frac=0.05) == []


# --- check_future_features (pure, deterministic corroboration) ----------------------------

def test_check_future_features_attaches_correlation_evidence():
    df = _forecast_df(with_future_leak=True)
    proposal = {"future_leaking_columns": [{"column": "actual_total", "reason": "same-day actual"}]}
    findings = check_future_features(df, "sales", proposal)
    assert len(findings) == 1
    assert findings[0]["column"] == "actual_total"
    assert findings[0]["correlation_with_target"] > 0.99


def test_check_future_features_drops_nonexistent_columns_without_crashing():
    df = _forecast_df()
    proposal = {"future_leaking_columns": [{"column": "does_not_exist", "reason": "x"}]}
    assert check_future_features(df, "sales", proposal) == []


# --- split_design_check (mocked AutoGluon) --------------------------------------------------

def _mock_train_and_evaluate(gap_when_naive_higher=True):
    """naive_* model_dirs score BETTER than embargo_* ones -- the lie split_design_check exists
    to catch. Consistent across every origin so the paired test has real power."""
    def fake(train, holdout, target, time_limit, model_dir, eval_metric=None, presets=None,
            sample_weight=None):
        is_naive = "naive" in model_dir
        base = 0.90 if is_naive else 0.70
        # tiny, deterministic per-call jitter so folds aren't bit-identical (still decisive)
        score = base + (0.01 if "0" in model_dir else (-0.01 if "1" in model_dir else 0.005))
        if not gap_when_naive_higher:
            score = 0.80  # both variants identical -> no gap
        predictor = type("P", (), {"eval_metric": type("M", (), {"greater_is_better": True})()})()
        return TrainingResult("regression", "r2", pd.DataFrame(), {"r2": score}, predictor=predictor)
    return fake


def test_split_design_check_detects_a_naive_optimistic_gap(monkeypatch):
    import maestra.engine as engine_mod
    monkeypatch.setattr(engine_mod, "train_and_evaluate", _mock_train_and_evaluate(True))

    df = _forecast_df()
    result = split_design_check(df, "sales", "date", n_origins=3, time_limit=1)
    assert result is not None
    assert result["direction"] == "optimistic (dangerous)"
    assert result["n_origins"] == 3


def test_split_design_check_no_gap_is_undecided(monkeypatch):
    import maestra.engine as engine_mod
    monkeypatch.setattr(engine_mod, "train_and_evaluate", _mock_train_and_evaluate(False))

    df = _forecast_df()
    result = split_design_check(df, "sales", "date", n_origins=3, time_limit=1)
    assert result is not None
    assert result["direction"] == "undecided"


def test_split_design_check_returns_none_for_too_little_data(monkeypatch):
    import maestra.engine as engine_mod
    monkeypatch.setattr(engine_mod, "train_and_evaluate", _mock_train_and_evaluate(True))
    df = _forecast_df(n=5)
    assert split_design_check(df, "sales", "date", n_origins=3, time_limit=1) is None


# --- series_leak_check (mocked AutoGluon) ---------------------------------------------------

def test_series_leak_check_reports_a_high_auc_when_series_separate_cleanly(monkeypatch):
    import maestra.validation as validation_mod
    monkeypatch.setattr(validation_mod, "adversarial_validation", lambda *a, **k: 0.93)

    df = _forecast_df()
    df["store_id"] = (np.arange(len(df)) % 5)
    auc = series_leak_check(df, "sales", "date", "store_id", time_limit=1)
    assert auc == pytest.approx(0.93)


def test_series_leak_check_none_for_too_little_data(monkeypatch):
    import maestra.validation as validation_mod
    monkeypatch.setattr(validation_mod, "adversarial_validation", lambda *a, **k: 0.93)
    df = _forecast_df(n=10)
    df["store_id"] = 0
    assert series_leak_check(df, "sales", "date", "store_id", time_limit=1) is None


# --- audit_backtest end-to-end (all three lies + a clean control) --------------------------

def _patch_clean(monkeypatch):
    import maestra.engine as engine_mod
    import maestra.target_framing as tf_mod
    import maestra.validation as validation_mod

    monkeypatch.setattr(ba_mod, "propose_future_features", lambda *a, **k: {"future_leaking_columns": []})
    monkeypatch.setattr(engine_mod, "train_and_evaluate", _mock_train_and_evaluate(False))
    monkeypatch.setattr(validation_mod, "adversarial_validation", lambda *a, **k: 0.50)
    monkeypatch.setattr(tf_mod, "propose_target_framing",
                        lambda *a, **k: {"transform": "none", "rationale": "not skewed"})


def test_audit_backtest_clean_dataset_is_low_risk_false_alarm_control(monkeypatch):
    _patch_clean(monkeypatch)
    df = _forecast_df()
    df["store_id"] = np.arange(len(df)) % 5
    report = audit_backtest(df, "sales", "date", model="m", series_column="store_id")
    assert isinstance(report, BacktestAuditReport)
    assert report.future_leaks == []
    assert report.split_design["direction"] == "undecided"
    assert report.series_leak_auc == pytest.approx(0.50)
    assert report.risk_level == "low"


def test_audit_backtest_finds_a_future_leak(monkeypatch):
    _patch_clean(monkeypatch)
    monkeypatch.setattr(ba_mod, "propose_future_features",
                        lambda *a, **k: {"future_leaking_columns":
                                        [{"column": "actual_total", "reason": "same-day actual"}]})
    df = _forecast_df(with_future_leak=True)
    report = audit_backtest(df, "sales", "date", model="m")
    assert len(report.future_leaks) == 1
    assert report.future_leaks[0]["column"] == "actual_total"
    assert report.risk_level == "high"


def test_audit_backtest_finds_a_missing_gap(monkeypatch):
    _patch_clean(monkeypatch)
    import maestra.engine as engine_mod
    monkeypatch.setattr(engine_mod, "train_and_evaluate", _mock_train_and_evaluate(True))
    df = _forecast_df()
    report = audit_backtest(df, "sales", "date", model="m")
    assert report.split_design["direction"] == "optimistic (dangerous)"
    assert report.risk_level == "high"


def test_audit_backtest_records_series_auc_but_does_not_escalate_risk(monkeypatch):
    """A high series-boundary AUC is RECORDED as diagnostic but must NOT drive risk_level: without
    a time-trend control it is ~1.0 for any trending series, so it carries no verdict meaning yet
    (the PR#6 review finding). Here nothing else is wrong (no future leak, undecided split), so a
    ~1.0 AUC alone must stay 'low', not escalate to 'high'."""
    _patch_clean(monkeypatch)
    import maestra.validation as validation_mod
    monkeypatch.setattr(validation_mod, "adversarial_validation", lambda *a, **k: 0.999)
    df = _forecast_df()
    df["store_id"] = np.arange(len(df)) % 5
    report = audit_backtest(df, "sales", "date", model="m", series_column="store_id")
    assert report.series_leak_auc == pytest.approx(0.999)  # still computed and surfaced
    assert report.risk_level == "low"                       # but NOT a verdict driver anymore


def test_risk_level_ignores_series_auc_across_the_whole_range():
    """The series AUC never changes the verdict, at any value -- a pure trending series with a
    near-perfect boundary AUC and no real leak/optimism stays 'low'."""
    for auc in (0.5, 0.65, 0.8, 0.999, 1.0):
        report = BacktestAuditReport(csv="x", n_rows=500, target="y", time_column="t",
                                     series_column="s", future_leaks=[], split_design=None,
                                     series_leak_auc=auc)
        assert report.risk_level == "low", f"series AUC {auc} must not escalate risk_level"


def test_audit_backtest_flags_target_framing_candidate(monkeypatch):
    _patch_clean(monkeypatch)
    import maestra.target_framing as tf_mod
    monkeypatch.setattr(tf_mod, "propose_target_framing",
                        lambda *a, **k: {"transform": "log1p", "rationale": "right-skewed counts"})
    df = _forecast_df()
    df["sales"] = df["sales"].abs() + 1  # non-negative, so the framing check runs
    report = audit_backtest(df, "sales", "date", model="m")
    assert report.target_framing["proposed"] == "log1p"
    assert report.target_framing["verified"] is True


def test_audit_backtest_raises_on_unknown_target():
    df = _forecast_df()
    with pytest.raises(ValueError, match="not in data"):
        audit_backtest(df, "no_such_column", "date", model="m")


def test_audit_backtest_raises_on_unknown_time_column():
    df = _forecast_df()
    with pytest.raises(ValueError, match="not in data"):
        audit_backtest(df, "sales", "no_such_column", model="m")
