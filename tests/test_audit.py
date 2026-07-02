"""Tests for maestra-audit. The Validation Strategist (LLM) and adversarial validation (AutoGluon)
are mocked; the structural flags and report rendering are exercised for real."""
import numpy as np
import pandas as pd

from maestra import audit as audit_mod
from maestra.audit import audit, render_report


def _df():
    return pd.DataFrame({
        "customer_id": np.repeat(range(20), 5),                 # id-like with repeats
        "row_id": range(100),                                   # id-like, unique
        "note": [f"free text {i}" for i in range(100)],         # high-cardinality text
        "empty": [np.nan] * 100,                                # all missing + constant
        "const": [1] * 100,                                     # constant
        "x": np.random.default_rng(0).normal(size=100),         # a real feature
        "y": [0, 1] * 50,                                        # target
    })


def _patch_strategist(monkeypatch, strategy="group", group_column="customer_id", warnings=None):
    monkeypatch.setattr(audit_mod, "propose_fold_strategy", lambda *a, **k: {
        "strategy": strategy, "group_column": group_column, "time_column": None,
        "rationale": "several rows per customer", "leakage_warnings": warnings or [],
    })


def test_structural_flags_are_deterministic(monkeypatch):
    _patch_strategist(monkeypatch)
    r = audit(_df(), "y", model="m")
    # unique columns (int id AND unique text) are id-like; the repeated customer_id is NOT — it is
    # the grouping entity, which the Strategist handles, not a column to drop.
    assert "row_id" in r.id_like and "note" in r.id_like
    assert "customer_id" not in r.id_like
    assert "const" in r.constant and "empty" in r.constant       # constant columns
    assert any(c == "empty" for c, _ in r.high_missing)          # 100% missing flagged
    assert r.adversarial_auc is None                             # no --test -> not run


def test_group_strategy_reaches_the_report(monkeypatch):
    _patch_strategist(monkeypatch, warnings=[{"column": "x", "reason": "measured after the outcome"}])
    r = audit(_df(), "y", model="m")
    md = render_report(r)
    assert "Group folds by `customer_id`" in md
    assert "measured after the outcome" in md          # leakage warning rendered
    assert "ID-like columns" in md and "`row_id`" in md
    assert "Not checked" in md                          # no test set


def test_adversarial_check_runs_only_with_test(monkeypatch):
    _patch_strategist(monkeypatch, strategy="random", group_column=None)
    monkeypatch.setattr(audit_mod, "adversarial_validation", lambda *a, **k: 0.92)
    test_df = _df().drop(columns=["y"])
    r = audit(_df(), "y", model="m", test_df=test_df)
    md = render_report(r)
    assert r.adversarial_auc == 0.92
    assert "STRONG shift" in md                          # 0.92 -> strong-shift verdict
    assert "Random folds are appropriate" in md


def test_unknown_target_raises(monkeypatch):
    _patch_strategist(monkeypatch)
    import pytest
    with pytest.raises(ValueError, match="not in CSV"):
        audit(_df(), "missing", model="m")
