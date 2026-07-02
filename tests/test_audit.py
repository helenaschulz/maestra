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


def test_deterministic_target_leak_scan(monkeypatch):
    """A numeric feature that is a near-copy of the target must be flagged without any LLM."""
    _patch_strategist(monkeypatch, strategy="random", group_column=None)
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    df = pd.DataFrame({
        "leaky": y + rng.normal(scale=0.01, size=200),   # |r| ~ 1.0 -> flagged
        "honest": rng.normal(size=200),                  # uncorrelated -> not flagged
        "y": y,
    })
    r = audit(df, "y", model="m")
    assert [c for c, _ in r.target_leaks] == ["leaky"]
    assert r.risk_level == "high"                        # leak evidence -> high
    assert "leaky" in render_report(r)


def test_risk_verdict_levels(monkeypatch):
    _patch_strategist(monkeypatch, strategy="random", group_column=None)
    clean = pd.DataFrame({"x": np.random.default_rng(0).normal(size=50), "y": [0, 1] * 25})
    assert audit(clean, "y", model="m").risk_level == "low"

    _patch_strategist(monkeypatch)                       # group strategy -> validation must change
    grouped = pd.DataFrame({"customer_id": np.repeat(range(10), 5),
                            "x": np.random.default_rng(0).normal(size=50), "y": [0, 1] * 25})
    assert audit(grouped, "y", model="m").risk_level == "elevated"


def test_german_report(monkeypatch):
    _patch_strategist(monkeypatch, warnings=[{"column": "x", "reason": "post-outcome"}])
    r = audit(_df(), "y", model="m")
    md = render_report(r, lang="de")
    assert "Datenrisiko-Audit" in md
    assert "Gesamtrisiko: HOCH" in md                    # leakage warning -> high
    assert "Gruppen-Folds nach `customer_id`" in md
    assert "Maßnahme" in md                              # actions rendered
    assert "post-outcome" in md                          # LLM rationale stays verbatim


def test_reports_carry_actions_and_summary(monkeypatch):
    _patch_strategist(monkeypatch)
    md = render_report(audit(_df(), "y", model="m"))
    assert "Executive summary" in md and "Overall risk" in md
    assert "→ Action:" in md                             # every main finding is actionable


def test_load_table_by_extension(tmp_path):
    from maestra.audit import _load_table
    df = pd.DataFrame({"a": [1, 2], "y": [0, 1]})
    csv, pq = tmp_path / "d.csv", tmp_path / "d.parquet"
    df.to_csv(csv, index=False)
    df.to_parquet(pq)
    assert _load_table(str(csv)).equals(df)
    assert _load_table(str(pq)).equals(df)
