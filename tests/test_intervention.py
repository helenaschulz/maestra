"""Tests for the intervention core (M4): CVBudget + run_counterfactual."""
import pytest

from maestra.intervention import CVBudget, run_counterfactual
from maestra.validation import CVResult


def _cv(scores, greater=True):
    import numpy as np
    return CVResult("m", "regression", list(scores), float(np.mean(scores)),
                    float(np.std(scores)), len(scores), False, greater)


# --- CVBudget ------------------------------------------------------------------------

def test_budget_unlimited_by_default_but_counts():
    b = CVBudget()
    assert all(b.try_spend() for _ in range(50))
    assert b.as_record() == {"limit": None, "trials_spent": 50}


def test_budget_refuses_beyond_limit_without_spending():
    b = CVBudget(limit=2)
    assert b.try_spend() and b.try_spend()
    assert not b.try_spend()
    assert b.as_record() == {"limit": 2, "trials_spent": 2}  # the refused call cost nothing


# --- run_counterfactual --------------------------------------------------------------

def test_accept_returns_trial_as_new_base():
    base = _cv([0.70, 0.71, 0.72])
    trial = _cv([0.80, 0.81, 0.82])
    outcome, new_base = run_counterfactual("feature:x", "generated_feature", "codegen",
                                           base=base, trial_fn=lambda: trial)
    assert outcome.accepted and outcome.reason == "improved"
    assert outcome.cv_delta == pytest.approx(0.10)
    assert outcome.base_scores == base.fold_scores
    assert outcome.trial_scores == trial.fold_scores  # the counterfactual is on the record
    assert new_base is trial


def test_reject_keeps_base_and_records_trial_mean():
    base = _cv([0.80, 0.81, 0.82])
    trial = _cv([0.80, 0.82, 0.80])  # noise, no majority improvement
    outcome, new_base = run_counterfactual("keep:col", "skeptic_keep", "skeptic",
                                           base=base, trial_fn=lambda: trial)
    assert not outcome.accepted and outcome.reason == "no_improvement"
    assert new_base is base
    assert outcome.trial_mean == pytest.approx(trial.mean)  # audit even on reject


def test_bit_identical_trial_is_no_effect():
    base = _cv([0.80, 0.81, 0.82])
    trial = _cv([0.80, 0.81, 0.82])
    outcome, _ = run_counterfactual("feature:dup", "generated_feature", "codegen",
                                    base=base, trial_fn=lambda: trial)
    assert outcome.reason == "no_effect" and not outcome.accepted


def test_exhausted_budget_never_runs_the_trial():
    calls = []
    base = _cv([0.80, 0.81, 0.82])
    outcome, new_base = run_counterfactual(
        "target:log1p", "target_framing", "target_framing", base=base,
        trial_fn=lambda: calls.append(1) or base, budget=CVBudget(limit=0))
    assert outcome.reason == "budget_exhausted" and not outcome.accepted
    assert outcome.cv_delta is None and outcome.trial_scores is None
    assert calls == []          # no AutoGluon money spent
    assert new_base is base


def test_budget_is_shared_across_interventions():
    base = _cv([0.70, 0.71, 0.72])
    better = _cv([0.80, 0.81, 0.82])
    budget = CVBudget(limit=1)
    o1, base = run_counterfactual("a", "k", "p", base=base, trial_fn=lambda: better, budget=budget)
    o2, _ = run_counterfactual("b", "k", "p", base=base, trial_fn=lambda: better, budget=budget)
    assert o1.reason == "improved" and o2.reason == "budget_exhausted"


# --- gates honour the budget ---------------------------------------------------------

def test_select_features_records_budget_exhausted(monkeypatch):
    import maestra.hybrid_features as hf
    from maestra.hybrid_features import GeneratedFeature, SandboxResult

    base = _cv([0.70, 0.71, 0.72])
    better = _cv([0.80, 0.81, 0.82])
    results = iter([base, better])  # gate's base CV, then the single affordable trial
    monkeypatch.setattr(hf, "_dry_run", lambda *a, **k: SandboxResult("ok"))
    monkeypatch.setattr(hf, "cross_validate", lambda *a, **k: next(results))

    cands = [GeneratedFeature("f1", "i", "c"), GeneratedFeature("f2", "i", "c")]
    kept, records, _ = hf.select_features(
        None, "y", cands, cleaning_plan=None, feature_plan=None, model_dir="x",
        time_limit=1, n_folds=3, seed=0, budget=CVBudget(limit=1))
    assert [f.name for f in kept] == ["f1"]
    assert [r.reason for r in records] == ["improved", "budget_exhausted"]


def test_skeptic_gate_marks_unmeasured_on_exhausted_budget(monkeypatch):
    import maestra.skeptic as sk

    base = _cv([0.70, 0.71, 0.72])
    monkeypatch.setattr(sk, "cross_validate", lambda *a, **k: base)

    plan = {"columns_to_drop": [{"column": "a"}, {"column": "b"}]}
    reviews = [{"column": "a", "risk": "high", "reason": "r"},
               {"column": "b", "risk": "high", "reason": "r"}]
    revised, records = sk.apply_skeptic_gate(
        None, "y", cleaning_plan=plan, feature_plan=None, reviews=reviews, model_dir="x",
        time_limit=1, n_folds=3, seed=0, budget=CVBudget(limit=0))
    assert revised == plan                       # unmeasured -> every drop stands
    assert all(not r.vetoed and not r.measured and r.cv_delta is None for r in records)
