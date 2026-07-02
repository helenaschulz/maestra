"""Tests for hybrid feature generation. The sandbox is exercised with REAL subprocesses
(it is the safety mechanism), the LLM and AutoGluon are mocked elsewhere."""
import numpy as np
import pandas as pd

from maestra import hybrid_features as hf
from maestra.hybrid_features import GeneratedFeature, SandboxResult, run_in_sandbox, select_features
from maestra.validation import CVResult


def _cv(mean, std=0.02):
    return CVResult("accuracy", "binary", [], mean, std, 3, True, True)

_TRAIN = pd.DataFrame({"a": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
_VAL = pd.DataFrame({"a": [10.0, 20.0], "y": [1, 0]})


def test_valid_feature_runs_and_returns_values():
    code = "def fit(train_df):\n    return {}\ndef transform(df, params):\n    return df['a'] * 2"
    res = run_in_sandbox(code, _TRAIN, _VAL, "y")
    assert res.status == "ok"
    assert list(res.train) == [2.0, 4.0, 6.0]
    assert list(res.val) == [20.0, 40.0]


def test_target_using_feature_fails_no_leak():
    """A feature that tries to read the target cannot: transform never gets the target column."""
    code = "def fit(train_df):\n    return {}\ndef transform(df, params):\n    return df['y']"
    res = run_in_sandbox(code, _TRAIN, _VAL, "y")
    assert res.status == "error"
    assert "y" in (res.error or "")          # KeyError on the missing target column


def test_crashing_candidate_is_caught():
    code = "def fit(train_df):\n    raise ValueError('boom')\ndef transform(df, params):\n    return df['a']"
    res = run_in_sandbox(code, _TRAIN, _VAL, "y")
    assert res.status == "error"
    assert "boom" in (res.error or "")


def test_infinite_loop_times_out_cleanly():
    code = "def fit(train_df):\n    while True:\n        pass\ndef transform(df, params):\n    return df['a']"
    res = run_in_sandbox(code, _TRAIN, _VAL, "y", timeout=2, cpu_seconds=1)
    assert res.status in {"timeout", "error"}   # killed, parent stays alive
    assert res.train is None


def test_network_access_is_blocked():
    code = (
        "def fit(train_df):\n    return {}\n"
        "def transform(df, params):\n"
        "    import socket\n"
        "    socket.socket()\n"
        "    return df['a']"
    )
    res = run_in_sandbox(code, _TRAIN, _VAL, "y")
    assert res.status == "error"               # socket import blocked / socket disabled


def test_wrong_length_output_is_invalid():
    code = "def fit(train_df):\n    return {}\ndef transform(df, params):\n    return df['a'].head(1)"
    res = run_in_sandbox(code, _TRAIN, _VAL, "y")
    assert res.status == "error"
    assert "length" in (res.error or "")


# --- CV gate (cross_validate mocked) ----------------------------------------------

def test_gate_keeps_improvement_discards_noise(monkeypatch):
    monkeypatch.setattr(hf, "_dry_run", lambda *a, **k: SandboxResult("ok"))
    results = iter([_cv(0.80), _cv(0.85), _cv(0.855)])  # base, trial_a (keep), trial_b (discard)
    monkeypatch.setattr(hf, "cross_validate", lambda *a, **k: next(results))
    cands = [GeneratedFeature("a", "ia", "ca"), GeneratedFeature("b", "ib", "cb")]

    kept, records, _ = select_features(pd.DataFrame({"y": [0, 1]}), "y", cands,
                                    cleaning_plan=None, feature_plan=None, model_dir="x",
                                    time_limit=1, n_folds=3, seed=0)

    assert [f.name for f in kept] == ["a"]            # +0.05 beats 1σ (0.02); b's +0.005 vs new base does not
    assert records[0].kept and records[0].reason == "improved"
    assert not records[1].kept and records[1].reason == "no_improvement"


def test_gate_flags_bit_identical_cv_as_no_effect(monkeypatch):
    """A trial CV that is bit-identical to the base (same mean AND fold scores) means the
    feature changed nothing — duplicate column or skipped in every fold — and must be
    distinguishable in the provenance from an honest 'trained on it, did not help'."""
    monkeypatch.setattr(hf, "_dry_run", lambda *a, **k: SandboxResult("ok"))
    base = CVResult("accuracy", "binary", [0.79, 0.80, 0.81], 0.80, 0.01, 3, True, True)
    identical = CVResult("accuracy", "binary", [0.79, 0.80, 0.81], 0.80, 0.01, 3, True, True)
    results = iter([base, identical])
    monkeypatch.setattr(hf, "cross_validate", lambda *a, **k: next(results))

    kept, records, _ = select_features(pd.DataFrame({"y": [0, 1]}), "y",
                                    [GeneratedFeature("dup", "i", "c")],
                                    cleaning_plan=None, feature_plan=None, model_dir="x",
                                    time_limit=1, n_folds=3, seed=0)

    assert kept == []
    assert records[0].reason == "no_effect" and records[0].cv_delta == 0.0


def test_gate_skips_cv_for_failed_dry_run(monkeypatch):
    monkeypatch.setattr(hf, "_dry_run", lambda *a, **k: SandboxResult("error", error="boom"))
    cv_calls = []
    monkeypatch.setattr(hf, "cross_validate", lambda *a, **k: cv_calls.append(1) or _cv(0.80))

    kept, records, _ = select_features(pd.DataFrame({"y": [0, 1]}), "y", [GeneratedFeature("a", "i", "c")],
                                    cleaning_plan=None, feature_plan=None, model_dir="x",
                                    time_limit=1, n_folds=2, seed=0)

    assert kept == []
    assert records[0].reason == "sandbox_error"
    assert len(cv_calls) == 1                          # only the baseline CV ran; no trial CV


def test_leaky_feature_rejected_by_real_sandbox(monkeypatch):
    """A constructed leakage attempt cannot improve the fold-wise CV — it fails the real
    sandbox dry-run (transform never gets the target) and is dropped before any trial CV."""
    monkeypatch.setattr(hf, "cross_validate", lambda *a, **k: _cv(0.80))
    leaky = GeneratedFeature("leak", "returns the label",
                             "def fit(train_df):\n    return {}\ndef transform(df, params):\n    return df['y']")
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6], "y": [0, 1, 0, 1, 0, 1]})

    kept, records, _ = select_features(df, "y", [leaky], cleaning_plan=None, feature_plan=None,
                                    model_dir="x", time_limit=1, n_folds=2, seed=0)

    assert kept == []
    assert records[0].reason == "sandbox_error"


# --- hardening: row-independence + env restriction --------------------------------

def test_row_context_dependent_feature_is_dropped(monkeypatch):
    """A feature using a batch-global statistic (df['a'].mean()) is rejected by the dry-run."""
    monkeypatch.setattr(hf, "cross_validate", lambda *a, **k: _cv(0.80))
    cand = GeneratedFeature("rc", "uses batch mean",
        "def fit(train_df):\n    return {}\ndef transform(df, params):\n    return df['a'] / df['a'].mean()")
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 7, 8], "y": [0, 1] * 4})

    kept, records, _ = select_features(df, "y", [cand], cleaning_plan=None, feature_plan=None,
                                       model_dir="x", time_limit=1, n_folds=2, seed=0)

    assert kept == []
    assert records[0].reason == "row_context_dependent"


def test_row_independent_feature_passes_dry_run(monkeypatch):
    results = iter([_cv(0.80), _cv(0.90)])  # base, trial (improves -> kept)
    monkeypatch.setattr(hf, "cross_validate", lambda *a, **k: next(results))
    cand = GeneratedFeature("rw", "row-wise double",
        "def fit(train_df):\n    return {}\ndef transform(df, params):\n    return df['a'] * 2")
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 7, 8], "y": [0, 1] * 4})

    kept, records, _ = select_features(df, "y", [cand], cleaning_plan=None, feature_plan=None,
                                       model_dir="x", time_limit=1, n_folds=2, seed=0)

    assert [f.name for f in kept] == ["rw"]      # row-wise feature is allowed (and here kept)
    assert records[0].reason == "improved"


def test_sandbox_env_excludes_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("TAVILY_API_KEY", "tv-secret")
    env = hf._sandbox_env()
    assert "OPENAI_API_KEY" not in env and "TAVILY_API_KEY" not in env
    assert all("KEY" not in k.upper() and "SECRET" not in k.upper() and "TOKEN" not in k.upper() for k in env)


def test_candidate_cannot_read_env_secret(monkeypatch):
    monkeypatch.setenv("MAESTRA_SECRET_TEST", "leak_me")
    code = (
        "def fit(train_df):\n    return {}\n"
        "def transform(df, params):\n"
        "    import os\n"  # blocked import -> cannot reach the environment at all
        "    return df['a'] * 0 + len(os.environ.get('MAESTRA_SECRET_TEST', ''))"
    )
    res = run_in_sandbox(code, _TRAIN, _VAL, "y")
    assert res.status == "error"   # the candidate does not get the secret
