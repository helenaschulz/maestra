"""Tests for the free-text featurization lane (M10)."""
import numpy as np
import pandas as pd
import pytest

import maestra.text_features as tf
from maestra.hybrid_features import run_in_sandbox
from maestra.text_features import detect_text_columns, propose_text_feature_code, text_profile

_N = 60  # above _MIN_NON_NULL


def _text(i: int) -> str:
    return (f"Spacious {i}-bedroom apartment with granite countertops, "
            f"hardwood floors and a {i % 3 + 1} car garage. Listing #{i:04d}.")


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "description": [_text(i) for i in range(_N)],           # free text
        "quality": ["Good", "Bad", "Excellent"] * (_N // 3),     # categorical (low unique frac)
        "code": [f"A{i}" for i in range(_N)],                    # unique but short
        "price": np.linspace(100, 200, _N),                      # numeric
        "y": np.arange(_N),
    })


# --- detection -----------------------------------------------------------------------

def test_detects_prose_and_ignores_categorical_short_numeric_target():
    assert detect_text_columns(_df(), "y") == ["description"]


def test_never_returns_the_target_even_if_texty():
    df = _df().rename(columns={"description": "y", "y": "label"})
    assert detect_text_columns(df, "y") == []


def test_too_few_rows_is_not_text():
    df = _df().head(10)
    assert detect_text_columns(df, "y") == []


# --- text profile --------------------------------------------------------------------

def test_profile_truncates_examples_and_is_deterministic():
    df = _df()
    df.loc[0, "description"] = "x" * 500
    prof = text_profile(df, ["description"])["description"]
    assert prof["n_non_null"] == _N
    assert all(len(e) <= tf._EXAMPLE_CHARS for e in prof["examples"])
    assert len(prof["examples"]) <= tf._N_EXAMPLES
    assert prof == text_profile(df, ["description"])["description"]


# --- proposal (mocked LLM) -----------------------------------------------------------

def test_propose_tags_source_text_and_caps_candidates(monkeypatch):
    out = {"features": [{"name": f"f{i}", "idea": "i", "code": "c"} for i in range(9)]
           + [{"name": "", "code": "c"}, "garbage"]}
    monkeypatch.setattr(tf, "call_structured", lambda **k: out)
    feats = propose_text_feature_code("m", _df(), "y", ["description"], max_candidates=3)
    assert len(feats) == 3
    assert all(f.source == "text" for f in feats)


def test_prompt_contains_verbatim_samples(monkeypatch):
    seen = {}
    monkeypatch.setattr(tf, "call_structured",
                        lambda **k: seen.update(k) or {"features": []})
    propose_text_feature_code("m", _df(), "y", ["description"])
    assert "granite countertops" in seen["user_prompt"]  # the LLM sees real text
    assert "n-gram" in seen["system_prompt"]              # and the anti-n-gram constraint


# --- sandbox: a realistic extractor actually runs -------------------------------------

def test_text_extractor_code_runs_in_sandbox():
    code = (
        "import re\n"
        "def fit(train_df):\n"
        "    return {}\n"
        "def transform(df, params):\n"
        "    s = df['description'].fillna('').astype(str)\n"
        "    return s.str.extract(r'(\\d+)-bedroom', expand=False).astype(float).fillna(0)\n"
    )
    df = _df()
    res = run_in_sandbox(code, df.iloc[:40], df.iloc[40:], "y")
    assert res.status == "ok"
    assert np.allclose(res.val, np.arange(40, _N, dtype=float))


# --- pipeline validation ---------------------------------------------------------------

def test_text_features_requires_cv():
    from maestra.pipeline import run_pipeline
    with pytest.raises(ValueError, match="--text-features requires --cv"):
        run_pipeline(_df(), "y", model="m", test_size=0.25, time_limit=1, seed=0,
                     model_dir="x", text_features=True)
