"""Free-text featurization (M10) — the LLM reads sample text and writes extraction code.

AutoGluon already handles free-text columns with n-gram vectorisation, so raw token counts are
covered ground. What the engine cannot do is *semantic* extraction: knowing that "granite
countertops" and "stainless appliances" both signal a renovated kitchen, that "2 car garage"
contains a number worth parsing, or that an all-caps title means something different from a
sentence-case one. The hypothesis under test: LLM-written semantic extractors beat (or add to)
the engine's n-grams. The negative feature-engineering results (structured lanes) do not cover
this lane — it is the one place the FE thesis is still open.

Design constraints, in-frame by construction:

* **No per-row LLM calls.** The LLM writes deterministic ``fit``/``transform`` code ONCE; the
  code runs in the same sandbox as every other generated feature (network blocked, secrets
  stripped, import whitelist). Inference cost is zero at prediction time.
* **Same arbiter.** Candidates go through the same greedy CV gate (``select_features``,
  paired 2-SEM rule) as the structured hybrid lane — a semantic extractor is kept only if it
  beats the n-gram baseline beyond fold noise. The thesis is falsifiable per candidate.
* **The LLM must SEE the text.** The column profile carries only lengths and counts; this
  module builds a dedicated text profile with verbatim (truncated) sample values, because a
  keyword extractor written blind is noise.
"""
from __future__ import annotations

import json

import pandas as pd

from maestra.hybrid_features import FEATURE_CODE_SCHEMA, GeneratedFeature
from maestra.llm import call_structured

# A column is "free text" when it is string-typed, mostly distinct (not categorical levels)
# and long enough that n-grams/semantics plausibly matter. Conservative thresholds: false
# negatives only cost a missed lane, false positives waste LLM candidates on id-like columns.
_MIN_AVG_LEN = 25
_MIN_UNIQUE_FRAC = 0.5
_MIN_NON_NULL = 30

_N_EXAMPLES = 6
_EXAMPLE_CHARS = 160


def detect_text_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Deterministically find free-text feature columns (never the target).

    String-typed, at least ``_MIN_NON_NULL`` non-null values, mean length >=
    ``_MIN_AVG_LEN`` and mostly distinct (unique fraction >= ``_MIN_UNIQUE_FRAC`` —
    separates prose from categorical levels like "Excellent"/"Good").
    """
    cols = []
    for col in df.columns:
        if col == target:
            continue
        s = df[col].dropna()
        if len(s) < _MIN_NON_NULL:
            continue
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        s = s.astype(str)
        if float(s.str.len().mean()) < _MIN_AVG_LEN:
            continue
        if s.nunique() / len(s) < _MIN_UNIQUE_FRAC:
            continue
        cols.append(col)
    return cols


def text_profile(df: pd.DataFrame, columns: list[str]) -> dict:
    """Per-column text summary WITH verbatim sample values (truncated).

    Examples are the first ``_N_EXAMPLES`` distinct values in order of appearance —
    deterministic, so the same data yields the same prompt.
    """
    out = {}
    for col in columns:
        s = df[col].dropna().astype(str)
        lengths = s.str.len()
        out[col] = {
            "n_non_null": int(len(s)),
            "n_unique": int(s.nunique()),
            "len_mean": round(float(lengths.mean()), 1),
            "len_max": int(lengths.max()),
            "examples": [v[:_EXAMPLE_CHARS] for v in s.drop_duplicates().head(_N_EXAMPLES)],
        }
    return out


_SYSTEM_PROMPT = (
    "You write Python feature-extraction code for FREE-TEXT columns of a tabular ML problem. "
    "For each feature define EXACTLY two functions:\n"
    "  def fit(train_df): -> params      # may use the target column (train ONLY)\n"
    "  def transform(df, params): -> pandas.Series of length len(df)\n"
    "RULES: transform receives df WITHOUT the target column and must NOT access it. Use only "
    "pandas (as pd), numpy (as np), math and re; no other imports. No file or network access "
    "(the sandbox blocks both). The output of transform must be NUMERIC and each row's value "
    "must depend only on that row's text (no batch statistics in transform — vocabulary or "
    "thresholds belong in fit's params). Guard against missing values (fillna('') first).\n\n"
    "IMPORTANT — the engine already builds n-gram features from raw text. Do NOT propose "
    "plain token/word counts, text length, or single-keyword flags an n-gram model already "
    "represents. Propose features that need UNDERSTANDING of the domain, e.g.:\n"
    "* semantic keyword GROUPS (many surface forms, one concept: luxury terms, damage terms, "
    "  urgency markers) scored as one signal;\n"
    "* structured values parsed out of prose with re (counts like '3 bed', units, years, "
    "  prices, percentages) returned as numbers;\n"
    "* style/register signals (all-caps ratio, exclamation density, digit density) when they "
    "  plausibly relate to the target.\n"
    "Every candidate must state in 'idea' why an n-gram model would NOT already capture it. "
    "Propose only features that plausibly carry signal for THIS target; fewer good candidates "
    "beat many weak ones."
)


def propose_text_feature_code(
    model: str,
    df: pd.DataFrame,
    target: str,
    text_columns: list[str],
    research_context: str | None = None,
    max_candidates: int = 5,
) -> list[GeneratedFeature]:
    """Ask the LLM for text-extraction feature code; the prompt shows real sample values.

    Returns ``GeneratedFeature`` items with ``source="text"`` so the gate's provenance
    records distinguish this lane from the structured hybrid lane.
    """
    user_prompt = (
        f"Target column: {target}\n"
        f"Free-text columns with sample values (JSON):\n"
        f"{json.dumps(text_profile(df, text_columns), ensure_ascii=False, indent=2)}\n\n"
        f"Generate at most {max_candidates} feature candidates."
    )
    if research_context:
        user_prompt += "\n\n" + research_context
    out = call_structured(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_name="generate_text_feature_code",
        tool_description="Generate deterministic text-extraction feature code (fit/transform).",
        parameters_schema=FEATURE_CODE_SCHEMA,
    )
    features = []
    for item in out.get("features", [])[:max_candidates]:
        if isinstance(item, dict) and item.get("name") and item.get("code"):
            features.append(
                GeneratedFeature(item["name"], item.get("idea", ""), item["code"], source="text"))
    return features
