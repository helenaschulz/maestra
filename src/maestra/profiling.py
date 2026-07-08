"""Deterministic column profiling — the LLM's view of the data.

The LLM never sees raw rows. It decides the cleaning plan from this compact profile:
per-column dtype, missingness, cardinality and a few example values. Keeping the LLM's
input small and structured makes its decisions cheap, reproducible and auditable.
Datetime-like columns also carry ``period_candidates`` (F2): profile-only month/week/
day-of-week tokens the Validation Strategist may propose for ``time_local`` before any
feature engineering has materialised those parts onto the DataFrame itself.
"""
from __future__ import annotations

import pandas as pd

# How many example values to surface per column, and how long each may be.
_MAX_EXAMPLES = 5
_MAX_EXAMPLE_LEN = 40

# A column is flagged id_like (a strong drop hint) when nearly every value is distinct
# AND it is not a continuous numeric column. High cardinality is normal for floats
# (measurements, coordinates) — only integer/text columns that are unique per row look
# like identifiers. This keeps the LLM from dropping real continuous features.
_ID_LIKE_UNIQUE_FRAC = 0.99


def _is_id_like(series: pd.Series, n_unique: int, n_rows: int) -> bool:
    if not n_rows or series.dtype.kind == "f":  # float == continuous measurement, never an id
        return False
    return n_unique / n_rows >= _ID_LIKE_UNIQUE_FRAC


# Same tolerance validate_fold_strategy uses for a proposed time_column: a datetime-like
# column may have a few unparseable values without disqualifying it.
_MAX_UNPARSEABLE_FRAC = 0.05

# format="mixed" parsing falls back to per-element dateutil parsing, which is slow on long
# free-text columns (K2's Kaggle description-style columns run 15k+ rows) — bound the probe to
# a fixed sample so profiling stays cheap regardless of row count, matching _examples' own
# bounded-sample philosophy.
_MAX_PARSE_SAMPLE = 200


def _is_datetime_like(series: pd.Series) -> bool:
    """True if ``series`` is (or parses as) a date/time axis worth offering period hints for.

    Numeric dtypes are excluded even though ``pd.to_datetime`` would happily reinterpret them
    as epoch timestamps — a plain numeric time axis (e.g. decimal years) is already handled by
    the `time` strategy and slicing it into month/week/day-of-week is meaningless.
    """
    if series.dtype.kind == "M":
        return True
    if series.dtype.kind != "O":
        return False
    sample = series.dropna()
    if sample.empty:
        return False
    if len(sample) > _MAX_PARSE_SAMPLE:
        sample = sample.sample(_MAX_PARSE_SAMPLE, random_state=0)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.isna().mean() <= _MAX_UNPARSEABLE_FRAC


def _period_candidates(col: str, series: pd.Series) -> list[str]:
    """Profile-only hints for the Validation Strategist: tokens naming a period DERIVABLE from
    a datetime-like column (e.g. ``month_of:datetime``), without materialising it onto ``df`` —
    that only happens later, and only for fold-building, via
    :func:`~maestra.validation._materialize_period` (see that function's docstring)."""
    if not _is_datetime_like(series):
        return []
    return [f"month_of:{col}", f"week_of:{col}", f"dayofweek_of:{col}"]


def _examples(series: pd.Series) -> list[str]:
    """Up to ``_MAX_EXAMPLES`` distinct non-null values, stringified and truncated.

    Examples give the LLM a feel for a column (free text vs. category vs. id) without
    leaking the full data or blowing up the prompt.
    """
    values = series.dropna().unique()[:_MAX_EXAMPLES]
    out: list[str] = []
    for v in values:
        s = str(v)
        out.append(s if len(s) <= _MAX_EXAMPLE_LEN else s[: _MAX_EXAMPLE_LEN - 3] + "...")
    return out


def profile_dataframe(df: pd.DataFrame, target: str) -> dict:
    """Build a compact, JSON-serialisable profile of ``df``.

    Args:
        df: The full dataset (features + target).
        target: Name of the target column; flagged so the LLM never plans to touch it.

    Returns:
        A dict ``{"n_rows", "target", "columns": [...]}`` where each column carries
        ``name, is_target, dtype, n_missing, missing_frac, n_unique, unique_frac,
        examples, period_candidates``. ``*_frac`` values are rounded to 3 decimals.
        ``period_candidates`` (F2) is a PROFILE-ONLY hint — tokens the Strategist may propose
        as ``period_column`` for ``time_local`` (see ``validation_strategist.py``); the
        DataFrame itself is never modified here.
    """
    n = len(df)
    columns = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        columns.append(
            {
                "name": col,
                "is_target": col == target,
                "dtype": str(s.dtype),
                "n_missing": n_missing,
                "missing_frac": round(n_missing / n, 3) if n else 0.0,
                "n_unique": n_unique,
                "unique_frac": round(n_unique / n, 3) if n else 0.0,
                "id_like": _is_id_like(s, n_unique, n),
                "examples": _examples(s),
                "period_candidates": _period_candidates(col, s),
            }
        )
    return {"n_rows": n, "target": target, "columns": columns}


# Dataset descriptions (e.g. Kaggle's data_description.txt) carry the column SEMANTICS the
# profile's statistics cannot: what a column means, its units, ordinal orders. Feeding them to
# the judgment nodes is the single cheapest quality lever on semantic-rich data (CAAFE's key
# mechanism). Capped so a book-length description cannot blow up the prompt.
_MAX_DESCRIPTION_CHARS = 4000


def description_context(text: str | None, max_chars: int = _MAX_DESCRIPTION_CHARS) -> str | None:
    """Prompt-ready block for a provider-written dataset description (None if there is none)."""
    if not text or not text.strip():
        return None
    body = text.strip()
    if len(body) > max_chars:
        body = body[:max_chars] + "\n[... truncated]"
    return (
        "Dataset description (verbatim, from the data provider — use it to understand what "
        "columns MEAN, e.g. units, ordinal orders, identifiers):\n" + body
    )
