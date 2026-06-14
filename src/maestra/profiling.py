"""Deterministic column profiling — the LLM's view of the data.

The LLM never sees raw rows. It decides the cleaning plan from this compact profile:
per-column dtype, missingness, cardinality and a few example values. Keeping the LLM's
input small and structured makes its decisions cheap, reproducible and auditable.
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
        examples``. ``*_frac`` values are rounded to 3 decimals.
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
            }
        )
    return {"n_rows": n, "target": target, "columns": columns}
