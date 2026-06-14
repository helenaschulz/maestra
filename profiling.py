"""Deterministisches Spalten-Profil als kompakter Input fuer das LLM.

Reines Python, keine Modellrechnung. Liefert pro Spalte: dtype, Fehlend-Anteil,
Kardinalitaet und ein paar Beispielwerte -- genug, damit das LLM einen Cleaning-Plan
entscheiden kann, ohne die Rohdaten zu sehen.
"""
from __future__ import annotations

import pandas as pd


def _examples(series: pd.Series, k: int = 5) -> list[str]:
    vals = series.dropna().unique()[:k]
    out = []
    for v in vals:
        s = str(v)
        out.append(s if len(s) <= 40 else s[:37] + "...")
    return out


def profile_dataframe(df: pd.DataFrame, target: str) -> dict:
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
                "examples": _examples(s),
            }
        )
    return {"n_rows": n, "target": target, "columns": columns}
