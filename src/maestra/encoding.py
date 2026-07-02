"""Ordinal encoding — the one feature transformation that INJECTS information.

AutoGluon handles unordered categoricals natively, and its trees recover most arithmetic
feature combinations on their own (measured: the hybrid layer's generated features never beat
it). Ordinal *order* is different: `KitchenQual` is Po < Fa < TA < Gd < Ex, an ordering the
trees cannot infer from unordered category labels but an LLM knows from the column's meaning.
Mapping such a column to its rank injects knowledge the engine does not have — so this is the
FE type with a real chance of beating the baseline (M2 in STRATEGY.md).

The LLM proposes the order (ideally reading the dataset description); deterministic code applies
it. The mapping is a pure function of the LLM-provided order — it uses neither the target nor any
data statistic — so it is trivially leakage-free and identical on train, holdout and test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from maestra.llm import call_structured

ORDINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "encodings": {
            "type": "array",
            "description": "Ordinal columns only — categoricals with a genuine order.",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "order": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The category values from WORST/LOWEST to BEST/HIGHEST.",
                    },
                    "reason": {"type": "string"},
                },
                "required": ["column", "order", "reason"],
            },
        }
    },
    "required": ["encodings"],
}

_SYSTEM_PROMPT = (
    "You map ORDINAL categorical columns to a numeric rank for a tabular ML task. An ordinal "
    "column has a genuine order (quality/condition ratings like Ex>Gd>TA>Fa>Po; sizes S<M<L; "
    "education levels; frequencies never<rarely<often<always). Gradient-boosted trees cannot "
    "recover this order from unordered labels, so encoding it injects real information. For each "
    "TRULY ordinal column give its values from WORST/LOWEST to BEST/HIGHEST. Be strict: do NOT "
    "encode nominal columns (neighbourhood, colour, brand, zip) — they have no order and forcing "
    "one hurts. Prefer the dataset description when present; it usually states the levels. Only "
    "name columns present in the profile. Return an empty list if none are ordinal."
)


@dataclass
class OrdinalEncoding:
    """A fitted set of ordinal maps. ``transform`` replaces each column with its integer rank
    (values outside the given order become NaN, which AutoGluon handles as missing)."""

    maps: dict[str, dict] = field(default_factory=dict)  # column -> {value: rank}
    log: list[str] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)  # provenance: column/n_levels/coverage/reason

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.maps:
            return df
        out = df.copy()
        for col, mapping in self.maps.items():
            if col in out.columns:
                # float, never int: an unseen/absent value must stay NaN so AutoGluon treats it as
                # missing (median-imputed), not as the lowest rank. As ints, a fully-covered train
                # column becomes int-without-nulls and AutoGluon imputes inference nulls to 0 (the
                # WORST level — e.g. "no pool" would read as "worst pool"), a systematic bias.
                out[col] = out[col].astype("object").map(mapping).astype("float")
        return out


def propose_ordinal_encodings(model: str, profile: dict, target: str, context: str | None = None) -> dict:
    """Ask the LLM which columns are ordinal and in what order. Matches ORDINAL_SCHEMA."""
    user_prompt = (
        f"Target column: {target}\n"
        f"Column profile (JSON):\n{json.dumps(profile, ensure_ascii=False, indent=2)}"
    )
    if context:
        user_prompt += "\n\n" + context
    return call_structured(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_name="propose_ordinal_encodings",
        tool_description="Map ordinal categorical columns to a worst->best rank order.",
        parameters_schema=ORDINAL_SCHEMA,
    )


def fit_ordinal_encodings(df: pd.DataFrame, encodings: list[dict], target: str) -> OrdinalEncoding:
    """Verify each proposed encoding against the data and build the maps. Defensive: a column
    that does not exist, is the target, is already numeric, has a degenerate order, or whose
    order matches almost none of the observed values is skipped with a log line — the LLM can
    inform the encoding but never corrupt it."""
    enc = OrdinalEncoding()
    for item in encodings or []:
        col = item.get("column")
        order = item.get("order") or []
        reason = item.get("reason", "")
        if not col or col not in df.columns:
            enc.log.append(f"SKIP ordinal {col!r}: column not present")
            continue
        if col == target:
            enc.log.append(f"SKIP ordinal {col!r}: is the target column")
            continue
        if df[col].dtype.kind in "iuf":
            enc.log.append(f"SKIP ordinal {col!r}: already numeric")
            continue
        if len(order) < 2 or len(set(order)) != len(order):
            enc.log.append(f"SKIP ordinal {col!r}: order must be >= 2 distinct values")
            continue
        mapping = {str(v): i for i, v in enumerate(order)}
        observed = df[col].dropna().astype(str)
        coverage = observed.isin(mapping).mean() if len(observed) else 0.0
        if coverage < 0.5:  # the proposed order barely matches reality -> likely hallucinated
            enc.log.append(f"SKIP ordinal {col!r}: order matches only {coverage:.0%} of values")
            continue
        enc.maps[col] = mapping
        enc.records.append({"column": col, "n_levels": len(order),
                            "coverage": round(float(coverage), 3), "reason": reason})
        enc.log.append(f"ORDINAL {col!r} -> rank[0..{len(order) - 1}] "
                       f"({coverage:.0%} covered) -- {reason}")
    return enc
