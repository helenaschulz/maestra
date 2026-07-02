"""Feature engineering: the LLM proposes, deterministic code computes — same contract as
cleaning.

The LLM picks new features from a FIXED vocabulary (date parts, quantile binning,
log-transform, ratio, difference). Application goes through the same fit/transform split
as :class:`~maestra.cleaning.CleaningTransform`: anything that learns a statistic (here,
the quantile bin edges) is fitted on **train only** and replayed unchanged on the holdout
and test sets. No LLM-generated code is executed; the applier is defensive and never
crashes on a malformed or hallucinated plan.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from maestra.llm import call_structured

_DATE_PARTS = ["year", "month", "weekday"]
#: The fixed feature-engineering vocabulary. Mirrored in ``FE_SCHEMA``.
OPS = ["date_parts", "bin", "log_transform", "ratio", "difference"]

FE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": OPS},
                    "column": {"type": "string"},
                    "parts": {"type": "array", "items": {"type": "string", "enum": _DATE_PARTS}},
                    "n_bins": {"type": "integer"},
                    "numerator": {"type": "string"},
                    "denominator": {"type": "string"},
                    "left": {"type": "string"},
                    "right": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["op", "reason"],
            },
        },
        "overall_rationale": {"type": "string"},
    },
    "required": ["features", "overall_rationale"],
}

_SYSTEM_PROMPT = (
    "You are an experienced data scientist proposing NEW features for an AutoML training "
    "run (AutoGluon). You are given a column profile of already-cleaned training data. "
    "Choose sensible transformations from the FIXED vocabulary: "
    "date_parts (date -> year/month/weekday), bin (a numeric column into n_bins "
    "quantile-based classes), log_transform (log1p of a NON-NEGATIVE, skewed numeric "
    "column), ratio (ratio of two numeric columns), difference (difference of two numeric "
    "columns). Only numeric columns for bin/log/ratio/difference; date_parts only for date "
    "columns. NEVER use the target column. Be conservative -- propose only features that "
    "plausibly carry signal. AutoGluon handles encoding/scaling itself."
)


def propose_feature_plan(
    model: str, profile: dict, target: str, research_context: str | None = None
) -> dict:
    """Ask the LLM for a feature-engineering plan for the (cleaned) train profile.

    ``research_context`` (optional) carries non-binding strategy hypotheses from the
    research node."""
    user_prompt = (
        f"Target column: {target}\n"
        f"Column profile (JSON):\n{json.dumps(profile, ensure_ascii=False, indent=2)}"
    )
    if research_context:
        user_prompt += "\n\n" + research_context
    return call_structured(
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_name="feature_plan",
        tool_description="Strukturierter Feature-Engineering-Plan aus festem Vokabular.",
        parameters_schema=FE_SCHEMA,
    )


# --- vocabulary helpers -------------------------------------------------------------

def _required_cols(op: dict) -> list:
    k = op.get("op")
    if k in ("date_parts", "bin", "log_transform"):
        return [op.get("column")]
    if k == "ratio":
        return [op.get("numerator"), op.get("denominator")]
    if k == "difference":
        return [op.get("left"), op.get("right")]
    return []


def _new_cols(op: dict) -> list[str]:
    k = op.get("op")
    if k == "date_parts":
        return [f"{op['column']}_{p}" for p in op.get("parts", [])]
    if k == "bin":
        return [f"{op['column']}_bin"]
    if k == "log_transform":
        return [f"{op['column']}_log"]
    if k == "ratio":
        return [f"{op['numerator']}_per_{op['denominator']}"]
    if k == "difference":
        return [f"{op['left']}_minus_{op['right']}"]
    return []


def _bin_edges(series: pd.Series, n_bins: int) -> list[float] | None:
    """Quantile bin edges fitted on ``series``; outer edges are ±inf so out-of-range
    holdout/test values still bin. Returns None if there aren't enough distinct edges."""
    values = series.dropna()
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1))).astype(float)
    if len(edges) < 3:  # fewer than 2 usable bins
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    return list(edges)


def _date_part(dt: pd.Series, part: str) -> pd.Series:
    return {"year": dt.dt.year, "month": dt.dt.month, "weekday": dt.dt.weekday}[part]


def _apply_op(df: pd.DataFrame, op: dict) -> pd.DataFrame:
    """Apply one fitted op to ``df`` (mutates a copy the caller already made)."""
    k = op["op"]
    if k == "date_parts":
        col = op["column"]
        dt = pd.to_datetime(df[col], errors="coerce")
        for part in op["parts"]:
            df[f"{col}_{part}"] = _date_part(dt, part)
        df = df.drop(columns=[col])
    elif k == "bin":
        col = op["column"]
        df[f"{col}_bin"] = pd.cut(df[col], bins=op["_edges"], labels=False, include_lowest=True)
    elif k == "log_transform":
        col = op["column"]
        df[f"{col}_log"] = np.log1p(df[col])
    elif k == "ratio":
        n, d = op["numerator"], op["denominator"]
        df[f"{n}_per_{d}"] = df[n] / df[d].replace(0, np.nan)
    elif k == "difference":
        left, right = op["left"], op["right"]
        df[f"{left}_minus_{right}"] = df[left] - df[right]
    return df


@dataclass
class FeatureTransform:
    """A fitted feature-engineering step: an ordered list of validated ops (``bin`` ops
    carry their train-fitted ``_edges``). Fitted on train, replayed on any DataFrame."""

    operations: list[dict]
    log: list[str] = field(default_factory=list)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for op in self.operations:
            df = _apply_op(df, op)
        return df


def _fit_op(work: pd.DataFrame, item: dict, target: str) -> tuple[bool, dict | None, str]:
    """Validate one proposed op against the current frame and fit its state (bin edges)."""
    k = item.get("op")
    if k not in OPS:
        return False, None, f"unknown op '{k}'"
    req = _required_cols(item)
    if any(c is None for c in req):
        return False, None, "missing column specification"
    if target in req:
        return False, None, "uses the target column"
    missing = [c for c in req if c not in work.columns]
    if missing:
        return False, None, f"column(s) not present: {missing}"
    if k in ("bin", "log_transform", "ratio", "difference"):
        non_numeric = [c for c in req if work[c].dtype.kind not in "iuf"]
        if non_numeric:
            return False, None, f"not numeric: {non_numeric}"
    if k == "log_transform" and (work[item["column"]] < 0).any():
        return False, None, "negative values (log1p needs >= 0)"
    if k == "date_parts":
        parts = item.get("parts") or []
        if not parts or any(p not in _DATE_PARTS for p in parts):
            return False, None, "invalid/empty parts"
        if pd.to_datetime(work[item["column"]], errors="coerce").notna().sum() == 0:
            return False, None, "not parseable as a date"
    clash = [c for c in _new_cols(item) if c in work.columns]
    if clash:
        return False, None, f"new column already exists: {clash}"

    fitted = {"op": k, "reason": item.get("reason", "")}
    for key in ("column", "parts", "numerator", "denominator", "left", "right"):
        if item.get(key) is not None:
            fitted[key] = item[key]
    if k == "bin":
        n_bins = item.get("n_bins") or 4
        edges = _bin_edges(work[item["column"]], n_bins)
        if edges is None:
            return False, None, "zu wenige eindeutige Bin-Grenzen"
        fitted["_edges"], fitted["n_bins"] = edges, n_bins
    return True, fitted, ""


def fit_feature_plan(train: pd.DataFrame, plan: dict, target: str) -> FeatureTransform:
    """Validate + fit a feature plan on ``train`` only, returning a replayable transform.

    Ops are validated against the *evolving* schema (so an op that uses a column an earlier
    op dropped is skipped), and bin edges are computed from train — making the transform
    leakage-free when applied to the holdout/test sets.
    """
    work = train.copy()
    operations: list[dict] = []
    log: list[str] = []
    for item in plan.get("features", []):
        ok, fitted, message = _fit_op(work, item, target)
        if not ok:
            log.append(f"SKIP {item.get('op')}: {message}")
            continue
        work = _apply_op(work, fitted)
        log.append(f"FEATURE [{fitted['op']}] -> {', '.join(_new_cols(fitted))} -- {fitted['reason']}")
        operations.append(fitted)
    return FeatureTransform(operations=operations, log=log)
