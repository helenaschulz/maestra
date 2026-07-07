"""Backtest audit (F1) — temporal-leakage detection and backtest-design review for an EXISTING
forecasting setup. No model building: the deliverable is a verdict, not a deployable model —
the diagnostic fits used here (a quick naive-vs-corrected backtest comparison, an adversarial
train/test-boundary classifier) are the same kind of measurement machinery `audit.py` already
uses for its own "trains no model [for the user]" audit, not the forecasting model itself.

Same three-part contract as every judgment node in this project:

* **propose** — the LLM reads column semantics and flags candidates (future-leaking columns);
  one structured call, temperature 0.
* **verify** — deterministic code corroborates each candidate (a correlation-with-target check,
  the same evidence `audit.py::_target_leak_scan` uses).
* **measure** — the split-design question ("how much does a naive backtest overstate quality?")
  is answered by :func:`quantify_backtest_lie`, comparing a naive backtest against an embargoed
  one over several rolling origins, with the SAME Nadeau-Bengio-corrected paired test
  (`validation.py::paired_delta_test`) every other gate in this project uses. The LLM never
  decides the verdict — only the naive-vs-corrected NUMBER does.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from maestra.llm import call_structured
from maestra.validation import paired_delta_mde, paired_delta_test

FUTURE_FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "future_leaking_columns": {
            "type": "array",
            "description": "Columns whose value would NOT actually be known at the moment a "
                           "forecast is made -- computed from, or only recorded after, the "
                           "outcome (e.g. 'actual_ship_date' when forecasting demand, a "
                           "same-day total that already includes the target).",
            "items": {
                "type": "object",
                "properties": {"column": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["column", "reason"],
            },
        },
    },
    "required": ["future_leaking_columns"],
}

_SYSTEM_PROMPT = (
    "You are a senior data scientist reviewing a FORECASTING setup for backtest leakage. A "
    "column leaks the future if its value would not actually be knowable at the moment a real "
    "forecast is made -- e.g. a same-period actual that already embeds the outcome, a status "
    "field only set after the event, a column literally named for a later date than the "
    "forecast origin. Do NOT flag ordinary predictive features (price, weather, calendar, "
    "promotions known in advance) even if they correlate with the target -- correlation is not "
    "the test, AVAILABILITY AT FORECAST TIME is. Only name columns that exist in the profile. "
    "If nothing looks like a future leak, return an empty list -- most forecasting setups have "
    "none, and a false alarm here is as costly as a miss."
)


def propose_future_features(model: str, profile: dict, target: str, time_column: str,
                            context: str | None = None) -> dict:
    """Ask the LLM which columns look like they would leak future information. Returns a dict
    matching :data:`FUTURE_FEATURE_SCHEMA`."""
    import json

    user_prompt = (
        f"Target column: {target}\nTime/ordering column: {time_column}\n"
        f"Column profile (JSON):\n{json.dumps(profile, ensure_ascii=False, indent=2)}"
    )
    if context:
        user_prompt += "\n\n" + context
    return call_structured(
        model=model, system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt,
        tool_name="propose_future_features",
        tool_description="Flag columns that would not be known at real forecast time.",
        parameters_schema=FUTURE_FEATURE_SCHEMA,
    )


def _correlation_with_target(df: pd.DataFrame, target: str, column: str) -> float | None:
    """|correlation| of a numeric column with the target — the same corroborating evidence
    `audit.py::_target_leak_scan` uses. Returns None when not computable (non-numeric either
    side, or a degenerate column)."""
    y = df[target]
    if y.dtype.kind not in "iufb":
        if y.dropna().nunique() != 2:
            return None
        y = (y == y.dropna().unique()[0]).astype(float)
    x = df[column]
    if x.dtype.kind not in "iufb":
        return None
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2:
        return None
    corr = np.corrcoef(x[valid].astype(float), y[valid].astype(float))[0, 1]
    return None if np.isnan(corr) else float(abs(corr))


def check_future_features(df: pd.DataFrame, target: str, proposal: dict) -> list[dict]:
    """Deterministically corroborate the LLM's future-leak candidates: attach each flagged
    column's |correlation| with the target (when computable) as supporting evidence. A
    nonexistent column is dropped, never crashed on — the LLM cannot break the audit."""
    findings = []
    for item in proposal.get("future_leaking_columns", []) or []:
        col = item.get("column")
        if not col or col not in df.columns or col == target:
            continue
        findings.append({
            "column": col, "reason": item.get("reason", ""),
            "correlation_with_target": _correlation_with_target(df, target, col),
        })
    return findings


def _time_sorted_order(df: pd.DataFrame, time_column: str) -> np.ndarray:
    values = df[time_column]
    if values.dtype.kind not in "iufM":
        values = pd.to_datetime(values, errors="coerce", format="mixed")
    return np.argsort(values.to_numpy(), kind="stable")


def rolling_origins(n_rows: int, *, n_origins: int = 3, test_frac: float = 0.1,
                    embargo_frac: float = 0.05) -> list[tuple[slice, slice, slice]]:
    """Positions (into a time-sorted order) for ``n_origins`` expanding-window backtest
    replications, tiling the back of the timeline. Each origin yields
    ``(naive_train, embargo_train, test)`` position slices: ``naive_train`` runs right up to the
    test block (no gap — the naive, leakage-prone shape); ``embargo_train`` stops
    ``embargo_frac`` of the data earlier (the honest, gapped shape). Both share the same test
    block, so their scores are PAIRED per origin.
    """
    test_len = max(1, int(n_rows * test_frac))
    embargo_len = max(0, int(n_rows * embargo_frac))
    region_start = max(0, n_rows - n_origins * test_len)
    origins = []
    for i in range(n_origins):
        test_start = region_start + i * test_len
        test_end = min(n_rows, test_start + test_len)
        if test_end <= test_start:
            continue
        embargo_end = max(0, test_start - embargo_len)
        if embargo_end < 10:  # too little train data left to fit anything meaningful
            continue
        origins.append((slice(0, test_start), slice(0, embargo_end), slice(test_start, test_end)))
    return origins


def _fit_and_score(train: pd.DataFrame, test: pd.DataFrame, target: str, time_limit: int,
                   model_dir: str, eval_metric: str | None):
    """One holdout fit+score — the same primitive `engine.py::train_and_evaluate` provides,
    called directly here (no cleaning/FE layer: a backtest audit measures the SPLIT, not the
    feature pipeline)."""
    from maestra.engine import train_and_evaluate

    result = train_and_evaluate(train, test, target, time_limit, model_dir, eval_metric=eval_metric)
    metric = eval_metric or result.eval_metric
    score = result.metrics.get(metric)
    if score is None:
        score = next(iter(result.metrics.values()))
    return float(score), result.eval_metric, getattr(result.predictor.eval_metric, "greater_is_better", True)


def quantify_backtest_lie(naive_scores: list[float], corrected_scores: list[float], *,
                          greater_is_better: bool) -> dict:
    """The core F1 measurement primitive: how much does a naive (no-embargo) backtest overstate
    quality versus a corrected (embargoed) one, across paired rolling origins?

    Signed so a positive ``mean_gap`` always means "the naive backtest is optimistic (dangerous)"
    — the same convention `mcp_server.py::check_validation`'s optimism gap uses. Uses the
    project's own Nadeau-Bengio-corrected paired test (N1) to say whether the gap clears noise,
    with the minimum-detectable-effect reported alongside so a small/negligible gap reads as "no
    effect at least this large", not a bare non-result.
    """
    if len(naive_scores) != len(corrected_scores) or not naive_scores:
        raise ValueError("naive_scores and corrected_scores must be non-empty and equal length")
    sign = 1.0 if greater_is_better else -1.0
    deltas = [sign * (n - c) for n, c in zip(naive_scores, corrected_scores)]
    n = len(deltas)
    mean_gap = float(np.mean(deltas))
    ratio = 1.0 / (n - 1) if n > 1 else 0.0
    std = float(np.std(deltas, ddof=1)) if n >= 2 else 0.0
    decided_optimistic = n >= 2 and paired_delta_test(deltas, test_train_ratio=ratio)
    decided_pessimistic = n >= 2 and paired_delta_test([-d for d in deltas], test_train_ratio=ratio)
    if decided_optimistic:
        direction = "optimistic (dangerous)"
    elif decided_pessimistic:
        direction = "pessimistic (safe)"
    else:
        direction = "undecided"
    mde = paired_delta_mde(std, n, test_train_ratio=ratio) if n >= 2 else float("inf")
    return {
        "mean_gap": mean_gap, "direction": direction, "n_origins": n, "mde": mde,
        "naive_scores": naive_scores, "corrected_scores": corrected_scores,
    }


def split_design_check(df: pd.DataFrame, target: str, time_column: str, *,
                       n_origins: int = 3, test_frac: float = 0.1, embargo_frac: float = 0.05,
                       time_limit: int = 15, model_dir: str = "AutogluonModels/backtest_audit",
                       eval_metric: str | None = None) -> dict | None:
    """Fit a naive and an embargoed backtest at several rolling origins and quantify the gap.
    Returns ``None`` if there isn't enough data for even one origin (too few rows)."""
    order = _time_sorted_order(df, time_column)
    origins = rolling_origins(len(df), n_origins=n_origins, test_frac=test_frac,
                              embargo_frac=embargo_frac)
    if not origins:
        return None
    naive_scores, corrected_scores = [], []
    metric_name, greater_is_better = eval_metric, True
    for i, (naive_train_sl, embargo_train_sl, test_sl) in enumerate(origins):
        test_rows = df.iloc[order[test_sl]]
        naive_score, metric_name, greater_is_better = _fit_and_score(
            df.iloc[order[naive_train_sl]], test_rows, target, time_limit,
            f"{model_dir}/naive_{i}", eval_metric)
        corrected_score, _, _ = _fit_and_score(
            df.iloc[order[embargo_train_sl]], test_rows, target, time_limit,
            f"{model_dir}/embargo_{i}", eval_metric or metric_name)
        naive_scores.append(naive_score)
        corrected_scores.append(corrected_score)
    lie = quantify_backtest_lie(naive_scores, corrected_scores, greater_is_better=greater_is_better)
    lie["eval_metric"] = metric_name
    return lie


def series_leak_check(df: pd.DataFrame, target: str, time_column: str, series_column: str, *,
                      test_frac: float = 0.2, time_limit: int = 15,
                      model_dir: str = "AutogluonModels/backtest_audit/series") -> float | None:
    """Does a single global model leak across SERIES at the time boundary? Splits at the naive
    (no-embargo) boundary and runs the project's existing adversarial-validation machinery on
    the two halves — an AUC near 0.5 means the "before" and "after" blocks (including their
    series membership) look interchangeable; a high AUC means the model could distinguish rows
    from either side of the boundary beyond just time itself, a warning sign for series-level
    leakage in a global model.
    """
    from maestra.validation import adversarial_validation

    order = _time_sorted_order(df, time_column)
    cut = int(len(df) * (1 - test_frac))
    if cut < 10 or len(df) - cut < 10:
        return None
    before = df.iloc[order[:cut]].drop(columns=[series_column])
    after = df.iloc[order[cut:]].drop(columns=[series_column])
    return adversarial_validation(before, after, target, cleaning_plan=None,
                                  model_dir=model_dir, time_limit=time_limit)


def target_framing_flag(model: str, profile: dict, df: pd.DataFrame, target: str,
                        context: str | None = None) -> dict:
    """Reuses `target_framing.py`'s propose+verify (M11/K1) to flag a log1p opportunity on a
    skewed count/regression target. Deliberately does NOT re-run the CV arbiter here — that
    measurement already exists and works (`pipeline.py --target-framing`); this only surfaces
    the candidate so a backtest audit's reader knows to ask for it, avoiding a second
    implementation of the same measurement loop. Verification assumes
    ``root_mean_squared_error`` (the common default for a numeric forecasting target) since,
    unlike the pipeline's own use of this check, no CV has run yet here to report AutoGluon's
    actual inferred metric.
    """
    from maestra.target_framing import propose_target_framing, validate_target_framing

    proposal = propose_target_framing(model, profile, df, target, context)
    transform, log = validate_target_framing(proposal, df, target, problem_type="regression",
                                             eval_metric="root_mean_squared_error")
    return {"proposed": proposal.get("transform", "none"), "verified": transform is not None,
           "rationale": proposal.get("rationale", ""), "log": log}


@dataclass
class BacktestAuditReport:
    csv: str
    n_rows: int
    target: str
    time_column: str
    series_column: str | None
    future_leaks: list[dict] = field(default_factory=list)
    split_design: dict | None = None
    series_leak_auc: float | None = None
    target_framing: dict | None = None

    @property
    def risk_level(self) -> str:
        """High: a future leak found, the naive backtest is a DECIDED-optimistic lie, or a
        series-leak AUC is strong (>0.75). Elevated: a series-leak AUC is moderate (0.6-0.75) --
        ambiguous, worth a second look. Otherwise low, INCLUDING an "undecided" split-design
        (no measurable gap found is the safe, not the risky, outcome -- distinct from "not
        enough data to measure at all", which returns split_design=None and is also low: a
        missing measurement isn't evidence of a problem, `feasibility`-style tools already
        surface small-n concerns separately)."""
        if self.future_leaks:
            return "high"
        if self.split_design and self.split_design["direction"] == "optimistic (dangerous)":
            return "high"
        if self.series_leak_auc is not None and self.series_leak_auc > 0.75:
            return "high"
        if self.series_leak_auc is not None and self.series_leak_auc > 0.6:
            return "elevated"
        return "low"


def audit_backtest(df: pd.DataFrame, target: str, time_column: str, *, model: str,
                   series_column: str | None = None, description: str | None = None,
                   time_limit: int = 15, csv: str = "data") -> BacktestAuditReport:
    """Produce a :class:`BacktestAuditReport` for an existing forecasting setup. One LLM call
    (future-feature semantics); the split-design measurement and the optional series-leak check
    are real, budget-bounded fits — see :func:`split_design_check`/:func:`series_leak_check`.
    """
    from maestra.profiling import description_context, profile_dataframe

    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not in data. Columns: {list(df.columns)}")
    if time_column not in df.columns:
        raise ValueError(f"Time column {time_column!r} not in data. Columns: {list(df.columns)}")

    profile = profile_dataframe(df, target)
    context = description_context(description)
    proposal = propose_future_features(model, profile, target, time_column, context)
    future_leaks = check_future_features(df, target, proposal)

    split_design = split_design_check(df, target, time_column, time_limit=time_limit)

    series_leak_auc = None
    if series_column is not None and series_column in df.columns:
        series_leak_auc = series_leak_check(df, target, time_column, series_column,
                                            time_limit=time_limit)

    framing = None
    if df[target].dtype.kind in "iuf" and float(df[target].min()) >= 0:
        framing = target_framing_flag(model, profile, df, target, context)

    return BacktestAuditReport(
        csv=csv, n_rows=len(df), target=target, time_column=time_column,
        series_column=series_column, future_leaks=future_leaks, split_design=split_design,
        series_leak_auc=series_leak_auc, target_framing=framing,
    )


def write_backtest_audit_html(report: BacktestAuditReport, path: str, *,
                              verdict_sentence: str | None = None) -> None:
    """Render the backtest audit on the shared HTML layer (P1's rendering, reused for F1) and
    write it to ``path`` — a verdict-first, dependency-free static file."""
    from maestra.dossier import render_backtest_audit
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_backtest_audit(report, verdict_sentence=verdict_sentence))
