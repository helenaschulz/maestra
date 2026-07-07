"""Maestra as an MCP tool backend for LLM frontends (Claude Desktop/Code) — the non-DS channel:
consume verdicts, never build models. Three opinionated, budget-capped tools, each wrapping
existing Maestra machinery (`audit.py`, `validation_strategist.py`, `pipeline.py`). Every return
value is a structured verdict record (a plain dict), never a model or a raw DataFrame; rejection
(too few rows, missing target, budget exceeded) is a first-class, reasoned result, not a
traceback. No selectable parameters beyond path/target/model — conservative CV settings and a
hard time budget are baked in, not exposed as knobs.

Guardrail design: every real tool call is already internally bounded by AutoGluon's own
``time_limit`` (the actual compute cost). ``_with_budget`` adds a second, best-effort wall-clock
backstop (a thread with a result timeout) against anything ELSE running long — a hung LLM call,
a network stall. Python cannot forcibly kill a thread, so on timeout the tool returns a guardrail
rejection immediately while the underlying call may keep running in the background until it
finishes on its own; this is an honest limitation, not hidden.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeoutError

import pandas as pd

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without the optional group
    raise ImportError(
        "maestra-mcp needs the optional 'mcp' dependency group: pip install 'maestra[mcp]'"
    ) from exc

mcp = FastMCP("maestra")

_MIN_ROWS = 50  # below this, any judgment (LLM or CV) is noise, not signal -- a reasoned refusal

# Per-tool wall-clock backstops (seconds). audit_csv is one LLM call + a profile (cheap);
# check_validation runs two small CVs (more real compute); feasibility runs a full conservative
# pipeline (the most expensive). Overridable by the caller for slower environments, not by the
# LLM frontend (the tool signatures below take no budget parameter — opinionated by design).
#
# check_validation/feasibility's nominal AutoGluon time (folds x time_limit) previously left
# almost no headroom under their own backstop -- found via a real rehearsal run against
# docs/examples/demo/demand.csv (~10.9k rows, 2026-07-07): check_validation actually hit its 90s
# backstop (6 fold-fits x 15s nominal = 90s already, before overhead), and feasibility finished
# at ~253s against a 300s ceiling (~16% slack). Both were tightened (less nominal AutoGluon time)
# and given real headroom (a higher backstop) so the backstop stays a genuine safety net, not a
# routine trip on ordinary real-world-sized data.
_BUDGETS = {"audit_csv": 60.0, "check_validation": 150.0, "feasibility": 360.0}
_CHECK_VALIDATION_TIME_LIMIT = 10  # seconds per fold, both CV arms
_CHECK_VALIDATION_FOLDS = 3
_FEASIBILITY_TIME_LIMIT = 45  # seconds per fold
_FEASIBILITY_FOLDS = 3


def _reject(reason: str, **extra) -> dict:
    """A structured, reasoned refusal — the project's own 'no' as a first-class result."""
    return {"verdict": "rejected", "reason": reason, **extra}


def _load_and_check(path: str, target: str) -> tuple[pd.DataFrame | None, dict | None]:
    """Shared entry guard for every tool: load the CSV, check the target exists and there are
    enough rows. Returns ``(df, None)`` on success or ``(None, rejection_dict)`` on failure."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None, _reject(f"file not found: {path}")
    except pd.errors.EmptyDataError:
        return None, _reject(f"CSV is empty: {path}")
    if target not in df.columns:
        return None, _reject(f"target column {target!r} not in {path}",
                             columns=list(df.columns))
    if len(df) < _MIN_ROWS:
        return None, _reject(
            f"only {len(df)} rows — too few for a reliable judgment (need >= {_MIN_ROWS})",
            n_rows=len(df))
    return df, None


def _with_budget(fn, *, tool: str, **kwargs) -> dict:
    """Run ``fn(**kwargs)`` under ``tool``'s wall-clock backstop (see module docstring)."""
    max_seconds = _BUDGETS[tool]
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, **kwargs)
        try:
            return future.result(timeout=max_seconds)
        except _FutureTimeoutError:
            return _reject(
                f"{tool} exceeded its {max_seconds:.0f}s budget",
                elapsed_seconds=round(time.monotonic() - start, 1))


# --- Tool 1: audit_csv -----------------------------------------------------------------

def _audit_csv(path: str, target: str, model: str) -> dict:
    df, err = _load_and_check(path, target)
    if err:
        return err
    from maestra.audit import audit, write_audit_html

    report = audit(df, target, model=model, csv=path)
    html_path = f"{path}.audit.html"
    write_audit_html(report, html_path)
    return {
        "verdict": "ok",
        "risk_level": report.risk_level,
        "fold_strategy": report.fold_strategy.get("strategy"),
        "fold_rationale": report.fold_strategy.get("rationale"),
        "target_leaks": report.target_leaks,
        "leakage_warnings": report.leakage_warnings,
        "structural_flags": {
            "id_like": report.id_like, "constant": report.constant,
            "high_missing": report.high_missing, "high_card_text": report.high_card_text,
        },
        "html_report": html_path,
    }


@mcp.tool()
def audit_csv(path: str, target: str, model: str = "gpt-4o") -> dict:
    """Run Maestra's pre-modelling data-risk audit on a CSV. Checks validation design (should
    folds be random/group/time?), scans for target leakage (LLM-flagged proxy columns PLUS a
    deterministic correlation check), and flags structural traps (id-like, constant, high-missing,
    free-text columns). Trains NO predictive model — one LLM call plus a column profile, fast and
    cheap. Use this FIRST, before asking whether a model can even be built.

    Returns a verdict record: {"verdict": "ok", "risk_level": "low"|"elevated"|"high",
    "fold_strategy": ..., "target_leaks": [...], "leakage_warnings": [...],
    "structural_flags": {...}, "html_report": "<path>"} — or {"verdict": "rejected",
    "reason": "..."} if the file/target is unusable or the audit ran too long.

    Example: audit_csv("data/churn.csv", "churned") might return
    {"verdict": "ok", "risk_level": "elevated", "fold_strategy": "group", ...}
    """
    return _with_budget(_audit_csv, tool="audit_csv", path=path, target=target, model=model)


# --- Tool 2: check_validation ------------------------------------------------------------

def _check_validation(path: str, target: str, model: str) -> dict:
    df, err = _load_and_check(path, target)
    if err:
        return err
    from maestra.profiling import profile_dataframe
    from maestra.validation import _is_classification, cross_validate
    from maestra.validation_strategist import propose_fold_strategy, validate_fold_strategy

    proposal = propose_fold_strategy(model, profile_dataframe(df, target), target)
    verified, log = validate_fold_strategy(proposal, df, target)

    if verified["strategy"] == "random":
        return {
            "verdict": "ok",
            "recommended_strategy": {"strategy": "random", "column": None},
            "rationale": verified.get("rationale"),
            "message": "No group/time structure detected — a standard random split is "
                       "representative; the naive split is not measurably optimistic.",
        }

    classification = _is_classification(df[target])
    common = dict(cleaning_plan=None, feature_plan=None, time_limit=_CHECK_VALIDATION_TIME_LIMIT,
                  n_folds=_CHECK_VALIDATION_FOLDS, seed=42, stratified=classification)
    naive_cv = cross_validate(df, target, model_dir="AutogluonModels/mcp_naive", **common)
    recommended_cv = cross_validate(
        df, target, model_dir="AutogluonModels/mcp_recommended",
        group_column=verified.get("group_column"), time_column=verified.get("time_column"),
        period_column=verified.get("period_column"), **common)

    # Signed so gap > 0 always means "the naive split reports an optimistic (too favourable)
    # score", regardless of whether the metric is higher-is-better (accuracy) or an error
    # (rmse) — the same convention used throughout the project's own ledger (docs/RESULTS.md).
    sign = 1.0 if naive_cv.greater_is_better else -1.0
    gap = sign * (naive_cv.mean - recommended_cv.mean)
    direction = ("optimistic (dangerous)" if gap > 1e-9 else
                "pessimistic (safe)" if gap < -1e-9 else "negligible")

    return {
        "verdict": "ok",
        "recommended_strategy": {
            "strategy": verified["strategy"],
            "column": (verified.get("group_column") or verified.get("time_column")
                      or verified.get("period_column")),
        },
        "rationale": verified.get("rationale"),
        "naive_cv": {"metric": naive_cv.eval_metric, "mean": naive_cv.mean},
        "recommended_cv": {"metric": recommended_cv.eval_metric, "mean": recommended_cv.mean},
        "optimism_gap": round(gap, 4),
        "direction": direction,
        "log": log,
    }


@mcp.tool()
def check_validation(path: str, target: str, model: str = "gpt-4o") -> dict:
    """Recommend how cross-validation folds must be built for a CSV, and MEASURE how optimistic
    a naive random split actually is — never just assert it. Runs the Validation Strategist
    (reads column semantics for group/time structure), then, only when structure is found, two
    small paired cross-validations (random vs. the recommended fold strategy, same data, same
    seed) to quantify the gap. A negligible gap or no detected structure is reported as such, not
    hidden.

    Returns {"verdict": "ok", "recommended_strategy": {"strategy": "group"|"time"|"time_local"|
    "random", "column": "..."|None}, "rationale": "...", "naive_cv": {...}, "recommended_cv":
    {...}, "optimism_gap": <float>, "direction": "optimistic (dangerous)"|"pessimistic (safe)"|
    "negligible"} — or {"verdict": "rejected", "reason": "..."}.

    Example: check_validation("data/patients.csv", "readmitted") might return
    {"verdict": "ok", "recommended_strategy": {"strategy": "group", "column": "patient_id"},
    "optimism_gap": 0.42, "direction": "optimistic (dangerous)", ...}
    """
    return _with_budget(_check_validation, tool="check_validation",
                       path=path, target=target, model=model)


# --- Tool 3: feasibility -----------------------------------------------------------------

def _feasibility(path: str, target: str, model: str) -> dict:
    df, err = _load_and_check(path, target)
    if err:
        return err
    from maestra.audit import audit
    from maestra.pipeline import run_pipeline

    # Opinionated, conservative, fixed flags: leakage-free CV (the trustworthy path) + the
    # Validation Strategist on, a short hard time budget. No hybrid/text-features/ordinal — those
    # FE lanes are measured-null and frozen project-wide (docs/RESULTS.md), not worth their cost
    # in a tool meant to answer FAST, not to search for the best possible model.
    result = run_pipeline(
        df, target, model=model, test_size=0.2, time_limit=_FEASIBILITY_TIME_LIMIT, seed=42,
        model_dir="AutogluonModels/mcp_feasibility", cv_folds=_FEASIBILITY_FOLDS,
        cv_time_limit=_FEASIBILITY_TIME_LIMIT, fold_advisor=True)
    risk = audit(df, target, model=model, csv=path)
    cv = result.cv

    drivers: list[dict] = []
    predictor = getattr(result.training, "predictor", None)
    if predictor is not None:
        try:
            # feature_stage="transformed" scores AutoGluon's OWN post-processed feature names
            # (e.g. "datetime_hour") using its internal validation split -- no external `data`
            # needed. feature_stage="original" (the default) demands a dataset matching Maestra's
            # cleaned+engineered schema, which isn't available here (PipelineResult doesn't expose
            # the transformed frame); found via a real rehearsal run against docs/examples/demo
            # (2026-07-07), where passing the raw input df raised a KeyError on AutoGluon's own
            # derived columns.
            imp = predictor.feature_importance(
                feature_stage="transformed", subsample_size=2000, num_shuffle_sets=1,
                time_limit=30, silent=True)
            drivers = [{"feature": name, "importance": round(float(row["importance"]), 4)}
                       for name, row in imp.head(5).iterrows()]
        except Exception:  # noqa: BLE001 - importance is a bonus, never block the verdict on it
            drivers = []

    return {
        "verdict": "ok" if cv is not None else "rejected",
        "achievable_quality": ({"metric": cv.eval_metric, "mean": cv.mean, "std": cv.std}
                               if cv is not None else None),
        "fold_strategy": (result.fold_strategy or {}).get("strategy"),
        "strongest_drivers": drivers,
        "risk_level": risk.risk_level,
        "biggest_risks": {
            "target_leaks": risk.target_leaks,
            "leakage_warnings": risk.leakage_warnings,
        },
        "reason": None if cv is not None else "the pipeline produced no cross-validation estimate",
    }


@mcp.tool()
def feasibility(path: str, target: str, model: str = "gpt-4o") -> dict:
    """Ask whether a CSV can support predicting ``target`` at all, and how well. Runs one
    internal, conservative Maestra pipeline (leakage-free CV, Validation Strategist on, a fixed
    hard time budget — no tunable knobs). The return is the ANSWER, not the model: achievable
    quality in the target's own metric, the strongest engine-reported feature drivers, and the
    biggest data risks (from the same audit as `audit_csv`). On an intractable setup (too few
    rows, no usable target), returns a structured rejection with reasoning, never a traceback.

    Returns {"verdict": "ok", "achievable_quality": {"metric": ..., "mean": ..., "std": ...},
    "fold_strategy": ..., "strongest_drivers": [{"feature": ..., "importance": ...}, ...],
    "risk_level": ..., "biggest_risks": {...}} — or {"verdict": "rejected", "reason": "..."}.

    Example: feasibility("data/churn.csv", "churned") might return
    {"verdict": "ok", "achievable_quality": {"metric": "accuracy", "mean": 0.87, "std": 0.01},
    "strongest_drivers": [{"feature": "tenure_months", "importance": 0.31}, ...], ...}
    """
    return _with_budget(_feasibility, tool="feasibility", path=path, target=target, model=model)


def main() -> None:
    """Entry point for the ``maestra-mcp`` console script — starts the server over stdio.

    Loads ``.env`` first, same as every other entry point (`cli.py`, `audit.py`): an MCP
    frontend launches this as a bare subprocess, not through a shell that already sourced it.
    """
    from maestra.config import load_dotenv

    load_dotenv()
    mcp.run()


if __name__ == "__main__":
    main()
