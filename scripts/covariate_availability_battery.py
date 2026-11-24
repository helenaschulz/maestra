"""F3-1 measurement: is the per-covariate availability verdict right on real forecasting data,
and does treating a not-known-in-advance column as known-future produce a measurable,
dishonest backtest gap?

Same three F1 tasks and data as ``scripts/backtest_audit_battery.py``, but with the
competition's ``test.csv`` as the real future window (the strongest availability evidence) and
WITHOUT dropping Rossmann's ``Customers`` up front -- it is the textbook not-known-at-forecast-
time column, so here it is the positive control the verdict has to catch.

Part 1 (one LLM call per task, gpt-4o by default): ``propose_future_features`` ->
``check_covariate_availability`` against the future window. Part 2 (Rossmann only, local
AutoGluon fits): rolling-origin backtest with vs. without ``Customers`` on IDENTICAL origins,
paired with the project's NB-corrected test (``quantify_backtest_lie``): the "with" arm is what
a forecaster who treated it as known-future would report; the deployment can't have it.

    ./.venv16/bin/python scripts/covariate_availability_battery.py --out covariate_availability.jsonl
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd

from maestra.backtest_audit import (
    _time_sorted_order,
    check_covariate_availability,
    propose_future_features,
    quantify_backtest_lie,
    rolling_origins,
)
from maestra.config import load_dotenv
from maestra.engine import autogluon_version, train_and_evaluate
from maestra.profiling import profile_dataframe

TASKS = [
    dict(name="rossmann", path="data/kaggle_rossmann/train.csv",
         future="data/kaggle_rossmann/test.csv", target="Sales", time_column="Date", drop=[],
         expect={"Customers": ("leaking", "past_only"), "Promo": ("known_future",),
                 "Open": ("known_future",), "StateHoliday": ("known_future",),
                 "SchoolHoliday": ("known_future",), "DayOfWeek": ("known_future",)}),
    dict(name="walmart", path="data/kaggle_walmart/train.csv",
         future="data/kaggle_walmart/test.csv", target="Weekly_Sales", time_column="Date",
         drop=[], expect={"IsHoliday": ("known_future",)}),
    dict(name="store-sales", path="data/kaggle_store_sales/train.csv",
         future="data/kaggle_store_sales/test.csv", target="sales", time_column="date",
         drop=["id"], expect={"onpromotion": ("known_future",)}),
]


def _gap(df: pd.DataFrame, target: str, time_column: str, column: str, *, n_origins: int,
         time_limit: int) -> dict:
    """Backtest with vs. without ``column`` on identical rolling origins (no embargo -- the
    question is the column, not the gap). Positive mean_gap = 'with' looks better = the
    optimism a forecaster would report by treating ``column`` as known in advance."""
    order = _time_sorted_order(df, time_column)
    with_s, without_s, gib, metric = [], [], True, None
    for i, (train_sl, _, test_sl) in enumerate(rolling_origins(len(df), n_origins=n_origins)):
        tr, te = df.iloc[order[train_sl]], df.iloc[order[test_sl]]
        for arm, cols in (("with", None), ("without", [column])):
            a, b = (tr, te) if cols is None else (tr.drop(columns=cols), te.drop(columns=cols))
            res = train_and_evaluate(a, b, target, time_limit,
                                     f"AutogluonModels/f3_cov/{arm}_{i}")
            metric = res.eval_metric
            gib = getattr(res.predictor.eval_metric, "greater_is_better", True)
            (with_s if arm == "with" else without_s).append(float(res.metrics[metric]))
    lie = quantify_backtest_lie(with_s, without_s, greater_is_better=gib)
    return {**lie, "eval_metric": metric, "column": column}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="covariate_availability.jsonl")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--sample", type=int, default=15000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--origins", type=int, default=5)
    p.add_argument("--time-limit", type=int, default=10)
    p.add_argument("--skip-gap", action="store_true")
    args = p.parse_args()
    load_dotenv()

    for spec in TASKS:
        df = pd.read_csv(spec["path"], low_memory=False).drop(columns=spec["drop"])
        if len(df) > args.sample:
            df = df.sample(n=args.sample, random_state=args.seed)
        future = pd.read_csv(spec["future"], low_memory=False)
        proposal = propose_future_features(args.model, profile_dataframe(df, spec["target"]),
                                           spec["target"], spec["time_column"])
        cov = check_covariate_availability(df, spec["target"], spec["time_column"], proposal,
                                           future_df=future)
        checks = {c["column"]: c["availability"] in spec["expect"][c["column"]]
                  for c in cov if c["column"] in spec["expect"]}
        rec = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "kind": "f3_covariate_availability", "task": spec["name"], "model": args.model,
            "autogluon_version": autogluon_version(), "n_rows": len(df),
            "covariates": cov, "expected_ok": checks,
        }
        for c in cov:
            print(f"{spec['name']:12s} {c['column']:14s} llm={str(c['llm']):13s} "
                  f"final={c['availability']:13s} "
                  f"{'OK ' if checks.get(c['column'], True) else 'MISS'} {c['reason'][-60:]}",
                  flush=True)
        if spec["name"] == "rossmann" and not args.skip_gap:
            rec["gap_customers"] = _gap(df, spec["target"], spec["time_column"], "Customers",
                                        n_origins=args.origins, time_limit=args.time_limit)
            g = rec["gap_customers"]
            print(f"rossmann Customers gap: {g['mean_gap']:+.4g} ({g['direction']}, "
                  f"mde {g['mde']:.4g}, n={g['n_origins']})", flush=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")


if __name__ == "__main__":
    main()
