"""F1 battery: run backtest_audit.py against real forecasting competitions with Kaggle LB
history, reusing K1/K2's already-downloaded data (`scripts/kaggle_battery.py` — reused, not
duplicated: same file paths, same target/series columns already established there).

Three tasks, all local (no download needed): rossmann and walmart are Kaggle-only (already
joined for K1/K2); store-sales ("Getting Started", Favorita's grocery data) is open by default.

  Store-sales:  data/kaggle_store_sales/train.csv  target=sales      time=date  series=store_nbr
  Rossmann:     data/kaggle_rossmann/train.csv      target=Sales     time=Date  series=Store
  Walmart:      data/kaggle_walmart/train.csv       target=Weekly_Sales time=Date series=Store

Each task's raw train.csv is large (0.4M-3M rows) -- sampled down (random, matching
kaggle_battery.py's own `sample=15000` convention) so the origin fits stay fast; this trades
series-density for speed, an explicit, documented simplification, not a silent one.

    ./.venv/bin/python scripts/backtest_audit_battery.py --list
    ./.venv/bin/python scripts/backtest_audit_battery.py --task rossmann
    ./.venv/bin/python scripts/backtest_audit_battery.py --task all
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from maestra.backtest_audit import audit_backtest, write_backtest_audit_html
from maestra.config import load_dotenv

TASKS = [
    dict(name="store-sales", path="data/kaggle_store_sales/train.csv", target="sales",
        time_column="date", series_column="store_nbr",
        drop=["id"]),  # a plain row index, not a feature
    dict(name="rossmann", path="data/kaggle_rossmann/train.csv", target="Sales",
        time_column="Date", series_column="Store",
        drop=["Customers"]),  # known future leak: only known AFTER the day closes -- dropped up
        # front the same way K1 dropped bike-sharing's casual+registered, so the AUDIT's LLM-flagged
        # candidate (if any) is tested on the REMAINING columns, not this already-known one
    dict(name="walmart", path="data/kaggle_walmart/train.csv", target="Weekly_Sales",
        time_column="Date", series_column="Store", drop=[]),
]


def run_task(spec: dict, *, model: str, sample: int, seed: int, time_limit: int) -> dict:
    df = pd.read_csv(spec["path"])
    if spec["drop"]:
        df = df.drop(columns=[c for c in spec["drop"] if c in df.columns])
    if len(df) > sample:
        df = df.sample(n=sample, random_state=seed)
    report = audit_backtest(df, spec["target"], spec["time_column"], model=model,
                            series_column=spec["series_column"], time_limit=time_limit,
                            csv=spec["path"])
    html_path = f"data/backtest_audit_{spec['name']}.html"
    write_backtest_audit_html(report, html_path)
    return {
        "name": spec["name"], "n_rows_sampled": len(df), "risk_level": report.risk_level,
        "future_leaks": report.future_leaks, "split_design": report.split_design,
        "series_leak_auc": report.series_leak_auc, "target_framing": report.target_framing,
        "html_report": html_path,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", help="Task name from the catalog, or 'all'.")
    p.add_argument("--list", action="store_true", help="List catalog tasks and exit.")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--sample", type=int, default=15000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-limit", type=int, default=10, help="AutoGluon budget per origin fit.")
    args = p.parse_args(argv)

    if args.list:
        for t in TASKS:
            print(f"{t['name']:14s} target={t['target']:14s} time={t['time_column']:8s} "
                 f"series={t['series_column']}")
        return 0

    if not args.task:
        p.error("--task TASK (or --list)")
    load_dotenv()
    names = [t["name"] for t in TASKS] if args.task == "all" else [args.task]
    specs = [t for t in TASKS if t["name"] in names]
    if len(specs) != len(names):
        p.error(f"unknown task(s); choices: {[t['name'] for t in TASKS]}")

    results = []
    for spec in specs:
        print(f"=== {spec['name']} ===")
        result = run_task(spec, model=args.model, sample=args.sample, seed=args.seed,
                          time_limit=args.time_limit)
        print(json.dumps(result, indent=2, default=str))
        results.append(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
