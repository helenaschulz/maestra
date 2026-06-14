"""Orchestrator: die schlichte Dirigent-Schleife. Eine Funktion pro Schritt, kein Framework.

  CSV laden -> profilieren -> LLM-Cleaning-Plan -> Plan anwenden -> trainieren -> Metriken.

Das LLM entscheidet (Cleaning-Plan als strukturiertes JSON). Gerechnet wird nur in
profiling/cleaning (deterministisch) und AutoGluon (Modelle/Metriken).
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from automl import split, train_and_evaluate
from cleaning import apply_cleaning_plan, propose_cleaning_plan
from profiling import profile_dataframe


def load_dotenv(path: str = ".env") -> None:
    """Minimaler .env-Loader: KEY=VALUE-Zeilen in os.environ, ohne Extra-Dependency."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def main() -> None:
    p = argparse.ArgumentParser(description="Agentisches AutoML: LLM-Cleaning + AutoGluon.")
    p.add_argument("--csv", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--model", default=os.environ.get("AUTOML_MODEL", "gpt-4o"),
                   help="LiteLLM-Modell-String (default gpt-4o oder $AUTOML_MODEL)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--time-limit", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-dir", default="AutogluonModels")
    p.add_argument("--no-llm", action="store_true", help="Cleaning-Schritt ueberspringen (Baseline)")
    args = p.parse_args()

    load_dotenv()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise SystemExit(f"Zielspalte '{args.target}' nicht im CSV. Spalten: {list(df.columns)}")
    print(f"Geladen: {args.csv}  Zeilen={len(df)}  Spalten={len(df.columns)}")

    if args.no_llm:
        print("\n[--no-llm] Cleaning uebersprungen.")
        clean = df
    else:
        profile = profile_dataframe(df, args.target)
        print(f"\n=== LLM-Cleaning-Plan ({args.model}) ===")
        plan = propose_cleaning_plan(args.model, profile, args.target)
        print(json.dumps(plan, ensure_ascii=False, indent=2))

        clean, log = apply_cleaning_plan(df, plan, args.target)
        print("\n=== Angewendet ===")
        for line in log:
            print(f"  {line}")
        print(f"Spalten nach Cleaning: {len(clean.columns)} (vorher {len(df.columns)})")

    train, test = split(clean, args.test_size, args.seed)
    print(f"\nSplit: train={len(train)}  holdout={len(test)}")
    train_and_evaluate(train, test, args.target, args.time_limit, args.model_dir)


if __name__ == "__main__":
    main()
