"""M2 decision experiment: does ordinal encoding beat AutoGluon?

The last open feature-engineering hypothesis. AutoGluon handles unordered categoricals natively
and its trees recover arithmetic feature combinations on their own (the hybrid layer never beat
it). Ordinal ORDER is the one thing trees cannot infer from unordered labels — `KitchenQual` is
Po < Fa < TA < Gd < Ex — but an LLM knows it from the column's meaning (and House Prices ships a
data_description.txt stating every level). Encoding it injects information the engine lacks.

Arms (identical pipeline, one variable; NO other LLM cleaning/FE, so the delta is pure ordinal):
  A) baseline   : use_llm=False, ordinal=False  -> AutoGluon sees raw unordered categories
  B) ordinal    : use_llm=False, ordinal=True   -> quality columns mapped to worst->best ranks

Run (needs the Kaggle House Prices data locally under data/house-prices/):
    ./.venv/bin/python scripts/ordinal_encoding_experiment.py --model gpt-4o
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from maestra.benchmark import _carve_answer_key, grade
from maestra.config import load_dotenv
from maestra.pipeline import run_pipeline

DATA = "data/house-prices/train.csv"
DESC = "data/house-prices/data_description.txt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--time-limit", type=int, default=60)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    args = p.parse_args()
    load_dotenv()

    if not os.path.exists(DATA):
        sys.exit(f"Missing {DATA}. Download the Kaggle House Prices competition data first "
                 "(kaggle competitions download -c house-prices-advanced-regression-techniques).")
    description = open(DESC).read() if os.path.exists(DESC) else None
    df = pd.read_csv(DATA)
    print(f"dataset: {len(df)} rows, {df.shape[1]} columns; "
          f"description: {'yes' if description else 'no'}\n")

    rows = []
    for seed in args.seeds:
        work, test_features, answer = _carve_answer_key(df, "SalePrice", "Id", 0.25, seed)
        for ordinal in (False, True):
            arm = "ordinal" if ordinal else "baseline"
            result = run_pipeline(
                work, "SalePrice", model=args.model, test_size=0.2, time_limit=args.time_limit,
                seed=seed, model_dir=f"AutogluonModels/ordinal_{arm}_s{seed}", cv_folds=3,
                use_llm=False, ordinal=ordinal, test_df=test_features, id_col="Id",
                dataset_description=description if ordinal else None,
            )
            rmse = grade(result.submission, answer, metric="rmse", id_col="Id", target="SalePrice")
            n_enc = len(result.ordinal["encodings"]) if result.ordinal else 0
            if ordinal and result.ordinal:
                for rec in result.ordinal["encodings"]:
                    print(f"  [encoded s{seed}] {rec['column']} ({rec['n_levels']} levels, "
                          f"{rec['coverage']:.0%} covered)")
            rows.append((seed, arm, n_enc, rmse))
            print(f"seed {seed}  {arm:9s} encoded={n_enc:2d}  rmse={rmse:.1f}\n")

    print("seed  arm        #enc   rmse")
    for seed, arm, n_enc, rmse in rows:
        print(f"{seed:<5} {arm:9s} {n_enc:4d}   {rmse:.1f}")
    # per-seed deltas (ordinal - baseline); negative = ordinal wins (lower rmse)
    print("\nper-seed delta (ordinal - baseline; negative = ordinal wins):")
    for seed in args.seeds:
        base = next(r[3] for r in rows if r[0] == seed and r[1] == "baseline")
        ordv = next(r[3] for r in rows if r[0] == seed and r[1] == "ordinal")
        print(f"  seed {seed}: {ordv - base:+.1f}")
    print("\nReading: ordinal encoding is the one FE type that INJECTS information (an order the "
          "trees\ncannot infer). If it still does not beat the baseline here, the honest "
          "conclusion is that\nAutoGluon's native categorical handling closes even this gap — "
          "and FE is dead across the board.")


if __name__ == "__main__":
    main()
