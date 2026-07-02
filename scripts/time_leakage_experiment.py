"""M1 (time) on real data: does the Validation Strategist keep the CV honest on temporal data?

The second of AutoML's validation blind spots. On trending time series, a random-fold CV lets the
model *interpolate between known dates* — trivially easy — while the deployed model must
*extrapolate into the future*. On the classic `economics` dataset (ggplot2; 574 months of US
macro data, 1967–2015; predict the unemployment level) a quick sklearn check shows random-fold
CV is ~9x too optimistic versus a train-on-past/test-on-future split, while time-ordered CV is
within ~1.4x.

Arms (identical pipeline, one variable):
  A) random folds  -> CV interpolates and reports a fantasy error
  B) fold-advisor  -> the Strategist should detect the `date` column, switch to time-ordered
                      folds, and report an error in the truth's ballpark

Truth = the last 30% of months (the future), never seen by either arm. Run:

    ./.venv/bin/python scripts/time_leakage_experiment.py --model gpt-4o
"""
from __future__ import annotations

import argparse
import os
import urllib.request

import pandas as pd

from maestra.benchmark import grade
from maestra.config import load_dotenv
from maestra.pipeline import run_pipeline

URL = "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/economics.csv"
DESCRIPTION = (
    "US monthly macroeconomic time series, 1967-2015. Each row is ONE MONTH ('date'). Columns: "
    "pce = personal consumption expenditures, pop = population, psavert = personal savings rate, "
    "uempmed = median duration of unemployment. The target 'unemploy' is the number of unemployed "
    "(thousands) in that month. The practical task is FORECASTING: estimate unemployment for "
    "future months from past data."
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--time-limit", type=int, default=30)
    args = p.parse_args()
    load_dotenv()

    path = "data/economics.csv"
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    df = pd.read_csv(path)[["date", "pce", "pop", "psavert", "uempmed", "unemploy"]]
    df = df.sort_values("date").reset_index(drop=True)
    df["row_id"] = range(len(df))

    cut = int(len(df) * 0.7)                       # truth = the FUTURE, never seen by either arm
    work, future = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    answer = future[["row_id", "unemploy"]]
    test_features = future.drop(columns=["unemploy"])
    print(f"dataset: {len(df)} months ({df['date'].iloc[0]} .. {df['date'].iloc[-1]}); "
          f"truth = last {len(future)} months (the future)\n")

    rows = []
    for fold_advisor in (False, True):
        arm = "fold-advisor" if fold_advisor else "random-folds"
        result = run_pipeline(
            work, "unemploy", model=args.model, test_size=0.2, time_limit=args.time_limit,
            seed=42, model_dir=f"AutogluonModels/time_leak_{arm}", cv_folds=3,
            fold_advisor=fold_advisor, use_llm=False, test_df=test_features, id_col="row_id",
            dataset_description=DESCRIPTION,
        )
        truth = grade(result.submission, answer, metric="rmse", id_col="row_id", target="unemploy")
        strategy = (result.fold_strategy or {}).get("strategy", "random")
        if result.fold_strategy:
            for line in result.fold_strategy["log"]:
                print(f"  [strategist] {line}")
        cv_rmse = abs(result.cv.mean)  # AutoGluon reports rmse negated (higher-is-better)
        rows.append((arm, strategy, cv_rmse, truth, cv_rmse - truth))
        print(f"  {arm:13s} folds={strategy:6s} CV rmse={cv_rmse:8.0f}  truth={truth:8.0f}  "
              f"gap={cv_rmse - truth:+8.0f}\n")

    print("arm            folds   CV rmse   truth     CV-truth gap")
    for arm, strategy, cv, truth, gap in rows:
        print(f"{arm:13s} {strategy:7s} {cv:8.0f}  {truth:8.0f}  {gap:+8.0f}")
    print("\nReading: with random folds the CV interpolates between known months and reports a "
          "fantasy\nerror; the fold-advisor detects `date`, validates on the future only, and its "
          "CV lands in the\ntruth's ballpark. The advisor's value IS the optimism it removes.")


if __name__ == "__main__":
    main()
