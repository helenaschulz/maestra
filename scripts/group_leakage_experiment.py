"""M1 decision experiment: does the Validation Strategist prevent the group-leakage disaster?

The setup is the classic silent killer of deployed models. Rows are grouped by an entity
(customer): each customer appears in ~8 rows, the features are a noisy per-customer
fingerprint, and the label is a PER-CUSTOMER coin flip — there is NOTHING generalizable to
learn. A random-fold CV puts rows of the same customer in train and validation, the model
memorizes fingerprint -> label, and the CV reports near-perfect accuracy. On truly unseen
customers the honest score is ~0.5 (chance).

Arms (identical pipeline, one variable):
  A) --cv 3 with default random folds        -> CV lies (~1.0), truth ~0.5
  B) --cv 3 --fold-advisor                   -> the Strategist must detect the group column,
                                                switch to GroupKFold, and report ~0.5 honestly

Both arms are graded against a GROUP-DISJOINT answer key (the truth). Run:

    ./.venv/bin/python scripts/group_leakage_experiment.py --model gpt-4o
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from maestra.benchmark import grade
from maestra.config import load_dotenv
from maestra.pipeline import run_pipeline

DESCRIPTION = (
    "Customer churn snapshots. Each row is one MONTHLY VISIT of a customer; most customers "
    "appear in several rows (customer_id). x1..x5 are behavioural scores measured at the "
    "visit. The target 'churned' is the customer's final churn outcome (identical across all "
    "rows of one customer)."
)


def make_grouped_dataset(n_customers=150, rows_per_customer=8, seed=7):
    """Labels are a per-customer coin flip; features a noisy per-customer fingerprint.

    Any model can reach ~100% accuracy on rows of KNOWN customers (memorize the fingerprint)
    and only ~50% on unseen ones — exactly the structure random-fold CV cannot see."""
    rng = np.random.default_rng(seed)
    fingerprint = rng.normal(size=(n_customers, 5))          # who the customer "looks like"
    label = rng.integers(0, 2, size=n_customers)             # coin flip per customer
    rows = []
    for c in range(n_customers):
        for _ in range(rows_per_customer):
            x = fingerprint[c] + rng.normal(scale=0.05, size=5)
            rows.append([c, *x, label[c]])
    df = pd.DataFrame(rows, columns=["customer_id", "x1", "x2", "x3", "x4", "x5", "churned"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def carve_group_disjoint(df, holdout_frac=0.25, seed=7):
    """The answer key: WHOLE customers held out — the only honest test for grouped data."""
    rng = np.random.default_rng(seed)
    customers = df["customer_id"].unique()
    held = set(rng.choice(customers, size=int(len(customers) * holdout_frac), replace=False))
    mask = df["customer_id"].isin(held)
    work = df[~mask].reset_index(drop=True)
    test_features = df[mask].drop(columns=["churned"]).reset_index(drop=True)
    return work, test_features, held


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--time-limit", type=int, default=30)
    args = p.parse_args()
    load_dotenv()

    df = make_grouped_dataset()
    # A row id for grading (customer_id is not unique per row).
    df["row_id"] = range(len(df))
    work, test_features, held = carve_group_disjoint(df)
    answer = df[df["customer_id"].isin(held)][["row_id", "churned"]]

    print(f"dataset: {len(df)} rows, {df['customer_id'].nunique()} customers "
          f"({len(held)} held out entirely as the answer key)\n")

    rows = []
    for fold_advisor in (False, True):
        arm = "fold-advisor" if fold_advisor else "random-folds"
        result = run_pipeline(
            work, "churned", model=args.model, test_size=0.2, time_limit=args.time_limit,
            seed=42, model_dir=f"AutogluonModels/group_leakage_{arm}", cv_folds=3,
            fold_advisor=fold_advisor, use_fe=False, test_df=test_features, id_col="row_id",
            dataset_description=DESCRIPTION,
        )
        lb = grade(result.submission, answer, metric="accuracy", id_col="row_id", target="churned")
        strategy = (result.fold_strategy or {}).get("strategy", "random")
        if result.fold_strategy:
            for line in result.fold_strategy["log"]:
                print(f"  [strategist] {line}")
        rows.append((arm, strategy, result.cv.mean, lb, result.cv.mean - lb))
        print(f"{arm:14s} folds={strategy:6s} CV={result.cv.mean:.3f}  truth(LB)={lb:.3f}  "
              f"gap={result.cv.mean - lb:+.3f}\n")

    print("arm            folds   CV     truth  CV-truth gap")
    for arm, strategy, cv, lb, gap in rows:
        print(f"{arm:14s} {strategy:7s} {cv:.3f}  {lb:.3f}  {gap:+.3f}")
    print("\nReading: with random folds the CV should be wildly optimistic (~1.0 vs ~0.5 truth);"
          "\nwith the fold advisor the CV should approximately match the truth. The advisor's"
          "\nvalue IS the gap it removes.")


if __name__ == "__main__":
    main()
