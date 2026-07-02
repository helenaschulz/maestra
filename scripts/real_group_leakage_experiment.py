"""M1 on REAL data: does the Validation Strategist keep the CV honest on genuine grouped datasets?

The synthetic experiment (group_leakage_experiment.py) proved the mechanism on maximally
adversarial data (+0.499 CV lie). This one answers the fair-minded reader's next question —
"show me on real data" — on two recognizable grouped datasets that bracket the real-world range:

  * Grunfeld  (plm, 200 rows, 10 firms): firm-year investment. Firm scale dominates the target,
    so entity identity leaks hard — a textbook panel-data case.
  * MathAchieve (nlme, 7185 rows, 160 schools): student math achievement. Schools explain only
    ~16% of variance, so the leak is real but modest — a textbook multilevel case.

Both are downloaded from the public Rdatasets mirror (no auth). For each, two arms run the same
pipeline; the only variable is `--fold-advisor`:

  A) random folds  -> the CV should be OPTIMISTIC (entity rows leak across folds)
  B) fold-advisor  -> the Strategist should detect the entity column, switch to group folds,
                      and report a CV that matches the group-disjoint truth

Truth = a group-disjoint answer key (whole entities held out — the only honest test). Run:

    ./.venv/bin/python scripts/real_group_leakage_experiment.py --model gpt-4o
"""
from __future__ import annotations

import argparse
import os
import urllib.request

import numpy as np
import pandas as pd

from maestra.benchmark import grade
from maestra.config import load_dotenv
from maestra.pipeline import run_pipeline

DATASETS = [
    {
        "name": "grunfeld",
        "url": "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Grunfeld.csv",
        "target": "inv", "entity": "firm", "features": ["firm", "year", "value", "capital"],
        "description": "Panel data: each row is one FIRM in one YEAR (firm repeats across years). "
                       "Predict gross investment 'inv' from firm value and capital stock.",
    },
    {
        "name": "mathachieve",
        "url": "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/MathAchieve.csv",
        "target": "MathAch", "entity": "School", "features": ["School", "Minority", "Sex", "SES"],
        "description": "Multilevel education data: each row is one STUDENT nested in a SCHOOL "
                       "(many students per school). Predict the student's math achievement 'MathAch'. "
                       "The goal is to generalize to students in NEW schools.",
    },
]


def _load(spec):
    path = f"data/{spec['name']}.csv"
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(spec["url"], path)
    df = pd.read_csv(path)[spec["features"] + [spec["target"]]].copy()
    df["row_id"] = range(len(df))
    return df


def _carve_group_disjoint(df, entity, seed, holdout_frac=0.3):
    """Hold out WHOLE entities — the only honest test for grouped data. Returns (work, held, n)."""
    rng = np.random.default_rng(seed)
    entities = df[entity].unique()
    held = set(rng.choice(entities, size=max(2, int(len(entities) * holdout_frac)), replace=False))
    mask = df[entity].isin(held)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True), len(held)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--time-limit", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    load_dotenv()

    summary = []
    for spec in DATASETS:
        df = _load(spec)
        target, entity = spec["target"], spec["entity"]
        work, held_df, n_held = _carve_group_disjoint(df, entity, args.seed)
        answer = held_df[["row_id", target]]
        test_features = held_df.drop(columns=[target])
        print(f"\n=== {spec['name']}: {len(df)} rows, {df[entity].nunique()} {entity}s "
              f"({n_held} held out as unseen-entity truth) ===")

        for fold_advisor in (False, True):
            arm = "fold-advisor" if fold_advisor else "random-folds"
            result = run_pipeline(
                work, target, model=args.model, test_size=0.2, time_limit=args.time_limit,
                seed=args.seed, model_dir=f"AutogluonModels/real_grp_{spec['name']}_{arm}", cv_folds=3,
                fold_advisor=fold_advisor, use_llm=False, test_df=test_features, id_col="row_id",
                dataset_description=spec["description"],
            )
            truth = grade(result.submission, answer, metric="rmse", id_col="row_id", target=target)
            strategy = (result.fold_strategy or {}).get("strategy", "random")
            if result.fold_strategy:
                for line in result.fold_strategy["log"]:
                    print(f"  [strategist] {line}")
            gap = result.cv.mean - truth
            summary.append((spec["name"], arm, strategy, result.cv.mean, truth, gap))
            print(f"  {arm:13s} folds={strategy:6s} CV={result.cv.mean:8.2f}  truth={truth:8.2f}  "
                  f"CV-truth gap={gap:+8.2f}\n")

    print("\ndataset      arm           folds   CV        truth     CV-truth gap")
    for name, arm, strategy, cv, truth, gap in summary:
        print(f"{name:12s} {arm:13s} {strategy:6s} {cv:8.2f}  {truth:8.2f}  {gap:+8.2f}")
    print("\nReading: with random folds the CV overstates real accuracy (rows of the same entity "
          "leak\nacross folds); the fold-advisor detects the entity, uses group folds, and its CV "
          "tracks the\ngroup-disjoint truth. The advisor's value IS the optimism it removes — the "
          "gap between the\ntwo arms' CVs. Grunfeld shows a large leak (entity-dominated target), "
          "MathAchieve a modest one.")


if __name__ == "__main__":
    main()
