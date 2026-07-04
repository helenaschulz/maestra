"""K1: the Kaggle battery — E2's verdict framework on real competition data.

The E2 battery spans the semantic spectrum on classic Rdatasets/UCI tables; this battery re-runs
the same instrument (5 seeds, `run_multi_seed`, three-way paired verdict) on REAL Kaggle
competition data — messier columns, known leaks, competition metrics. The axes mirror E2 plus
the project's blind spots:

  * rich semantics (spaceship-titanic, house-prices) — where the thesis predicts wins,
  * mixed (titanic: semantic names but 891 rows),
  * **anonymized control** (allstate: cat1..cat116/cont1..cont14, the Kaggle analogue of the
    friedman/twin controls — the thesis predicts inert),
  * **temporal** (bike-sharing: a datetime axis; also a natural `--target-framing` candidate —
    `count` is right-skewed, and the competition metric is RMSLE).

Kaggle data cannot be fetched anonymously: join each competition once in the web UI (accept the
rules), then this script's printed `kaggle competitions download` command works. Local files are
checked first, so already-downloaded tasks run offline.

Known-leak hygiene (the diamonds lesson, applied up front): bike-sharing's `casual`+`registered`
sum exactly to the target `count` and are absent from the competition's test set — dropped in
the loader, not left for the arbiter to stumble over.

    ./.venv/bin/python scripts/kaggle_battery.py --list
    ./.venv/bin/python scripts/kaggle_battery.py --task titanic
    ./.venv/bin/python scripts/kaggle_battery.py --task all

Real submissions: `--make-submission TASK` trains Maestra on the FULL train set (leakage-free CV
for the honest estimate), predicts the competition's own test.csv and writes
`data/submission_<task>.csv` plus the exact `kaggle competitions submit` command. The printed CV
estimate is the number to compare against the public LB — the CV↔LB gap on a real leaderboard.
For the RMSLE competitions (house-prices, bike-sharing) `--target-framing` is enabled and
metric-aligned by construction (training on log1p under RMSE optimises RMSLE): M11 paying off on
a real leaderboard. For other regression tasks framing stays on too — the arbiter decides.
"""
from __future__ import annotations

import argparse
import glob
import os
import zipfile
from datetime import datetime

import pandas as pd

from maestra.benchmark import append_multi_seed, append_result, run_multi_seed
from maestra.config import load_dotenv

_SEEDS = [42, 7, 1, 2, 3]

CATALOG = [
    dict(name="titanic", semantics="mixed", target="Survived", metric="balanced_accuracy",
         id_col="PassengerId", path="data/titanic.csv", competition="titanic",
         test_path="data/kaggle_titanic/test.csv", eval_metric="accuracy", framing=False,
         note="891 rows; semantic names (Name/Ticket/Cabin) but tiny — E2's small-data anchor. "
              "Single-seed result exists (baseline 0.793 > maestra 0.732); this is the honest "
              "5-seed re-measurement in the paired-verdict framework"),
    dict(name="house-prices", semantics="rich", target="SalePrice", metric="rmse",
         id_col="Id", path="data/house-prices/train.csv",
         competition="house-prices-advanced-regression-techniques",
         test_path="data/house-prices/test.csv",
         eval_metric="root_mean_squared_error", framing=True,  # comp metric RMSLE: log1p aligns
         note="the M6 anchor (Maestra 5/5 seeds, mean +1285) — in the catalog for completeness; "
              "re-running burns ~30 min for a verdict that already exists in benchmark.jsonl"),
    dict(name="spaceship-titanic", semantics="rich", target="Transported", metric="accuracy",
         id_col="PassengerId", path="data/kaggle_spaceship/train.csv",
         competition="spaceship-titanic",
         test_path="data/kaggle_spaceship/test.csv", eval_metric="accuracy", framing=False,
         note="8.7k rows, rich semantics (HomePlanet/CryoSleep/Cabin/VIP/RoomService...) — the "
              "modern Kaggle rich-semantics test; competition metric is accuracy"),
    dict(name="bike-sharing", semantics="rich+time", target="count", metric="rmse",
         id_col="rowid", path="data/kaggle_bike/train.csv", competition="bike-sharing-demand",
         test_path="data/kaggle_bike/test.csv", submit_id="datetime",
         eval_metric="root_mean_squared_error", framing=True,  # comp metric RMSLE: log1p aligns
         drop=["casual", "registered"],  # sum EXACTLY to `count` and absent from comp test set
         note="10.9k rows with a datetime axis — the temporal Kaggle case; `count` is "
              "right-skewed (competition metric is RMSLE), making it the natural "
              "--target-framing candidate for a follow-up"),
    dict(name="allstate", semantics="poor", target="loss", metric="mae",
         id_col="id", path="data/kaggle_allstate/train.csv",
         competition="allstate-claims-severity", sample=8000,
         test_path="data/kaggle_allstate/test.csv",
         eval_metric="mean_absolute_error", framing=True,  # skewed loss; the arbiter decides
         note="fully anonymized (cat1..cat116, cont1..cont14) — the Kaggle analogue of the "
              "friedman/anonymized-twin controls; thesis predicts inert. Subsampled to 8k rows; "
              "competition metric is MAE (natively supported)"),
]


def _materialize(spec: dict) -> str | None:
    """Return the task's CSV path, unzipping a downloaded archive if needed; None if absent."""
    path = spec["path"]
    if not os.path.exists(path):
        folder = os.path.dirname(path) or "data"
        for z in glob.glob(os.path.join(folder, "*.zip")):
            with zipfile.ZipFile(z) as zh:
                zh.extractall(folder)
        if not os.path.exists(path):
            return None
    prepared = f"data/kbattery_{spec['name']}.csv"
    if os.path.exists(prepared):
        return prepared
    df = pd.read_csv(path)
    for col in spec.get("drop", []):
        if col in df.columns:
            df = df.drop(columns=[col])
    if spec.get("sample") and len(df) > spec["sample"]:
        df = df.sample(spec["sample"], random_state=0).reset_index(drop=True)
    if spec["id_col"] not in df.columns:
        df.insert(0, spec["id_col"], range(len(df)))
    df.to_csv(prepared, index=False)
    return prepared


def _download_help(spec: dict) -> str:
    folder = os.path.dirname(spec["path"]) or "data"
    return (f"  1) join once in the web UI: https://www.kaggle.com/competitions/{spec['competition']}\n"
            f"  2) .venv311/bin/kaggle competitions download -c {spec['competition']} -p {folder}")


def make_submission(spec: dict, *, model: str, time_limit: int, cv: int) -> None:
    """Train Maestra on the FULL train set and write a submittable prediction file.

    Uses the leakage-free CV for the honest estimate (that is the number the public LB gets
    compared against — the CV↔LB gap on a real leaderboard), the competition's eval metric,
    and target framing where enabled in the catalog (RMSLE comps: metric-aligned log1p; other
    regression: the arbiter decides). The competition's own test.csv is predicted through the
    same fitted transforms; the submission id is the competition's, not the battery's row id.
    """
    from maestra.pipeline import run_pipeline
    from maestra.runlog import append_run

    train_csv = _materialize(spec)
    test_path = spec["test_path"]
    if train_csv is None or not os.path.exists(test_path):
        print(f"\n=== {spec['name']}: competition data missing — to fetch:")
        print(_download_help(spec))
        return
    submit_id = spec.get("submit_id", spec["id_col"])
    train = pd.read_csv(spec["path"])          # FULL train (no battery subsample)
    for col in spec.get("drop", []):
        if col in train.columns:
            train = train.drop(columns=[col])
    test_df = pd.read_csv(test_path)

    print(f"\n=== {spec['name']}: full-train submission run "
          f"(eval_metric={spec['eval_metric']}, framing={spec['framing']}) ===")
    result = run_pipeline(
        train, spec["target"], model=model, test_size=0.2, time_limit=time_limit,
        seed=42, model_dir=f"AutogluonModels/kaggle_{spec['name']}", cv_folds=cv,
        eval_metric=spec["eval_metric"], target_framing=spec["framing"],
        test_df=test_df, id_col=submit_id)

    out = f"data/submission_{spec['name']}.csv"
    result.submission.to_csv(out, index=False)
    append_run("runs.jsonl", result, csv=spec["path"], target=spec["target"], model=model,
               no_llm=False, max_attempts=1,
               timestamp=datetime.now().isoformat(timespec="seconds"))

    cv_est = result.cv
    framing_note = ""
    if result.target_framing:
        framing_note = f" | framing: {result.target_framing['transform']}" \
                       f" accepted={result.target_framing['accepted']}"
    print(f"\nsubmission written: {out}  ({len(result.submission)} rows)")
    print(f"CV estimate ({cv_est.eval_metric}): {cv_est.mean:.4f} ± {cv_est.std:.4f}{framing_note}")
    print("submit with:")
    print(f"  .venv311/bin/kaggle competitions submit -c {spec['competition']} "
          f"-f {out} -m 'maestra cv={cv_est.mean:.4f}'")
    print("then compare the public LB score against the CV estimate — the real CV<->LB gap.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", help="Task name from the catalog, or 'all'.")
    p.add_argument("--list", action="store_true", help="List catalog tasks and exit.")
    p.add_argument("--make-submission", metavar="TASK",
                   help="Full-train run on TASK (or 'all'), predicting the competition's own "
                        "test.csv into data/submission_<task>.csv + the submit command.")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--time-limit", type=int, default=60)
    p.add_argument("--cv", type=int, default=3)
    args = p.parse_args()
    load_dotenv()

    if args.make_submission:
        todo = CATALOG if args.make_submission == "all" \
            else [s for s in CATALOG if s["name"] == args.make_submission]
        if not todo:
            raise SystemExit(f"unknown task {args.make_submission!r} — see --list")
        for spec in todo:
            make_submission(spec, model=args.model, time_limit=args.time_limit, cv=args.cv)
        return

    if args.list or not args.task:
        for spec in CATALOG:
            status = "ready" if _materialize(spec) else "MISSING (join + download)"
            print(f"{spec['name']:18s} {spec['semantics']:10s} {spec['metric']:18s} [{status}]")
            print(f"    {spec['note']}")
        return

    todo = CATALOG if args.task == "all" else [s for s in CATALOG if s["name"] == args.task]
    if not todo:
        raise SystemExit(f"unknown task {args.task!r} — see --list")

    for spec in todo:
        csv = _materialize(spec)
        if csv is None:
            print(f"\n=== {spec['name']}: data missing — to fetch:")
            print(_download_help(spec))
            continue
        print(f"\n=== {spec['name']} ({spec['semantics']}) — {len(_SEEDS)} seeds ===")
        ms = run_multi_seed(csv, spec["target"], metric=spec["metric"], seeds=_SEEDS,
                            id_col=spec["id_col"], model=args.model,
                            time_limit=args.time_limit, cv_folds=args.cv,
                            name=f"kaggle-{spec['name']}")
        ts = datetime.now().isoformat(timespec="seconds")
        for r in ms.per_seed:
            append_result("benchmark.jsonl", r, timestamp=ts)
        append_multi_seed("benchmark.jsonl", ms, timestamp=ts)


if __name__ == "__main__":
    main()
