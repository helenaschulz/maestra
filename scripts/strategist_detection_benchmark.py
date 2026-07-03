"""E1: detection benchmark for the Validation Strategist — precision, recall, false alarms.

The Strategist's headline results (group/time leaks detected unaided) rest on 5 anecdotes, all on
datasets that HAD structure. This benchmark quantifies detection on a catalog of classic datasets
with known ground truth — including iid datasets, measuring the never-before-measured
**false-alarm rate** (does the agent hallucinate structure where none exists?). One entry
(PlantGrowth) is a deliberate trap: it has a column literally named `group` that is a treatment
factor, not an entity.

Design choices:
  * **Profile-only.** No dataset descriptions are passed — the claim under test is "detection from
    the column profile alone", the hardest, cleanest version.
  * **Acceptable-answer sets.** Panel data is legitimately group OR time (Grunfeld: by `firm` or
    by `year`); repeated-measures data likewise. Scoring accepts any defensible answer, so the
    benchmark measures detection, not agreement with one arbitrary label.
  * **Model-parametric.** `--model` runs the same catalog under any LiteLLM backbone — this is the
    yardstick M9 (model-robustness matrix) reuses.

Cost: one structured LLM call per dataset (~17 calls), no AutoGluon training. Run:

    ./.venv/bin/python scripts/strategist_detection_benchmark.py --model gpt-4o
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime

import pandas as pd

from maestra.config import load_dotenv
from maestra.profiling import profile_dataframe
from maestra.validation_strategist import propose_fold_strategy, validate_fold_strategy

_BASE = "https://vincentarelbundock.github.io/Rdatasets/csv"

# Each entry: name, source (url path or local), target, and the ACCEPTABLE truths — a dict
# strategy -> set of acceptable columns ("random" maps to an empty set). "primary" names the
# canonical truth used in the per-class recall breakdown.
CATALOG = [
    # --- grouped / repeated-measures (entities repeat across rows) --------------------
    dict(name="grunfeld", src=f"{_BASE}/plm/Grunfeld.csv", target="inv",
         primary="group", truth={"group": {"firm"}, "time": {"year"}}),
    dict(name="emplUK", src=f"{_BASE}/plm/EmplUK.csv", target="emp",
         primary="group", truth={"group": {"firm"}, "time": {"year"}}),
    dict(name="mathachieve", src=f"{_BASE}/nlme/MathAchieve.csv", target="MathAch",
         primary="group", truth={"group": {"School"}}),
    dict(name="sleepstudy", src=f"{_BASE}/lme4/sleepstudy.csv", target="Reaction",
         primary="group", truth={"group": {"Subject"}, "time": {"Days"}}),
    dict(name="oxboys", src=f"{_BASE}/nlme/Oxboys.csv", target="height",
         primary="group", truth={"group": {"Subject"}, "time": {"age", "Occasion"}}),
    dict(name="chickweight", src=f"{_BASE}/datasets/ChickWeight.csv", target="weight",
         primary="group", truth={"group": {"Chick"}, "time": {"Time"}}),
    # --- temporal (predict the future of an ordered series) ---------------------------
    dict(name="economics", src=f"{_BASE}/ggplot2/economics.csv", target="unemploy",
         primary="time", truth={"time": {"date"}}),
    dict(name="nottem", src=f"{_BASE}/datasets/nottem.csv", target="value",
         primary="time", truth={"time": {"time"}}),
    dict(name="airpassengers", src=f"{_BASE}/datasets/AirPassengers.csv", target="value",
         primary="time", truth={"time": {"time"}}),
    dict(name="ukgas", src=f"{_BASE}/datasets/UKgas.csv", target="value",
         primary="time", truth={"time": {"time"}}),
    dict(name="lynx", src=f"{_BASE}/datasets/lynx.csv", target="value",
         primary="time", truth={"time": {"time"}}),
    # --- iid (independent rows; anything but random is a FALSE ALARM) -----------------
    dict(name="titanic", src="data/titanic.csv", target="Survived",
         primary="random", truth={"random": set()}),
    dict(name="iris", src=f"{_BASE}/datasets/iris.csv", target="Species",
         primary="random", truth={"random": set()}),
    dict(name="mtcars", src=f"{_BASE}/datasets/mtcars.csv", target="mpg",
         primary="random", truth={"random": set()}),
    dict(name="swiss", src=f"{_BASE}/datasets/swiss.csv", target="Fertility",
         primary="random", truth={"random": set()}),
    # the trap: a column literally named 'group' that is a 3-level TREATMENT, not an entity
    dict(name="plantgrowth", src=f"{_BASE}/datasets/PlantGrowth.csv", target="weight",
         primary="random", truth={"random": set()}),
    dict(name="boston", src=f"{_BASE}/MASS/Boston.csv", target="medv",
         primary="random", truth={"random": set()}),
]


def load_dataset(spec: dict) -> pd.DataFrame:
    src = spec["src"]
    if src.startswith("http"):
        path = f"data/detection_{spec['name']}.csv"
        if not os.path.exists(path):
            os.makedirs("data", exist_ok=True)
            urllib.request.urlretrieve(src, path)
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(src)
    return df.drop(columns=[c for c in ("rownames",) if c in df.columns])


def score(spec: dict, verified: dict) -> dict:
    """One row of the result table: what the agent said, whether it is acceptable."""
    strategy = verified["strategy"]
    column = verified.get("group_column") or verified.get("time_column")
    acceptable = spec["truth"]
    if strategy == "random":
        correct = "random" in acceptable
    else:
        correct = strategy in acceptable and column in acceptable[strategy]
    return dict(name=spec["name"], primary=spec["primary"], predicted=strategy,
                column=column, correct=bool(correct))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--out", default="detection_benchmark.jsonl",
                   help="Append one JSON line per run (model, per-dataset results, metrics).")
    args = p.parse_args()
    load_dotenv()

    rows = []
    for spec in CATALOG:
        df = load_dataset(spec)
        proposal = propose_fold_strategy(args.model, profile_dataframe(df, spec["target"]),
                                         spec["target"])  # profile-only: no description
        verified, _ = validate_fold_strategy(proposal, df, spec["target"])
        row = score(spec, verified)
        rows.append(row)
        mark = "OK " if row["correct"] else "MISS"
        print(f"  [{mark}] {spec['name']:14s} truth={spec['primary']:6s} "
              f"predicted={row['predicted']:6s} column={row['column']}")

    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    by_class = {}
    for cls in ("group", "time", "random"):
        cls_rows = [r for r in rows if r["primary"] == cls]
        by_class[cls] = (sum(r["correct"] for r in cls_rows), len(cls_rows))
    iid_rows = [r for r in rows if r["primary"] == "random"]
    false_alarms = sum(1 for r in iid_rows if r["predicted"] != "random")

    print(f"\nmodel: {args.model}")
    print(f"overall: {correct}/{n} acceptable ({correct / n:.0%})")
    for cls, (c, t) in by_class.items():
        print(f"  {cls:6s} recall: {c}/{t}")
    print(f"  FALSE-ALARM rate on iid data: {false_alarms}/{len(iid_rows)}"
          f"  (structure hallucinated where none exists)")

    with open(args.out, "a") as fh:
        fh.write(json.dumps({
            "timestamp": datetime.now().isoformat(timespec="seconds"), "kind": "detection_benchmark",
            "model": args.model, "n": n, "correct": correct,
            "recall": {k: f"{c}/{t}" for k, (c, t) in by_class.items()},
            "false_alarms_iid": false_alarms, "n_iid": len(iid_rows), "rows": rows,
        }, default=str) + "\n")
    print(f"\nLogged to {args.out}. Re-run with another --model for the robustness matrix (M9).")


if __name__ == "__main__":
    main()
