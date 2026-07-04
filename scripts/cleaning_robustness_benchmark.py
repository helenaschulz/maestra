"""M9-extend: model-robustness benchmark for the CLEANING node — is the drop judgment stable?

M9 quantified model robustness for the Validation Strategist (fold-strategy detection). This is
its sibling for the other blind-spot agent — the cleaning node that decides which columns to
DROP. It isolates the *judgment* the same way E1 does: profile-only, one structured LLM call per
dataset, no AutoGluon training, scored against known ground truth. That keeps it cheap and
model-parametric (`--model`), so it reuses M9's yardstick.

Two directions are scored separately, because they carry very different costs:

  * **drop recall** — of the columns that genuinely SHOULD go (running ids, constants, target
    leaks), how many did the model catch? A miss here is a nuisance (a useless column survives).
  * **KEEP violations** — how many genuine features did the model wrongly drop? This is the
    dangerous direction: it is the exact Stellar bug that motivated the whole "validate against a
    baseline" philosophy (the photometric bands `u,g,r,i,z` dropped as "unique per row"). The
    highlighted subset are **traps**: high-cardinality *float* measurements that look id-like but
    are real. The deterministic `id_like` flag already exempts floats, so any trap drop is pure
    model over-eagerness — the cleanest possible probe of the judgment.

Datasets are mostly controlled synthetics (exact ground truth), plus raw Rdatasets diamonds as a
real anchor whose `rownames` index is a genuine drop. One dataset (`all_real`) has NOTHING to
drop — the over-eager-dropping control, the cleaning analogue of E1's iid false-alarm test.

    ./.venv/bin/python scripts/cleaning_robustness_benchmark.py --model gpt-4o
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from maestra.cleaning import propose_cleaning_plan
from maestra.config import load_dotenv
from maestra.profiling import profile_dataframe


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_house(seed: int = 0):
    """Everyday tabular regression with two clear junk columns and no traps."""
    r = _rng(seed)
    n = 400
    area = r.normal(150, 40, n).round(1)
    rooms = r.integers(1, 8, n)
    neighborhood = r.choice(["Old Town", "Riverside", "Hillcrest", "Docklands"], n)
    year_built = r.integers(1950, 2021, n)
    price = (area * 900 + rooms * 5000 + year_built * 50
             + r.normal(0, 8000, n)).round(0)
    df = pd.DataFrame({
        "row_id": np.arange(n),                       # running id -> DROP
        "schema_version": "v3",                        # constant -> DROP
        "area_sqm": area, "rooms": rooms,
        "neighborhood": neighborhood, "year_built": year_built,
        "price": price,
    })
    return df, "price", {"row_id", "schema_version"}, set()


def build_sensor(seed: int = 1):
    """The Stellar analogue: five high-cardinality FLOAT measurement bands (all real) + an id."""
    r = _rng(seed)
    n = 500
    bands = {b: r.normal(18, 2.5, n).round(6) for b in
             ("band_u", "band_g", "band_r", "band_i", "band_z")}
    redshift = sum(bands.values()) / 5 + r.normal(0, 0.5, n)
    star_class = np.where(redshift > redshift.mean(), "GALAXY", "STAR")
    df = pd.DataFrame({
        "obs_id": np.arange(n),                        # running id -> DROP
        "survey_field": r.choice(["F1", "F2", "F3"], n),
        **bands,                                        # all five -> KEEP (traps)
        "star_class": star_class,
    })
    traps = set(bands)
    return df, "star_class", {"obs_id"}, traps


def build_leak(seed: int = 2):
    """A near-perfect target leak among real features — the drop the arbiter cares about most."""
    r = _rng(seed)
    n = 400
    x1, x2, x3 = r.normal(0, 1, n), r.normal(5, 2, n), r.integers(0, 4, n)
    y = (3 * x1 - 2 * x2 + x3 + r.normal(0, 0.3, n)).round(3)
    df = pd.DataFrame({
        "record_id": np.arange(n),                     # running id -> DROP
        "feat_a": x1.round(3), "feat_b": x2.round(3), "segment": x3,
        "target_snapshot": (y + r.normal(0, 0.01, n)).round(3),  # |corr|~0.999 -> DROP (leak)
        "y": y,
    })
    return df, "y", {"record_id", "target_snapshot"}, set()


def build_geo(seed: int = 3):
    """Latitude/longitude: high-cardinality floats that read like ids but carry real signal."""
    r = _rng(seed)
    n = 450
    lat = r.uniform(47.0, 55.0, n).round(6)
    lon = r.uniform(6.0, 15.0, n).round(6)
    pop = r.integers(500, 90000, n)
    income = r.normal(42000, 9000, n).round(0)
    price = (lat * 1000 - lon * 400 + pop * 0.1 + income * 0.05
             + r.normal(0, 3000, n)).round(0)
    df = pd.DataFrame({
        "point_id": np.arange(n),                      # running id -> DROP
        "latitude": lat, "longitude": lon,             # KEEP (traps)
        "population": pop, "median_income": income,
        "listing_price": price,
    })
    return df, "listing_price", {"point_id"}, {"latitude", "longitude"}


def build_all_real(seed: int = 4):
    """Every column is a genuine feature — the over-eager-dropping control. Ideal: drop NOTHING."""
    r = _rng(seed)
    n = 400
    age = r.integers(18, 80, n)
    bmi = r.normal(26, 4, n).round(1)
    smoker = r.choice(["yes", "no"], n, p=[0.2, 0.8])
    region = r.choice(["north", "south", "east", "west"], n)
    children = r.integers(0, 5, n)
    charges = (age * 60 + bmi * 200 + (smoker == "yes") * 15000
               + children * 500 + r.normal(0, 2000, n)).round(2)
    df = pd.DataFrame({
        "age": age, "bmi": bmi, "smoker": smoker,
        "region": region, "children": children, "charges": charges,
    })
    return df, "charges", set(), set()


def load_diamonds_raw():
    """Real anchor: raw Rdatasets diamonds keeps the source row index as `rownames` (a drop)."""
    path = "data/clean_diamonds_raw.csv"
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(
            "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/diamonds.csv", path)
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "rownames"}) if df.columns[0].startswith("Unnamed") \
        else df
    df = df.sample(3000, random_state=0).reset_index(drop=True)
    # rownames: the source index, unique + id-like -> DROP. x/y/z: high-card float traps.
    return df, "price", {"rownames"}, {"x", "y", "z"}


CATALOG = [
    ("house", build_house), ("sensor_stellar", build_sensor), ("leak", build_leak),
    ("geo", build_geo), ("all_real", build_all_real), ("diamonds_raw", load_diamonds_raw),
]


def score(spec_name, df, target, must_drop, traps, proposed_drops) -> dict:
    """One result row: recall on the junk, violations on the genuine features."""
    proposed = {c for c in proposed_drops if c in df.columns and c != target}
    keep = set(df.columns) - {target} - must_drop
    caught = must_drop & proposed
    keep_violations = keep & proposed          # genuine features wrongly dropped (dangerous)
    trap_violations = traps & proposed          # the highlighted subset of keep_violations
    return dict(
        dataset=spec_name,
        drop_recall=(len(caught), len(must_drop)),
        keep_violations=sorted(keep_violations),
        trap_violations=sorted(trap_violations),
        proposed=sorted(proposed),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--out", default="cleaning_benchmark.jsonl")
    args = p.parse_args()
    load_dotenv()

    rows = []
    for name, builder in CATALOG:
        df, target, must_drop, traps = builder()
        plan = propose_cleaning_plan(args.model, profile_dataframe(df, target), target)
        drops = [d.get("column") for d in plan.get("columns_to_drop", []) if isinstance(d, dict)]
        row = score(name, df, target, must_drop, traps, drops)
        rows.append(row)
        c, t = row["drop_recall"]
        kv, tv = row["keep_violations"], row["trap_violations"]
        flag = "OK  " if not kv else "KEEP-VIOL"
        print(f"  [{flag}] {name:15s} drop-recall={c}/{t}  "
              f"keep-violations={kv or '—'}  traps-dropped={tv or '—'}")

    tot_caught = sum(r["drop_recall"][0] for r in rows)
    tot_drop = sum(r["drop_recall"][1] for r in rows)
    tot_keepv = sum(len(r["keep_violations"]) for r in rows)
    tot_trapv = sum(len(r["trap_violations"]) for r in rows)
    n_traps = sum(len(t) for _, b in CATALOG for *_, t in [b()])

    print(f"\nmodel: {args.model}")
    print(f"drop recall (junk caught):     {tot_caught}/{tot_drop}")
    print(f"KEEP violations (real dropped): {tot_keepv}   <- the dangerous direction")
    print(f"  of which trap columns:        {tot_trapv}/{n_traps} high-card float measurements dropped")

    with open(args.out, "a") as fh:
        fh.write(json.dumps({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "kind": "cleaning_benchmark", "model": args.model,
            "drop_recall": [tot_caught, tot_drop],
            "keep_violations": tot_keepv, "trap_violations": [tot_trapv, n_traps],
            "rows": rows,
        }) + "\n")


if __name__ == "__main__":
    main()
