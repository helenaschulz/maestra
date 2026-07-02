"""M3 probe: does the Skeptic + arbiter save a real feature the cleaner wrongly drops?

Recreates the project's historical failure in miniature (Stellar: the cleaner dropped the
photometric bands u,g,r,i,z as "unique per row", costing balanced_accuracy 0.955 -> 0.919).
The bait dataset has:
  * `flux`      — a CONTINUOUS measurement, unique per row, and the ONLY real signal
                   (y = flux > median + noise). Exactly the column the id-heuristic trap eats.
  * `sample_id` — a genuine running id (deserves dropping),
  * `noise_a/b` — uninformative numerics.

Two arms, same pipeline:
  A) --cv 3                (cleaner alone; if it drops `flux`, accuracy collapses to ~0.5)
  B) --cv 3 --skeptic      (the Skeptic risk-rates the drops; every high-risk drop is put to the
                            CV arbiter and vetoed only if KEEPING the column measurably helps)

Honest note: since the cleaning prompt was hardened after Stellar, the cleaner may well keep
`flux` on its own — then both arms match and the net simply did not need to fire; the Skeptic's
provenance still shows what it reviewed. Either outcome is informative. Run:

    ./.venv/bin/python scripts/skeptic_probe_experiment.py --model gpt-4o
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from maestra.benchmark import _carve_answer_key, grade
from maestra.config import load_dotenv
from maestra.pipeline import run_pipeline

DESCRIPTION = (
    "Astronomical survey snapshots. Each row is one object. 'flux' is a continuous photometric "
    "measurement (float, effectively unique per object — it is a MEASUREMENT, not an identifier); "
    "'sample_id' is the running catalogue id; 'noise_a'/'noise_b' are auxiliary channels. The "
    "target 'bright' marks objects above the survey's brightness threshold."
)


def make_bait(n=1200, seed=3):
    rng = np.random.default_rng(seed)
    flux = rng.normal(size=n) * 10 + 100          # continuous, unique per row, THE signal
    y = ((flux > np.median(flux)) ^ (rng.random(n) < 0.08)).astype(int)  # 8% label noise
    return pd.DataFrame({
        "sample_id": range(1, n + 1),
        "flux": flux,
        "noise_a": rng.normal(size=n),
        "noise_b": rng.normal(size=n),
        "bright": y,
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--time-limit", type=int, default=30)
    args = p.parse_args()
    load_dotenv()

    df = make_bait()
    work, test_features, answer = _carve_answer_key(df, "bright", "sample_id", 0.25, 42)
    print(f"bait dataset: {len(df)} rows; only `flux` (continuous, unique per row) carries signal\n")

    rows = []
    for skeptic in (False, True):
        arm = "skeptic" if skeptic else "cleaner-only"
        result = run_pipeline(
            work, "bright", model=args.model, test_size=0.2, time_limit=args.time_limit,
            seed=42, model_dir=f"AutogluonModels/skeptic_probe_{arm}", cv_folds=3,
            skeptic=skeptic, use_fe=False, test_df=test_features, id_col="sample_id",
            dataset_description=DESCRIPTION,
        )
        acc = grade(result.submission, answer, metric="accuracy", id_col="sample_id", target="bright")
        dropped = [d.get("column") for d in (result.plan or {}).get("columns_to_drop", [])]
        print(f"  [{arm}] cleaner drops: {dropped}")
        if result.skeptic:
            for r in result.skeptic:
                verdict = ("VETOED (kept, Δcv=%+.4f)" % r["cv_delta"]) if r["vetoed"] else \
                          (f"upheld (Δcv={r['cv_delta']})" if r["risk"] == "high" else "low risk")
                print(f"  [skeptic] {r['column']}: {verdict} -- {r['reason'][:80]}")
        rows.append((arm, "flux" in dropped, acc))
        print(f"  {arm:13s} flux dropped={'flux' in dropped!s:5s} truth accuracy={acc:.3f}\n")

    print("arm            flux dropped   truth accuracy")
    for arm, fd, acc in rows:
        print(f"{arm:13s} {str(fd):13s} {acc:.3f}")
    print("\nReading: if the cleaner eats `flux`, accuracy collapses toward 0.5 and the Skeptic arm"
          "\nmust rescue it via the arbiter (veto with a large positive Δcv). If the cleaner keeps"
          "\n`flux` on its own, both arms match — the safety net simply had nothing to catch.")


if __name__ == "__main__":
    main()
