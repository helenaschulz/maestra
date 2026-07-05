"""N2 decision experiment: does time_local close the CV<->truth gap on a SECOND repeating-period
task (the plan's gate for promoting the bike-sharing fix from one-off to a real capability)?

Synthetic "monthly demand" data: a strong across-period TREND (like bike-sharing's year-over-year
growth) plus a smooth within-period ramp (like bike's within-day commute pattern), so a
tree-based AutoML model must EXTRAPOLATE the trend when evaluated on periods it never saw during
training. The real deployment split, mirroring the actual bike-sharing-demand competition, is
LOCAL and REPEATING: predict the last third of every period from the first two-thirds of that
SAME period, repeated across all periods -- every period appears on both sides of the true split.

Three CV arms (identical data, one fold-construction variable), each measured against the TRUE
held-out score (fit on the local-train rows, score on the local-truth rows):

  A) random folds       -- expected OPTIMISTIC (every period already seen in training; no
                            extrapolation needed at all, unlike the real deployment split)
  B) global time split  -- expected PESSIMISTIC (expanding-window bias: early folds train on
                            only the lowest-trend periods and must extrapolate the trend far
                            beyond their training range for later-period validation rows --
                            AutoGluon's tree ensembles cannot extrapolate past observed values)
  C) time_local folds   -- expected CLOSEST to truth (every fold sees every period's trend
                            level in training; only the LOCAL within-period day pattern needs
                            a small extrapolation, matching what the true split actually asks)

No LLM calls -- this isolates the FOLD CONSTRUCTION mechanism (already covered by
validate_fold_strategy's own tests); the Strategist's ability to DETECT this structure from
column semantics is a separate, already-tested capability (M1, E1/M9).

    ./.venv/bin/python scripts/time_local_experiment.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from maestra.engine import train_and_evaluate
from maestra.validation import cross_validate

_MODEL_DIR = "AutogluonModels/n2_time_local_experiment"


def make_repeating_period_dataset(n_periods=10, period_len=30, trend_slope=8.0,
                                  day_amplitude=30.0, noise_std=5.0, seed=7):
    """Target = period trend + within-period ramp + noise. Only period/day_in_period/noise are
    given as features (feature engineering is not under test here)."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_periods):
        for d in range(period_len):
            demand = (100 + trend_slope * p + day_amplitude * (d / period_len)
                     + rng.normal(scale=noise_std))
            rows.append([p, d, rng.normal(), demand])
    df = pd.DataFrame(rows, columns=["period", "day_in_period", "noise_feature", "demand"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # arrival order shuffled


def local_split(df, period_col="period", day_col="day_in_period", train_frac=2 / 3):
    """The real deployment shape: the first train_frac of every period's days train, the rest
    is truth -- mirrors bike-sharing-demand's actual day 1-19 (train) / day 20-end (test) split,
    repeated every month."""
    cutoff = df.groupby(period_col)[day_col].transform(lambda s: s.quantile(train_frac))
    work = df[df[day_col] <= cutoff].reset_index(drop=True)
    truth = df[df[day_col] > cutoff].reset_index(drop=True)
    return work, truth


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--time-limit", type=int, default=20, help="AutoGluon budget per fit (seconds).")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()

    df = make_repeating_period_dataset()
    work, truth = local_split(df)
    print(f"dataset: {len(df)} rows, {df['period'].nunique()} periods "
          f"({len(work)} local-train rows, {len(truth)} local-truth rows)\n")

    truth_result = train_and_evaluate(work, truth, "demand", args.time_limit, f"{_MODEL_DIR}/truth",
                                      eval_metric="root_mean_squared_error")
    truth_rmse = abs(truth_result.metrics["root_mean_squared_error"])
    print(f"TRUTH (fit on local-train, score on local-truth): rmse={truth_rmse:.3f}\n")

    arms = {
        "random": {},
        "global time": dict(time_column="period"),
        "time_local": dict(time_column="day_in_period", period_column="period"),
    }
    print(f"{'arm':14s}{'CV rmse':>10s}{'truth rmse':>12s}{'gap':>10s}  direction")
    for name, kwargs in arms.items():
        cv = cross_validate(df, "demand", cleaning_plan=None, feature_plan=None,
                            model_dir=f"{_MODEL_DIR}/{name.replace(' ', '_')}",
                            time_limit=args.time_limit, n_folds=args.folds, seed=0,
                            eval_metric="root_mean_squared_error", **kwargs)
        cv_rmse = -cv.mean  # AutoGluon's negated-error CV convention (see mlebench_runner)
        gap = cv_rmse - truth_rmse
        direction = "optimistic (dangerous)" if gap < 0 else "pessimistic (safe)"
        print(f"{name:14s}{cv_rmse:10.3f}{truth_rmse:12.3f}{gap:+10.3f}  {direction}")


if __name__ == "__main__":
    main()
