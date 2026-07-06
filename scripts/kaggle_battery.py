"""K1/K2: the Kaggle battery — E2's verdict framework on real competition data.

The E2 battery spans the semantic spectrum on classic Rdatasets/UCI tables; this battery re-runs
the same instrument (5 seeds, `run_multi_seed`, three-way paired verdict) on REAL Kaggle
competition data — messier columns, known leaks, competition metrics. The axes mirror E2 plus
the project's blind spots:

  * rich semantics (spaceship-titanic, house-prices) — where the thesis predicts wins,
  * mixed (titanic: semantic names but 891 rows),
  * **anonymized control** (allstate, santander-transaction: the Kaggle analogue of the
    friedman/twin controls — the thesis predicts inert),
  * **temporal** (bike-sharing, store-sales, rossmann, walmart, ieee-fraud) — a datetime axis;
    several are also a natural `--target-framing` candidate (skewed sales/revenue targets),
  * **group** (rossmann/walmart's Store, two-sigma's manager_id) — a repeating entity the
    Validation Strategist can detect WITHOUT the N2 raw-timestamp gap (the group column already
    exists in the raw profile, unlike a derived month).

K2 (2026-07-05) extends K1 from 5 to 13 tasks, all real Kaggle competitions, to turn the
setup-wins from single case (bike-sharing) to pattern — see docs/RESULTS.md's K2 section for the
verdicts. Competition metrics that this harness doesn't natively support (RMSPE, NDCG@k, log
loss on a >2-class label) are approximated with a supported label metric (rmse/balanced_accuracy)
for the internal battery verdict; this is a deliberate simplification consistent with how K1
already handled titanic/spaceship-titanic, and is noted per task below. It does not affect
`--make-submission`, which always uses each task's real `eval_metric`.

Kaggle data cannot be fetched anonymously: join each competition once in the web UI (accept the
rules), then this script's printed `kaggle competitions download` command works — EXCEPT the
"Getting Started" competitions (titanic, house-prices, spaceship-titanic, store-sales), which are
open by default. Local files are checked first, so already-downloaded tasks run offline.

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

import numpy as np
import pandas as pd

from maestra.benchmark import append_multi_seed, append_result, run_multi_seed
from maestra.config import load_dotenv

_SEEDS = [42, 7, 1, 2, 3]


def _rmspe(y_true, y_pred):
    """Root Mean Square Percentage Error — Rossmann's real metric. Zero-sales (closed-store)
    rows are ignored, exactly as the competition scores them."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def _custom_scorers() -> dict:
    """AutoGluon Scorer objects for competition metrics with no native AutoGluon scorer.
    Built lazily so importing this module never needs AutoGluon."""
    from autogluon.core.metrics import make_scorer
    return {
        "rmspe": make_scorer("rmspe", _rmspe, optimum=0.0, greater_is_better=False),
    }


def _resolve_metric(metric):
    """A plain metric string passes through; a key of :func:`_custom_scorers` becomes its
    AutoGluon Scorer object (which ``run_pipeline`` accepts as ``eval_metric``)."""
    if isinstance(metric, str) and metric in _custom_scorers():
        return _custom_scorers()[metric]
    return metric


def _topk_long(wide: pd.DataFrame, id_col: str, out_col: str, k: int) -> pd.DataFrame:
    """Reshape a wide class-probability frame (id + one column per class) into the LONG top-k
    ranking format (id repeated k times, one class per row, best first) that NDCG@k competitions
    like airbnb expect."""
    class_cols = [c for c in wide.columns if c != id_col]
    classes = np.array(class_cols)
    probs = wide[class_cols].to_numpy()
    topk_idx = np.argsort(-probs, axis=1)[:, :k]          # k highest-prob classes per row, desc
    ids = np.repeat(wide[id_col].to_numpy(), topk_idx.shape[1])
    picks = classes[topk_idx].reshape(-1)
    return pd.DataFrame({id_col: ids, out_col: picks})

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

    # --- K2 (2026-07-05): temporal/group structure, real competitions, 8 new tasks ---
    dict(name="store-sales", semantics="rich+time+group", target="sales", metric="rmse",
         id_col="id", path="data/kaggle_store_sales/train.csv",
         competition="store-sales-time-series-forecasting", sample=15000,
         test_path="data/kaggle_store_sales/test.csv",
         submit_sample=200000,  # full train is 3M rows -> infeasible; cap for the submission run
         eval_metric="root_mean_squared_error", framing=True,  # sales is right-skewed
         note="Getting-Started (open by default, no join needed). 3M rows -> subsampled to 15k. "
              "store_nbr (54 stores) + family (33 categories) repeat densely — group AND time "
              "axes both present in a single un-joined table (holidays_events.csv/oil.csv/"
              "stores.csv/transactions.csv exist but are ignored: multi-table join is a "
              "deliberate non-goal). Real metric is RMSLE; framing=True aligns it. Verified via "
              "a live single-seed smoke run (2026-07-05): baseline 411.5, maestra 408.2."),
    dict(name="rossmann", semantics="rich+time+group", target="Sales", metric="rmse",
         id_col="row_id", path="data/kaggle_rossmann/train.csv",  # Store repeats -- NOT a valid
         # row id for grade()'s merge; row_id doesn't exist in the raw columns, so _materialize
         # auto-inserts a unique range index (the same pattern bike-sharing's "rowid" uses)
         competition="rossmann-store-sales", sample=15000,
         test_path="data/kaggle_rossmann/test.csv",
         submit_id="Id",  # the real test.csv keys on "Id" (row_id is battery-only, not in test)
         submit_sample=200000,  # full train is 1M rows
         submit_eval_metric="rmspe",  # the competition's real metric (custom AutoGluon scorer)
         submit_framing=False,  # a custom RMSPE scorer works in ORIGINAL space -> no log1p
         eval_metric="root_mean_squared_error", framing=True,  # sales right-skewed, many closed-day zeros
         drop=["Customers"],  # leak: near-perfect proxy for Sales, absent from the real test set
         note="1017209 rows -> subsampled to 15k. Store (1115 stores, ~942 days each) is a "
              "genuine repeating GROUP entity -- unlike bike-sharing's raw datetime, this exists "
              "as a column already, so the N2 timing gap (period not materialized before FE) "
              "does not apply here; this tests whether --fold-advisor picks GROUP correctly on "
              "real, messy data. The submission now trains on the competition's real RMSPE metric "
              "(custom scorer, closed-day zeros ignored); the battery verdict still uses rmse."),
    dict(name="walmart", semantics="rich+time+group", target="Weekly_Sales", metric="rmse",
         id_col="row_id", path="data/kaggle_walmart/train.csv",  # Store/Dept both repeat -- same
         # non-unique-id fix as rossmann above
         competition="walmart-recruiting-store-sales-forecasting", sample=15000,
         test_path="data/kaggle_walmart/test.csv",
         submit_id="Id", submit_id_construct=["Store", "Dept", "Date"],  # LB id = "Store_Dept_Date"
         submit_sample=200000,  # full train is 421k rows
         submit_eval_metric="mean_absolute_error",  # + sample weights below => the LB's WMAE
         submit_weight_from="IsHoliday", submit_weight=5.0,  # holiday weeks count ×5
         submit_use_llm=False,  # keep the weight column safe from cleaning (walmart's 4 columns
         # gain little from LLM judgment anyway) and let AutoGluon weight the metric directly
         submit_framing=False,  # weighted MAE is defined in original space -> no log1p
         eval_metric="root_mean_squared_error", framing=True,
         note="421570 rows -> subsampled to 15k. Store x Dept (~3331 combinations) repeats "
              "densely -- a second, independent group+time real task. Weekly_Sales can be "
              "genuinely NEGATIVE (returns > sales) -- clip_nonneg correctly stays off. The "
              "submission now optimises the competition's real weighted MAE (holiday weeks ×5 "
              "via AutoGluon sample weights, LLM off to protect the weight column); the battery "
              "verdict still uses plain rmse."),
    dict(name="ieee-fraud", semantics="poor+time", target="isFraud", metric="balanced_accuracy",
         id_col="TransactionID", path="data/kaggle_ieee/train_transaction.csv",
         competition="ieee-fraud-detection", sample=15000,
         test_path="data/kaggle_ieee/test_transaction.csv",
         eval_metric="accuracy", framing=False,
         submit_proba=True, submit_col="isFraud",  # LB metric is AUC -> submit P(fraud), not a label
         submit_eval_metric="roc_auc",
         submit_sample=50000,   # 393 wide columns kept -> rows are the memory lever; best_quality
         submit_presets="high_quality",  # bagging at 200k×393 OOMs on ~10GB (verified: the run
         # failed silently). high_quality bags lighter; 50k rows × 393 cols fits. Keeping the V
         # signal (below) matters more than best_quality here -- an honest memory/quality trade.
         submit_use_llm=False,  # keep all V1-V339 for the submission (real AUC signal); with the
         # LLM off there is no profiling call to blow the token limit, so nothing needs dropping
         llm_only_drop=[f"V{i}" for i in range(1, 340)],  # 339 anonymized PCA columns: dropped
         # ONLY for LLM runs (battery + any --llm submission) to fit gpt-4o's 30k TPM budget
         # (verified: FAILED live, RateLimitError, 31805 requested) -- not a leak, a token bound.
         note="590540 rows. Battery/LLM runs drop V1-V339 (token budget) -> ~55 cols; the "
              "submission keeps all 393 columns with the LLM off (the V-block carries real AUC "
              "signal, and the measured thesis says LLM cleaning adds little anyway). "
              "TransactionDT (time) + card1-6/addr1-2 (weak repeating group) on real fraud data. "
              "train_transaction.csv ONLY (train_identity.csv ignored -- multi-table non-goal). "
              "Real metric is AUC; balanced_accuracy used internally for the battery verdict."),
    dict(name="santander-transaction", semantics="poor", target="target",
         metric="balanced_accuracy", id_col="ID_code",
         path="data/kaggle_santander_transaction/train.csv",
         competition="santander-customer-transaction-prediction", sample=15000,
         test_path="data/kaggle_santander_transaction/test.csv",
         eval_metric="accuracy", framing=False,
         submit_proba=True, submit_col="target",  # LB metric is AUC -> submit P(target=1)
         submit_eval_metric="roc_auc", submit_sample=200000,  # full train is 200k rows
         note="200000 rows, 200 fully anonymized numeric columns (var_0..var_199) -> subsampled "
              "to 15k. A THIRD anonymized control (after friedman-synth/E2 and allstate/K1) on "
              "modern Kaggle data -- the thesis predicts inert, same as the others. Real metric "
              "is AUC; balanced_accuracy used internally."),
    dict(name="restaurant-revenue", semantics="rich+time", target="revenue", metric="rmse",
         id_col="Id", path="data/kaggle_restaurant/train.csv",
         competition="restaurant-revenue-prediction",
         test_path="data/kaggle_restaurant/test.csv",
         submit_col="Prediction",  # sample submission column is "Prediction", not "revenue"
         eval_metric="root_mean_squared_error", framing=True,  # revenue is right-skewed
         note="Only 137 rows -- no subsampling. 'Open Date' (temporal) + City/City Group/Type "
              "(rich semantics) + P1-P37 (anonymized census-style features). Famous for being "
              "small enough to overfit trivially -- a genuine small-n stress test, distinct from "
              "titanic's 891 rows."),
    dict(name="airbnb", semantics="rich+time", target="country_destination",
         metric="balanced_accuracy", id_col="id",
         path="data/kaggle_airbnb/train_users_2.csv",
         competition="airbnb-recruiting-new-user-bookings", sample=15000,
         test_path="data/kaggle_airbnb/test_users.csv",
         eval_metric="accuracy", framing=False,
         submit_col="country",   # sample submission column is "country"
         submit_topk=5,          # NDCG@5: emit the 5 most-probable countries per user, ranked
         submit_sample=200000,   # full train is 213k rows
         note="213451 rows -> subsampled to 15k. 12-class target (country_destination: NDF/US/"
              "other/FR/CA/GB/ES/IT/PT/NL/DE/AU), rich semantics (gender/age/signup_method/"
              "language/affiliate_channel/first_device_type) + weak time (date_account_created, "
              "timestamp_first_active). sessions.csv/countries.csv/age_gender_bkts.csv ignored "
              "(multi-table non-goal). Real metric is NDCG@5; the submission now emits the ranked "
              "top-5 (submit_topk), the battery still uses balanced_accuracy on the top label."),
    dict(name="two-sigma-rental", semantics="rich+group", target="interest_level",
         metric="balanced_accuracy", id_col="listing_id",
         path="data/kaggle_twosigma/train_flat.csv",
         competition="two-sigma-connect-rental-listing-inquiries", sample=15000,
         test_path="data/kaggle_twosigma/test_flat.csv",  # test.json flattened once, like train
         submit_id="listing_id", submit_proba=True,
         submit_proba_columns=["high", "medium", "low"],  # sample: listing_id,high,medium,low
         submit_eval_metric="log_loss",  # the competition's real multi-class log loss
         eval_metric="accuracy", framing=False,
         note="49352 rows. Ships as train.json with list-valued 'features'/'photos' columns -- "
              "flattened once to a plain CSV (train_flat.csv / test_flat.csv), those two columns "
              "dropped (not a leak, just an unsupported column type for this harness; the "
              "remaining bathrooms/bedrooms/price/manager_id/created/description/lat/long stay). "
              "manager_id is a genuine repeating GROUP entity (a landlord/agency posts many "
              "listings) on real semantic-rich data -- a second, structurally different group "
              "test from Rossmann/Walmart's Store. 3-class target (low/medium/high); real metric "
              "is multi-class log loss. Battery verdict uses balanced_accuracy; the submission "
              "emits the 3-class probability matrix (submit_proba_columns)."),
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
    # The battery always runs with the LLM (the maestra arm), so it drops both real leaks
    # (`drop`) and the token-budget columns (`llm_only_drop`) — the latter only exist to keep
    # the LLM profiling call under its rate limit, exactly the case the battery is in.
    for col in list(spec.get("drop", [])) + list(spec.get("llm_only_drop", [])):
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


def make_submission(spec: dict, *, model: str, time_limit: int, cv: int,
                    fold_advisor: bool = False, presets: str | None = "best_quality") -> None:
    """Train Maestra on the FULL train set and write a submittable prediction file.

    Uses the leakage-free CV for the honest estimate (that is the number the public LB gets
    compared against — the CV↔LB gap on a real leaderboard), the competition's eval metric,
    and target framing where enabled in the catalog (RMSLE comps: metric-aligned log1p; other
    regression: the arbiter decides). The competition's own test.csv is predicted through the
    same fitted transforms; the submission id is the competition's, not the battery's row id.

    Submission-only spec fields (never touch the battery run, which grades on carved keys):
      * ``submit_id`` — the id column of the competition's test.csv (default: the battery id_col).
      * ``submit_col`` — the prediction column name the sample submission expects (default: the
        target name; some competitions want e.g. ``Prediction`` not ``revenue``).
      * ``submit_proba`` — True for probability metrics (AUC): output P(positive class) instead
        of a hard label, in a column named ``submit_col``.
      * ``submit_topk`` — for ranking metrics (airbnb NDCG@5): emit the top-k classes per row in
        LONG format (id repeated k times, one class per row) instead of a single label. Implies
        multiclass probabilities under the hood.
      * ``submit_eval_metric`` — eval metric override for the submission run. A plain string, OR
        one of the custom-scorer keys in ``_CUSTOM_SCORERS`` (e.g. ``"rmspe"`` for Rossmann's
        real metric, which AutoGluon has no native scorer for).
      * ``submit_framing`` — override target framing for the submission (default: the catalog
        ``framing``). A custom scorer works in original space, so framing is turned off with it.
      * ``submit_use_llm`` — run with the LLM cleaning/FE off (default True). Set False to keep a
        very wide table the LLM profiling can't fit in its token budget (ieee-fraud's V1-V339).
      * ``llm_only_drop`` — columns dropped ONLY when the LLM runs (a token-budget bound, not a
        leak); kept when ``submit_use_llm`` is False so their signal reaches the engine.
      * ``submit_sample`` — cap the training rows (huge tables where full-train is infeasible);
        a large capped sample, not the battery's tiny 15k.
      * ``submit_presets`` — per-task override of the CLI ``--presets``. Set a lighter preset for
        a memory-heavy task (ieee-fraud: 393 wide columns × bagging OOMs at best_quality).
    """
    from maestra.pipeline import run_pipeline
    from maestra.runlog import append_run

    train_csv = _materialize(spec)
    test_path = spec.get("test_path")
    if train_csv is None or not test_path or not os.path.exists(test_path):
        print(f"\n=== {spec['name']}: not submittable "
              f"({'no test_path wired' if not test_path else 'competition data missing'}) ===")
        if test_path:
            print(_download_help(spec))
        return
    submit_id = spec.get("submit_id", spec["id_col"])
    submit_col = spec.get("submit_col", spec["target"])
    submit_proba = spec.get("submit_proba", False)
    submit_topk = spec.get("submit_topk")
    submit_use_llm = spec.get("submit_use_llm", True)
    submit_framing = spec.get("submit_framing", spec["framing"])
    submit_eval = _resolve_metric(spec.get("submit_eval_metric", spec["eval_metric"]))
    train = pd.read_csv(spec["path"])          # FULL train (no battery subsample)
    test_df = pd.read_csv(test_path)
    # `drop` = real leaks, always removed. `llm_only_drop` = a token-budget bound, removed only
    # when the LLM profiling actually runs (kept for a --no-llm submission so the engine sees it).
    to_drop = list(spec.get("drop", []))
    if submit_use_llm:
        to_drop += list(spec.get("llm_only_drop", []))
    for col in to_drop:                        # same columns dropped from train AND test
        train = train.drop(columns=[col], errors="ignore")
        test_df = test_df.drop(columns=[col], errors="ignore")
    # Some competitions key the submission on a COMPOSITE id the test set does not carry as a
    # column (Walmart: "Store_Dept_Date"). Build it by joining the named columns with "_".
    construct = spec.get("submit_id_construct")
    if construct:
        test_df[submit_id] = test_df[construct[0]].astype(str)
        for c in construct[1:]:
            test_df[submit_id] = test_df[submit_id] + "_" + test_df[c].astype(str)
    submit_sample = spec.get("submit_sample")
    if submit_sample and len(train) > submit_sample:
        train = train.sample(submit_sample, random_state=42).reset_index(drop=True)
        print(f"  (train capped to {submit_sample} rows for a feasible submission run)")
    presets = spec.get("submit_presets", presets)  # per-task override (memory-heavy tasks)
    # best_quality (and other bagging presets) do ~8-fold bagging + stacking; on very few rows
    # that overfits and AutoGluon can crash ("Learner is already fit"). Below ~2k rows a bagging
    # preset is worse, not "higher" -- fall back to the plain preset. Only tiny tasks hit this.
    if presets and presets != "medium_quality" and len(train) < 2000:
        print(f"  ({len(train)} rows is too few for '{presets}' bagging -> using medium_quality)")
        presets = "medium_quality"

    # Per-row training weights (Walmart's weighted MAE: holiday weeks count ×5). Built from a
    # truthy source column and handed to AutoGluon as a sample_weight column, which weights BOTH
    # training and the metric — the only way to optimise a WMAE the engine has no scorer for.
    sample_weight = None
    weight_from = spec.get("submit_weight_from")
    if weight_from and weight_from in train.columns:
        sample_weight = "__sample_weight__"
        heavy = train[weight_from].astype(str).str.upper().isin(["TRUE", "1", "T", "YES"])
        train[sample_weight] = np.where(heavy, float(spec.get("submit_weight", 5.0)), 1.0)
        print(f"  (sample weights from {weight_from!r}: {int(heavy.sum())} heavy rows ×"
              f"{spec.get('submit_weight', 5.0)})")

    # Probability-submission column plan:
    #  * submit_topk        -> full class matrix (wide), reshaped to long top-k below;
    #  * submit_proba_columns -> an EXPLICIT multiclass wide format the sample dictates, in its
    #    exact column order (two-sigma: listing_id,high,medium,low);
    #  * submit_proba       -> single positive-class probability (binary AUC).
    proba_columns = None
    if submit_topk:
        proba_columns = [str(c) for c in sorted(train[spec["target"]].dropna().unique())]
    elif spec.get("submit_proba_columns"):
        proba_columns = list(spec["submit_proba_columns"])
    elif submit_proba:
        proba_columns = [submit_col]

    print(f"\n=== {spec['name']}: submission run "
          f"(eval_metric={submit_eval}, framing={submit_framing}, proba={bool(proba_columns)}, "
          f"use_llm={submit_use_llm}, topk={submit_topk}, fold_advisor={fold_advisor}, "
          f"presets={presets}, train_rows={len(train)}) ===")
    result = run_pipeline(
        train, spec["target"], model=model, test_size=0.2, time_limit=time_limit,
        seed=42, model_dir=f"AutogluonModels/kaggle_{spec['name']}", cv_folds=cv,
        eval_metric=submit_eval, target_framing=submit_framing, use_llm=submit_use_llm,
        fold_advisor=fold_advisor, test_df=test_df, id_col=submit_id,
        proba=bool(proba_columns), proba_columns=proba_columns,
        presets=presets, sample_weight=sample_weight)

    submission = result.submission
    if submit_topk:  # wide class-prob frame -> long top-k rows (id repeated, one class each)
        submission = _topk_long(submission, submit_id, submit_col, submit_topk)
    elif not submit_proba and submit_col != spec["target"] and spec["target"] in submission.columns:
        # Label path names the prediction column after the target; rename to the sample's column.
        # (The plain proba path already emits submit_col directly.)
        submission = submission.rename(columns={spec["target"]: submit_col})
    out = f"data/submission_{spec['name']}.csv"
    submission.to_csv(out, index=False)
    append_run("runs.jsonl", result, csv=spec["path"], target=spec["target"], model=model,
               no_llm=not submit_use_llm, max_attempts=1,
               timestamp=datetime.now().isoformat(timespec="seconds"))

    cv_est = result.cv
    framing_note = ""
    if result.target_framing:
        framing_note = f" | framing: {result.target_framing['transform']}" \
                       f" accepted={result.target_framing['accepted']}"
    if result.fold_strategy:
        framing_note += f" | folds: {result.fold_strategy['strategy']}"
    print(f"\nsubmission written: {out}  ({len(submission)} rows, columns {list(submission.columns)})")
    print(f"CV estimate ({cv_est.eval_metric}): {cv_est.mean:.4f} ± {cv_est.std:.4f}{framing_note}")
    if result.fold_strategy:
        for line in result.fold_strategy["log"]:
            print(f"  {line}")
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
    p.add_argument("--fold-advisor", action="store_true",
                   help="Validation Strategist on both arms (N2, 2026-07-05) — lets the "
                        "bike-sharing temporal task pick up time_local instead of random folds.")
    p.add_argument("--presets", default="best_quality",
                   help="AutoGluon quality preset for --make-submission runs (default "
                        "best_quality: multi-layer stacking + bagging, strong but slow). Pass "
                        "'medium_quality' for a fast draft, or '' to use AutoGluon's own default.")
    args = p.parse_args()
    load_dotenv()

    if args.make_submission:
        todo = CATALOG if args.make_submission == "all" \
            else [s for s in CATALOG if s["name"] == args.make_submission]
        if not todo:
            raise SystemExit(f"unknown task {args.make_submission!r} — see --list")
        for spec in todo:
            make_submission(spec, model=args.model, time_limit=args.time_limit, cv=args.cv,
                            fold_advisor=args.fold_advisor, presets=args.presets or None)
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
                            fold_advisor=args.fold_advisor,
                            name=f"kaggle-{spec['name']}")
        ts = datetime.now().isoformat(timespec="seconds")
        for r in ms.per_seed:
            append_result("benchmark.jsonl", r, timestamp=ts)
        append_multi_seed("benchmark.jsonl", ms, timestamp=ts)


if __name__ == "__main__":
    main()
