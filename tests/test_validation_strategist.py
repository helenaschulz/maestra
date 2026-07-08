"""Tests for the Validation Strategist: the LLM proposes a fold strategy, deterministic code
verifies it, and the fold builders honour it. The LLM is never called (proposals are dicts)."""
import numpy as np
import pandas as pd

from maestra import validation
from maestra.validation_strategist import check_validation, validate_fold_strategy


def _grouped_df(n_groups=10, rows_per_group=6):
    rng = np.random.default_rng(0)
    gid = np.repeat(np.arange(n_groups), rows_per_group)
    return pd.DataFrame({
        "customer_id": gid,
        "x": rng.normal(size=n_groups * rows_per_group),
        "y": (gid % 2),  # target depends on the entity — the leaky setup
    })


# --- fold builders -----------------------------------------------------------------

def test_group_folds_never_split_an_entity():
    df = _grouped_df()
    folds = validation._make_folds(df, "y", 3, seed=0, stratified=False, group_column="customer_id")
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        train_groups = set(df["customer_id"].iloc[train_idx])
        val_groups = set(df["customer_id"].iloc[val_idx])
        assert not (train_groups & val_groups)  # an entity lives in exactly one side


def test_time_folds_validate_strictly_later_than_train():
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=30, freq="D")[np.random.default_rng(1).permutation(30)],
        "x": range(30),
        "y": [0, 1] * 15,
    })
    folds = validation._make_folds(df, "y", 3, seed=0, stratified=False, time_column="ts")
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        assert df["ts"].iloc[train_idx].max() < df["ts"].iloc[val_idx].min()  # past -> future only


def _repeating_period_df(n_periods=4, rows_per_period=12):
    """Two months' worth of daily rows, repeated -- the bike-sharing shape: a within-period
    time order (day) plus a repeating period (month)."""
    rng = np.random.default_rng(2)
    rows = []
    for p in range(n_periods):
        ts = pd.date_range(f"2024-{p + 1:02d}-01", periods=rows_per_period, freq="D")
        order = rng.permutation(rows_per_period)  # rows arrive out of order, like real data
        rows.append(pd.DataFrame({"month": p, "ts": ts[order], "y": range(rows_per_period)}))
    return pd.concat(rows, ignore_index=True)


def test_time_local_folds_pool_every_period_and_validate_strictly_later_within_it():
    df = _repeating_period_df()
    folds = validation._make_folds(df, "y", 3, seed=0, stratified=False,
                                   time_column="ts", period_column="month")
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        # every fold is exposed to every period (curing the global-time-split's expanding bias)
        assert set(df["month"].iloc[train_idx]) == set(df["month"].iloc[val_idx]) == set(df["month"])
        for p in df["month"].unique():
            p_train = df[(df.index.isin(train_idx)) & (df["month"] == p)]
            p_val = df[(df.index.isin(val_idx)) & (df["month"] == p)]
            if len(p_train) and len(p_val):
                assert p_train["ts"].max() < p_val["ts"].min()  # local past -> future, per period


def test_materialize_period_derives_month_week_dayofweek_tokens():
    from maestra.validation import _materialize_period

    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=10, freq="D")})
    assert _materialize_period(df, "month_of:ts").tolist() == [1] * 10
    assert _materialize_period(df, "dayofweek_of:ts").tolist() == \
        df["ts"].dt.dayofweek.tolist()
    assert _materialize_period(df, "week_of:ts").tolist() == \
        df["ts"].dt.isocalendar().week.astype(int).tolist()


def test_time_local_folds_accept_a_period_token_without_a_materialised_column():
    """F2: _time_local_folds works from a period_candidates token alone (e.g. bike-sharing's
    raw 'datetime'), the same shape validate_fold_strategy now verifies."""
    df = _repeating_period_df()
    df = df.rename(columns={"ts": "datetime"}).drop(columns=["month"])
    folds = validation._make_folds(df, "y", 3, seed=0, stratified=False,
                                   time_column="datetime", period_column="month_of:datetime")
    assert len(folds) == 3
    months = df["datetime"].dt.month
    for train_idx, val_idx in folds:
        assert set(months.iloc[train_idx]) == set(months.iloc[val_idx]) == set(months)
        for m in months.unique():
            p_train = df[(df.index.isin(train_idx)) & (months == m)]
            p_val = df[(df.index.isin(val_idx)) & (months == m)]
            if len(p_train) and len(p_val):
                assert p_train["datetime"].max() < p_val["datetime"].min()


def test_time_local_folds_are_deterministic():
    df = _repeating_period_df()
    a = validation._make_folds(df, "y", 3, seed=0, stratified=False, time_column="ts", period_column="month")
    b = validation._make_folds(df, "y", 3, seed=0, stratified=False, time_column="ts", period_column="month")
    assert [v.tolist() for _, v in a] == [v.tolist() for _, v in b]


# --- deterministic verification of proposals ----------------------------------------

def test_group_proposal_verified(monkeypatch):
    df = _grouped_df()
    proposal = {"strategy": "group", "group_column": "customer_id",
                "rationale": "six rows per customer"}
    verified, log = validate_fold_strategy(proposal, df, "y")
    assert verified["strategy"] == "group" and verified["group_column"] == "customer_id"
    assert any("FOLDS group by 'customer_id'" in line for line in log)


def test_nonexistent_column_falls_back_to_random():
    df = _grouped_df()
    verified, log = validate_fold_strategy(
        {"strategy": "group", "group_column": "patient_id", "rationale": "r"}, df, "y")
    assert verified["strategy"] == "random" and verified["group_column"] is None
    assert any("fallback" in line for line in log)


def test_group_column_without_repeats_falls_back():
    df = pd.DataFrame({"rowid": range(10), "x": range(10), "y": [0, 1] * 5})
    verified, log = validate_fold_strategy(
        {"strategy": "group", "group_column": "rowid", "rationale": "r"}, df, "y")
    assert verified["strategy"] == "random"
    assert any("no repeated entities" in line for line in log)


def test_unparseable_time_column_falls_back():
    df = pd.DataFrame({"when": ["yesterday", "??", "soon", "later"], "y": [0, 1, 0, 1]})
    verified, log = validate_fold_strategy(
        {"strategy": "time", "time_column": "when", "rationale": "r"}, df, "y")
    assert verified["strategy"] == "random"
    assert any("not sortable" in line for line in log)


def test_time_local_proposal_verified():
    df = _repeating_period_df()
    proposal = {"strategy": "time_local", "time_column": "ts", "period_column": "month",
                "rationale": "repeating within-month split"}
    verified, log = validate_fold_strategy(proposal, df, "y")
    assert verified["strategy"] == "time_local"
    assert verified["time_column"] == "ts" and verified["period_column"] == "month"
    assert any("FOLDS time-local within 'month'" in line for line in log)


def test_time_local_missing_period_column_falls_back_to_random():
    df = _repeating_period_df()
    proposal = {"strategy": "time_local", "time_column": "ts", "period_column": "quarter",
                "rationale": "r"}
    verified, log = validate_fold_strategy(proposal, df, "y")
    assert verified["strategy"] == "random" and verified["period_column"] is None
    assert any("period column" in line and "fallback" in line for line in log)


def test_time_local_too_few_periods_falls_back():
    df = pd.DataFrame({"month": [0] * 10, "ts": pd.date_range("2024-01-01", periods=10),
                       "y": range(10)})
    proposal = {"strategy": "time_local", "time_column": "ts", "period_column": "month",
                "rationale": "r"}
    verified, log = validate_fold_strategy(proposal, df, "y")
    assert verified["strategy"] == "random"
    assert any("fewer than 2 periods" in line for line in log)


def _bike_sharing_shape_df(n_months=4, rows_per_month=12):
    """The N2 integration-gap shape: only a RAW datetime column, no separate period column —
    period_candidates tokens (F2) are the only way to name a period here."""
    rng = np.random.default_rng(3)
    rows = []
    for m in range(1, n_months + 1):
        ts = pd.date_range(f"2024-{m:02d}-01", periods=rows_per_month, freq="D")
        order = rng.permutation(rows_per_month)
        rows.append(pd.DataFrame({"datetime": ts[order], "y": range(rows_per_month)}))
    return pd.concat(rows, ignore_index=True)


def test_time_local_proposal_verified_with_a_period_token():
    """F2: period_column may be a period_candidates token naming a period not yet materialised
    onto df (closes the N2 gap — the Strategist sees this BEFORE any FE datetime split)."""
    df = _bike_sharing_shape_df()
    proposal = {"strategy": "time_local", "time_column": "datetime",
                "period_column": "month_of:datetime", "rationale": "repeats monthly"}
    verified, log = validate_fold_strategy(proposal, df, "y")
    assert verified["strategy"] == "time_local"
    assert verified["time_column"] == "datetime"
    assert verified["period_column"] == "month_of:datetime"
    assert any("FOLDS time-local within 'month_of:datetime'" in line for line in log)
    assert "month_of:datetime" not in df.columns  # verification never materialises onto df


def test_time_local_period_token_referencing_nonexistent_column_falls_back():
    df = _bike_sharing_shape_df()
    proposal = {"strategy": "time_local", "time_column": "datetime",
                "period_column": "month_of:no_such_column", "rationale": "r"}
    verified, log = validate_fold_strategy(proposal, df, "y")
    assert verified["strategy"] == "random"
    assert any("nonexistent column" in line for line in log)


def test_time_local_period_token_too_few_periods_falls_back():
    df = _bike_sharing_shape_df(n_months=1)  # a single month -> only 1 period, not >= 2
    proposal = {"strategy": "time_local", "time_column": "datetime",
                "period_column": "month_of:datetime", "rationale": "r"}
    verified, log = validate_fold_strategy(proposal, df, "y")
    assert verified["strategy"] == "random"
    assert any("fewer than 2 periods" in line for line in log)


def test_time_local_unparseable_time_column_falls_back():
    df = pd.DataFrame({"month": [0, 0, 1, 1], "when": ["yesterday", "??", "soon", "later"],
                       "y": [0, 1, 0, 1]})
    proposal = {"strategy": "time_local", "time_column": "when", "period_column": "month",
                "rationale": "r"}
    verified, log = validate_fold_strategy(proposal, df, "y")
    assert verified["strategy"] == "random"
    assert any("not sortable" in line for line in log)


def test_leakage_warnings_are_logged_not_applied():
    df = _grouped_df()
    proposal = {"strategy": "random", "rationale": "iid",
                "leakage_warnings": [{"column": "x", "reason": "recorded after the outcome"}]}
    verified, log = validate_fold_strategy(proposal, df, "y")
    assert verified["strategy"] == "random"
    assert any("LEAKAGE WARNING 'x'" in line for line in log)
    assert "x" in df.columns  # advice only — nothing was dropped


# --- pipeline wiring (LLM + engine mocked) ------------------------------------------

def test_pipeline_threads_group_column_into_cv(monkeypatch):
    from maestra import pipeline
    from maestra.validation import CVResult

    df = _grouped_df()
    captured = {}

    def fake_cross_validate(df_, target, **kwargs):
        captured.update(kwargs)
        return CVResult("accuracy", "binary", [0.8, 0.8], 0.8, 0.0, 2, False, True)

    monkeypatch.setattr(pipeline, "propose_fold_strategy",
                        lambda *a, **k: {"strategy": "group", "group_column": "customer_id",
                                         "rationale": "entities repeat"})
    monkeypatch.setattr(pipeline, "propose_cleaning_plan",
                        lambda *a, **k: {"columns_to_drop": [], "imputations": []})
    monkeypatch.setattr(pipeline, "cross_validate", fake_cross_validate)
    monkeypatch.setattr(pipeline, "fit_predictor",
                        lambda *a, **k: __import__("maestra.engine", fromlist=["TrainingResult"]).TrainingResult(
                            "binary", "accuracy", pd.DataFrame(), {}))

    result = pipeline.run_pipeline(df, "y", model="m", test_size=0.2, time_limit=1, seed=0,
                                   model_dir="x", cv_folds=2, fold_advisor=True, use_fe=False)

    assert captured["group_column"] == "customer_id"          # the verified strategy reached the CV
    assert result.fold_strategy["strategy"] == "group"        # and is reported for the log
    assert any("FOLDS group" in line for line in result.fold_strategy["log"])


def test_pipeline_threads_period_column_into_cv(monkeypatch):
    """time_local's period_column reaches cross_validate the same way group_column does."""
    from maestra import pipeline
    from maestra.validation import CVResult

    df = _repeating_period_df()
    captured = {}

    def fake_cross_validate(df_, target, **kwargs):
        captured.update(kwargs)
        return CVResult("rmse", "regression", [0.8, 0.8], 0.8, 0.0, 2, False, False)

    monkeypatch.setattr(pipeline, "propose_fold_strategy",
                        lambda *a, **k: {"strategy": "time_local", "time_column": "ts",
                                         "period_column": "month", "rationale": "repeats monthly"})
    monkeypatch.setattr(pipeline, "propose_cleaning_plan",
                        lambda *a, **k: {"columns_to_drop": [], "imputations": []})
    monkeypatch.setattr(pipeline, "cross_validate", fake_cross_validate)
    monkeypatch.setattr(pipeline, "fit_predictor",
                        lambda *a, **k: __import__("maestra.engine", fromlist=["TrainingResult"]).TrainingResult(
                            "regression", "root_mean_squared_error", pd.DataFrame(), {}))

    result = pipeline.run_pipeline(df, "y", model="m", test_size=0.2, time_limit=1, seed=0,
                                   model_dir="x", cv_folds=2, fold_advisor=True, use_fe=False)

    assert captured["period_column"] == "month"                    # verified strategy reached the CV
    assert result.fold_strategy["strategy"] == "time_local"
    assert any("FOLDS time-local" in line for line in result.fold_strategy["log"])


def test_fold_advisor_runs_without_llm_cleaning(monkeypatch):
    """The Strategist is independent of use_llm — a --no-llm baseline can still get smart folds
    (regression: it was wrongly gated on use_llm, so the real-data experiment silently used random)."""
    from maestra import pipeline
    from maestra.engine import TrainingResult
    from maestra.validation import CVResult

    df = _grouped_df()
    captured = {}
    monkeypatch.setattr(pipeline, "propose_fold_strategy",
                        lambda *a, **k: {"strategy": "group", "group_column": "customer_id",
                                         "rationale": "entities repeat"})
    monkeypatch.setattr(pipeline, "cross_validate",
                        lambda df_, target, **k: captured.update(k) or
                        CVResult("accuracy", "binary", [0.8], 0.8, 0.0, 2, False, True))
    monkeypatch.setattr(pipeline, "fit_predictor",
                        lambda *a, **k: TrainingResult("binary", "accuracy", pd.DataFrame(), {}))

    result = pipeline.run_pipeline(df, "y", model="m", test_size=0.2, time_limit=1, seed=0,
                                   model_dir="x", cv_folds=2, fold_advisor=True, use_llm=False)

    assert captured["group_column"] == "customer_id"      # advisor ran despite use_llm=False
    assert result.fold_strategy["strategy"] == "group"


def test_fold_advisor_requires_cv():
    import pytest

    from maestra.pipeline import run_pipeline
    df = _grouped_df()
    with pytest.raises(ValueError, match="requires --cv"):
        run_pipeline(df, "y", model="m", test_size=0.2, time_limit=1, seed=0,
                     model_dir="x", fold_advisor=True)


def test_group_folds_stay_stratified_for_classification():
    """With a class target, group folds must balance the class mix (StratifiedGroupKFold)
    while still never splitting an entity."""
    df = _grouped_df(n_groups=12, rows_per_group=5)
    folds = validation._make_folds(df, "y", 3, seed=0, stratified=True, group_column="customer_id")
    for train_idx, val_idx in folds:
        assert not (set(df["customer_id"].iloc[train_idx]) & set(df["customer_id"].iloc[val_idx]))
        # labels are entity-bound here, so perfect balance is impossible — but every fold must
        # contain BOTH classes (an unstratified group split can produce single-class folds)
        assert 0.0 < df["y"].iloc[val_idx].mean() < 1.0


def test_few_entity_group_gets_a_treatment_factor_warning():
    """The PlantGrowth trap: a verified group column with very few entities is legitimate
    (small panels exist) but gets an explicit verify-this note in the log."""
    df = pd.DataFrame({"arm": ["ctrl", "trt1", "trt2"] * 10, "x": range(30), "y": [0, 1] * 15})
    verified, log = validate_fold_strategy(
        {"strategy": "group", "group_column": "arm", "rationale": "r"}, df, "y")
    assert verified["strategy"] == "group"                       # judgment not overridden
    assert any("treatment/design factor" in line for line in log)


def test_check_validation_is_a_thin_dataframe_wrapper(monkeypatch):
    """P3 public API: DataFrame in, no CSV/CLI involved, same two calls audit() makes
    internally (propose_fold_strategy + validate_fold_strategy), plus the log."""
    from maestra import validation_strategist as vs_mod

    captured = {}

    def fake_propose(model, profile, target, context=None):
        captured["model"], captured["context"] = model, context
        return {"strategy": "group", "group_column": "customer_id", "rationale": "r"}

    monkeypatch.setattr(vs_mod, "propose_fold_strategy", fake_propose)
    df = _grouped_df()
    result = check_validation(df, "y", model="gpt-4o", description="a churn dataset")
    assert result["strategy"] == "group" and result["group_column"] == "customer_id"
    assert any("FOLDS group by 'customer_id'" in line for line in result["log"])
    assert captured["model"] == "gpt-4o" and "churn dataset" in captured["context"]


def test_check_validation_reachable_as_public_maestra_import(monkeypatch):
    from maestra import check_validation as public_check_validation
    from maestra import validation_strategist as vs_mod

    monkeypatch.setattr(vs_mod, "propose_fold_strategy",
                        lambda model, profile, target, context=None: {"strategy": "random"})
    result = public_check_validation(_grouped_df(), "y")
    assert result["strategy"] == "random"
