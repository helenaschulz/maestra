"""Pipeline wiring tests. AutoGluon and the LLM are mocked so the suite is fast and
offline — we assert the conductor calls the right steps and threads data correctly."""
import pandas as pd
import pytest

from maestra import pipeline
from maestra.engine import TrainingResult
from maestra.pipeline import run_pipeline


@pytest.fixture
def df():
    return pd.DataFrame({"id": [1, 2, 3, 4], "f": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})


@pytest.fixture
def fake_training():
    return TrainingResult(
        problem_type="binary",
        eval_metric="accuracy",
        leaderboard=pd.DataFrame({"model": ["m"], "score_test": [0.9]}),
        metrics={"accuracy": 0.9},
    )


def _patch_engine(monkeypatch, fake_training):
    monkeypatch.setattr(
        pipeline, "train_and_evaluate", lambda *a, **k: fake_training
    )


class _FakePredictor:
    def predict(self, X):
        return pd.Series(["A"] * len(X), index=X.index)


def test_submission_built_from_test_set(df, monkeypatch):
    training = TrainingResult("binary", "accuracy", pd.DataFrame({"model": ["m"]}),
                              {"accuracy": 0.9}, predictor=_FakePredictor())
    monkeypatch.setattr(pipeline, "train_and_evaluate", lambda *a, **k: training)

    test_df = pd.DataFrame({"id": [101, 102, 103], "f": [9.0, 8.0, 7.0]})
    result = run_pipeline(df, "y", model="m", test_size=0.25, time_limit=1, seed=0,
                          model_dir="x", use_llm=False, test_df=test_df, id_col="id")

    sub = result.submission
    assert list(sub.columns) == ["id", "y"]
    assert sub["id"].tolist() == [101, 102, 103]   # ids preserved, in test order
    assert sub["y"].tolist() == ["A", "A", "A"]


class _FakeBinaryProbaPredictor:
    positive_class = 1  # AutoGluon's designated positive class

    def predict_proba(self, X):  # columns are class labels (here ints, as from AutoGluon)
        return pd.DataFrame({0: [0.3] * len(X), 1: [0.7] * len(X)}, index=X.index)


class _FakeMulticlassProbaPredictor:
    def predict_proba(self, X):
        n = len(X)
        return pd.DataFrame({"A": [0.2] * n, "B": [0.3] * n, "C": [0.5] * n}, index=X.index)


def _run_proba(df, predictor, proba_columns, monkeypatch):
    training = TrainingResult("binary", "roc_auc", pd.DataFrame({"model": ["m"]}),
                              {"roc_auc": 0.9}, predictor=predictor)
    monkeypatch.setattr(pipeline, "train_and_evaluate", lambda *a, **k: training)
    test_df = pd.DataFrame({"id": [101, 102, 103], "f": [9.0, 8.0, 7.0]})
    return run_pipeline(df, "y", model="m", test_size=0.25, time_limit=1, seed=0,
                        model_dir="x", use_llm=False, test_df=test_df, id_col="id",
                        proba=True, proba_columns=proba_columns).submission


def test_binary_proba_submission_single_column(df, monkeypatch):
    # One non-id column that is not the class set -> positive-class probability.
    sub = _run_proba(df, _FakeBinaryProbaPredictor(), ["y"], monkeypatch)
    assert list(sub.columns) == ["id", "y"]
    assert sub["id"].tolist() == [101, 102, 103]
    assert sub["y"].tolist() == [0.7, 0.7, 0.7]   # P(positive class = 1)


def test_multiclass_proba_submission_per_class_in_order(df, monkeypatch):
    # Columns equal the class labels -> one probability per class, in sample-submission order.
    sub = _run_proba(df, _FakeMulticlassProbaPredictor(), ["C", "A", "B"], monkeypatch)
    assert list(sub.columns) == ["id", "C", "A", "B"]          # exactly the requested order
    assert sub["A"].tolist() == [0.2, 0.2, 0.2]
    row_sums = sub[["C", "A", "B"]].sum(axis=1)
    assert all(s == pytest.approx(1.0) for s in row_sums)       # probabilities sum to ~1


def test_proba_submission_format_mismatch_aborts(df, monkeypatch):
    # Two columns that are neither the full class set nor a single column -> clear abort.
    with pytest.raises(pipeline.PipelineError, match="match neither"):
        _run_proba(df, _FakeMulticlassProbaPredictor(), ["A", "B"], monkeypatch)


def test_missing_id_col_raises(df, fake_training, monkeypatch):
    _patch_engine(monkeypatch, fake_training)
    test_df = pd.DataFrame({"row": [1], "f": [9.0]})  # no 'id' column
    with pytest.raises(pipeline.PipelineError, match="id column"):
        run_pipeline(df, "y", model="m", test_size=0.25, time_limit=1, seed=0,
                     model_dir="x", use_llm=False, test_df=test_df, id_col="id")


def test_unknown_target_raises(df):
    with pytest.raises(ValueError, match="not in CSV"):
        run_pipeline(df, "missing", model="m", test_size=0.25, time_limit=1, seed=0, model_dir="x")


def test_no_llm_skips_cleaning(df, fake_training, monkeypatch):
    _patch_engine(monkeypatch, fake_training)

    def boom(*a, **k):
        raise AssertionError("LLM must not be called with use_llm=False")

    monkeypatch.setattr(pipeline, "propose_cleaning_plan", boom)

    result = run_pipeline(
        df, "y", model="m", test_size=0.25, time_limit=1, seed=0, model_dir="x", use_llm=False
    )
    assert result.plan is None
    assert result.n_cols_before == result.n_cols_after == 3
    assert result.training.metrics["accuracy"] == 0.9


def test_llm_path_applies_plan_before_training(df, fake_training, monkeypatch):
    _patch_engine(monkeypatch, fake_training)
    monkeypatch.setattr(
        pipeline,
        "propose_cleaning_plan",
        lambda *a, **k: {"columns_to_drop": [{"column": "id", "reason": "ID"}], "imputations": []},
    )

    result = run_pipeline(
        df, "y", model="gpt-4o", test_size=0.25, time_limit=1, seed=0, model_dir="x", use_fe=False
    )
    assert result.plan is not None
    assert result.n_cols_before == 3
    assert result.n_cols_after == 2  # 'id' dropped
    assert any("DROP 'id'" in line for line in result.cleaning_log)


def test_dataset_description_reaches_the_planner(df, fake_training, monkeypatch):
    """M0: a provider-written description flows into the planners' shared context channel."""
    _patch_engine(monkeypatch, fake_training)
    captured = {}

    def fake_clean(model, profile, target, research_context=None):
        captured["ctx"] = research_context
        return {"columns_to_drop": [], "imputations": []}

    monkeypatch.setattr(pipeline, "propose_cleaning_plan", fake_clean)

    run_pipeline(df, "y", model="m", test_size=0.25, time_limit=1, seed=0, model_dir="x",
                 use_fe=False, dataset_description="f: the fare in USD; id: row number")

    assert "the fare in USD" in captured["ctx"]          # description text arrived
    assert "Dataset description" in captured["ctx"]      # wrapped in the prompt-ready block


def test_hybrid_with_no_candidates_is_equivalent_to_plain_cv(df, monkeypatch):
    """Closes an old suspicion: on leaf, hybrid runs with kept=0 scored differently from plain
    runs. Verified cause was LLM plan drift between invocations, NOT a path difference — with
    identical plans, the hybrid path's base CV must receive exactly the same arguments as the
    plain CV path (model_dir aside), and the final fit must see identical data."""
    from maestra.engine import TrainingResult

    captured = []

    def fake_cross_validate(df_, target, **kwargs):
        from maestra.validation import CVResult
        captured.append({**kwargs, "n_rows": len(df_)})
        return CVResult("accuracy", "binary", [0.8], 0.8, 0.0, 2, True, True)

    fitted = []
    monkeypatch.setattr(pipeline, "propose_cleaning_plan",
                        lambda *a, **k: {"columns_to_drop": [], "imputations": []})
    monkeypatch.setattr(pipeline, "propose_feature_code", lambda *a, **k: [])  # no candidates
    monkeypatch.setattr(pipeline, "cross_validate", fake_cross_validate)
    monkeypatch.setattr("maestra.hybrid_features.cross_validate", fake_cross_validate)
    monkeypatch.setattr(pipeline, "fit_predictor",
                        lambda train, *a, **k: fitted.append(list(train.columns)) or
                        TrainingResult("binary", "accuracy", pd.DataFrame(), {}))

    common = dict(model="m", test_size=0.25, time_limit=1, seed=0, cv_folds=2, use_fe=False)
    pipeline.run_pipeline(df, "y", model_dir="plain", **common)
    pipeline.run_pipeline(df, "y", model_dir="hyb", hybrid=True, **common)

    plain_kwargs, hybrid_kwargs = captured[0], captured[1]
    plain_kwargs.pop("model_dir"), hybrid_kwargs.pop("model_dir")   # the only allowed difference
    assert plain_kwargs == hybrid_kwargs                            # identical CV inputs
    assert fitted[0] == fitted[1]                                   # identical final-fit columns


# --- opt-in research wiring --------------------------------------------------------

def test_research_runs_feeds_planning_and_logs(df, fake_training, monkeypatch):
    from maestra.research import ResearchResult

    _patch_engine(monkeypatch, fake_training)
    rr = ResearchResult(
        brief={"summary": "s", "references": [{"url": "https://x/1"}], "grounded": True},
        rules_mode="live",
    )
    rules_seen = []

    def fake_research(model, problem, *, profile, rules_mode):
        rules_seen.append(rules_mode)
        return rr

    captured = {}

    def fake_clean(model, profile, target, research_context=None):
        captured["ctx"] = research_context
        return {"columns_to_drop": [], "imputations": []}

    monkeypatch.setattr(pipeline, "research_strategy", fake_research)
    monkeypatch.setattr(pipeline, "propose_cleaning_plan", fake_clean)

    result = run_pipeline(df, "y", model="m", test_size=0.25, time_limit=1, seed=0,
                          model_dir="x", research=True, rules_mode="live", use_fe=False)

    assert rules_seen == ["live"]                 # research ran, in the requested mode
    assert captured["ctx"] is not None            # brief context reached the cleaning planner
    assert result.research == {"rules_mode": "live", "references": ["https://x/1"], "grounded": True}


def test_no_research_by_default_leaves_path_unchanged(df, fake_training, monkeypatch):
    _patch_engine(monkeypatch, fake_training)

    def boom(*a, **k):
        raise AssertionError("research_strategy must not run without research=True")

    monkeypatch.setattr(pipeline, "research_strategy", boom)
    monkeypatch.setattr(pipeline, "propose_cleaning_plan",
                        lambda *a, **k: {"columns_to_drop": [], "imputations": []})

    result = run_pipeline(df, "y", model="m", test_size=0.25, time_limit=1, seed=0,
                          model_dir="x", use_fe=False)  # research defaults to False
    assert result.research is None


# --- hybrid features (gate mocked) -------------------------------------------------

def test_hybrid_requires_cv(df):
    with pytest.raises(ValueError, match="requires --cv"):
        run_pipeline(df, "y", model="m", test_size=0.25, time_limit=1, seed=0, model_dir="x", hybrid=True)


def test_hybrid_runs_gate_and_logs_provenance(df, fake_training, monkeypatch):
    from dataclasses import asdict

    from maestra.hybrid_features import CandidateRecord, GeneratedFeature
    from maestra.validation import CVResult

    cv = CVResult("accuracy", "binary", [0.80, 0.82], 0.81, 0.01, 2, True, True)
    rec = CandidateRecord("g", "an idea", "profile", 0.05, True, "improved")

    monkeypatch.setattr(pipeline, "propose_cleaning_plan",
                        lambda *a, **k: {"columns_to_drop": [], "imputations": []})
    monkeypatch.setattr(pipeline, "propose_feature_code", lambda *a, **k: [GeneratedFeature("g", "i", "c")])
    monkeypatch.setattr(pipeline, "select_features",
                        lambda *a, **k: ([GeneratedFeature("g", "i", "c")], [rec], cv))
    monkeypatch.setattr(pipeline, "apply_generated_features", lambda train, other, target, feats: (train, other))
    monkeypatch.setattr(pipeline, "fit_predictor", lambda *a, **k: fake_training)

    result = run_pipeline(df, "y", model="m", test_size=0.25, time_limit=1, seed=0, model_dir="x",
                          cv_folds=2, hybrid=True, use_fe=False)

    assert result.cv is cv                          # the gated CV (with kept features) is reported
    assert result.hybrid == [asdict(rec)]           # provenance logged (kept, delta, reason, source)


def test_text_lane_feeds_the_shared_gate(df, fake_training, monkeypatch):
    """Text candidates go through the SAME select_features gate; the summary records the lane."""
    from maestra.hybrid_features import CandidateRecord, GeneratedFeature
    from maestra.validation import CVResult

    cv = CVResult("accuracy", "binary", [0.80, 0.82], 0.81, 0.01, 2, True, True)
    text_feat = GeneratedFeature("kw_group", "luxury terms", "code", source="text")
    gate_calls = {}

    def fake_gate(_df, _target, candidates, **kwargs):
        gate_calls["candidates"] = candidates
        rec = CandidateRecord("kw_group", "luxury terms", "text", 0.02, True, "improved")
        return [text_feat], [rec], cv

    monkeypatch.setattr(pipeline, "propose_cleaning_plan",
                        lambda *a, **k: {"columns_to_drop": [], "imputations": []})
    monkeypatch.setattr(pipeline, "detect_text_columns", lambda *a, **k: ["description"])
    monkeypatch.setattr(pipeline, "propose_text_feature_code", lambda *a, **k: [text_feat])
    monkeypatch.setattr(pipeline, "select_features", fake_gate)
    monkeypatch.setattr(pipeline, "apply_generated_features", lambda train, other, target, feats: (train, other))
    monkeypatch.setattr(pipeline, "fit_predictor", lambda *a, **k: fake_training)

    result = run_pipeline(df, "y", model="m", test_size=0.25, time_limit=1, seed=0, model_dir="x",
                          cv_folds=2, text_features=True, use_fe=False)

    assert gate_calls["candidates"] == [text_feat]   # the text lane fed the shared gate
    assert result.text_features == {"columns": ["description"], "n_candidates": 1}
    assert result.hybrid[0]["source"] == "text"      # provenance distinguishes the lane


def test_text_lane_skips_gate_when_no_text_columns(df, fake_training, monkeypatch):
    """No detected text columns -> no LLM call, no extra gate CV; summary still recorded."""
    from maestra.validation import CVResult

    cv = CVResult("accuracy", "binary", [0.80, 0.82], 0.81, 0.01, 2, True, True)
    monkeypatch.setattr(pipeline, "cross_validate", lambda *a, **k: cv)
    monkeypatch.setattr(pipeline, "propose_cleaning_plan",
                        lambda *a, **k: {"columns_to_drop": [], "imputations": []})
    monkeypatch.setattr(pipeline, "detect_text_columns", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "propose_text_feature_code",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    monkeypatch.setattr(pipeline, "select_features",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    monkeypatch.setattr(pipeline, "fit_predictor", lambda *a, **k: fake_training)

    result = run_pipeline(df, "y", model="m", test_size=0.25, time_limit=1, seed=0, model_dir="x",
                          cv_folds=2, text_features=True, use_fe=False)

    assert result.text_features == {"columns": [], "n_candidates": 0}
    assert result.hybrid is None
