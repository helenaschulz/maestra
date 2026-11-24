"""Offline tests for N6: the outer fold strategy reaches AutoGluon's INTERNAL validation via
``validation_structure``. AutoGluon is never imported for real -- ``TabularPredictor`` is replaced
by a recording fake in ``sys.modules``, and ``cross_validate``'s fold fit is monkeypatched, same
convention as test_validation.py."""
import sys
import types

import pandas as pd
import pytest

from maestra import engine as engine_mod
from maestra import validation
from maestra.engine import (
    TrainingResult,
    inner_validation_label,
    inner_validation_structure,
    outer_strategy,
    resolve_inner_validation,
)


class _FakePredictor:
    """Records constructor + fit kwargs; answers the few calls train_and_evaluate makes."""

    calls: list = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.problem_type = "regression"
        self.eval_metric = types.SimpleNamespace(name="r2", greater_is_better=True)

    def fit(self, train, **kwargs):
        _FakePredictor.calls.append({**kwargs, "_train": train, "_init": self.init_kwargs})
        return self

    def leaderboard(self, *args, **kwargs):
        return pd.DataFrame({"model": ["m"], "score_val": [0.5]})

    def evaluate(self, *args, **kwargs):
        return {"r2": 0.4}


@pytest.fixture
def fake_ag(monkeypatch):
    _FakePredictor.calls = []
    mod = types.ModuleType("autogluon.tabular")
    mod.TabularPredictor = _FakePredictor
    monkeypatch.setitem(sys.modules, "autogluon.tabular", mod)
    return _FakePredictor


def _df():
    return pd.DataFrame({"g": [1, 1, 2, 2, 3, 3], "t": range(6), "x": range(6),
                         "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})


# (a) None -> AutoGluon's fit sees no new keyword at all (call identical to pre-N6)
def test_none_structure_passes_no_keyword(fake_ag):
    engine_mod.train_and_evaluate(_df(), _df(), "y", 5, "m")
    engine_mod.fit_predictor(_df(), "y", 5, "m")
    assert len(fake_ag.calls) == 2
    for kwargs in fake_ag.calls:
        assert "validation_structure" not in kwargs
        assert set(kwargs) == {"time_limit", "presets", "_train", "_init"}
        assert "learner_kwargs" not in kwargs["_init"]
        assert list(kwargs["_train"].columns) == list(_df().columns)  # no carrier column


def test_structure_is_forwarded_to_fit(fake_ag):
    engine_mod.train_and_evaluate(_df(), _df(), "y", 5, "m",
                                  validation_structure={"group_on": "g"})
    engine_mod.fit_predictor(_df(), "y", 5, "m", validation_structure={"group_on": "g"})
    carrier = "__maestra_group_on__"
    for k in fake_ag.calls:
        # declared on an IGNORED carrier copy, so the feature set is exactly pre-N6
        assert k["validation_structure"] == {"group_on": carrier}
        assert k["_init"]["learner_kwargs"] == {"ignored_columns": [carrier]}
        assert k["_train"][carrier].tolist() == _df()["g"].tolist()
        assert "g" in k["_train"].columns  # the real column is untouched


def test_string_dates_become_an_int_time_carrier(fake_ag):
    df = _df().assign(t=pd.date_range("2020-01-01", periods=6).strftime("%Y-%m-%d"))
    engine_mod.fit_predictor(df, "y", 5, "m",
                             validation_structure={"time_on": "t", "temporal_forward_only": True})
    k = fake_ag.calls[0]
    assert k["validation_structure"] == {"time_on": "__maestra_time_on__",
                                         "temporal_forward_only": True}
    carrier = k["_train"]["__maestra_time_on__"]
    assert carrier.is_monotonic_increasing and carrier.dtype.kind == "i"
    assert k["_train"]["t"].dtype == object  # real column NOT converted (no new datetime features)


def test_unparseable_time_column_raises_clearly(fake_ag):
    df = _df().assign(t=["a", "b", "c", "d", "e", "f"])
    with pytest.raises(ValueError, match="don't parse as time"):
        engine_mod.fit_predictor(df, "y", 5, "m", validation_structure={"time_on": "t"})


def test_structure_column_dropped_by_cleaning_is_carried_from_the_source(fake_ag):
    raw = _df()
    cleaned = raw.drop(columns=["g"])
    engine_mod.fit_predictor(cleaned, "y", 5, "m", validation_structure={"group_on": "g"},
                             structure_source=raw)
    k = fake_ag.calls[0]
    assert "g" not in k["_train"].columns
    assert k["_train"]["__maestra_group_on__"].tolist() == raw["g"].tolist()

def test_groups_init_argument_is_never_used(fake_ag, monkeypatch):
    """validation_structure is exclusive with TabularPredictor(groups=...) in AG 1.6."""
    inits = []
    orig = _FakePredictor.__init__

    def spy(self, **kwargs):
        inits.append(kwargs)
        orig(self, **kwargs)

    monkeypatch.setattr(_FakePredictor, "__init__", spy)
    engine_mod.fit_predictor(_df(), "y", 5, "m", validation_structure={"group_on": "g"})
    assert "groups" not in inits[0]


# (b) group -> group_on
def test_group_maps_to_group_on():
    s, note = inner_validation_structure("group", group_column="g")
    assert s == {"group_on": "g"}
    assert "g" in note


# (c) time / time_local / rolling origin -> time_on + temporal_forward_only
@pytest.mark.parametrize("strategy", ["time", "time_local", "rolling_origin"])
def test_time_maps_to_time_on_forward(strategy):
    s, _ = inner_validation_structure(strategy, time_column="t")
    assert s == {"time_on": "t", "temporal_forward_only": True}


# (d) stacking -> no temporal_forward_only, and the record says so
def test_stacking_drops_forward_only_with_a_note():
    s, rec = resolve_inner_validation(["t", "x"], time_column="t", presets="best_quality")
    assert s == {"time_on": "t"}
    assert "stacking" in rec["note"]
    assert "leave-one-block-out" in rec["note"]


def test_random_is_none():
    assert inner_validation_structure("random") == (None, "random")
    assert resolve_inner_validation(["x"])[0] is None


def test_outer_strategy_mirrors_make_folds_precedence():
    assert outer_strategy(group_column="g", time_column="t") == "group"
    assert outer_strategy(time_column="t", period_column="month_of:t") == "time_local"
    assert outer_strategy(time_column="t") == "time"
    assert outer_strategy(splitter=object(), time_column="t", group_column="g") == "rolling_origin"
    assert outer_strategy() == "random"
    with pytest.raises(ValueError):
        outer_strategy(period_column="p")  # N6 stop condition: period without time


def test_time_local_uses_the_time_column_inside():
    s, rec = resolve_inner_validation(["t", "x"], time_column="t", period_column="month_of:t")
    assert s == {"time_on": "t", "temporal_forward_only": True}
    assert rec["strategy"] == "time_local"


def test_dropped_structure_column_falls_back_loudly():
    s, rec = resolve_inner_validation(["x"], group_column="g")
    assert s is None
    assert "not available" in rec["note"]
    assert inner_validation_label(rec) == "zufällig"


def test_disabled_is_the_pre_n6_arm():
    s, rec = resolve_inner_validation(["g"], group_column="g", enabled=False)
    assert s is None and rec["strategy"] == "group"


def test_labels():
    assert inner_validation_label({"structure": {"group_on": "g"}}) == "gruppen-disjunkt auf g"
    assert inner_validation_label(
        {"structure": {"time_on": "t", "temporal_forward_only": True}}) == "zeitlich vorwärts auf t"
    assert inner_validation_label(None) == "zufällig"


# (e) cross_validate hands the structure to EVERY fold fit
def _spy_cv(monkeypatch):
    seen = []

    def fake(train, val, target, time_limit, model_dir, eval_metric=None, presets=None,
             sample_weight=None, validation_structure=None, structure_source=None):
        seen.append(validation_structure)
        return TrainingResult("regression", "r2", pd.DataFrame(), {"r2": 0.5}, val_score=0.6)

    monkeypatch.setattr(validation, "train_and_evaluate", fake)
    return seen


def _grouped(n=60):
    return pd.DataFrame({"g": [i % 6 for i in range(n)], "t": range(n),
                         "x": [i * 0.5 for i in range(n)], "y": [float(i % 7) for i in range(n)]})


def test_cross_validate_passes_group_on_into_every_fold(monkeypatch):
    seen = _spy_cv(monkeypatch)
    cv = validation.cross_validate(_grouped(), "y", cleaning_plan=None, feature_plan=None,
                                   model_dir="m", time_limit=1, n_folds=3, group_column="g")
    assert seen == [{"group_on": "g"}] * 3
    assert cv.inner_validation["structure"] == {"group_on": "g"}
    assert cv.val_scores == [0.6, 0.6, 0.6]


def test_cross_validate_passes_time_on_into_every_fold(monkeypatch):
    seen = _spy_cv(monkeypatch)
    validation.cross_validate(_grouped(), "y", cleaning_plan=None, feature_plan=None,
                              model_dir="m", time_limit=1, n_folds=3, time_column="t")
    assert seen == [{"time_on": "t", "temporal_forward_only": True}] * 3


def test_cross_validate_random_passes_nothing(monkeypatch):
    seen = _spy_cv(monkeypatch)
    cv = validation.cross_validate(_grouped(), "y", cleaning_plan=None, feature_plan=None,
                                   model_dir="m", time_limit=1, n_folds=3)
    assert seen == [None] * 3
    assert cv.inner_validation["note"] == "random"


def test_cross_validate_carries_a_column_cleaning_drops(monkeypatch):
    """The group id is dropped by cleaning (typical for an id column) -- the structure still
    reaches every fold, read from the raw rows."""
    sources = []

    def fake(train, val, target, time_limit, model_dir, eval_metric=None, presets=None,
             sample_weight=None, validation_structure=None, structure_source=None):
        sources.append((validation_structure, "g" in train.columns,
                        structure_source is not None and "g" in structure_source.columns))
        return TrainingResult("regression", "r2", pd.DataFrame(), {"r2": 0.5})

    monkeypatch.setattr(validation, "train_and_evaluate", fake)
    plan = {"columns_to_drop": ["g"], "imputations": []}
    validation.cross_validate(_grouped(), "y", cleaning_plan=plan, feature_plan=None,
                              model_dir="m", time_limit=1, n_folds=3, group_column="g")
    assert sources == [({"group_on": "g"}, False, True)] * 3


def test_cross_validate_opt_out_arm(monkeypatch):
    seen = _spy_cv(monkeypatch)
    validation.cross_validate(_grouped(), "y", cleaning_plan=None, feature_plan=None,
                              model_dir="m", time_limit=1, n_folds=3, group_column="g",
                              inner_structure=False)
    assert seen == [None] * 3


def test_runlog_records_version_and_inner_validation(tmp_path):
    import json

    from maestra.pipeline import PipelineResult
    from maestra.runlog import append_run

    res = PipelineResult(n_cols_before=3, n_cols_after=3, plan=None, autogluon_version="1.6.3",
                         inner_validation={"cv": {"structure": {"group_on": "g"}}, "final": None})
    path = tmp_path / "runs.jsonl"
    append_run(str(path), res, csv="c", target="y", model="m", no_llm=True, max_attempts=1,
               timestamp="t")
    rec = json.loads(path.read_text())
    assert rec["autogluon_version"] == "1.6.3"
    assert rec["inner_validation"]["cv"]["structure"] == {"group_on": "g"}


def test_dossier_shows_the_inner_validation_line():
    from maestra.dossier import render_dossier
    from maestra.pipeline import PipelineResult

    res = PipelineResult(n_cols_before=3, n_cols_after=3, plan=None, autogluon_version="1.6.3",
                         inner_validation={"cv": {"structure": {"group_on": "firm"},
                                                  "note": "group-disjoint on firm"},
                                           "final": None})
    html = render_dossier(res)
    assert "Interne Modellauswahl" in html
    assert "gruppen-disjunkt auf firm" in html
    assert "AutoGluon 1.6.3" in html
