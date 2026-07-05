"""Tests for run memory (N4): only DECIDED verdicts are retrievable, formatted as non-binding
context. No LLM/AutoGluon involved -- pure parsing/formatting + pipeline wiring."""
import json

from maestra.run_memory import format_memory_context, load_decided_verdicts, memory_context


def _write(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_load_decided_verdicts_excludes_undecided_and_non_multi_seed_rows(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    _write(path, [
        {"kind": None, "name": "single-run", "verdict": None},          # not multi_seed -> excluded
        {"kind": "multi_seed", "name": "a", "verdict": "undecided"},    # excluded: the whole point
        {"kind": "multi_seed", "name": "b", "verdict": "maestra"},
        {"kind": "multi_seed", "name": "c", "verdict": "baseline"},
    ])
    decided = load_decided_verdicts(str(path))
    assert [r["name"] for r in decided] == ["b", "c"]


def test_load_decided_verdicts_missing_file_returns_empty_not_an_error():
    assert load_decided_verdicts("does/not/exist.jsonl") == []


def test_format_memory_context_none_when_nothing_decided():
    assert format_memory_context([]) is None


def test_format_memory_context_caps_and_labels_non_binding():
    records = [{"name": f"task-{i}", "verdict": "maestra", "delta": 1.0 + i,
               "metric": "rmse", "seeds": [1, 2, 3]} for i in range(8)]
    ctx = format_memory_context(records, max_entries=5)
    assert ctx.count("task-") == 5                    # capped, not all 8
    assert "task-7" in ctx and "task-0" not in ctx     # keeps the most recent
    assert "non-binding" in ctx.lower() or "context only" in ctx.lower()
    assert "never treat these as a reason to skip" in ctx.lower()


def test_format_memory_context_states_direction_for_both_verdicts():
    records = [
        {"name": "won", "verdict": "maestra", "delta": 5.0, "metric": "rmse", "seeds": [1, 2, 3]},
        {"name": "lost", "verdict": "baseline", "delta": -5.0, "metric": "rmse", "seeds": [1, 2, 3]},
    ]
    ctx = format_memory_context(records)
    assert "won" in ctx and "Maestra beat the baseline" in ctx
    assert "lost" in ctx and "the baseline beat Maestra" in ctx


def test_memory_context_convenience_roundtrip(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    _write(path, [{"kind": "multi_seed", "name": "x", "verdict": "maestra", "delta": 2.0,
                  "metric": "rmse", "seeds": [1, 2, 3]}])
    ctx = memory_context(str(path))
    assert ctx is not None and "x" in ctx


def test_pipeline_threads_memory_context_into_judgment_nodes(tmp_path, monkeypatch):
    """run_memory=True must reach the SAME shared context channel as dataset_description --
    every judgment node (cleaning, FE, ...) is fed the same picture."""
    import pandas as pd

    from maestra import pipeline
    from maestra.engine import TrainingResult

    path = tmp_path / "benchmark.jsonl"
    _write(path, [{"kind": "multi_seed", "name": "past-task", "verdict": "maestra", "delta": 3.0,
                  "metric": "rmse", "seeds": [1, 2, 3]}])

    captured = {}
    df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
    training = TrainingResult("binary", "accuracy", pd.DataFrame({"model": ["m"]}), {"accuracy": 0.9})

    def fake_propose_cleaning(model, profile, target, context):
        captured["context"] = context
        return {"columns_to_drop": [], "imputations": []}

    monkeypatch.setattr(pipeline, "propose_cleaning_plan", fake_propose_cleaning)
    monkeypatch.setattr(pipeline, "train_and_evaluate", lambda *a, **k: training)

    pipeline.run_pipeline(df, "y", model="m", test_size=0.2, time_limit=1, seed=0,
                          model_dir="x", use_fe=False, run_memory=True, memory_path=str(path))

    assert captured["context"] is not None
    assert "past-task" in captured["context"]
    assert "Maestra beat the baseline" in captured["context"]


def test_pipeline_run_memory_off_by_default_leaves_context_untouched(monkeypatch):
    import pandas as pd

    from maestra import pipeline
    from maestra.engine import TrainingResult

    captured = {}
    df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
    training = TrainingResult("binary", "accuracy", pd.DataFrame({"model": ["m"]}), {"accuracy": 0.9})

    def fake_propose_cleaning(model, profile, target, context):
        captured["context"] = context
        return {"columns_to_drop": [], "imputations": []}

    monkeypatch.setattr(pipeline, "propose_cleaning_plan", fake_propose_cleaning)
    monkeypatch.setattr(pipeline, "train_and_evaluate", lambda *a, **k: training)

    pipeline.run_pipeline(df, "y", model="m", test_size=0.2, time_limit=1, seed=0,
                          model_dir="x", use_fe=False)  # run_memory defaults to False

    assert captured["context"] is None
