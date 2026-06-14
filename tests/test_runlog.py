"""Tests for the run log + baseline comparison. Pure file I/O, no network or AutoGluon."""
import pandas as pd

from maestra.engine import TrainingResult
from maestra.pipeline import PipelineResult
from maestra.runlog import append_run, compare_runs, _read_runs


def _result(metrics):
    training = TrainingResult("binary", "accuracy", pd.DataFrame(), metrics)
    return PipelineResult(n_cols_before=12, n_cols_after=8, plan=None, training=training)


def _log(path, *, no_llm, metrics, ts):
    append_run(path, _result(metrics), csv="data/titanic.csv", target="Survived",
               model="gpt-4o", no_llm=no_llm, max_attempts=1, timestamp=ts)


def test_append_writes_one_json_line_per_run(tmp_path):
    p = str(tmp_path / "runs.jsonl")
    _log(p, no_llm=True, metrics={"accuracy": 0.80}, ts="2026-06-14T10:00:00")
    _log(p, no_llm=False, metrics={"accuracy": 0.83}, ts="2026-06-14T10:05:00")

    runs = _read_runs(p)
    assert len(runs) == 2
    assert runs[0]["no_llm"] is True and runs[0]["metrics"]["accuracy"] == 0.80
    assert runs[1]["target"] == "Survived" and runs[1]["attempts"] == 1


def test_compare_shows_metric_diff(tmp_path):
    p = str(tmp_path / "runs.jsonl")
    _log(p, no_llm=True, metrics={"accuracy": 0.80, "roc_auc": 0.87}, ts="2026-06-14T10:00:00")
    _log(p, no_llm=False, metrics={"accuracy": 0.83, "roc_auc": 0.88}, ts="2026-06-14T10:05:00")

    out = compare_runs(p, "data/titanic.csv", "Survived")
    assert "0.8000 -> 0.8300  (+0.0300)" in out
    assert "0.8700 -> 0.8800  (+0.0100)" in out


def test_compare_uses_latest_of_each(tmp_path):
    p = str(tmp_path / "runs.jsonl")
    _log(p, no_llm=False, metrics={"accuracy": 0.50}, ts="2026-06-14T09:00:00")  # stale llm
    _log(p, no_llm=True, metrics={"accuracy": 0.80}, ts="2026-06-14T10:00:00")
    _log(p, no_llm=False, metrics={"accuracy": 0.83}, ts="2026-06-14T10:05:00")  # latest llm

    out = compare_runs(p, "data/titanic.csv", "Survived")
    assert "0.8000 -> 0.8300" in out  # uses latest llm (0.83), not the stale 0.50


def test_compare_needs_both_sides(tmp_path):
    p = str(tmp_path / "runs.jsonl")
    _log(p, no_llm=False, metrics={"accuracy": 0.83}, ts="2026-06-14T10:00:00")
    out = compare_runs(p, "data/titanic.csv", "Survived")
    assert "Need one run with and one without" in out


def test_compare_missing_log_is_graceful(tmp_path):
    out = compare_runs(str(tmp_path / "nope.jsonl"), "data/titanic.csv", "Survived")
    assert "No run log" in out
