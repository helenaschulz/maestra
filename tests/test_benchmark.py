"""Tests for the benchmark harness. grade() is checked against known metric values; run_task
is wired against a mocked run_pipeline (no network, no AutoGluon)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from maestra import benchmark
from maestra.benchmark import BenchResult, append_result, grade, run_task, summary


def _answer_and_submission():
    answer = pd.DataFrame({"id": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    submission = pd.DataFrame({"id": [1, 2, 3, 4], "y": [0, 1, 1, 1]})  # one wrong (id 3)
    return answer, submission


def test_grade_accuracy_and_balanced_accuracy():
    answer, sub = _answer_and_submission()
    assert grade(sub, answer, metric="accuracy", id_col="id", target="y") == 0.75
    assert grade(sub, answer, metric="balanced_accuracy", id_col="id", target="y") == 0.75


def test_grade_unknown_metric_raises():
    answer, sub = _answer_and_submission()
    with pytest.raises(ValueError, match="Unknown metric"):
        grade(sub, answer, metric="roc_auc", id_col="id", target="y")


def test_grade_incomplete_submission_raises():
    answer, sub = _answer_and_submission()
    with pytest.raises(ValueError, match="covers"):
        grade(sub.iloc[:2], answer, metric="accuracy", id_col="id", target="y")


def test_run_task_grades_maestra_and_baseline(tmp_path, monkeypatch):
    df = pd.DataFrame({"id": range(8), "f": [0, 1, 0, 1, 0, 1, 0, 1], "y": [0, 1] * 4})
    csv = tmp_path / "toy.csv"
    df.to_csv(csv, index=False)
    truth = dict(zip(df["id"], df["y"]))

    def fake_run(work, target, *, use_llm, test_df, id_col, **kwargs):
        ids = test_df[id_col].tolist()
        preds = [truth[i] for i in ids] if use_llm else [0] * len(ids)  # maestra perfect, baseline all-0
        return SimpleNamespace(submission=pd.DataFrame({id_col: ids, target: preds}))

    monkeypatch.setattr(benchmark, "run_pipeline", fake_run)
    r = run_task(str(csv), "y", metric="accuracy", id_col="id", time_limit=1, seed=0, holdout_frac=0.5)

    assert r.maestra == 1.0          # perfect predictions
    assert r.baseline == 0.5         # all-0 on a balanced answer key
    assert r.delta > 0
    assert benchmark._winner(r.delta, r.higher_is_better) == "maestra"


def test_summary_renders_and_handles_missing(tmp_path):
    p = str(tmp_path / "b.jsonl")
    assert "No benchmark results" in summary(p)
    append_result(p, BenchResult("toy", "accuracy", 0.50, 0.80, 0.30, True, 6, 2), timestamp="t")
    out = summary(p)
    assert "toy" in out and "maestra" in out
