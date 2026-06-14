"""Tests for the MLE-bench adapter. mlebench (grading) and run_pipeline (AutoGluon) are both
mocked, so this verifies the adapter plumbing offline with a tiny synthetic task."""
import builtins
import json

import pandas as pd
import pytest

from maestra import mlebench_runner as mr
from maestra.mlebench_runner import MleBenchError, read_task


def _make_task(dir_path):
    dir_path.mkdir()
    pd.DataFrame({"id": [1, 2, 3, 4], "f": [1, 2, 3, 4], "label": [0, 1, 0, 1]}).to_csv(
        dir_path / "train.csv", index=False)
    pd.DataFrame({"id": [5, 6], "f": [5, 6]}).to_csv(dir_path / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "label": [0, 0]}).to_csv(dir_path / "sample_submission.csv", index=False)


def test_read_task_derives_id_and_target(tmp_path):
    _make_task(tmp_path / "toycomp")
    task = read_task(str(tmp_path / "toycomp"))
    assert task.id_col == "id"          # the sample-submission column also present in test.csv
    assert task.target_col == "label"   # the other column
    assert task.name == "toycomp"


def _result(cv):
    from maestra.pipeline import PipelineResult
    submission = pd.DataFrame({"id": [5, 6], "label": [1, 0]})
    return PipelineResult(n_cols_before=3, n_cols_after=2, plan=None, submission=submission, cv=cv)


def test_run_mlebench_task_smoke_aligned_metric(tmp_path, monkeypatch):
    from maestra.validation import CVResult

    _make_task(tmp_path / "toycomp")
    cv = CVResult("accuracy", "binary", [0.90, 0.92], 0.91, 0.01, 2, True, True)
    monkeypatch.setattr(mr, "run_pipeline", lambda *a, **k: _result(cv))
    monkeypatch.setattr(mr, "grade_submission",
                        lambda *a, **k: mr.GradeReport(score=0.88, gold=0.95, silver=0.90,
                                                       bronze=0.85, medal="bronze"))
    out_dir, runs_log = tmp_path / "out", tmp_path / "runs.jsonl"

    record = mr.run_mlebench_task(str(tmp_path / "toycomp"), "toycomp", metric="accuracy",
                                  out_dir=str(out_dir), runs_log=str(runs_log))

    assert record["cv_score"] == 0.91 and record["mle_score"] == 0.88   # CV (in the metric) + LB
    assert record["cv_lb_gap"] == pytest.approx(0.03)                   # comparable: aligned metric
    assert record["metric_mode"] == "aligned" and record["medal"] == "bronze"

    written = pd.read_csv(out_dir / "toycomp_maestra_submission.csv")
    assert list(written.columns) == ["id", "label"] and written["id"].tolist() == [5, 6]
    logged = json.loads(runs_log.read_text().splitlines()[-1])
    assert logged["kind"] == "mlebench" and logged["cv_lb_gap"] == pytest.approx(0.03)


def test_oof_metric_computed_on_predictions(tmp_path, monkeypatch):
    """For a metric with no AutoGluon equivalent (quadratic_weighted_kappa), the CV score is
    computed on the out-of-fold predictions — making the gap comparable."""
    from maestra import benchmark
    from maestra.validation import CVResult

    _make_task(tmp_path / "toycomp")  # train labels = [0, 1, 0, 1] at index 0..3
    oof = pd.Series([0, 1, 1, 1])     # out-of-fold predictions
    cv = CVResult("accuracy", "binary", [0.5, 0.5], 0.5, 0.0, 2, True, True, oof_pred=oof)
    monkeypatch.setattr(mr, "run_pipeline", lambda *a, **k: _result(cv))
    monkeypatch.setattr(mr, "grade_submission",
                        lambda *a, **k: mr.GradeReport(score=0.40, gold=None, silver=None, bronze=None, medal=None))

    record = mr.run_mlebench_task(str(tmp_path / "toycomp"), "toycomp", metric="quadratic_weighted_kappa",
                                  out_dir=str(tmp_path / "out"), runs_log=str(tmp_path / "runs.jsonl"))

    expected = benchmark._METRICS["quadratic_weighted_kappa"]([0, 1, 0, 1], [0, 1, 1, 1])
    assert record["metric_mode"] == "oof"
    assert record["cv_score"] == pytest.approx(expected)          # computed on OOF preds, not AG default
    assert record["cv_lb_gap"] == pytest.approx(expected - 0.40)  # comparable


def test_multitarget_submission_aborts(tmp_path):
    d = tmp_path / "multi"
    d.mkdir()
    pd.DataFrame({"id": [1], "a": [1], "b": [2]}).to_csv(d / "train.csv", index=False)
    pd.DataFrame({"id": [2]}).to_csv(d / "test.csv", index=False)
    pd.DataFrame({"id": [2], "a": [0], "b": [0]}).to_csv(d / "sample_submission.csv", index=False)
    with pytest.raises(MleBenchError, match="unsupported submission shape"):
        read_task(str(d))


def test_grade_submission_without_mlebench_raises(monkeypatch):
    real_import = builtins.__import__

    def no_mlebench(name, *args, **kwargs):
        if name.startswith("mlebench"):
            raise ImportError("mlebench not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mlebench)
    with pytest.raises(MleBenchError, match="not installed"):
        mr.grade_submission("/tmp/sub.csv", "toycomp")
