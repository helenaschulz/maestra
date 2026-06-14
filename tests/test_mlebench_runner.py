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


def test_run_mlebench_task_smoke(tmp_path, monkeypatch):
    from maestra.pipeline import PipelineResult
    from maestra.validation import CVResult

    _make_task(tmp_path / "toycomp")
    submission = pd.DataFrame({"id": [5, 6], "label": [1, 0]})
    cv = CVResult("roc_auc", "binary", [0.90, 0.92], 0.91, 0.01, 2, True, True)

    monkeypatch.setattr(mr, "run_pipeline",
                        lambda *a, **k: PipelineResult(n_cols_before=3, n_cols_after=2, plan=None,
                                                       submission=submission, cv=cv))
    monkeypatch.setattr(mr, "grade_submission",
                        lambda *a, **k: mr.GradeReport(score=0.88, gold=0.95, silver=0.90,
                                                       bronze=0.85, medal="bronze"))

    out_dir = tmp_path / "out"
    runs_log = tmp_path / "runs.jsonl"
    record = mr.run_mlebench_task(str(tmp_path / "toycomp"), "toycomp", eval_metric="roc_auc",
                                  out_dir=str(out_dir), runs_log=str(runs_log))

    # CV score + graded score + the all-important CV↔LB gap
    assert record["cv_score"] == 0.91 and record["mle_score"] == 0.88
    assert record["cv_lb_gap"] == pytest.approx(0.03)
    assert record["medal"] == "bronze" and record["metric_aligned"] is True
    assert record["cv_metric"] == "roc_auc"

    # a valid submission was written in the sample-submission format
    written = pd.read_csv(out_dir / "toycomp_maestra_submission.csv")
    assert list(written.columns) == ["id", "label"]
    assert written["id"].tolist() == [5, 6]

    # and the record is in runs.jsonl
    logged = json.loads(runs_log.read_text().splitlines()[-1])
    assert logged["kind"] == "mlebench" and logged["cv_lb_gap"] == pytest.approx(0.03)


def test_grade_submission_without_mlebench_raises(monkeypatch):
    real_import = builtins.__import__

    def no_mlebench(name, *args, **kwargs):
        if name.startswith("mlebench"):
            raise ImportError("mlebench not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mlebench)
    with pytest.raises(MleBenchError, match="not installed"):
        mr.grade_submission("/tmp/sub.csv", "toycomp")
