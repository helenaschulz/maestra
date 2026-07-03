"""Smoke tests for the CLI layer: argument parsing and the fail-fast error paths that run
before any LLM/AutoGluon work. The heavy paths are covered via the pipeline tests."""
import pandas as pd
import pytest

from maestra import cli
from maestra.cli import _parse_args, _resolve_fold_advisor


def test_parse_args_defaults():
    args = _parse_args(["--csv", "x.csv", "--target", "y"])
    assert args.model_dir == "AutogluonModels" and args.seed == 42
    assert args.cv is None and not args.hybrid and not args.skeptic
    assert args.fold_advisor is None             # tri-state "auto" (resolved against --cv)
    assert args.hybrid_threshold == 2.0          # the hardened gate default
    assert args.id_col == "id" and args.runs_log == "runs.jsonl"


@pytest.mark.parametrize("argv, expected", [
    ([], False),                                 # no cv, no flag -> off (never errors)
    (["--cv", "3"], True),                       # cv active -> ON by default (the M9 promotion)
    (["--cv", "3", "--no-fold-advisor"], False),  # explicit opt-out wins
    (["--cv", "3", "--fold-advisor"], True),     # explicit opt-in
    (["--fold-advisor"], True),                  # explicit on w/o cv -> True (pipeline then errors)
    (["--no-fold-advisor"], False),              # explicit off w/o cv
])
def test_fold_advisor_default_on_with_cv(argv, expected):
    args = _parse_args(["--csv", "x.csv", "--target", "y", *argv])
    assert _resolve_fold_advisor(args.fold_advisor, args.cv) is expected


def test_missing_csv_fails_fast(capsys):
    rc = cli.main(["--csv", "does_not_exist.csv", "--target", "y"])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_submission_requires_test(tmp_path, capsys):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2], "y": [0, 1]}).to_csv(csv, index=False)
    rc = cli.main(["--csv", str(csv), "--target", "y", "--submission", "out.csv"])
    assert rc == 1
    assert "--submission requires --test" in capsys.readouterr().err


def test_missing_description_file_fails_fast(tmp_path, capsys):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2], "y": [0, 1]}).to_csv(csv, index=False)
    rc = cli.main(["--csv", str(csv), "--target", "y", "--description", "missing.txt"])
    assert rc == 1
    assert "description file not found" in capsys.readouterr().err
