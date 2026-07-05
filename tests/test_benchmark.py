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


def test_proba_metrics_score_probabilities():
    """roc_auc / log_loss are computed on class probabilities (one column per class), not
    labels — binary uses the positive-class column, multiclass the full matrix in column order."""
    from sklearn.metrics import log_loss, roc_auc_score

    y = [0, 1, 0, 1]
    binary = pd.DataFrame({0: [0.8, 0.3, 0.6, 0.2], 1: [0.2, 0.7, 0.4, 0.8]})
    assert benchmark._PROBA_METRICS["roc_auc"](y, binary, positive_class=1) == pytest.approx(
        roc_auc_score(y, binary[1]))

    ym = ["A", "B", "C", "A"]
    multi = pd.DataFrame({"A": [0.7, 0.1, 0.2, 0.6], "B": [0.2, 0.8, 0.2, 0.3], "C": [0.1, 0.1, 0.6, 0.1]})
    assert benchmark._PROBA_METRICS["log_loss"](ym, multi, positive_class=None) == pytest.approx(
        log_loss(ym, multi, labels=["A", "B", "C"]))


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
        return SimpleNamespace(submission=pd.DataFrame({id_col: ids, target: preds}), hybrid=None)

    monkeypatch.setattr(benchmark, "run_pipeline", fake_run)
    r = run_task(str(csv), "y", metric="accuracy", id_col="id", time_limit=1, seed=0, holdout_frac=0.5)

    assert r.maestra == 1.0          # perfect predictions
    assert r.baseline == 0.5         # all-0 on a balanced answer key
    assert r.delta > 0
    assert benchmark._winner(r.delta, r.higher_is_better) == "maestra"


def test_run_task_threads_fold_advisor_into_both_arms(tmp_path, monkeypatch):
    """N2 (2026-07-05): fold_advisor must reach BOTH arms via run_task's `common` dict, or a
    random-vs-advised comparison would be confounded with use_llm (the bug this closes: the K1
    battery/--make-submission path never exercised the fold advisor at all)."""
    df = pd.DataFrame({"id": range(8), "f": [0, 1, 0, 1, 0, 1, 0, 1], "y": [0, 1] * 4})
    csv = tmp_path / "toy.csv"
    df.to_csv(csv, index=False)
    captured = []

    def fake_run(work, target, *, use_llm, test_df, id_col, **kwargs):
        captured.append(kwargs.get("fold_advisor"))
        return SimpleNamespace(submission=pd.DataFrame(
            {id_col: test_df[id_col].tolist(), target: [0] * len(test_df)}), hybrid=None)

    monkeypatch.setattr(benchmark, "run_pipeline", fake_run)
    run_task(str(csv), "y", metric="accuracy", id_col="id", time_limit=1, seed=0,
            holdout_frac=0.5, cv_folds=3, fold_advisor=True)

    assert captured == [True, True]      # both arms (maestra, baseline) got it


def test_summary_renders_and_handles_missing(tmp_path):
    p = str(tmp_path / "b.jsonl")
    assert "No benchmark results" in summary(p)
    append_result(p, BenchResult("toy", "accuracy", 0.50, 0.80, 0.30, True, 6, 2), timestamp="t")
    out = summary(p)
    assert "toy" in out and "maestra" in out


# --- M8: multi-seed mode (run_task mocked; the paired verdict logic is the unit under test) ---

def _bench(seed, baseline, maestra, higher=True):
    from maestra.benchmark import BenchResult
    return BenchResult(name="toy", metric="m", baseline=baseline, maestra=maestra,
                       delta=maestra - baseline, higher_is_better=higher,
                       n_train=80, n_grade=20, seed=seed)


def test_multi_seed_clear_win_is_maestra(monkeypatch):
    from maestra import benchmark as B
    results = {1: _bench(1, 0.80, 0.85), 2: _bench(2, 0.81, 0.86), 3: _bench(3, 0.79, 0.84)}
    monkeypatch.setattr(B, "run_task", lambda csv, target, *, metric, seed, **k: results[seed])

    ms = B.run_multi_seed("x.csv", "y", metric="m", seeds=[1, 2, 3])
    assert ms.verdict == "maestra"
    assert ms.mean_delta == pytest.approx(0.05)
    assert [r.seed for r in ms.per_seed] == [1, 2, 3]


def test_multi_seed_noisy_result_is_undecided(monkeypatch):
    from maestra import benchmark as B
    # Maestra ahead twice by a hair, behind once by a lot -> neither side passes the paired rule
    results = {1: _bench(1, 0.80, 0.805), 2: _bench(2, 0.80, 0.803), 3: _bench(3, 0.80, 0.76)}
    monkeypatch.setattr(B, "run_task", lambda csv, target, *, metric, seed, **k: results[seed])

    ms = B.run_multi_seed("x.csv", "y", metric="m", seeds=[1, 2, 3])
    assert ms.verdict == "undecided"          # within noise is a first-class outcome


def test_multi_seed_lower_is_better_baseline_win(monkeypatch):
    from maestra import benchmark as B
    # rmse-style (lower is better): maestra consistently worse -> baseline verdict
    results = {s: _bench(s, 100.0, 110.0 + s, higher=False) for s in (1, 2, 3)}
    monkeypatch.setattr(B, "run_task", lambda csv, target, *, metric, seed, **k: results[seed])

    ms = B.run_multi_seed("x.csv", "y", metric="rmse", seeds=[1, 2, 3])
    assert ms.verdict == "baseline"


def test_multi_seed_one_seed_crashing_does_not_void_the_run(monkeypatch):
    """A single seed's AutoGluon/LLM crash is caught and recorded, not fatal to the whole run --
    found on real Kaggle data (an AutoGluon internal fragility on one seed's fold shape)."""
    from maestra import benchmark as B
    results = {1: _bench(1, 0.80, 0.85), 3: _bench(3, 0.79, 0.84)}

    def flaky(csv, target, *, metric, seed, **k):
        if seed == 2:
            raise ValueError("boom: setting an array element with a sequence")
        return results[seed]

    monkeypatch.setattr(B, "run_task", flaky)
    ms = B.run_multi_seed("x.csv", "y", metric="m", seeds=[1, 2, 3])

    assert [r.seed for r in ms.per_seed] == [1, 3]           # the two survivors
    assert ms.failed_seeds == [{"seed": 2, "error": "ValueError: boom: setting an array "
                                "element with a sequence"}]
    assert ms.verdict == "maestra"                            # the survivors still settle it


def test_multi_seed_all_seeds_failing_raises_not_silent_undecided(monkeypatch):
    from maestra import benchmark as B

    def always_fails(csv, target, *, metric, seed, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(B, "run_task", always_fails)
    with pytest.raises(RuntimeError, match="all 2 seeds failed"):
        B.run_multi_seed("x.csv", "y", metric="m", seeds=[1, 2])


def test_multi_seed_reports_mde_alongside_the_verdict(monkeypatch):
    """N1 (2026-07-05): every multi-seed result carries the minimum detectable effect at this
    seed count/spread, so 'undecided' is interpretable as 'no effect this large', not a bare
    non-result."""
    from maestra import benchmark as B
    results = {1: _bench(1, 0.80, 0.805), 2: _bench(2, 0.80, 0.803), 3: _bench(3, 0.80, 0.76)}
    monkeypatch.setattr(B, "run_task", lambda csv, target, *, metric, seed, **k: results[seed])

    ms = B.run_multi_seed("x.csv", "y", metric="m", seeds=[1, 2, 3])
    assert ms.verdict == "undecided"
    assert ms.mde > abs(ms.mean_delta)  # undecided: the mean fell short of the bar it needed


def test_multi_seed_nb_correction_can_flip_a_marginal_win_to_undecided(monkeypatch):
    """The Nadeau-Bengio inflation (test_train_ratio from holdout_frac) only raises the bar --
    reproduces the real M6 finding (House Prices, 5 seeds) at benchmark scale: a delta that
    passed the naive rule "narrowly" fails once the seed replications' shared training pool is
    accounted for."""
    from maestra import benchmark as B

    deltas = [625.16, 1401.56, 376.93, 3653.06, 369.998]  # M6's logged per-seed rmse deltas
    results = {i + 1: _bench(i + 1, 100000.0 + d, 100000.0, higher=False)
              for i, d in enumerate(deltas)}
    monkeypatch.setattr(B, "run_task", lambda csv, target, *, metric, seed, **k: results[seed])

    ms_default = B.run_multi_seed("x.csv", "y", metric="rmse", seeds=[1, 2, 3, 4, 5])
    assert ms_default.verdict == "undecided"  # holdout_frac defaults to 0.25 -> ratio 1/3

    ms_naive = B.run_multi_seed("x.csv", "y", metric="rmse", seeds=[1, 2, 3, 4, 5],
                                holdout_frac=0.0)  # ratio 0 == the pre-N1 naive rule
    assert ms_naive.verdict == "maestra"  # the naive rule accepted this exact case (M6, logged)


def test_multi_seed_aggregate_row_feeds_summary(monkeypatch, tmp_path):
    from maestra import benchmark as B
    results = {1: _bench(1, 0.80, 0.85), 2: _bench(2, 0.81, 0.86), 3: _bench(3, 0.79, 0.84)}
    monkeypatch.setattr(B, "run_task", lambda csv, target, *, metric, seed, **k: results[seed])

    ms = B.run_multi_seed("x.csv", "y", metric="m", seeds=[1, 2, 3])
    log = tmp_path / "bench.jsonl"
    B.append_multi_seed(str(log), ms, timestamp="t")
    board = B.summary(str(log))
    assert "n=3 seeds" in board and "maestra" in board   # verdict rendered on the board
