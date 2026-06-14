"""The conductor loop — one function per step, no framework.

    split -> profile(train) -> LLM cleaning plan -> fit+apply -> train -> evaluate

With ``max_attempts > 1`` this becomes an agentic loop: if an attempt fails, the LLM
diagnoses the (truncated) traceback and picks a bounded recovery action — revise the
plan, raise the time budget, or give up — and we retry. The loop is deliberately
side-effect-free (no printing, no arg parsing) so the whole thing, including the
recovery path, is unit-testable with a mocked LLM/engine.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field

import pandas as pd

from maestra.cleaning import fit_cleaning_plan, propose_cleaning_plan
from maestra.diagnosis import diagnose_failure
from maestra.engine import TrainingResult, predict, split, train_and_evaluate
from maestra.profiling import profile_dataframe

# How much of the traceback to hand the LLM (keep the tail — the actual error is there).
_MAX_ERROR_CHARS = 1500


class PipelineError(RuntimeError):
    """Raised for pipeline-level failures (e.g. no trainable features, LLM gave up)."""


@dataclass
class PipelineResult:
    """Everything a run produced, ready for the CLI to render."""

    n_cols_before: int
    n_cols_after: int
    plan: dict | None  # None when the cleaning step was skipped
    cleaning_log: list[str] = field(default_factory=list)
    training: TrainingResult | None = None
    attempts: int = 1
    diagnosis_log: list[dict] = field(default_factory=list)
    submission: pd.DataFrame | None = None  # id + prediction, when a test set was given


def _build_submission(transform, predictor, test_df, target, id_col):
    """Predict on the test set and return a Kaggle-style ``id``/``target`` frame.

    The test set is cleaned with the *same* fitted transform as training (so drops and
    train-fitted fills match), but its identifier column is preserved separately for the
    submission even though cleaning drops it from the features.
    """
    if id_col not in test_df.columns:
        raise PipelineError(f"id column {id_col!r} not in test set. Columns: {list(test_df.columns)}")
    ids = test_df[id_col]
    features = transform.transform(test_df) if transform is not None else test_df
    preds = predict(predictor, features)
    return pd.DataFrame({id_col: ids.to_numpy(), target: preds.to_numpy()})


def _validate_trainable(train: pd.DataFrame, target: str) -> None:
    """Fail fast (and clearly) on inputs AutoGluon can't train on.

    Catches the realistic ``revise_plan`` trigger: an over-aggressive plan that drops
    every feature column, leaving only the target.
    """
    features = [c for c in train.columns if c != target]
    if not features:
        raise PipelineError("No feature columns remain after cleaning — the plan dropped them all.")


def _format_error(exc: Exception) -> str:
    """Last ``_MAX_ERROR_CHARS`` characters of the traceback for the diagnosis prompt."""
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return text[-_MAX_ERROR_CHARS:]


def run_pipeline(
    df: pd.DataFrame,
    target: str,
    *,
    model: str,
    test_size: float,
    time_limit: int,
    seed: int,
    model_dir: str,
    use_llm: bool = True,
    max_attempts: int = 1,
    test_df: pd.DataFrame | None = None,
    id_col: str = "id",
) -> PipelineResult:
    """Run the conductor loop on ``df`` and return a :class:`PipelineResult`.

    Args:
        df: Raw input data including the target column.
        target: Name of the target column (must exist in ``df``).
        model: LiteLLM model string for the cleaning and diagnosis steps.
        test_size: Fraction held out for evaluation.
        time_limit: AutoGluon training budget in seconds.
        seed: Random seed for the split.
        model_dir: Directory for AutoGluon's model artefacts.
        use_llm: If ``False``, skip the LLM cleaning step (baseline run).
        max_attempts: Upper bound on attempts. ``1`` disables the diagnosis loop; with
            ``> 1`` a failed attempt is diagnosed by the LLM and retried.
        test_df: Optional unlabeled test set. If given, the trained model predicts on it
            and the result carries a Kaggle-style ``submission`` frame. The model is the
            one trained on the train split (the holdout is *not* added back) — a maximal
            leaderboard score would refit on all labeled rows.
        id_col: Identifier column carried from ``test_df`` into the submission.

    Raises:
        ValueError: If ``target`` is not a column of ``df``.
        PipelineError: If the LLM gives up, no recovery succeeds, or ``id_col`` is missing.
    """
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not in CSV. Columns: {list(df.columns)}")

    n_before = len(df.columns)

    # Split first, then fit cleaning on train only — otherwise imputation statistics
    # would see the holdout and leak test information into the features.
    train_raw, holdout_raw = split(df, test_size, seed)

    plan: dict | None = None
    if use_llm:
        profile = profile_dataframe(train_raw, target)
        plan = propose_cleaning_plan(model, profile, target)

    current_time_limit = time_limit
    diagnosis_log: list[dict] = []

    for attempt in range(1, max_attempts + 1):
        try:
            transform = None
            if plan is not None:
                transform = fit_cleaning_plan(train_raw, plan, target)
                train = transform.transform(train_raw)
                holdout = transform.transform(holdout_raw)
                cleaning_log = transform.log
            else:
                train, holdout, cleaning_log = train_raw, holdout_raw, []

            _validate_trainable(train, target)
            training = train_and_evaluate(train, holdout, target, current_time_limit, model_dir)

            submission = None
            if test_df is not None:
                submission = _build_submission(transform, training.predictor, test_df, target, id_col)

            return PipelineResult(
                n_cols_before=n_before,
                n_cols_after=len(train.columns),
                plan=plan,
                cleaning_log=cleaning_log,
                training=training,
                attempts=attempt,
                diagnosis_log=diagnosis_log,
                submission=submission,
            )
        except Exception as exc:
            if attempt == max_attempts:
                raise  # budget exhausted — surface the real failure

            diagnosis = diagnose_failure(
                model,
                _format_error(exc),
                profile=profile_dataframe(train_raw, target),
                plan=plan,
                time_limit=current_time_limit,
                target=target,
            )
            diagnosis_log.append(diagnosis)
            action = diagnosis.get("action")
            if action == "give_up":
                raise PipelineError(
                    f"LLM gave up after attempt {attempt}: {diagnosis.get('diagnosis')}"
                ) from exc
            if action == "increase_time_limit" and diagnosis.get("new_time_limit"):
                current_time_limit = int(diagnosis["new_time_limit"])
            elif action == "revise_plan" and diagnosis.get("new_plan"):
                plan = diagnosis["new_plan"]
            # else: no actionable recovery — the next iteration retries unchanged and,
            # if it fails again, the exhausted-budget branch above surfaces the error.

    # Unreachable for max_attempts >= 1, but keeps type checkers and readers honest.
    raise PipelineError("Pipeline did not run any attempts (max_attempts must be >= 1).")
