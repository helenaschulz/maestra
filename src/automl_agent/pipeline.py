"""The conductor loop — one function per step, no framework.

    profile -> LLM cleaning plan -> apply plan -> train -> evaluate

This module is pure orchestration over the other modules and returns structured data;
the CLI is responsible for presenting it. Keeping it side-effect-free (no printing, no
arg parsing) is what makes the whole pipeline unit-testable with a mocked LLM/engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from automl_agent.cleaning import apply_cleaning_plan, propose_cleaning_plan
from automl_agent.engine import TrainingResult, split, train_and_evaluate
from automl_agent.profiling import profile_dataframe


@dataclass
class PipelineResult:
    """Everything a run produced, ready for the CLI to render."""

    n_cols_before: int
    n_cols_after: int
    plan: dict | None  # None when the cleaning step was skipped
    cleaning_log: list[str] = field(default_factory=list)
    training: TrainingResult | None = None


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
) -> PipelineResult:
    """Run the full conductor loop on ``df`` and return a :class:`PipelineResult`.

    Args:
        df: Raw input data including the target column.
        target: Name of the target column (must exist in ``df``).
        model: LiteLLM model string for the cleaning step.
        test_size: Fraction held out for evaluation.
        time_limit: AutoGluon training budget in seconds.
        seed: Random seed for the split.
        model_dir: Directory for AutoGluon's model artefacts.
        use_llm: If ``False``, skip the LLM cleaning step (baseline run).

    Raises:
        ValueError: If ``target`` is not a column of ``df``.
    """
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not in CSV. Columns: {list(df.columns)}")

    n_before = len(df.columns)

    if use_llm:
        profile = profile_dataframe(df, target)
        plan = propose_cleaning_plan(model, profile, target)
        clean, log = apply_cleaning_plan(df, plan, target)
    else:
        plan, log, clean = None, [], df

    train, holdout = split(clean, test_size, seed)
    training = train_and_evaluate(train, holdout, target, time_limit, model_dir)

    return PipelineResult(
        n_cols_before=n_before,
        n_cols_after=len(clean.columns),
        plan=plan,
        cleaning_log=log,
        training=training,
    )
