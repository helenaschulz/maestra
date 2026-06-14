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
from dataclasses import asdict, dataclass, field

import pandas as pd

from maestra.cleaning import fit_cleaning_plan, propose_cleaning_plan
from maestra.diagnosis import diagnose_failure
from maestra.engine import TrainingResult, fit_predictor, predict, split, train_and_evaluate
from maestra.feature_engineering import fit_feature_plan, propose_feature_plan
from maestra.hybrid_features import apply_generated_features, propose_feature_code, select_features
from maestra.profiling import profile_dataframe
from maestra.research import brief_context, research_strategy
from maestra.validation import CVResult, adversarial_validation, cross_validate

# How much of the traceback to hand the LLM (keep the tail — the actual error is there).
_MAX_ERROR_CHARS = 1500


class PipelineError(RuntimeError):
    """Raised for pipeline-level failures (e.g. no trainable features, LLM gave up)."""


@dataclass
class PipelineResult:
    """Everything a run produced, ready for the CLI to render."""

    n_cols_before: int
    n_cols_after: int  # final column count (after cleaning + feature engineering)
    plan: dict | None  # None when the cleaning step was skipped
    n_cols_clean: int = 0  # column count after cleaning, before feature engineering
    cleaning_log: list[str] = field(default_factory=list)
    training: TrainingResult | None = None
    attempts: int = 1
    diagnosis_log: list[dict] = field(default_factory=list)
    submission: pd.DataFrame | None = None  # id + prediction, when a test set was given
    feature_plan: dict | None = None  # None when feature engineering was skipped
    feature_log: list[str] = field(default_factory=list)
    cv: CVResult | None = None  # cross-validation estimate, when --cv was used
    adversarial_auc: float | None = None  # train/test shift AUC, when a test set was given
    research: dict | None = None  # strategy-research summary, when --research was used
    hybrid: list | None = None  # generated-feature candidate provenance, when --hybrid was used


def _do_research(model, df, target, rules_mode):
    """Run the (opt-in) research node and return ``(context_string, log_summary)``.

    The context is fed to the planning nodes as non-binding hypotheses; the summary
    (rules_mode, reference URLs, grounded flag) is logged. Nothing here bypasses
    validation — the brief only shapes what the LLM proposes.
    """
    rr = research_strategy(
        model,
        f"Tabular machine-learning task: predict the column '{target}'.",
        profile=profile_dataframe(df, target),
        rules_mode=rules_mode,
    )
    summary = {
        "rules_mode": rr.rules_mode,
        "references": [r.get("url") for r in rr.brief.get("references", [])],
        "grounded": rr.brief.get("grounded"),
    }
    return brief_context(rr.brief), summary


def _build_submission(transforms, predictor, test_df, target, id_col,
                      generated_features=None, fit_df=None):
    """Predict on the test set and return a Kaggle-style ``id``/``target`` frame.

    The test set is run through the *same* fitted transforms as training (cleaning, then
    feature engineering, then any kept generated features fitted on ``fit_df``), but its
    identifier column is preserved separately even though cleaning drops it from the features.
    """
    if id_col not in test_df.columns:
        raise PipelineError(f"id column {id_col!r} not in test set. Columns: {list(test_df.columns)}")
    ids = test_df[id_col]
    features = test_df
    for transform in transforms:
        features = transform.transform(features)
    if generated_features:
        _, features = apply_generated_features(fit_df, features, target, generated_features)
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


def _run_with_cv(df, target, *, model, time_limit, cv_time_limit, seed, model_dir,
                 use_llm, use_fe, n_folds, test_df, id_col, n_before,
                 research_context=None, research_summary=None,
                 hybrid=False, hybrid_max_candidates=5, hybrid_threshold=1.0,
                 eval_metric=None) -> PipelineResult:
    """Cross-validation path (opt-in via --cv). No holdout, no retry/quality loop.

    The cleaning/FE plan structure is proposed once on the full data; cross_validate re-fits
    the plan *parameters* per fold for an honest, leakage-free score. With ``hybrid`` the LLM
    also generates feature code, gated fold-wise by the CV. A final model is trained on all
    data (plus any kept generated features) for prediction/submission.
    """
    cleaning_plan = (
        propose_cleaning_plan(model, profile_dataframe(df, target), target, research_context)
        if use_llm else None
    )

    # Build the cleaned + feature-engineered full dataset (also the profile for code-gen).
    transforms, cleaning_log, feature_log = [], [], []
    full = df
    if cleaning_plan is not None:
        ct = fit_cleaning_plan(df, cleaning_plan, target)
        full, cleaning_log = ct.transform(df), ct.log
        transforms.append(ct)
    n_clean = len(full.columns)
    feature_plan = None
    if use_llm and use_fe:
        feature_plan = propose_feature_plan(model, profile_dataframe(full, target), target, research_context)
        ft = fit_feature_plan(full, feature_plan, target)
        full, feature_log = ft.transform(full), ft.log
        transforms.append(ft)

    # Hybrid feature generation (opt-in): generate code, gate fold-wise by CV, keep what helps.
    generated_features, hybrid_records = [], None
    if hybrid and use_llm:
        candidates = propose_feature_code(
            model, profile_dataframe(full, target), research_context, hybrid_max_candidates)
        generated_features, records, cv = select_features(
            df, target, candidates, cleaning_plan=cleaning_plan, feature_plan=feature_plan,
            model_dir=f"{model_dir}/hybrid", time_limit=cv_time_limit, n_folds=n_folds, seed=seed,
            sigma_mult=hybrid_threshold, eval_metric=eval_metric)
        hybrid_records = [asdict(r) for r in records]
    else:
        cv = cross_validate(df, target, cleaning_plan=cleaning_plan, feature_plan=feature_plan,
                            model_dir=f"{model_dir}/cv", time_limit=cv_time_limit, n_folds=n_folds, seed=seed,
                            eval_metric=eval_metric)

    # Final model on all data, plus kept generated features (fitted on the full cleaned+FE data).
    full_for_fit = full
    if generated_features:
        full_for_fit, _ = apply_generated_features(full, full, target, generated_features)
    _validate_trainable(full_for_fit, target)
    training = fit_predictor(full_for_fit, target, time_limit, f"{model_dir}/final", eval_metric)

    adversarial_auc = None
    submission = None
    if test_df is not None:
        adversarial_auc = adversarial_validation(df, test_df, target, cleaning_plan=cleaning_plan,
                                                  model_dir=f"{model_dir}/adversarial")
        submission = _build_submission(transforms, training.predictor, test_df, target, id_col,
                                       generated_features=generated_features or None, fit_df=full)

    return PipelineResult(
        n_cols_before=n_before, n_cols_after=len(full_for_fit.columns), n_cols_clean=n_clean,
        plan=cleaning_plan, cleaning_log=cleaning_log, training=training,
        feature_plan=feature_plan, feature_log=feature_log, submission=submission,
        cv=cv, adversarial_auc=adversarial_auc, research=research_summary, hybrid=hybrid_records,
    )


def _weak_context(val_score: float, floor: float, metric: str) -> str:
    """Synthetic 'failure' context for the quality gate, reusing the diagnosis tool."""
    return (
        f"Das Training lief erfolgreich durch, aber der interne Validierungs-Score "
        f"({metric}) = {val_score:.4f} liegt unter dem konservativen Floor {floor}. Der "
        f"Lauf gilt als zu schwach. Falls der Cleaning-Plan nutzbares Signal verwirft, "
        f"revidiere ihn (revise_plan); wenn mehr Zeit helfen koennte, increase_time_limit; "
        f"sonst give_up."
    )


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
    use_fe: bool = True,
    max_attempts: int = 1,
    revise_below: float | None = None,
    cv_folds: int | None = None,
    cv_time_limit: int | None = None,
    hybrid: bool = False,
    hybrid_max_candidates: int = 5,
    hybrid_threshold: float = 1.0,
    research: bool = False,
    rules_mode: str = "offline",
    eval_metric: str | None = None,
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
        use_fe: If ``False`` (or ``use_llm`` is ``False``), skip LLM feature engineering.
        max_attempts: Upper bound on attempts. ``1`` disables the diagnosis loop; with
            ``> 1`` a failed attempt is diagnosed by the LLM and retried.
        revise_below: Conservative floor on AutoGluon's internal validation score. If set
            and a successful run scores below it, the LLM may revise the plan ONCE and
            retrain (within ``max_attempts``). The gate reads the internal val score, not
            the holdout, so the holdout estimate stays unbiased. ``None`` disables it.
        cv_folds: If set (>= 2), run leakage-free k-fold cross-validation instead of the
            single holdout — the trustworthy path. The cleaning/FE plan *parameters* are
            re-fitted per fold; the final model is trained on all data for prediction. The
            diagnosis/retry and quality-revision loops apply only to the holdout path.
        cv_time_limit: Training budget per CV fold (defaults to ``time_limit``).
        research: If ``True`` (and ``use_llm``), run the web strategy-research node first and
            feed its brief to the cleaning/FE planners as *non-binding hypotheses*. Nothing
            bypasses validation; off by default (no web/LLM research calls).
        rules_mode: ``"offline"`` (default) or ``"live"`` for the research node — in live
            mode the brief must not recommend external data or third-party solutions.
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
    if hybrid and not (cv_folds and cv_folds >= 2):
        raise ValueError("--hybrid requires --cv (the CV is the gate that keeps/drops features).")

    n_before = len(df.columns)

    # Opt-in strategy research (before planning). Produces non-binding hypotheses for the
    # planners + a summary to log. Skipped entirely without --research / use_llm.
    research_context, research_summary = None, None
    if research and use_llm:
        research_context, research_summary = _do_research(model, df, target, rules_mode)

    if cv_folds is not None and cv_folds >= 2:
        return _run_with_cv(
            df, target, model=model, time_limit=time_limit, cv_time_limit=cv_time_limit or time_limit,
            seed=seed, model_dir=model_dir, use_llm=use_llm, use_fe=use_fe, n_folds=cv_folds,
            test_df=test_df, id_col=id_col, n_before=n_before,
            research_context=research_context, research_summary=research_summary,
            hybrid=hybrid, hybrid_max_candidates=hybrid_max_candidates, hybrid_threshold=hybrid_threshold,
            eval_metric=eval_metric,
        )

    # Split first, then fit cleaning on train only — otherwise imputation statistics
    # would see the holdout and leak test information into the features.
    train_raw, holdout_raw = split(df, test_size, seed)

    plan: dict | None = None
    if use_llm:
        profile = profile_dataframe(train_raw, target)
        plan = propose_cleaning_plan(model, profile, target, research_context)

    current_time_limit = time_limit
    diagnosis_log: list[dict] = []
    feature_plan: dict | None = None  # proposed once on the cleaned train, then cached
    quality_revised = False  # the success-but-weak gate fires at most once

    for attempt in range(1, max_attempts + 1):
        try:
            transforms = []  # fitted transforms, applied in order to train/holdout/test
            if plan is not None:
                ctransform = fit_cleaning_plan(train_raw, plan, target)
                train = ctransform.transform(train_raw)
                holdout = ctransform.transform(holdout_raw)
                cleaning_log = ctransform.log
                transforms.append(ctransform)
            else:
                train, holdout, cleaning_log = train_raw, holdout_raw, []

            n_clean = len(train.columns)
            feature_log: list[str] = []
            if use_llm and use_fe:
                if feature_plan is None:
                    feature_plan = propose_feature_plan(
                        model, profile_dataframe(train, target), target, research_context)
                ftransform = fit_feature_plan(train, feature_plan, target)
                train = ftransform.transform(train)
                holdout = ftransform.transform(holdout)
                feature_log = ftransform.log
                transforms.append(ftransform)

            _validate_trainable(train, target)
            training = train_and_evaluate(train, holdout, target, current_time_limit, model_dir, eval_metric)

            # Quality gate (Point 3): if the run is essentially degenerate, let the LLM
            # revise the plan ONCE and retrain. The decision uses AutoGluon's internal
            # validation score, never the holdout — so the holdout stays an honest estimate.
            if (
                revise_below is not None
                and use_llm
                and not quality_revised
                and attempt < max_attempts
                and training.val_score is not None
                and training.val_score < revise_below
            ):
                diagnosis = diagnose_failure(
                    model,
                    _weak_context(training.val_score, revise_below, training.eval_metric),
                    profile=profile_dataframe(train_raw, target),
                    plan=plan,
                    time_limit=current_time_limit,
                    target=target,
                )
                diagnosis_log.append({**diagnosis, "trigger": "weak_metric", "val_score": training.val_score})
                action = diagnosis.get("action")
                if action == "revise_plan" and diagnosis.get("new_plan"):
                    plan, quality_revised = diagnosis["new_plan"], True
                    continue
                if action == "increase_time_limit" and diagnosis.get("new_time_limit"):
                    current_time_limit, quality_revised = int(diagnosis["new_time_limit"]), True
                    continue
                # give_up or no actionable revision -> accept this (weak) run

            submission = None
            if test_df is not None:
                submission = _build_submission(transforms, training.predictor, test_df, target, id_col)

            return PipelineResult(
                n_cols_before=n_before,
                n_cols_after=len(train.columns),
                n_cols_clean=n_clean,
                plan=plan,
                cleaning_log=cleaning_log,
                training=training,
                attempts=attempt,
                diagnosis_log=diagnosis_log,
                submission=submission,
                feature_plan=feature_plan,
                feature_log=feature_log,
                research=research_summary,
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
