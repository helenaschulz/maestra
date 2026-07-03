"""The intervention core (M4) — one primitive for every measured change to the run.

Every judgment agent in Maestra ends the same way: a proposal becomes an *intervention*
(keep a column, add a generated feature, reframe the target), and the intervention is
adopted only if a paired counterfactual — trial CV vs. the current base CV on identical
folds — clears ``improves_beyond_noise``. Before this module, that loop existed three
times (Skeptic, generated-feature gate, target framing), each ad hoc. Here it exists
once, with two properties the copies lacked:

* **A built-in counterfactual.** Every outcome records the per-fold base and trial
  scores, so any accepted (or rejected) intervention can be audited from the ledger
  alone — what exactly was compared, and by how much it moved.
* **A first-class cost budget.** Trial CVs are the multiplier that makes intervention
  loops expensive (each is a full k-fold AutoGluon run). ``CVBudget`` caps how many
  trials a run may spend, across ALL gates; an intervention that finds the budget
  exhausted is recorded as such — visibly skipped, never silently dropped. Base CVs
  are not counted: every gate runs at most one, and the reported CV would run anyway.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from maestra.validation import CVResult, improves_beyond_noise

_MIN_ABS_DELTA = 1e-4


@dataclass
class CVBudget:
    """Per-run cap on counterfactual trial CVs, shared by every intervention gate.

    ``limit=None`` means unlimited (the pre-M4 behaviour). ``spent`` counts trials
    actually run, so ``{"limit": ..., "trials_spent": ...}`` in the run log states the
    run's real intervention cost.
    """

    limit: int | None = None
    spent: int = 0

    def try_spend(self) -> bool:
        """Reserve one trial CV; False (and no change) when the budget is exhausted."""
        if self.limit is not None and self.spent >= self.limit:
            return False
        self.spent += 1
        return True

    def as_record(self) -> dict:
        return {"limit": self.limit, "trials_spent": self.spent}


@dataclass
class InterventionOutcome:
    """One intervention's audit record: what was proposed, what was measured, what won."""

    name: str                 # e.g. "keep:Fence", "feature:kw_group", "target:log1p"
    kind: str                 # "skeptic_keep" | "generated_feature" | "target_framing"
    proposed_by: str          # which agent nominated it
    accepted: bool
    cv_delta: float | None    # paired mean delta (None if never measured)
    reason: str               # improved | no_improvement | no_effect | budget_exhausted
    base_scores: list[float] | None = None   # the counterfactual, per fold
    trial_scores: list[float] | None = None
    base_mean: float | None = None
    trial_mean: float | None = None


def run_counterfactual(
    name: str,
    kind: str,
    proposed_by: str,
    *,
    base: CVResult,
    trial_fn: Callable[[], CVResult],
    budget: CVBudget | None = None,
    sigma_mult: float = 2.0,
    min_abs: float = _MIN_ABS_DELTA,
) -> tuple[InterventionOutcome, CVResult]:
    """Measure one intervention against the current base; the measurement decides.

    Runs ``trial_fn`` (a full CV of the intervened configuration, on the SAME folds as
    ``base``) unless the budget is exhausted, then applies the shared accept rule.
    Returns ``(outcome, new_base)`` where ``new_base`` is the trial CV if accepted —
    greedy loops raise the bar by threading it into the next call — and ``base``
    otherwise. Never trains anything when the budget refuses: the skip is free.
    """
    if budget is not None and not budget.try_spend():
        return (
            InterventionOutcome(name, kind, proposed_by, False, None, "budget_exhausted"),
            base,
        )
    trial = trial_fn()
    accepted, delta = improves_beyond_noise(base=base, trial=trial,
                                            sigma_mult=sigma_mult, min_abs=min_abs)
    if not accepted and delta == 0.0 and trial.fold_scores == base.fold_scores:
        # Bit-identical folds mean the intervention changed NOTHING (duplicate column,
        # feature skipped in every fold) — distinct from an honest "measured, didn't help".
        reason = "no_effect"
    else:
        reason = "improved" if accepted else "no_improvement"
    outcome = InterventionOutcome(
        name, kind, proposed_by, accepted, float(delta), reason,
        list(base.fold_scores) if base.fold_scores else None,
        list(trial.fold_scores) if trial.fold_scores else None,
        base_mean=float(base.mean), trial_mean=float(trial.mean),
    )
    return outcome, (trial if accepted else base)
