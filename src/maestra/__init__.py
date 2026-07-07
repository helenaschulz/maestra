"""Maestra: an LLM conductor over AutoGluon for tabular AutoML.

The LLM *decides* (it proposes a structured cleaning plan); the engine *computes*
(AutoGluon does all model search and metric calculation). The two never blur:
the LLM emits constrained JSON via function-calling, deterministic code applies it.
"""

from maestra.compare import CompareResult, compare
from maestra.pipeline import PipelineResult, run_pipeline
from maestra.validation_strategist import check_validation

# audit() is deliberately NOT re-exported here as `maestra.audit`: `audit.py` is itself a
# submodule of this package, and `from maestra.audit import audit` at package level would
# shadow it (the function would win over the submodule for `from maestra import audit`),
# breaking the existing `from maestra import audit as audit_mod` pattern the test suite (and
# potentially other code) uses to reach the MODULE. audit() is already public and DataFrame-input
# (P3) via `from maestra.audit import audit` -- no re-export needed for that to be true.
__all__ = ["CompareResult", "PipelineResult", "check_validation", "compare", "run_pipeline"]
__version__ = "0.1.0"
