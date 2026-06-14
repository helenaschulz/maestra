"""automl-agent: an LLM conductor over AutoGluon for tabular AutoML.

The LLM *decides* (it proposes a structured cleaning plan); the engine *computes*
(AutoGluon does all model search and metric calculation). The two never blur:
the LLM emits constrained JSON via function-calling, deterministic code applies it.
"""

from automl_agent.pipeline import PipelineResult, run_pipeline

__all__ = ["PipelineResult", "run_pipeline"]
__version__ = "0.1.0"
