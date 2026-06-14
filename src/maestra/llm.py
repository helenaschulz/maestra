"""Thin LiteLLM wrapper — model-agnostic structured output via function-calling.

We force the model to call a single tool whose ``parameters`` are a JSON schema, then
read the arguments as JSON. This is deliberate: the LLM's decision arrives as validated
structure, never as free text we have to parse. Swapping the backbone is just a model
string (``gpt-4o``, ``claude-3-5-sonnet-latest``, ``ollama/qwen2.5``, ...).
"""
from __future__ import annotations

import json

import litellm

# Per-call network timeout and retry budget. LiteLLM retries transient errors itself.
_TIMEOUT_S = 60
_NUM_RETRIES = 2


class LLMError(RuntimeError):
    """Raised when the model returns no usable structured tool call."""


def call_structured(
    model: str,
    system_prompt: str,
    user_prompt: str,
    tool_name: str,
    tool_description: str,
    parameters_schema: dict,
) -> dict:
    """Call ``model`` and return the arguments of a forced tool call as a dict.

    Args:
        model: LiteLLM model string (e.g. ``"gpt-4o"``).
        system_prompt: Role / instructions for the model.
        user_prompt: The task payload (here: the column profile).
        tool_name: Name of the single tool the model is forced to call.
        tool_description: Human-readable description of the tool.
        parameters_schema: JSON schema describing the tool's arguments — this is the
            contract the model's output is validated against by the provider.

    Returns:
        The parsed tool-call arguments.

    Raises:
        LLMError: If the model answered with text instead of calling the tool, or the
            arguments were not valid JSON.
    """
    tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": parameters_schema,
        },
    }
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        temperature=0,
        timeout=_TIMEOUT_S,
        num_retries=_NUM_RETRIES,
    )
    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        raise LLMError(
            f"Model {model!r} did not call tool {tool_name!r}. "
            f"Text response was: {message.content!r}"
        )
    try:
        return json.loads(tool_calls[0].function.arguments)
    except json.JSONDecodeError as exc:  # pragma: no cover - provider-dependent
        raise LLMError(f"Tool arguments were not valid JSON: {exc}") from exc
