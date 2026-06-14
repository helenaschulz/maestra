"""Duenner LiteLLM-Wrapper. Modell-agnostisch, strukturiertes Output via Function-Calling.

Das LLM MUSS das definierte Tool aufrufen; wir lesen die Argumente als JSON aus.
Kein Parsen aus Freitext. Backbone ist per Modell-String umschaltbar
(z.B. "gpt-4o", "claude-3-5-sonnet-latest", "ollama/qwen2.5").
"""
from __future__ import annotations

import json

import litellm


def call_structured(
    model: str,
    system_prompt: str,
    user_prompt: str,
    tool_name: str,
    tool_description: str,
    parameters_schema: dict,
) -> dict:
    tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": parameters_schema,
        },
    }
    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        temperature=0,
    )
    msg = resp.choices[0].message
    if not getattr(msg, "tool_calls", None):
        raise RuntimeError(f"LLM hat das Tool '{tool_name}' nicht aufgerufen. Antwort: {msg.content!r}")
    return json.loads(msg.tool_calls[0].function.arguments)
