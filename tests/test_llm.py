"""Tests for the LLM wrapper's schema-guided argument repair."""
import json

from maestra.llm import _repair_stringified

SCHEMA = {
    "type": "object",
    "properties": {
        "columns_to_drop": {"type": "array"},
        "imputations": {"type": "array"},
        "overall_rationale": {"type": "string"},
    },
}

_DROPS = [{"column": "Id", "reason": "running index"}]


def test_decodes_stringified_array():
    args = {"columns_to_drop": json.dumps(_DROPS), "overall_rationale": "r"}
    out = _repair_stringified(args, SCHEMA)
    assert out["columns_to_drop"] == _DROPS


def test_unwraps_whole_object_rewrap():
    # The observed claude-sonnet-5 failure: the value re-wraps the arguments object itself.
    args = {"columns_to_drop": json.dumps({"columns_to_drop": _DROPS}), "overall_rationale": "r"}
    out = _repair_stringified(args, SCHEMA)
    assert out["columns_to_drop"] == _DROPS


def test_leaves_valid_and_unparseable_values_alone():
    args = {"columns_to_drop": _DROPS, "imputations": "not json {", "overall_rationale": "r"}
    out = _repair_stringified(args, SCHEMA)
    assert out["columns_to_drop"] == _DROPS          # already correct: untouched
    assert out["imputations"] == "not json {"        # unparseable: untouched (processors skip it)
    assert out["overall_rationale"] == "r"           # declared string: never decoded


def test_string_typed_fields_are_never_decoded():
    # A rationale that happens to BE valid JSON must stay a string.
    args = {"overall_rationale": '["not", "a", "list"]'}
    out = _repair_stringified(args, SCHEMA)
    assert out["overall_rationale"] == '["not", "a", "list"]'
