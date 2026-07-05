"""Run memory (N4, 2026-07-05): the project's own DECIDED past verdicts as non-binding context.

The MALMAS/Agent-K pattern (see RELATED_WORK.md) is retrieved run history fed back as guidance
for future runs. The risk, flagged since E2 and sharpened by N1: an "undecided" or statistically
noisy past result fed back as precedent would let the LLM rationalize repeating exactly the
mistake the arbiter exists to catch (self-reinforcement). So this module retrieves ONLY DECIDED
verdicts -- multi-seed comparisons that cleared the (N1-hardened) accept bar, never
"undecided" -- and feeds them the same way the dataset description and research briefs are fed:
as non-binding context for the judgment nodes (`profiling.description_context`'s pattern). The
arbiter re-measures on THIS task regardless; memory can suggest, never skip, verification.
"""
from __future__ import annotations

import json

_MAX_ENTRIES = 5
_DECIDED_VERDICTS = ("maestra", "baseline")


def load_decided_verdicts(path: str = "benchmark.jsonl") -> list[dict]:
    """Every multi-seed verdict logged in ``path`` that is NOT "undecided" -- the only outcomes
    this project treats as a settled finding (each gets its own RESULTS.md provenance line).
    Missing file -> no memory, not an error (a fresh checkout has nothing to retrieve yet)."""
    try:
        with open(path) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        return []
    return [r for r in rows if r.get("kind") == "multi_seed" and r.get("verdict") in _DECIDED_VERDICTS]


def format_memory_context(records: list[dict], max_entries: int = _MAX_ENTRIES) -> str | None:
    """Prompt-ready block of past DECIDED verdicts, capped and explicitly non-binding.

    ``None`` when there is nothing decided yet -- an empty/absent memory must never itself
    become a signal (e.g. "no precedent" is not evidence of anything).
    """
    if not records:
        return None
    lines = []
    for r in records[-max_entries:]:
        direction = "Maestra beat the baseline" if r["verdict"] == "maestra" else "the baseline beat Maestra"
        n_seeds = len(r.get("seeds") or [])
        lines.append(f"- {r.get('name')}: {direction} (mean delta {r.get('delta'):+.4g} "
                     f"{r.get('metric')}, {n_seeds} seeds)")
    return (
        "Past DECIDED results from this project's own measured runs, most recent last (context "
        "only -- these are OTHER datasets; they may or may not transfer to the CURRENT one, and "
        "this run's own arbiter decides regardless. Never treat these as a reason to skip "
        "measurement or to assume a similar outcome here):\n" + "\n".join(lines)
    )


def memory_context(path: str = "benchmark.jsonl", max_entries: int = _MAX_ENTRIES) -> str | None:
    """Convenience: load + format in one call, for wiring into the shared context channel."""
    return format_memory_context(load_decided_verdicts(path), max_entries)
