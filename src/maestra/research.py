"""Strategy-research node: full web research -> a structured strategy brief.

The project's first *outward-looking* agentic step. Given an ML problem (a short
description plus, optionally, the column profile from :mod:`maestra.profiling`), the node:

  1. asks the LLM which questions to research — it decides the search agenda;
  2. runs those queries through :mod:`maestra.websearch` (provider-agnostic);
  3. lets the LLM pick the few most promising sources to read in full, then fetches them;
  4. asks the LLM to synthesise everything into a structured *strategy brief*.

Every LLM step goes through the same forced-tool interface as the rest of maestra
(:func:`maestra.llm.call_structured`): the model emits constrained JSON, deterministic
code drives the loop. The loop is bounded at every stage (max queries, results, pages),
so cost and latency stay predictable — the same discipline as the diagnosis retry loop.

The brief is plain, JSON-serialisable data. Its ``validation_strategy`` field is shaped to
later inform both the planner and the cross-validation gate; wiring it into the pipeline is
intentionally deferred until the CV branch lands, so this module touches nothing else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from maestra.llm import call_structured
from maestra.websearch import (
    DEFAULT_PROVIDER,
    SearchResult,
    WebSearchError,
    get_provider,
)

# Bounds — keep web research cheap and predictable.
_MAX_QUERIES = 4
_MAX_RESULTS_PER_QUERY = 5
_MAX_PAGES = 3
# Per-page text handed to the synthesis LLM is truncated; snippets stay short too.
_MAX_CONTENT_CHARS = 6000
_MAX_SNIPPET_CHARS = 500

# Progress callback: ``on_event(name, payload)``. Defaults to a no-op.
EventHook = Callable[[str, dict], None]


def _array_of(properties: dict, required: list[str], description: str) -> dict:
    """Build a JSON schema for an array of uniform objects (used across the brief)."""
    return {
        "type": "array",
        "description": description,
        "items": {"type": "object", "properties": properties, "required": required},
    }


#: What questions to research. The LLM owns the agenda; code only bounds it.
QUERY_PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "queries": _array_of(
            {
                "query": {"type": "string", "description": "A concrete web-search query."},
                "rationale": {"type": "string", "description": "Why this query helps."},
            },
            required=["query"],
            description="Search queries, most important first.",
        )
    },
    "required": ["queries"],
}

#: Which of the gathered hits are worth reading in full.
SOURCE_SELECTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "urls": {
            "type": "array",
            "items": {"type": "string"},
            "description": "URLs (verbatim, from the provided list) to read in full, best first.",
        }
    },
    "required": ["urls"],
}

#: The deliverable: a structured strategy brief that later feeds planning and the CV gate.
STRATEGY_BRIEF_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Two or three sentences, the gist."},
        "problem_framing": {
            "type": "string",
            "description": "Restated ML task / problem type and what success looks like.",
        },
        "recommended_models": _array_of(
            {
                "name": {"type": "string"},
                "rationale": {"type": "string"},
            },
            required=["name"],
            description="Candidate model families, most promising first.",
        ),
        "preprocessing": _array_of(
            {"step": {"type": "string"}, "rationale": {"type": "string"}},
            required=["step"],
            description="Cleaning / preprocessing steps to consider.",
        ),
        "feature_engineering": _array_of(
            {"idea": {"type": "string"}, "rationale": {"type": "string"}},
            required=["idea"],
            description="Feature-engineering ideas grounded in the research.",
        ),
        "validation_strategy": {
            "type": "object",
            "description": "How to validate — feeds the cross-validation gate later.",
            "properties": {
                "approach": {
                    "type": "string",
                    "description": "e.g. stratified k-fold, grouped, or time-based split.",
                },
                "rationale": {"type": "string"},
            },
            "required": ["approach"],
        },
        "evaluation_metrics": _array_of(
            {"metric": {"type": "string"}, "rationale": {"type": "string"}},
            required=["metric"],
            description="Metrics that fit the problem.",
        ),
        "pitfalls": _array_of(
            {"risk": {"type": "string"}, "mitigation": {"type": "string"}},
            required=["risk"],
            description="Likely failure modes (leakage, imbalance, drift) and mitigations.",
        ),
        "references": _array_of(
            {"title": {"type": "string"}, "url": {"type": "string"}},
            required=["url"],
            description="Sources the brief draws on. Cite only URLs you were given.",
        ),
    },
    "required": ["summary", "recommended_models", "validation_strategy"],
}


_PLAN_SYSTEM_PROMPT = (
    "Du bist Senior Data Scientist und planst eine Web-Recherche zu einem ML-Problem. "
    "Formuliere wenige, praezise Suchanfragen, die zusammen die Strategie abdecken: "
    "passende Modellfamilien, Preprocessing, Feature-Engineering, Validierungs-Setup, "
    "Metriken und typische Fallstricke (Leakage, Imbalance, Drift). Keine Floskeln, "
    "konkrete Suchbegriffe. Wichtigste Anfrage zuerst."
)

_SELECT_SYSTEM_PROMPT = (
    "Du waehlst aus einer Trefferliste die wenigen Quellen aus, die sich zum vollstaendigen "
    "Lesen lohnen. Bevorzuge fachlich fundierte, spezifische Quellen vor Werbung oder "
    "Allgemeinplaetzen. Gib AUSSCHLIESSLICH URLs aus der vorgelegten Liste zurueck, beste "
    "zuerst. Lieber wenige starke Quellen als viele schwache."
)

_SYNTH_SYSTEM_PROMPT = (
    "Du bist Senior Data Scientist und schreibst aus Recherche-Evidenz einen strukturierten "
    "Strategie-Brief fuer ein ML-Problem. Stuetze jede Empfehlung auf die uebergebene "
    "Evidenz (Snippets und Seiteninhalte); erfinde keine Fakten und keine Quellen. Zitiere "
    "unter references nur URLs, die in der Evidenz vorkommen. Sei konkret und entscheidbar: "
    "der Brief speist spaeter die Planung und das Validierungs-Gate. Wenn die Evidenz duenn "
    "ist, sag das im summary, statt zu spekulieren."
)


def _problem_block(problem_description: str, profile: Optional[dict]) -> str:
    """Render the problem context (description + optional profile) as a JSON-ish prompt block."""
    payload: dict = {"problem": problem_description}
    if profile is not None:
        payload["profile"] = profile
    return json.dumps(payload, ensure_ascii=False, indent=2, default=float)


def plan_queries(
    model: str,
    problem_description: str,
    profile: Optional[dict] = None,
    max_queries: int = _MAX_QUERIES,
) -> list[dict]:
    """Ask the LLM which queries to research. Returns up to ``max_queries`` query dicts."""
    out = call_structured(
        model=model,
        system_prompt=_PLAN_SYSTEM_PROMPT,
        user_prompt=(
            "ML-Problem (JSON):\n"
            f"{_problem_block(problem_description, profile)}\n\n"
            f"Plane hoechstens {max_queries} Suchanfragen."
        ),
        tool_name="plan_research_queries",
        tool_description="Plane Web-Suchanfragen zu einem ML-Problem.",
        parameters_schema=QUERY_PLAN_SCHEMA,
    )
    return list(out.get("queries", []))[:max_queries]


def select_sources(
    model: str,
    problem_description: str,
    results: list[SearchResult],
    max_pages: int = _MAX_PAGES,
) -> list[str]:
    """Ask the LLM which result URLs to read in full. Returns up to ``max_pages`` URLs."""
    listing = [
        {"url": r.url, "title": r.title, "snippet": r.snippet[:_MAX_SNIPPET_CHARS]}
        for r in results
    ]
    out = call_structured(
        model=model,
        system_prompt=_SELECT_SYSTEM_PROMPT,
        user_prompt=(
            f"Problem: {problem_description}\n\n"
            "Treffer (JSON):\n"
            f"{json.dumps(listing, ensure_ascii=False, indent=2)}\n\n"
            f"Waehle hoechstens {max_pages} URLs zum vollstaendigen Lesen."
        ),
        tool_name="select_sources",
        tool_description="Waehle die lesenswerten Quellen aus einer Trefferliste.",
        parameters_schema=SOURCE_SELECTION_SCHEMA,
    )
    return list(out.get("urls", []))[:max_pages]


def _evidence(results: list[SearchResult]) -> list[dict]:
    """Compact, truncated evidence for the synthesis prompt — snippets for all, full
    (capped) text for the pages that were fetched."""
    evidence = []
    for r in results:
        item = {"title": r.title, "url": r.url, "snippet": r.snippet[:_MAX_SNIPPET_CHARS]}
        if r.content:
            item["content"] = r.content[:_MAX_CONTENT_CHARS]
        evidence.append(item)
    return evidence


def synthesize_brief(
    model: str,
    problem_description: str,
    profile: Optional[dict],
    results: list[SearchResult],
) -> dict:
    """Ask the LLM to write the structured strategy brief from the gathered evidence."""
    return call_structured(
        model=model,
        system_prompt=_SYNTH_SYSTEM_PROMPT,
        user_prompt=(
            "ML-Problem (JSON):\n"
            f"{_problem_block(problem_description, profile)}\n\n"
            "Recherche-Evidenz (JSON):\n"
            f"{json.dumps(_evidence(results), ensure_ascii=False, indent=2)}"
        ),
        tool_name="write_strategy_brief",
        tool_description="Schreibe einen strukturierten Strategie-Brief aus Recherche-Evidenz.",
        parameters_schema=STRATEGY_BRIEF_SCHEMA,
    )


@dataclass
class ResearchResult:
    """The brief plus a small audit trail of what the node actually did."""

    brief: dict
    queries: list[str] = field(default_factory=list)
    sources_read: list[str] = field(default_factory=list)
    n_results: int = 0


def research_strategy(
    model: str,
    problem_description: str,
    *,
    profile: Optional[dict] = None,
    provider: str = DEFAULT_PROVIDER,
    max_queries: int = _MAX_QUERIES,
    max_results_per_query: int = _MAX_RESULTS_PER_QUERY,
    max_pages: int = _MAX_PAGES,
    on_event: Optional[EventHook] = None,
) -> ResearchResult:
    """Run full web research on an ML problem and return a structured strategy brief.

    Args:
        model: LiteLLM model string used for every LLM step.
        problem_description: Short natural-language statement of the ML problem.
        profile: Optional column profile (:func:`maestra.profiling.profile_dataframe`).
        provider: Web-search backend name (see :mod:`maestra.websearch`).
        max_queries / max_results_per_query / max_pages: bounds on the research loop.
        on_event: Optional ``(name, payload)`` progress callback.

    Returns:
        A :class:`ResearchResult` whose ``brief`` matches :data:`STRATEGY_BRIEF_SCHEMA`.

    Raises:
        WebSearchError: if the provider is misconfigured or no query returned any result.
    """
    emit = on_event or (lambda name, payload: None)
    backend = get_provider(provider)  # fail fast on a bad provider / missing key

    plan = plan_queries(model, problem_description, profile, max_queries)
    queries = [q["query"] for q in plan if q.get("query")]
    emit("queries_planned", {"queries": queries})

    results: list[SearchResult] = []
    seen: set[str] = set()
    for query in queries:
        try:
            hits = backend.search(query, max_results_per_query)
        except WebSearchError as exc:
            emit("search_failed", {"query": query, "error": str(exc)})
            continue
        for hit in hits:
            if hit.url and hit.url not in seen:
                seen.add(hit.url)
                results.append(hit)
        emit("search_done", {"query": query, "n_hits": len(hits)})

    if not results:
        raise WebSearchError("Web research returned no results for any planned query.")

    by_url = {r.url: r for r in results}
    selected = select_sources(model, problem_description, results, max_pages)
    sources_read: list[str] = []
    for url in selected:
        result = by_url.get(url)
        if result is None:  # LLM cited a URL outside the list — ignore it
            continue
        if not result.content:
            try:
                result.content = backend.fetch(url)
            except WebSearchError as exc:
                emit("fetch_failed", {"url": url, "error": str(exc)})
                continue
        sources_read.append(url)
        emit("fetched", {"url": url, "chars": len(result.content)})

    brief = synthesize_brief(model, problem_description, profile, results)
    emit("brief_ready", {"references": len(brief.get("references", []))})
    return ResearchResult(
        brief=brief,
        queries=queries,
        sources_read=sources_read,
        n_results=len(results),
    )
