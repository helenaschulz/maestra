"""Tests for the strategy-research node. The LLM (call_structured) and the search backend
are mocked, so the plan -> search -> select -> fetch -> synthesise loop is verified
deterministically and offline."""
import pytest

from maestra import research
from maestra.research import (
    QUERY_PLAN_SCHEMA,
    SOURCE_SELECTION_SCHEMA,
    STRATEGY_BRIEF_SCHEMA,
    ResearchResult,
    research_strategy,
)
from maestra.websearch import SearchResult, WebSearchError


# --- fakes -------------------------------------------------------------------------

def _brief():
    return {
        "summary": "use gradient boosting with stratified CV",
        "recommended_models": [{"name": "LightGBM"}],
        "validation_strategy": {"approach": "stratified k-fold"},
        "references": [{"url": "https://a/1"}],
    }


def _fake_llm(**scripts):
    """Return a call_structured stub that branches on tool_name; records every call."""
    calls = []
    defaults = {
        "plan_research_queries": {"queries": [{"query": "q1"}, {"query": "q2"}]},
        "select_sources": {"urls": ["https://a/1"]},
        "write_strategy_brief": _brief(),
    }
    defaults.update(scripts)

    def call(**kwargs):
        calls.append(kwargs)
        return defaults[kwargs["tool_name"]]

    call.calls = calls
    return call


class _FakeBackend:
    def __init__(self, hits_by_query=None, fetch_raises=False):
        self._hits = hits_by_query if hits_by_query is not None else {
            "q1": [SearchResult("A", "https://a/1", "snippet a")],
            "q2": [SearchResult("B", "https://b/2", "snippet b")],
        }
        self._fetch_raises = fetch_raises
        self.fetched = []
        self.searched = []

    def search(self, query, max_results):
        self.searched.append(query)
        return self._hits.get(query, [])

    def fetch(self, url):
        self.fetched.append(url)
        if self._fetch_raises:
            raise WebSearchError("boom")
        return f"full content of {url}"


def _patch(monkeypatch, llm, backend):
    monkeypatch.setattr(research, "call_structured", llm)
    monkeypatch.setattr(research, "get_provider", lambda name: backend)


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Every test gets its own empty research cache — no cross-test cache hits, and nothing
    is written to the real .maestra_cache dir."""
    monkeypatch.setattr(research, "_DEFAULT_CACHE_DIR", str(tmp_path / "cache"))


# --- happy path --------------------------------------------------------------------

def test_full_loop_produces_brief(monkeypatch):
    llm, backend = _fake_llm(), _FakeBackend()
    _patch(monkeypatch, llm, backend)
    events = []

    out = research_strategy("m", "predict churn", on_event=lambda n, p: events.append(n))

    assert isinstance(out, ResearchResult)
    assert out.brief["validation_strategy"]["approach"] == "stratified k-fold"
    assert out.queries == ["q1", "q2"]
    assert out.sources_read == ["https://a/1"]      # the selected URL was fetched
    assert out.n_results == 2                        # two deduped hits
    assert backend.fetched == ["https://a/1"]
    assert {"queries_planned", "search_done", "fetched", "brief_ready"} <= set(events)


def test_dedupes_urls_across_queries(monkeypatch):
    dup = {"q1": [SearchResult("A", "https://a/1", "x")],
           "q2": [SearchResult("A again", "https://a/1", "y")]}
    llm, backend = _fake_llm(), _FakeBackend(hits_by_query=dup)
    _patch(monkeypatch, llm, backend)

    out = research_strategy("m", "p")
    assert out.n_results == 1


def test_no_results_raises(monkeypatch):
    llm = _fake_llm()
    backend = _FakeBackend(hits_by_query={})  # every search comes back empty
    _patch(monkeypatch, llm, backend)
    with pytest.raises(WebSearchError, match="no results"):
        research_strategy("m", "p")


def test_fetch_failure_is_skipped(monkeypatch):
    llm = _fake_llm()
    backend = _FakeBackend(fetch_raises=True)
    _patch(monkeypatch, llm, backend)
    events = []

    out = research_strategy("m", "p", on_event=lambda n, p: events.append(n))
    assert out.sources_read == []        # the failed fetch did not count as read
    assert "fetch_failed" in events
    assert out.brief["summary"]          # synthesis still ran on snippet-only evidence


def test_selecting_url_outside_list_is_ignored(monkeypatch):
    llm = _fake_llm(select_sources={"urls": ["https://not-in-results/9"]})
    backend = _FakeBackend()
    _patch(monkeypatch, llm, backend)

    out = research_strategy("m", "p")
    assert out.sources_read == []
    assert backend.fetched == []


def test_uses_prefetched_content_without_fetching(monkeypatch):
    # Tavily-style hit that already carries content -> no fetch call needed.
    hits = {"q1": [SearchResult("A", "https://a/1", "snip", content="already here")],
            "q2": []}
    llm, backend = _fake_llm(), _FakeBackend(hits_by_query=hits)
    _patch(monkeypatch, llm, backend)

    out = research_strategy("m", "p")
    assert out.sources_read == ["https://a/1"]
    assert backend.fetched == []          # content was reused, not refetched


# --- bounds + delegation -----------------------------------------------------------

def test_query_and_page_bounds_respected(monkeypatch):
    llm = _fake_llm(
        plan_research_queries={"queries": [{"query": f"q{i}"} for i in range(10)]},
        select_sources={"urls": [f"https://a/{i}" for i in range(10)]},
    )
    hits = {f"q{i}": [SearchResult(f"T{i}", f"https://a/{i}", "s")] for i in range(10)}
    backend = _FakeBackend(hits_by_query=hits)
    _patch(monkeypatch, llm, backend)

    out = research_strategy("m", "p", max_queries=2, max_pages=1)
    assert out.queries == ["q0", "q1"]            # planning capped to 2
    assert out.sources_read == ["https://a/0"]    # reading capped to 1


def test_node_functions_delegate_with_their_schemas(monkeypatch):
    captured = {}

    def fake_call(**kwargs):
        captured[kwargs["tool_name"]] = kwargs
        return {
            "plan_research_queries": {"queries": [{"query": "x"}]},
            "select_sources": {"urls": []},
            "write_strategy_brief": _brief(),
        }[kwargs["tool_name"]]

    monkeypatch.setattr(research, "call_structured", fake_call)

    research.plan_queries("gpt-4o", "predict churn", profile={"n_rows": 5})
    research.select_sources("gpt-4o", "p", [SearchResult("t", "u", "s")])
    research.synthesize_brief("gpt-4o", "p", None, [SearchResult("t", "u", "s")])

    assert captured["plan_research_queries"]["parameters_schema"] is QUERY_PLAN_SCHEMA
    assert captured["select_sources"]["parameters_schema"] is SOURCE_SELECTION_SCHEMA
    assert captured["write_strategy_brief"]["parameters_schema"] is STRATEGY_BRIEF_SCHEMA
    assert "predict churn" in captured["plan_research_queries"]["user_prompt"]


# --- guardrails: rules_mode, caching, grounding ------------------------------------

def test_live_rules_mode_recorded_and_passed_to_synthesis(monkeypatch):
    llm, backend = _fake_llm(), _FakeBackend()
    _patch(monkeypatch, llm, backend)

    out = research_strategy("m", "p", rules_mode="live")

    assert out.rules_mode == "live"                 # audit trail
    assert out.brief["rules_mode"] == "live"        # brief flags the mode (deterministic)
    synth = next(c for c in llm.calls if c["tool_name"] == "write_strategy_brief")
    assert "live" in synth["user_prompt"].lower()   # constraint reached the synthesis prompt
    assert "external data" in synth["user_prompt"].lower()


def test_second_call_hits_cache_and_skips_all_work(monkeypatch):
    llm, backend = _fake_llm(), _FakeBackend()
    _patch(monkeypatch, llm, backend)

    events1 = []
    out1 = research_strategy("m", "p", on_event=lambda n, _p: events1.append(n))
    llm_calls_after_first = len(llm.calls)
    searches_after_first = len(backend.searched)
    assert "cache_miss" in events1

    events2 = []
    out2 = research_strategy("m", "p", on_event=lambda n, _p: events2.append(n))
    assert "cache_hit" in events2
    assert len(llm.calls) == llm_calls_after_first      # no new LLM calls
    assert len(backend.searched) == searches_after_first  # no new searches
    assert out2.brief == out1.brief

    research_strategy("m", "p", force_refresh=True)     # force_refresh bypasses the cache
    assert len(llm.calls) > llm_calls_after_first


def test_invented_reference_is_dropped_and_audited(monkeypatch):
    brief = {
        "summary": "s", "recommended_models": [{"name": "x"}],
        "validation_strategy": {"approach": "k-fold"},
        "references": [{"url": "https://a/1"}, {"url": "https://made-up/9"}],
    }
    # evidence URLs from _FakeBackend defaults: https://a/1, https://b/2
    _patch(monkeypatch, _fake_llm(write_strategy_brief=brief), _FakeBackend())

    out = research_strategy("m", "p")

    assert [r["url"] for r in out.brief["references"]] == ["https://a/1"]  # invented one removed
    assert {"url": "https://made-up/9"} in out.dropped_references
    assert out.brief["grounded"] is True


def test_all_references_invented_marks_brief_ungrounded(monkeypatch):
    brief = {
        "summary": "s", "recommended_models": [{"name": "x"}],
        "validation_strategy": {"approach": "k-fold"},
        "references": [{"url": "https://made-up/9"}],
    }
    _patch(monkeypatch, _fake_llm(write_strategy_brief=brief), _FakeBackend())

    out = research_strategy("m", "p")

    assert out.brief["references"] == []
    assert out.brief["grounded"] is False
