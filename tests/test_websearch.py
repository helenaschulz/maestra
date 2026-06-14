"""Tests for the provider-agnostic web-search wrapper. No network and no optional deps:
the dispatch, registry and error mapping are exercised with a fake provider."""
import pytest

from maestra import websearch
from maestra.websearch import (
    SearchResult,
    WebSearchError,
    get_provider,
    register_provider,
)


class _FakeProvider:
    def __init__(self):
        self.searched = []
        self.fetched = []

    def search(self, query, max_results):
        self.searched.append((query, max_results))
        return [SearchResult(title="t", url="https://x/1", snippet="s")]

    def fetch(self, url):
        self.fetched.append(url)
        return f"text of {url}"


def test_search_and_fetch_dispatch_to_provider():
    fake = _FakeProvider()
    register_provider("fake", lambda: fake)

    hits = websearch.search("ml validation", provider="fake", max_results=3)
    assert [h.url for h in hits] == ["https://x/1"]
    assert fake.searched == [("ml validation", 3)]

    assert websearch.fetch("https://x/1", provider="fake") == "text of https://x/1"
    assert fake.fetched == ["https://x/1"]


def test_unknown_provider_raises():
    with pytest.raises(WebSearchError, match="Unknown search provider"):
        get_provider("does-not-exist")


def test_searchresult_defaults():
    r = SearchResult(title="t", url="u")
    assert r.snippet == "" and r.content == ""


def test_missing_optional_dependency_maps_to_websearch_error():
    with pytest.raises(WebSearchError, match=r"pip install 'maestra\[research\]'"):
        websearch._import("maestra_no_such_optional_pkg_xyz")


def test_tavily_requires_api_key(monkeypatch):
    # Bypass the optional httpx import so we exercise the key check specifically.
    monkeypatch.setattr(websearch, "_import", lambda name: object())
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(WebSearchError, match="TAVILY_API_KEY"):
        get_provider("tavily")
