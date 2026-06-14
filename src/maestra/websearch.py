"""Provider-agnostic web search + fetch — the research node's eyes on the world.

Mirrors the philosophy of :mod:`maestra.llm`: a thin, swappable wrapper. There the
backbone is chosen by a *model* string; here the search backend is chosen by a *provider*
name (``"tavily"``, ``"duckduckgo"``, ...). The rest of maestra depends only on the two
verbs — :func:`search` and :func:`fetch` — plus the small :class:`SearchResult` record,
never on a concrete SDK. Adding a backend is registering one class.

Network access is an *optional* capability: the provider SDKs live in the ``research``
dependency group (``pip install 'maestra[research]'``). Imports are therefore lazy and a
missing package surfaces as a clear :class:`WebSearchError`, not an ``ImportError`` at
import time — so the rest of maestra keeps working without the extra deps installed.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Callable, Protocol

# Per-request network timeout and the default page-size of a single search.
_TIMEOUT_S = 30
_DEFAULT_MAX_RESULTS = 5
# Snippets are kept short on purpose; full text is fetched on demand via :func:`fetch`.
_SNIPPET_LEN = 500

#: Default backend. Tavily is LLM-oriented (returns ranked results *with* page content);
#: swap to ``"duckduckgo"`` for a keyless option.
DEFAULT_PROVIDER = "tavily"


class WebSearchError(RuntimeError):
    """Raised on any search/fetch failure, including a missing optional dependency."""


@dataclass
class SearchResult:
    """One hit. ``content`` is the full page text when the provider already returns it
    (Tavily does); otherwise it stays empty until :func:`fetch` fills it on demand."""

    title: str
    url: str
    snippet: str = ""
    content: str = ""


class SearchProvider(Protocol):
    """The contract every backend implements — two verbs, nothing else."""

    def search(self, query: str, max_results: int) -> list[SearchResult]: ...

    def fetch(self, url: str) -> str: ...


# --- provider registry -------------------------------------------------------------

#: name -> zero-arg factory. Factories build lazily so importing this module never
#: touches an optional dependency or an API key.
_PROVIDERS: dict[str, Callable[[], SearchProvider]] = {}


def register_provider(name: str, factory: Callable[[], SearchProvider]) -> None:
    """Register a backend factory under ``name`` (overwrites an existing one)."""
    _PROVIDERS[name] = factory


def get_provider(name: str = DEFAULT_PROVIDER) -> SearchProvider:
    """Instantiate the backend registered as ``name``.

    Raises:
        WebSearchError: if ``name`` is unknown, or the backend cannot initialise (missing
            optional dependency or unset API key).
    """
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        raise WebSearchError(
            f"Unknown search provider {name!r}. Known providers: {sorted(_PROVIDERS)}."
        ) from None
    return factory()


def _import(module: str):
    """Import an optional dependency, turning ``ImportError`` into a helpful WebSearchError."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise WebSearchError(
            f"The {module!r} package is required for this search provider. Install the "
            f"optional dependencies with: pip install 'maestra[research]'"
        ) from exc


# --- concrete providers ------------------------------------------------------------

class TavilyProvider:
    """Tavily — an LLM-oriented search API that returns ranked results *with* page content.

    Needs ``TAVILY_API_KEY`` in the environment. Talks to the REST API over httpx, so no
    Tavily SDK is required beyond ``httpx`` from the ``research`` group.
    """

    _SEARCH_URL = "https://api.tavily.com/search"
    _EXTRACT_URL = "https://api.tavily.com/extract"

    def __init__(self) -> None:
        self._httpx = _import("httpx")
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            raise WebSearchError("TAVILY_API_KEY is not set in the environment.")
        self._key = key

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            resp = self._httpx.post(
                self._SEARCH_URL,
                json={
                    "api_key": self._key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                    "include_raw_content": True,
                },
                timeout=_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - network/JSON errors all map to one type
            raise WebSearchError(f"Tavily search failed for {query!r}: {exc}") from exc
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=(item.get("content") or "")[:_SNIPPET_LEN],
                content=item.get("raw_content") or "",
            )
            for item in data.get("results", [])
        ]

    def fetch(self, url: str) -> str:
        try:
            resp = self._httpx.post(
                self._EXTRACT_URL,
                json={"api_key": self._key, "urls": [url]},
                timeout=_TIMEOUT_S,
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
        except Exception as exc:  # noqa: BLE001
            raise WebSearchError(f"Tavily extract failed for {url}: {exc}") from exc
        if not results:
            raise WebSearchError(f"Tavily returned no content for {url}.")
        return results[0].get("raw_content", "")


class DuckDuckGoProvider:
    """Keyless search via the ``ddgs`` package; pages fetched with httpx and reduced to
    readable plain text with trafilatura."""

    def __init__(self) -> None:
        self._ddgs = _import("ddgs")
        self._httpx = _import("httpx")
        self._trafilatura = _import("trafilatura")

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            with self._ddgs.DDGS() as client:
                hits = list(client.text(query, max_results=max_results))
        except Exception as exc:  # noqa: BLE001
            raise WebSearchError(f"DuckDuckGo search failed for {query!r}: {exc}") from exc
        return [
            SearchResult(
                title=h.get("title", ""),
                url=h.get("href", ""),
                snippet=(h.get("body") or "")[:_SNIPPET_LEN],
            )
            for h in hits
        ]

    def fetch(self, url: str) -> str:
        try:
            resp = self._httpx.get(url, timeout=_TIMEOUT_S, follow_redirects=True)
            resp.raise_for_status()
            return self._trafilatura.extract(resp.text) or ""
        except Exception as exc:  # noqa: BLE001
            raise WebSearchError(f"Fetch failed for {url}: {exc}") from exc


register_provider("tavily", TavilyProvider)
register_provider("duckduckgo", DuckDuckGoProvider)


# --- module-level convenience ------------------------------------------------------

def search(
    query: str,
    *,
    provider: str = DEFAULT_PROVIDER,
    max_results: int = _DEFAULT_MAX_RESULTS,
) -> list[SearchResult]:
    """One-shot search through ``provider``. For many queries, reuse :func:`get_provider`."""
    return get_provider(provider).search(query, max_results)


def fetch(url: str, *, provider: str = DEFAULT_PROVIDER) -> str:
    """Fetch ``url`` as readable plain text through ``provider``."""
    return get_provider(provider).fetch(url)
