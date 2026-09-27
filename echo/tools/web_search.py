"""Web search via DuckDuckGo (the only tool that talks to the internet)."""

from __future__ import annotations

import asyncio
from typing import Any

from echo.tools.base import Tool, ToolContext

SearchResult = dict[str, str]


def search_backend(query: str, max_results: int) -> list[SearchResult]:
    """Return ``[{title, href, body}]``. Isolated so tests can replace it."""
    from ddgs import DDGS

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def format_results(results: list[SearchResult]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        body = " ".join(str(r.get("body", "")).split())[:300]
        lines.append(f"{i}. {r.get('title', '').strip()} — {body} ({r.get('href', '')})")
    return "\n".join(lines)


class WebSearch(Tool):
    name = "web_search"
    description = (
        "Search the web for current or factual information. Returns the top results "
        "(title, snippet, link); summarize them for the user in your own words."
    )
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for."}},
        "required": ["query"],
    }

    async def run(self, ctx: ToolContext, query: str, **_: Any) -> str:
        try:
            results = await asyncio.to_thread(
                search_backend, query, ctx.config.tools.web_search_results
            )
        except ImportError:
            return "Web search is unavailable: install the 'ddgs' package."
        except Exception as exc:
            return f"Web search failed: {exc}"
        if not results:
            return f"No web results for '{query}'."
        return f"Web results for '{query}':\n{format_results(results)}"
