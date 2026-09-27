from __future__ import annotations

from collections.abc import Callable

import pytest

from echo.tools import web_search
from echo.tools.base import ToolContext
from echo.tools.web_search import WebSearch

RESULTS = [
    {"title": "Ollama", "href": "https://ollama.com", "body": "Run   large language\nmodels locally."},
    {"title": "Whisper", "href": "https://example.org", "body": "Speech recognition."},
]


async def test_formats_results(monkeypatch: pytest.MonkeyPatch, make_ctx: Callable[..., ToolContext]) -> None:
    calls: list[tuple[str, int]] = []

    def fake(query: str, max_results: int) -> list[dict[str, str]]:
        calls.append((query, max_results))
        return RESULTS

    monkeypatch.setattr(web_search, "search_backend", fake)
    result = await WebSearch().run(make_ctx(), query="local llm")
    assert calls == [("local llm", 5)]
    assert "1. Ollama — Run large language models locally. (https://ollama.com)" in result
    assert "2. Whisper" in result


async def test_no_results_and_failures(monkeypatch: pytest.MonkeyPatch, make_ctx: Callable[..., ToolContext]) -> None:
    monkeypatch.setattr(web_search, "search_backend", lambda q, n: [])
    assert "No web results" in await WebSearch().run(make_ctx(), query="zzz")

    def offline(q: str, n: int) -> list[dict[str, str]]:
        raise ConnectionError("network down")

    monkeypatch.setattr(web_search, "search_backend", offline)
    assert await WebSearch().run(make_ctx(), query="x") == "Web search failed: network down"
