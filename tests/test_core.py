from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from echo.core.config import ConfigError, config_from_dict, load_config
from echo.core.events import Event, EventBus
from echo.llm.errors import LLMError, LLMUnavailable
from echo.llm.intents import match_intent, parse_yes_no
from echo.llm.ollama_client import OllamaClient
from echo.llm.prompts import system_prompt

ROOT = Path(__file__).resolve().parent.parent


# --- config ---------------------------------------------------------------------------------

def test_repo_config_loads() -> None:
    config = load_config(ROOT / "config.yaml")
    assert config.llm.model == "llama3.2:3b"
    assert config.llm.vision_model == "moondream"


def test_partial_config_keeps_defaults() -> None:
    config = config_from_dict({"audio": {"mode": "wake_word"}})
    assert config.audio.mode == "wake_word"
    assert config.audio.stt_model == "base.en"


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="audio.*modee"):
        config_from_dict({"audio": {"modee": "x"}})


def test_missing_default_config_gives_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ECHO_CONFIG", raising=False)
    assert load_config().name == "Echo"


# --- event bus ------------------------------------------------------------------------------

async def test_bus_delivers_to_topic_and_wildcard() -> None:
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe("a", lambda e: seen.append("a:" + e.get("x")))
    bus.subscribe("*", lambda e: seen.append("*:" + e.topic))
    bus.publish("a", x="1")
    bus.publish("b")
    await bus.drain()
    assert sorted(seen) == ["*:a", "*:b", "a:1"]


async def test_bus_isolates_failing_handlers() -> None:
    bus = EventBus()
    seen: list[Event] = []

    def boom(_: Event) -> None:
        raise ValueError("bad handler")

    bus.subscribe("t", boom)
    bus.subscribe("t", seen.append)
    bus.publish("t")
    await bus.drain()
    assert len(seen) == 1


async def test_bus_wait_for_and_unsubscribe() -> None:
    bus = EventBus()
    waiter = asyncio.create_task(bus.wait_for("t", timeout=1, predicate=lambda e: e.get("n") == 2))
    await asyncio.sleep(0)
    bus.publish("t", n=1)
    bus.publish("t", n=2)
    event = await waiter
    assert event is not None and event.get("n") == 2
    assert await bus.wait_for("never", timeout=0.01) is None

    seen: list[Event] = []
    unsubscribe = bus.subscribe("u", seen.append)
    unsubscribe()
    bus.publish("u")
    await bus.drain()
    assert seen == []


# --- intents & prompts ----------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "tool", "args"),
    [
        ("What do you see?", "describe_scene", {}),
        ("read this", "read_text", {}),
        ("what time is it", "system_info", {"what": "time"}),
        ("how's my battery", "system_info", {"what": "battery"}),
        ("open Firefox", "open_app", {"app": "Firefox"}),
        ("search the web for local llms", "web_search", {"query": "local llms"}),
        ("search my notes for milk", "notes", {"action": "search", "query": "milk"}),
        ("summarize report.pdf", "summarize_pdf", {"path": "report.pdf"}),
    ],
)
def test_match_intent(text: str, tool: str, args: dict[str, str]) -> None:
    assert match_intent(text) == (tool, args)


def test_note_intent_and_no_match() -> None:
    tool, args = match_intent("take a note buy milk and eggs")  # type: ignore[misc]
    assert tool == "notes" and args["content"] == "buy milk and eggs"
    assert match_intent("tell me a joke") is None


@pytest.mark.parametrize(("text", "expected"), [
    ("yes", True), ("Yes please!", True), ("go ahead", True), ("no", False),
    ("cancel that", False), ("maybe later", None),
])
def test_parse_yes_no(text: str, expected: bool | None) -> None:
    assert parse_yes_no(text) is expected


def test_system_prompt_adapts_tone() -> None:
    assert "angry" in system_prompt("Echo", "angry")
    assert "friendly" in system_prompt("Echo", None)


# --- ollama client --------------------------------------------------------------------------

async def test_ollama_client_chat_and_tags() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})
        body = json.loads(request.content)
        assert body["stream"] is False and body["tools"][0]["function"]["name"] == "t"
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}})

    client = OllamaClient(transport=httpx.MockTransport(handler))
    assert await client.list_models() == ["llama3.2:3b"]
    tools = [{"type": "function", "function": {"name": "t"}}]
    reply = await client.chat("llama3.2:3b", [{"role": "user", "content": "x"}], tools=tools)
    assert reply["content"] == "hi"
    await client.aclose()


async def test_ollama_client_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "missing":
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(400, json={"error": "does not support tools"})

    client = OllamaClient(transport=httpx.MockTransport(handler))
    with pytest.raises(LLMUnavailable, match="ollama pull missing"):
        await client.generate("missing", "x")
    with pytest.raises(LLMError, match="does not support tools"):
        await client.generate("other", "x")

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    offline = OllamaClient(transport=httpx.MockTransport(refuse))
    with pytest.raises(LLMUnavailable):
        await offline.list_models()
