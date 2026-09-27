from __future__ import annotations

import asyncio
from typing import Any

import pytest

from echo.core.config import EchoConfig
from echo.core.events import Event, EventBus
from echo.core.module import Module, ModuleState, ModuleUnavailable
from echo.core.orchestrator import Orchestrator
from echo.llm.agent import Agent
from echo.tools.base import Tool, ToolContext, ToolRegistry
from tests.conftest import FakeLLM, answer, tool_call


class Launch(Tool):
    name = "launch"
    description = "needs confirmation"
    parameters = {"type": "object", "properties": {"what": {"type": "string"}}}
    requires_confirmation = True

    def __init__(self) -> None:
        self.ran = False

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        self.ran = True
        return "launched"


class DummyModule(Module):
    def __init__(self, bus: EventBus, name: str, fail: Exception | None = None,
                 depends_on: tuple[str, ...] = (), log: list[str] | None = None) -> None:
        self.name = name  # type: ignore[misc]
        self.depends_on = depends_on  # type: ignore[misc]
        super().__init__(bus)
        self.fail = fail
        self.log = log if log is not None else []
        self.stopped = False

    async def start(self) -> None:
        await asyncio.sleep(0.01)
        self.log.append(self.name)
        if self.fail:
            raise self.fail

    async def stop(self) -> None:
        self.stopped = True


def build(config: EchoConfig, llm: FakeLLM, tools: list[Tool] | None = None,
          modules: list[Module] | None = None,
          bus: EventBus | None = None) -> tuple[Orchestrator, EventBus, list[Event]]:
    bus = bus or EventBus()
    registry = ToolRegistry(tools or [])
    agent = Agent(llm, registry, ToolContext(config=config, llm=llm, bus=bus), config)
    orchestrator = Orchestrator(config, bus, agent, modules or [])
    replies: list[Event] = []
    bus.subscribe("assistant.reply", replies.append)
    return orchestrator, bus, replies


async def test_user_text_produces_reply(config: EchoConfig) -> None:
    llm = FakeLLM([answer("Hello there!")])
    orchestrator, bus, replies = build(config, llm)
    bus.publish("user.text", text="hi", source="keyboard")
    await bus.drain()
    assert [r.get("text") for r in replies] == ["Hello there!"]


async def test_turns_are_serialized(config: EchoConfig) -> None:
    llm = FakeLLM([answer("one"), answer("two")])
    orchestrator, bus, replies = build(config, llm)
    bus.publish("user.text", text="first")
    bus.publish("user.text", text="second")
    await bus.drain()
    assert [r.get("text") for r in replies] == ["one", "two"]
    # the second turn saw the first one in its history
    assert any(m["content"] == "first" for m in llm.chat_calls[1]["messages"])


async def test_mood_changes_system_prompt(config: EchoConfig) -> None:
    llm = FakeLLM([answer("there there")])
    orchestrator, bus, _ = build(config, llm)
    bus.publish("mood", emotion="sad", confidence=0.9)
    await bus.drain()
    bus.publish("user.text", text="rough day")
    await bus.drain()
    system = llm.chat_calls[0]["messages"][0]["content"]
    assert "sad" in system and "gentle" in system


async def test_thumbs_up_confirms_tool(config: EchoConfig) -> None:
    tool = Launch()
    llm = FakeLLM([tool_call("launch", what="x"), answer("Done.")])
    orchestrator, bus, replies = build(config, llm, [tool])
    bus.publish("user.text", text="launch x")
    await bus.wait_for("assistant.reply", timeout=1, predicate=lambda e: e.get("kind") == "confirm")
    bus.publish("gesture", name="thumbs_up")
    await bus.drain()
    assert tool.ran
    assert replies[-1].get("text") == "Done."


async def test_spoken_no_cancels_tool(config: EchoConfig) -> None:
    tool = Launch()
    llm = FakeLLM([tool_call("launch"), answer("Okay, not launching.")])
    orchestrator, bus, replies = build(config, llm, [tool])
    bus.publish("user.text", text="launch it")
    await bus.wait_for("assistant.reply", timeout=1, predicate=lambda e: e.get("kind") == "confirm")
    bus.publish("user.text", text="No thanks")
    await bus.drain()
    assert not tool.ran
    tool_msg = [m for m in llm.chat_calls[1]["messages"] if m["role"] == "tool"][0]
    assert "Cancelled" in tool_msg["content"]


async def test_confirmation_times_out(config: EchoConfig) -> None:
    config.tools.confirm_timeout_s = 0.05
    tool = Launch()
    orchestrator, bus, replies = build(config, FakeLLM([tool_call("launch"), answer("ok")]), [tool])
    await orchestrator.handle("launch it")
    await bus.drain()
    assert not tool.ran
    assert any(r.get("text") == "Okay, I won't." for r in replies)


@pytest.mark.parametrize(("gesture", "topic"), [("open_palm", "listen.start"), ("fist", "listen.stop")])
async def test_gestures_map_to_listening(config: EchoConfig, gesture: str, topic: str) -> None:
    orchestrator, bus, _ = build(config, FakeLLM())
    seen: list[Event] = []
    bus.subscribe(topic, seen.append)
    bus.publish("gesture", name=gesture)
    await bus.drain()
    assert len(seen) == 1 and seen[0].get("source") == "gesture"


async def test_modules_start_with_graceful_fallback(config: EchoConfig) -> None:
    bus = EventBus()
    order: list[str] = []
    camera = DummyModule(bus, "camera", fail=ModuleUnavailable("no camera"), log=order)
    gestures = DummyModule(bus, "gestures", depends_on=("camera",), log=order)
    broken = DummyModule(bus, "broken", fail=RuntimeError("boom"), log=order)
    disabled = DummyModule(bus, "off", log=order)
    disabled.enabled = False
    orchestrator, _, _ = build(config, FakeLLM(), modules=[gestures, camera, broken, disabled], bus=bus)

    await orchestrator.start()

    assert order.index("camera") < order.index("gestures")
    assert camera.state is ModuleState.UNAVAILABLE and camera.detail == "no camera"
    assert broken.state is ModuleState.ERROR
    assert gestures.state is ModuleState.READY
    assert disabled.state is ModuleState.DISABLED and "off" not in order
    assert orchestrator.started.is_set()


async def test_run_until_shutdown_stops_modules(config: EchoConfig) -> None:
    bus = EventBus()
    module = DummyModule(bus, "m")
    orchestrator, _, _ = build(config, FakeLLM(), modules=[module], bus=bus)
    task = asyncio.create_task(orchestrator.run())
    await asyncio.wait_for(orchestrator.started.wait(), 1)
    bus.publish("system.shutdown")
    await asyncio.wait_for(task, 1)
    assert module.stopped


async def test_agent_crash_still_replies(config: EchoConfig) -> None:
    orchestrator, bus, replies = build(config, FakeLLM())

    async def explode(text: str, emotion: str | None = None) -> Any:
        raise RuntimeError("kaput")

    orchestrator.agent.respond = explode  # type: ignore[method-assign]
    await orchestrator.handle("hi")
    await bus.drain()
    assert "kaput" in replies[-1].get("text")
