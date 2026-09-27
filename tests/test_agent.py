from __future__ import annotations

from typing import Any

from echo.core.config import EchoConfig
from echo.llm.agent import OFFLINE_REPLY, Agent
from echo.tools.base import Tool, ToolContext, ToolRegistry
from tests.conftest import FakeLLM, answer, tool_call


class Echoer(Tool):
    name = "echoer"
    description = "repeat"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def run(self, ctx: ToolContext, text: str, **_: Any) -> str:
        return f"echo:{text}"


class Clock(Tool):
    name = "system_info"
    description = "time"
    parameters = {"type": "object", "properties": {"what": {"type": "string"}}}

    async def run(self, ctx: ToolContext, what: str = "all", **_: Any) -> str:
        return f"clock:{what}"


class Eyes(Tool):
    name = "describe_scene"
    description = "look"

    async def run(self, ctx: ToolContext, **_: Any) -> str:
        return "a cat on a desk"


def make_agent(config: EchoConfig, llm: FakeLLM, *tools: Tool) -> Agent:
    return Agent(llm, ToolRegistry(tools), ToolContext(config=config, llm=llm), config)


async def test_plain_answer(config: EchoConfig) -> None:
    llm = FakeLLM([answer("**Paris** is the capital.")])
    reply = await make_agent(config, llm, Echoer()).respond("capital of France?")
    assert reply.text == "Paris is the capital."
    assert llm.chat_calls[0]["tools"][0]["function"]["name"] == "echoer"


async def test_tool_loop_feeds_results_back(config: EchoConfig) -> None:
    llm = FakeLLM([tool_call("echoer", text="hi"), answer("It said hi.")])
    reply = await make_agent(config, llm, Echoer()).respond("repeat hi")
    assert reply.text == "It said hi."
    assert [c.result for c in reply.tool_calls] == ["echo:hi"]
    tool_msg = llm.chat_calls[1]["messages"][-1]
    assert tool_msg == {"role": "tool", "tool_name": "echoer", "content": "echo:hi"}


async def test_inline_json_tool_call_from_small_model(config: EchoConfig) -> None:
    llm = FakeLLM([answer('{"name": "echoer", "parameters": {"text": "yo"}}'), answer("Done")])
    reply = await make_agent(config, llm, Echoer()).respond("say yo")
    assert reply.tool_calls[0].result == "echo:yo"


async def test_string_arguments_are_parsed(config: EchoConfig) -> None:
    msg = {"role": "assistant", "content": "",
           "tool_calls": [{"function": {"name": "echoer", "arguments": '{"text": "s"}'}}]}
    llm = FakeLLM([msg, answer("fine")])
    reply = await make_agent(config, llm, Echoer()).respond("x")
    assert reply.tool_calls[0].result == "echo:s"


async def test_max_steps_forces_final_answer(config: EchoConfig) -> None:
    config.llm.max_tool_steps = 2
    llm = FakeLLM([tool_call("echoer", text="a"), tool_call("echoer", text="b"), answer("final")])
    reply = await make_agent(config, llm, Echoer()).respond("loop")
    assert reply.text == "final"
    assert llm.chat_calls[-1]["tools"] is None


async def test_unknown_tool_is_reported_to_llm(config: EchoConfig) -> None:
    llm = FakeLLM([tool_call("nope"), answer("sorry")])
    await make_agent(config, llm, Echoer()).respond("x")
    assert "no tool named 'nope'" in llm.chat_calls[1]["messages"][-1]["content"]


async def test_offline_fallback_routes_simple_commands(config: EchoConfig) -> None:
    llm = FakeLLM()
    llm.offline = True
    agent = make_agent(config, llm, Clock())
    reply = await agent.respond("What's the time?")
    assert reply.offline and reply.text == "clock:time"
    reply = await agent.respond("write me a poem")
    assert reply.text == OFFLINE_REPLY


async def test_vision_fast_path_skips_llm(config: EchoConfig) -> None:
    llm = FakeLLM()
    reply = await make_agent(config, llm, Eyes()).respond("Hey, what do you see?")
    assert reply.text == "a cat on a desk"
    assert llm.chat_calls == []


async def test_history_is_bounded(config: EchoConfig) -> None:
    config.llm.context_turns = 2
    agent = make_agent(config, FakeLLM([answer(str(i)) for i in range(5)]))
    for i in range(5):
        await agent.respond(f"q{i}")
    assert len(agent.history) == 4
    assert agent.history[-1]["content"] == "4"
