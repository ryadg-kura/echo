"""The brain: a tool-calling loop where the LLM decides which tool to use."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from echo.core.config import EchoConfig
from echo.llm.errors import LLMError, LLMUnavailable
from echo.llm.intents import FAST_PATH_TOOLS, match_intent
from echo.llm.ollama_client import ChatBackend, Message
from echo.llm.prompts import system_prompt
from echo.tools.base import ConfirmFn, ToolContext, ToolRegistry

log = logging.getLogger(__name__)

OFFLINE_REPLY = (
    "My language model is offline, so I can only handle simple commands right now, "
    "like the time, opening an app, or taking a note. Start Ollama to unlock the rest."
)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result: str


@dataclass
class AgentReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    offline: bool = False


class Agent:
    def __init__(
        self,
        llm: ChatBackend,
        registry: ToolRegistry,
        ctx: ToolContext,
        config: EchoConfig,
        confirm: ConfirmFn | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.ctx = ctx
        self.config = config
        self.confirm = confirm
        self.history: list[Message] = []

    async def respond(self, text: str, emotion: str | None = None) -> AgentReply:
        calls: list[ToolCall] = []
        intent = match_intent(text)
        if intent and intent[0] in FAST_PATH_TOOLS and intent[0] in self.registry.names:
            result = await self._call_tool(intent[0], intent[1], calls)
            return self._remember(text, AgentReply(result, calls))
        try:
            reply = await self._llm_loop(text, emotion, calls)
        except LLMUnavailable as exc:
            log.info("LLM unavailable (%s); using offline fallback", exc)
            if intent and intent[0] in self.registry.names:
                result = await self._call_tool(intent[0], intent[1], calls)
                return AgentReply(result, calls, offline=True)
            return AgentReply(OFFLINE_REPLY, calls, offline=True)
        except LLMError as exc:
            log.error("LLM error: %s", exc)
            return AgentReply(f"Sorry, my language model returned an error: {exc}", calls)
        return self._remember(text, AgentReply(reply, calls))

    async def _llm_loop(self, text: str, emotion: str | None, calls: list[ToolCall]) -> str:
        cfg = self.config.llm
        turns = self.history[-2 * cfg.context_turns :] if cfg.context_turns > 0 else []
        messages: list[Message] = [
            {"role": "system", "content": system_prompt(self.config.name, emotion)},
            *turns,
            {"role": "user", "content": text},
        ]
        tools = self.registry.schemas()
        options = {"temperature": cfg.temperature}
        for _ in range(cfg.max_tool_steps):
            message = await self.llm.chat(cfg.model, messages, tools=tools, options=options)
            tool_calls = message.get("tool_calls") or self._inline_tool_calls(message)
            if not tool_calls:
                return _clean_reply(str(message.get("content", "")))
            messages.append(
                {"role": "assistant", "content": message.get("content", ""), "tool_calls": tool_calls}
            )
            for call in tool_calls:
                function = call.get("function", {})
                name = str(function.get("name", ""))
                result = await self._call_tool(name, _parse_args(function.get("arguments")), calls)
                messages.append({"role": "tool", "tool_name": name, "content": result})
        # Out of tool steps: ask for a final answer without tools.
        message = await self.llm.chat(cfg.model, messages, options=options)
        return _clean_reply(str(message.get("content", "")))

    async def _call_tool(self, name: str, args: dict[str, Any], calls: list[ToolCall]) -> str:
        if self.ctx.bus is not None:
            self.ctx.bus.publish("tool.call", name=name, args=args)
        result = await self.registry.execute(name, args, self.ctx, self.confirm)
        log.info("tool %s(%s) -> %s", name, args, result[:200])
        calls.append(ToolCall(name, args, result))
        return result

    def _inline_tool_calls(self, message: Message) -> list[dict[str, Any]]:
        """Small models sometimes print the tool call as JSON instead of using tool_calls."""
        content = str(message.get("content", "")).strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, dict) or data.get("name") not in self.registry.names:
            return []
        args = data.get("parameters", data.get("arguments", {}))
        return [{"function": {"name": data["name"], "arguments": args}}]

    def _remember(self, text: str, reply: AgentReply) -> AgentReply:
        self.history += [
            {"role": "user", "content": text},
            {"role": "assistant", "content": reply.text},
        ]
        limit = 2 * max(self.config.llm.context_turns, 1)
        self.history = self.history[-limit:]
        return reply


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _clean_reply(text: str) -> str:
    """Strip markdown that would sound odd when spoken."""
    text = re.sub(r"[*`#]+", "", text)
    return re.sub(r"\n{2,}", "\n", text).strip() or "Sorry, I don't have an answer for that."
