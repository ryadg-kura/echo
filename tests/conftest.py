from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from echo.core.config import EchoConfig
from echo.llm.errors import LLMUnavailable
from echo.llm.ollama_client import Message
from echo.tools.base import ToolContext


class FakeLLM:
    """Scripted ChatBackend: returns queued messages and records every request."""

    def __init__(self, replies: list[Message] | None = None, generate_reply: str = "summary") -> None:
        self.replies = list(replies or [])
        self.generate_reply = generate_reply
        self.chat_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.offline = False

    async def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Message:
        if self.offline:
            raise LLMUnavailable("offline")
        self.chat_calls.append({"model": model, "messages": [dict(m) for m in messages], "tools": tools})
        return self.replies.pop(0) if self.replies else {"role": "assistant", "content": "ok"}

    async def generate(
        self, model: str, prompt: str, system: str | None = None, images: list[str] | None = None
    ) -> str:
        if self.offline:
            raise LLMUnavailable("offline")
        self.generate_calls.append({"model": model, "prompt": prompt, "images": images})
        return self.generate_reply


def tool_call(name: str, **args: Any) -> Message:
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def answer(text: str) -> Message:
    return {"role": "assistant", "content": text}


@pytest.fixture
def config(tmp_path: Path) -> EchoConfig:
    cfg = EchoConfig()
    cfg.tools.notes_dir = str(tmp_path / "notes")
    cfg.logging.file = str(tmp_path / "logs" / "echo.log")
    return cfg


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def make_ctx(config: EchoConfig, fake_llm: FakeLLM) -> Callable[..., ToolContext]:
    def factory(**overrides: Any) -> ToolContext:
        return ToolContext(**{"config": config, "llm": fake_llm, **overrides})

    return factory
