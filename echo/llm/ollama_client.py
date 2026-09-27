"""Minimal async client for the local Ollama HTTP API."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import httpx

from echo.llm.errors import LLMError, LLMUnavailable

log = logging.getLogger(__name__)

Message = dict[str, Any]


class ChatBackend(Protocol):
    """What the agent needs from an LLM backend (lets tests plug in fakes)."""

    async def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Message: ...

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        images: list[str] | None = None,
    ) -> str: ...


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.host, timeout=httpx.Timeout(timeout, connect=3.0), transport=transport
        )

    async def list_models(self) -> list[str]:
        data = await self._request("GET", "/api/tags")
        return [m["name"] for m in data.get("models", [])]

    async def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Message:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
        }
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options
        data = await self._request("POST", "/api/chat", json=payload, model=model)
        message: Message = data.get("message") or {}
        return message

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        images: list[str] | None = None,
    ) -> str:
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        user: Message = {"role": "user", "content": prompt}
        if images:
            user["images"] = images
        messages.append(user)
        reply = await self.chat(model, messages)
        return str(reply.get("content", "")).strip()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self, method: str, path: str, json: dict[str, Any] | None = None, model: str | None = None
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, json=json)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMUnavailable(f"Ollama is not reachable at {self.host}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        if response.status_code == 404 and model:
            raise LLMUnavailable(f"model '{model}' is not installed (run: ollama pull {model})")
        if response.status_code >= 400:
            try:
                detail = response.json().get("error", response.text)
            except ValueError:
                detail = response.text
            raise LLMError(f"Ollama error {response.status_code}: {detail}")
        result: dict[str, Any] = response.json()
        return result
