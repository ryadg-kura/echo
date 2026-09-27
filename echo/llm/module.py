"""Tracks Ollama availability so the UI shows it and the user knows about the fallback."""

from __future__ import annotations

import asyncio

from echo.core.config import LLMConfig
from echo.core.events import EventBus
from echo.core.module import Module, ModuleState
from echo.llm.errors import LLMError
from echo.llm.ollama_client import OllamaClient


class LLMModule(Module):
    name = "llm"

    def __init__(self, bus: EventBus, client: OllamaClient, config: LLMConfig) -> None:
        super().__init__(bus)
        self.client = client
        self.config = config

    async def start(self) -> None:
        await self.check()
        self.spawn(self._monitor())

    async def check(self) -> None:
        try:
            installed = await self.client.list_models()
        except LLMError as exc:
            self.set_status(ModuleState.UNAVAILABLE, f"{exc} — offline fallback")
            return
        missing = [
            m for m in (self.config.model, self.config.vision_model) if not _installed(m, installed)
        ]
        if self.config.model in missing:
            self.set_status(ModuleState.UNAVAILABLE, f"run: ollama pull {self.config.model}")
        elif missing:
            self.set_status(ModuleState.READY, f"{self.config.model} (vision: ollama pull {missing[0]})")
        elif self.state not in (ModuleState.BUSY, ModuleState.ACTIVE):
            self.set_status(ModuleState.READY, f"{self.config.model} + {self.config.vision_model}")

    async def _monitor(self) -> None:
        while True:
            await asyncio.sleep(self.config.health_check_interval_s)
            await self.check()

    async def stop(self) -> None:
        await super().stop()
        await self.client.aclose()


def _installed(model: str, installed: list[str]) -> bool:
    names = set(installed)
    return model in names or f"{model}:latest" in names
