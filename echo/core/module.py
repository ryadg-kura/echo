"""Base class for long-running modules (voice, camera, gestures...)."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from enum import StrEnum
from typing import Any, ClassVar

from echo.core.events import EventBus

log = logging.getLogger(__name__)


class ModuleState(StrEnum):
    DISABLED = "disabled"
    STARTING = "starting"
    READY = "ready"
    ACTIVE = "active"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ModuleUnavailable(RuntimeError):
    """Raised by ``Module.start`` when hardware or a dependency is missing.

    Echo keeps running without the module (graceful fallback).
    """


class Module(ABC):
    name: ClassVar[str]
    depends_on: ClassVar[tuple[str, ...]] = ()

    def __init__(self, bus: EventBus, enabled: bool = True) -> None:
        self.bus = bus
        self.enabled = enabled
        self.state = ModuleState.STARTING if enabled else ModuleState.DISABLED
        self.detail = ""
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def available(self) -> bool:
        return self.state in (ModuleState.READY, ModuleState.ACTIVE, ModuleState.BUSY)

    def set_status(self, state: ModuleState, detail: str = "") -> None:
        if (state, detail) == (self.state, self.detail):
            return
        self.state, self.detail = state, detail
        log.debug("module %s -> %s %s", self.name, state.value, detail)
        try:
            self.bus.publish("module.status", module=self.name, state=state.value, detail=detail)
        except RuntimeError:  # no running loop (e.g. constructed outside asyncio)
            pass

    def spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Run a background task owned by this module; cancelled on ``stop``."""
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            log.error("module %s task crashed: %s", self.name, exc, exc_info=exc)
            self.set_status(ModuleState.ERROR, str(exc))

    @abstractmethod
    async def start(self) -> None:
        """Acquire resources. Raise ``ModuleUnavailable`` to fall back gracefully."""

    async def stop(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
