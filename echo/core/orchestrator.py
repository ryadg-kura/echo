"""Wires modules to the agent through the event bus and runs them concurrently."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from echo.core.config import EchoConfig
from echo.core.events import Event, EventBus
from echo.core.module import Module, ModuleState, ModuleUnavailable
from echo.llm.agent import Agent
from echo.llm.intents import parse_yes_no

log = logging.getLogger(__name__)

GESTURE_ACTIONS = {"open_palm": "listen.start", "fist": "listen.stop"}


class Orchestrator:
    def __init__(
        self,
        config: EchoConfig,
        bus: EventBus,
        agent: Agent,
        modules: Sequence[Module] = (),
    ) -> None:
        self.config = config
        self.bus = bus
        self.agent = agent
        self.modules = list(modules)
        self.emotion = "neutral"
        self._turn_lock = asyncio.Lock()
        self._pending_confirm: asyncio.Future[bool] | None = None
        self._shutdown = asyncio.Event()
        self.started = asyncio.Event()
        agent.confirm = self.confirm

        bus.subscribe("user.text", self._on_user_text)
        bus.subscribe("gesture", self._on_gesture)
        bus.subscribe("mood", self._on_mood)
        bus.subscribe("system.shutdown", lambda _: self._shutdown.set())

    def module(self, name: str) -> Module | None:
        return next((m for m in self.modules if m.name == name), None)

    # --- lifecycle ------------------------------------------------------------------------

    async def start(self) -> None:
        """Start every module concurrently, respecting ``depends_on`` ordering."""
        self.bus.bind_loop(asyncio.get_running_loop())
        tasks: dict[str, asyncio.Task[None]] = {}

        async def start_after_deps(module: Module) -> None:
            for dep in module.depends_on:
                if dep in tasks:
                    await asyncio.shield(tasks[dep])
            await self._start_module(module)

        for module in self.modules:
            tasks[module.name] = asyncio.create_task(start_after_deps(module))
        await asyncio.gather(*tasks.values())
        self.started.set()
        ready = [m.name for m in self.modules if m.available]
        log.info("Echo started; available modules: %s", ", ".join(ready) or "none")

    async def _start_module(self, module: Module) -> None:
        if not module.enabled:
            module.set_status(ModuleState.DISABLED, "disabled in config")
            return
        module.set_status(ModuleState.STARTING, module.detail)
        try:
            await module.start()
        except ModuleUnavailable as exc:
            log.info("%s unavailable: %s", module.name, exc)
            module.set_status(ModuleState.UNAVAILABLE, str(exc))
            return
        except Exception as exc:
            log.exception("%s failed to start", module.name)
            module.set_status(ModuleState.ERROR, str(exc))
            return
        if module.state is ModuleState.STARTING:
            module.set_status(ModuleState.READY, module.detail)

    async def run(self) -> None:
        """Start modules, then block until ``system.shutdown`` (or cancellation)."""
        await self.start()
        try:
            await self._shutdown.wait()
        finally:
            await self.stop()

    def request_shutdown(self) -> None:
        self._shutdown.set()

    async def stop(self) -> None:
        if self._pending_confirm and not self._pending_confirm.done():
            self._pending_confirm.set_result(False)
        for module in reversed(self.modules):
            try:
                await asyncio.wait_for(module.stop(), timeout=5)
            except Exception:
                log.exception("error stopping %s", module.name)

    # --- conversation ---------------------------------------------------------------------

    async def _on_user_text(self, event: Event) -> None:
        text = str(event.get("text", "")).strip()
        if not text:
            return
        pending = self._pending_confirm
        if pending is not None and not pending.done():
            answer = parse_yes_no(text)
            if answer is not None:
                pending.set_result(answer)
                return
        await self.handle(text)

    async def handle(self, text: str) -> str:
        """Run one conversational turn and publish the reply."""
        async with self._turn_lock:
            self.bus.publish("assistant.thinking")
            try:
                reply = await self.agent.respond(text, emotion=self.emotion)
                answer = reply.text
            except Exception as exc:
                log.exception("agent failed")
                answer = f"Sorry, something went wrong: {exc}"
            self.bus.publish("assistant.reply", text=answer, kind="answer")
            return answer

    async def confirm(self, prompt: str) -> bool:
        """Ask the user to confirm an action by thumbs up or by saying/typing yes."""
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending_confirm = future
        self.bus.publish(
            "assistant.reply", text=f"{prompt} Give me a thumbs up or say yes.", kind="confirm"
        )
        try:
            confirmed = await asyncio.wait_for(future, self.config.tools.confirm_timeout_s)
        except TimeoutError:
            confirmed = False
        finally:
            self._pending_confirm = None
        if not confirmed:
            self.bus.publish("assistant.reply", text="Okay, I won't.", kind="info")
        return confirmed

    @property
    def awaiting_confirmation(self) -> bool:
        return self._pending_confirm is not None and not self._pending_confirm.done()

    # --- gestures & mood ------------------------------------------------------------------

    async def _on_gesture(self, event: Event) -> None:
        name = event.get("name")
        if name == "thumbs_up":
            if self.awaiting_confirmation:
                assert self._pending_confirm is not None
                self._pending_confirm.set_result(True)
            return
        if name == "fist" and self.awaiting_confirmation:
            assert self._pending_confirm is not None
            self._pending_confirm.set_result(False)
            return
        if topic := GESTURE_ACTIONS.get(str(name)):
            self.bus.publish(topic, source="gesture")

    async def _on_mood(self, event: Event) -> None:
        self.emotion = str(event.get("emotion", "neutral"))
        log.info("mood -> %s", self.emotion)
