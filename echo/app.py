"""Builds the object graph and runs Echo."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass

from echo.audio.tts import TTSModule
from echo.audio.voice import VoiceModule
from echo.core.config import EchoConfig
from echo.core.events import EventBus
from echo.core.module import Module
from echo.core.orchestrator import Orchestrator
from echo.llm.agent import Agent
from echo.llm.module import LLMModule
from echo.llm.ollama_client import OllamaClient
from echo.tools.base import ToolContext, ToolRegistry
from echo.ui.dashboard import Dashboard, PlainUI
from echo.ui.keyboard import KeyboardReader
from echo.vision.camera import CameraModule
from echo.vision.gestures import GestureModule
from echo.vision.mood import MoodModule
from echo.vision.service import VisionService

log = logging.getLogger(__name__)


@dataclass
class EchoApp:
    config: EchoConfig
    bus: EventBus
    orchestrator: Orchestrator
    registry: ToolRegistry
    modules: list[Module]


def build_app(config: EchoConfig) -> EchoApp:
    bus = EventBus()
    llm = OllamaClient(config.llm.host, config.llm.timeout_s)
    camera = CameraModule(bus, config.vision)
    modules: list[Module] = [
        LLMModule(bus, llm, config.llm),
        VoiceModule(bus, config.audio),
        TTSModule(bus, config.tts),
        camera,
        GestureModule(bus, config.gestures, camera),
        MoodModule(bus, config.mood, camera),
    ]
    vision = VisionService(camera, llm, config)
    registry = ToolRegistry.discover(enabled=config.tools.enabled, disabled=config.tools.disabled)
    ctx = ToolContext(config=config, llm=llm, vision=vision, bus=bus)
    agent = Agent(llm, registry, ctx, config)
    orchestrator = Orchestrator(config, bus, agent, modules)
    return EchoApp(config, bus, orchestrator, registry, modules)


def use_dashboard(config: EchoConfig) -> bool:
    return config.ui.enabled and KeyboardReader.supported() and sys.stdout.isatty()


async def run(config: EchoConfig, dashboard: bool) -> None:
    app = build_app(config)
    orchestrator = app.orchestrator
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, orchestrator.request_shutdown)
        except (NotImplementedError, RuntimeError):  # Windows
            pass

    stop = asyncio.Event()
    if dashboard:
        ui: Dashboard | PlainUI = Dashboard(config, app.bus, app.modules, app.registry.names)
    else:
        ui = PlainUI(config, app.bus, app.modules)
    log.info("tools loaded: %s", ", ".join(app.registry.names))

    ui_task = asyncio.create_task(ui.run(stop, orchestrator.started))
    try:
        await orchestrator.run()
    finally:
        stop.set()
        ui_task.cancel()
        await asyncio.gather(ui_task, return_exceptions=True)
