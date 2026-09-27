from __future__ import annotations

import re

from rich.console import Console

from echo.app import build_app
from echo.core.config import EchoConfig
from echo.core.events import Event, EventBus
from echo.ui.dashboard import Dashboard, Line, PlainUI, handle_command


async def test_handle_command() -> None:
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe("*", lambda e: seen.append(e.topic))
    assert handle_command("", bus)
    assert handle_command("/listen", bus)
    assert handle_command("/quit", bus)
    assert not handle_command("hello", bus)
    await bus.drain()
    assert seen == ["listen.toggle", "listen.start", "system.shutdown"]


async def test_dashboard_renders_and_tracks_conversation(config: EchoConfig) -> None:
    app = build_app(config)
    bus = EventBus()  # isolated from the orchestrator: we only exercise the UI
    ui = Dashboard(config, bus, app.modules, app.registry.names)
    ui.console = Console(width=120, height=20, record=True, color_system=None)
    bus.publish("user.text", text="hello", source="voice")
    bus.publish("mood", emotion="happy", confidence=0.9)
    for i in range(40):
        bus.publish("assistant.reply", text=f"reply {i}", kind="answer")
    await bus.drain()
    ui.console.print(ui.render())
    screen = ui.console.export_text()
    assert "reply 39" in screen  # newest line stays visible in a short terminal
    assert not re.search(r"\breply 0\b", screen)  # oldest line scrolled out
    assert "happy" in screen and "web_search" in screen
    assert not ui.thinking


async def test_plain_ui_escapes_markup(config: EchoConfig) -> None:
    ui = PlainUI(config, EventBus(), [])
    ui.console = Console(record=True, width=80, color_system=None)
    ui.add(Line("Echo", "[not markup] ok", "cyan"))
    ui._on_status(Event("module.status", {"module": "llm", "state": "ready", "detail": "[x]"}))
    output = ui.console.export_text()
    assert "[not markup] ok" in output
    assert "[x]" in output
