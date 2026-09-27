"""Terminal UIs: a live rich dashboard, and a plain line-based fallback."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from echo.core.config import EchoConfig
from echo.core.events import Event, EventBus
from echo.core.module import Module, ModuleState
from echo.ui.keyboard import KeyboardReader

STATE_STYLE: dict[ModuleState, tuple[str, str]] = {
    ModuleState.READY: ("●", "green"),
    ModuleState.ACTIVE: ("◉", "bold cyan"),
    ModuleState.BUSY: ("◌", "yellow"),
    ModuleState.STARTING: ("…", "yellow"),
    ModuleState.UNAVAILABLE: ("○", "dim red"),
    ModuleState.ERROR: ("✖", "bold red"),
    ModuleState.DISABLED: ("–", "dim"),
}

MOOD_ICON = {"happy": "😊", "sad": "😔", "angry": "😠", "fearful": "😨", "surprised": "😮",
             "disgusted": "😖", "neutral": "😐"}

HELP = "Type a message · Enter on empty line = push-to-talk · /quit to exit"


@dataclass
class Line:
    who: str
    text: str
    style: str


def handle_command(text: str, bus: EventBus) -> bool:
    """Handle UI slash-commands and push-to-talk. Returns True if ``text`` was consumed."""
    stripped = text.strip()
    if not stripped:
        bus.publish("listen.toggle", source="keyboard")
        return True
    if stripped in ("/quit", "/exit", "/q"):
        bus.publish("system.shutdown")
        return True
    if stripped == "/listen":
        bus.publish("listen.start", source="keyboard")
        return True
    return False


class TranscriptMixin:
    """Collects conversation lines from bus events."""

    lines: deque[Line]
    thinking: bool
    mood: str
    last_warning: str

    def _init_transcript(self, bus: EventBus, history: int) -> None:
        self.lines = deque(maxlen=history)
        self.thinking = False
        self.mood = "neutral"
        self.last_warning = ""
        bus.subscribe("user.text", self._on_event)
        bus.subscribe("assistant.reply", self._on_event)
        bus.subscribe("assistant.thinking", self._on_event)
        bus.subscribe("tool.call", self._on_event)
        bus.subscribe("gesture", self._on_event)
        bus.subscribe("mood", self._on_event)

    def _on_event(self, event: Event) -> None:
        topic = event.topic
        if topic == "user.text":
            icon = "🎤" if event.get("source") == "voice" else "⌨"
            self.add(Line(f"{icon} You", str(event.get("text")), "bold white"))
        elif topic == "assistant.thinking":
            self.thinking = True
        elif topic == "assistant.reply":
            if event.get("kind") == "answer":
                self.thinking = False
            self.add(Line("Echo", str(event.get("text")), "cyan"))
        elif topic == "tool.call":
            args = ", ".join(f"{k}={v!r}" for k, v in (event.get("args") or {}).items())
            self.add(Line("tool", f"{event.get('name')}({args})", "dim magenta"))
        elif topic == "gesture":
            self.add(Line("gesture", str(event.get("name")), "dim yellow"))
        elif topic == "mood":
            self.mood = str(event.get("emotion"))

    def add(self, line: Line) -> None:
        self.lines.append(line)


class UIWarningHandler(logging.Handler):
    """Surfaces the latest warning in the dashboard instead of printing over it."""

    def __init__(self, target: TranscriptMixin) -> None:
        super().__init__(level=logging.WARNING)
        self.target = target

    def emit(self, record: logging.LogRecord) -> None:
        self.target.last_warning = f"{time.strftime('%H:%M:%S')} {record.getMessage()}"


class Dashboard(TranscriptMixin):
    def __init__(
        self,
        config: EchoConfig,
        bus: EventBus,
        modules: Sequence[Module],
        tool_names: Sequence[str] = (),
    ) -> None:
        self.config = config
        self.bus = bus
        self.modules = modules
        self.tool_names = list(tool_names)
        self._init_transcript(bus, config.ui.history)
        self._loop: asyncio.AbstractEventLoop | None = None
        self.console = Console()
        self.keyboard = KeyboardReader(self._submit, self._eof)

    def _submit(self, text: str) -> None:
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._dispatch, text)

    def _eof(self) -> None:
        assert self._loop is not None
        self._loop.call_soon_threadsafe(lambda: self.bus.publish("system.shutdown"))

    def _dispatch(self, text: str) -> None:
        if not handle_command(text, self.bus):
            self.bus.publish("user.text", text=text, source="keyboard")

    async def run(self, until: asyncio.Event, started: asyncio.Event | None = None) -> None:
        self._loop = asyncio.get_running_loop()
        console = self.console
        handler = UIWarningHandler(self)
        logging.getLogger().addHandler(handler)
        self.keyboard.start()
        try:
            with Live(self.render(), console=console, screen=True,
                      refresh_per_second=self.config.ui.refresh_per_second) as live:
                while not until.is_set():
                    live.update(self.render())
                    await asyncio.sleep(1 / self.config.ui.refresh_per_second)
        finally:
            self.keyboard.stop()
            logging.getLogger().removeHandler(handler)

    # --- rendering ------------------------------------------------------------------------

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(), name="header", size=3),
            Layout(name="body"),
            Layout(self._input(), name="input", size=3),
        )
        layout["body"].split_row(
            Layout(self._status_panel(), name="status", ratio=2),
            Layout(self._conversation(), name="chat", ratio=3),
        )
        return layout

    def _header(self) -> Panel:
        mood_icon = MOOD_ICON.get(self.mood, "🙂")
        title = Text.assemble(
            (f" {self.config.name} ", "bold black on cyan"),
            "  local-first multimodal assistant   ",
            (f"mood: {mood_icon} {self.mood}", "magenta"),
        )
        return Panel(title, style="cyan")

    def _status_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(width=2)
        table.add_column(style="bold", width=9)
        table.add_column(ratio=1, overflow="fold")
        for module in self.modules:
            icon, style = STATE_STYLE[module.state]
            detail = module.detail or module.state.value
            table.add_row(Text(icon, style), module.name, Text(detail, style="dim" if
                          module.state in (ModuleState.DISABLED, ModuleState.UNAVAILABLE) else ""))
        table.add_row(Text("●", "green"), "tools", Text(", ".join(self.tool_names), "dim"))
        parts: list[Table | Text] = [table]
        if self.last_warning:
            parts.append(Text(f"\n⚠ {self.last_warning}", style="yellow", overflow="fold"))
        return Panel(Group(*parts), title="modules", border_style="blue")

    def _visible_lines(self) -> list[Line]:
        """Newest lines that fit in the chat panel (rich would crop the newest otherwise)."""
        width = max(int(self.console.width * 3 / 5) - 4, 20)
        rows_left = max(self.console.height - 9, 3)
        visible: list[Line] = []
        for line in reversed(self.lines):
            rows = (len(line.who) + len(line.text) + 2) // width + 1
            if rows > rows_left and visible:
                break
            visible.append(line)
            rows_left -= rows
        return visible[::-1]

    def _conversation(self) -> Panel:
        body = Text()
        for line in self._visible_lines():
            body.append(f"{line.who}: ", style=f"bold {line.style}")
            body.append(f"{line.text}\n", style=line.style)
        if self.thinking:
            body.append("Echo is thinking…", style="italic dim cyan")
        return Panel(body, title="conversation", border_style="cyan")

    def _input(self) -> Panel:
        cursor = "█" if int(time.monotonic() * 2) % 2 == 0 else " "
        text = Text.assemble(("› ", "bold cyan"), self.keyboard.buffer, (cursor, "cyan"))
        return Panel(text, subtitle=HELP, subtitle_align="right", border_style="dim")


class PlainUI(TranscriptMixin):
    """Line-based UI for non-interactive terminals (pipes, CI, Windows consoles)."""

    def __init__(self, config: EchoConfig, bus: EventBus, modules: Sequence[Module]) -> None:
        self.config = config
        self.bus = bus
        self.modules = modules
        self.console = Console()
        self._init_transcript(bus, config.ui.history)
        self._reported: set[str] = set()
        bus.subscribe("module.status", self._on_status)

    def add(self, line: Line) -> None:
        super().add(line)
        if line.who.endswith("You"):
            return  # the user already sees what they typed
        self.console.print(f"[{line.style}]{line.who}:[/] {escape(line.text)}", highlight=False)

    def _on_status(self, event: Event) -> None:
        state = ModuleState(event.get("state"))
        module = str(event.get("module"))
        # Print each module's first real status, then only problems (no "speaking" spam).
        if state is ModuleState.STARTING:
            return
        if module in self._reported and state not in (ModuleState.ERROR, ModuleState.UNAVAILABLE):
            return
        self._reported.add(module)
        icon, style = STATE_STYLE[state]
        self.console.print(
            f"[{style}]{icon} {module}[/] [dim]{escape(str(event.get('detail') or state.value))}[/]",
            highlight=False,
        )

    async def run(self, until: asyncio.Event, started: asyncio.Event | None = None) -> None:
        if started is not None:
            await started.wait()
        self.console.print(f"[bold cyan]{self.config.name}[/] ready. {HELP}")
        loop = asyncio.get_running_loop()
        while not until.is_set():
            line = await loop.run_in_executor(None, _read_line)
            if line is None:
                # stdin closed: let in-flight replies finish, then exit.
                await self.bus.drain()
                self.bus.publish("system.shutdown")
                return
            if not handle_command(line, self.bus):
                self.bus.publish("user.text", text=line, source="keyboard")


def _read_line() -> str | None:
    try:
        return input()
    except EOFError:
        return None
