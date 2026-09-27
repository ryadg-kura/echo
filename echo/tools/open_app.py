"""Launch desktop applications (Linux, macOS, Windows)."""

from __future__ import annotations

import configparser
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from echo.tools.base import Tool, ToolContext

DEFAULT_ALIASES: dict[str, str] = {
    "browser": "firefox",
    "web browser": "firefox",
    "terminal": "gnome-terminal",
    "calculator": "gnome-calculator",
    "files": "nautilus",
    "file manager": "nautilus",
    "text editor": "gedit",
    "vs code": "code",
    "vscode": "code",
}

_SAFE_NAME = re.compile(r"^[\w .+-]{1,64}$")

DESKTOP_DIRS = (
    Path("/usr/share/applications"),
    Path("/usr/local/share/applications"),
    Path("/var/lib/flatpak/exports/share/applications"),
    Path.home() / ".local/share/applications",
    Path.home() / ".local/share/flatpak/exports/share/applications",
)


def find_desktop_entry(name: str, dirs: Iterable[Path] = DESKTOP_DIRS) -> str | None:
    """Return the desktop-file id whose ``Name=`` or file name matches ``name``."""
    wanted = name.lower()
    for directory in dirs:
        if not directory.is_dir():
            continue
        for entry in sorted(directory.glob("*.desktop")):
            if entry.stem.lower() == wanted or entry.stem.lower().endswith("." + wanted):
                return entry.name
            parser = configparser.ConfigParser(interpolation=None, strict=False)
            try:
                parser.read(entry, encoding="utf-8")
                display = parser.get("Desktop Entry", "Name", fallback="")
            except (configparser.Error, UnicodeDecodeError):
                continue
            if display.lower() == wanted:
                return entry.name
    return None


def resolve_command(
    app: str,
    aliases: dict[str, str] | None = None,
    system: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    desktop_lookup: Callable[[str], str | None] = find_desktop_entry,
) -> list[str] | None:
    """Turn a spoken app name into an argv list, or ``None`` if it can't be resolved safely."""
    name = app.strip().lower().removesuffix(".").strip()
    name = {**DEFAULT_ALIASES, **(aliases or {})}.get(name, name)
    if not _SAFE_NAME.match(name):
        return None
    system = system or platform.system()
    if system == "Darwin":
        return ["open", "-a", name]
    if system == "Windows":
        return ["cmd", "/c", "start", "", name]
    for candidate in dict.fromkeys([name, name.replace(" ", "-"), name.replace(" ", "")]):
        if path := which(candidate):
            return [path]
    if (desktop_id := desktop_lookup(name)) and which("gtk-launch"):
        return ["gtk-launch", desktop_id.removesuffix(".desktop")]
    return None


def launch(command: list[str]) -> None:
    subprocess.Popen(  # noqa: S603 - argv list, no shell, name validated above
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name != "nt",
    )


class OpenApp(Tool):
    name = "open_app"
    description = "Open (launch) a desktop application by name, e.g. firefox, calculator, terminal."
    parameters = {
        "type": "object",
        "properties": {"app": {"type": "string", "description": "Application name."}},
        "required": ["app"],
    }
    requires_confirmation = True

    def confirmation_prompt(self, app: str = "", **_: Any) -> str:
        return f"Should I open {app}?"

    async def run(self, ctx: ToolContext, app: str, **_: Any) -> str:
        command = resolve_command(app, ctx.config.tools.app_aliases)
        if command is None:
            return f"I couldn't find an application called {app}."
        launch(command)
        return f"Opening {app}."
