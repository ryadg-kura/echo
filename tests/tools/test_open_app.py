from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from echo.tools import open_app
from echo.tools.base import ToolContext
from echo.tools.open_app import OpenApp, find_desktop_entry, resolve_command


def fake_which(available: set[str]) -> Callable[[str], str | None]:
    return lambda name: f"/usr/bin/{name}" if name in available else None


def no_desktop(_: str) -> None:
    return None


def test_resolves_binary_and_alias() -> None:
    which = fake_which({"firefox", "gnome-calculator"})
    assert resolve_command("Firefox", system="Linux", which=which, desktop_lookup=no_desktop) == ["/usr/bin/firefox"]
    assert resolve_command("calculator", system="Linux", which=which, desktop_lookup=no_desktop) == [
        "/usr/bin/gnome-calculator"
    ]
    assert resolve_command("browser", {"browser": "gnome-calculator"}, "Linux", which, no_desktop) == [
        "/usr/bin/gnome-calculator"
    ]


def test_multiword_names_and_desktop_entries() -> None:
    which = fake_which({"visual-studio", "gtk-launch"})
    assert resolve_command("visual studio", system="Linux", which=which, desktop_lookup=no_desktop) == [
        "/usr/bin/visual-studio"
    ]
    lookup = {"spotify": "com.spotify.Client.desktop"}.get
    assert resolve_command("Spotify", system="Linux", which=which, desktop_lookup=lookup) == [
        "gtk-launch", "com.spotify.Client"
    ]


def test_other_platforms() -> None:
    assert resolve_command("Safari", system="Darwin") == ["open", "-a", "safari"]
    assert resolve_command("notepad", system="Windows") == ["cmd", "/c", "start", "", "notepad"]


@pytest.mark.parametrize("evil", ["rm -rf /; ls", "$(reboot)", "a|b", "../../bin/sh", "x" * 100])
def test_rejects_unsafe_names(evil: str) -> None:
    assert resolve_command(evil, system="Linux", which=lambda n: "/bin/x") is None


def test_find_desktop_entry(tmp_path: Path) -> None:
    (tmp_path / "org.gnome.Calculator.desktop").write_text("[Desktop Entry]\nName=Calculator\n")
    (tmp_path / "broken.desktop").write_text("not an ini file [[[")
    assert find_desktop_entry("calculator", [tmp_path]) == "org.gnome.Calculator.desktop"
    assert find_desktop_entry("nothing", [tmp_path, tmp_path / "missing"]) is None


async def test_run_launches(monkeypatch: pytest.MonkeyPatch, make_ctx: Callable[..., ToolContext]) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(open_app, "resolve_command", lambda app, aliases: ["/usr/bin/" + app])
    monkeypatch.setattr(open_app, "launch", launched.append)
    tool = OpenApp()
    assert await tool.run(make_ctx(), app="gedit") == "Opening gedit."
    assert launched == [["/usr/bin/gedit"]]
    assert tool.requires_confirmation
    assert tool.confirmation_prompt(app="gedit") == "Should I open gedit?"


async def test_run_unknown_app(monkeypatch: pytest.MonkeyPatch, make_ctx: Callable[..., ToolContext]) -> None:
    monkeypatch.setattr(open_app, "resolve_command", lambda app, aliases: None)
    assert "couldn't find" in await OpenApp().run(make_ctx(), app="nope")
