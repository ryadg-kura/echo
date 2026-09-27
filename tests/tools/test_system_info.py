from __future__ import annotations

import re
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from echo.tools import system_info
from echo.tools.base import ToolContext
from echo.tools.system_info import SystemInfo


async def test_time_and_date(make_ctx: Callable[..., ToolContext]) -> None:
    assert re.fullmatch(r"It is \d\d:\d\d\.", await SystemInfo().run(make_ctx(), what="time"))
    assert (await SystemInfo().run(make_ctx(), what="date")).startswith("Today is ")


async def test_cpu_and_memory(monkeypatch: pytest.MonkeyPatch, make_ctx: Callable[..., ToolContext]) -> None:
    monkeypatch.setattr(system_info.psutil, "cpu_percent", lambda interval: 42.0)
    monkeypatch.setattr(system_info.psutil, "cpu_count", lambda: 8)
    monkeypatch.setattr(system_info.psutil, "virtual_memory", lambda: SimpleNamespace(percent=61.2))
    assert await SystemInfo().run(make_ctx(), what="cpu") == "CPU usage is 42% across 8 cores."
    assert await SystemInfo().run(make_ctx(), what="memory") == "Memory usage is 61%."


@pytest.mark.parametrize(("battery", "expected"), [
    (None, "No battery detected."),
    (SimpleNamespace(percent=80.4, power_plugged=True), "Battery is at 80% and charging."),
    (SimpleNamespace(percent=15, power_plugged=False), "Battery is at 15% and on battery."),
])
async def test_battery(monkeypatch: pytest.MonkeyPatch, make_ctx: Callable[..., ToolContext],
                       battery: object, expected: str) -> None:
    monkeypatch.setattr(system_info.psutil, "sensors_battery", lambda: battery)
    assert await SystemInfo().run(make_ctx(), what="battery") == expected


async def test_all_sections(monkeypatch: pytest.MonkeyPatch, make_ctx: Callable[..., ToolContext]) -> None:
    monkeypatch.setattr(system_info.psutil, "cpu_percent", lambda interval: 1.0)
    result = await SystemInfo().run(make_ctx(), what="everything")  # unknown -> all
    for fragment in ("It is", "Today is", "CPU usage", "Memory usage"):
        assert fragment in result
