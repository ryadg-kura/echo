"""Time, date, battery, CPU and memory."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import psutil

from echo.tools.base import Tool, ToolContext

_SECTIONS = ("time", "date", "battery", "cpu", "memory")


class SystemInfo(Tool):
    name = "system_info"
    description = "Get the current time, date, battery level, CPU usage or memory usage."
    parameters = {
        "type": "object",
        "properties": {
            "what": {
                "type": "string",
                "enum": ["all", *_SECTIONS],
                "description": "Which information to return (default: all).",
            }
        },
    }

    async def run(self, ctx: ToolContext, what: str = "all", **_: Any) -> str:
        what = what.lower()
        sections = _SECTIONS if what not in _SECTIONS else (what,)
        parts: list[str] = []
        now = datetime.now()
        for section in sections:
            if section == "time":
                parts.append(f"It is {now.strftime('%H:%M')}.")
            elif section == "date":
                parts.append(f"Today is {now.strftime('%A, %B')} {now.day}, {now.year}.")
            elif section == "battery":
                parts.append(_battery())
            elif section == "cpu":
                cpu = await asyncio.to_thread(psutil.cpu_percent, 0.3)
                parts.append(f"CPU usage is {cpu:.0f}% across {psutil.cpu_count()} cores.")
            elif section == "memory":
                mem = psutil.virtual_memory()
                parts.append(f"Memory usage is {mem.percent:.0f}%.")
        return " ".join(parts)


def _battery() -> str:
    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    if battery is None:
        return "No battery detected."
    state = "charging" if battery.power_plugged else "on battery"
    return f"Battery is at {battery.percent:.0f}% and {state}."
