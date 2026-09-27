"""'Read this' — OCR the text held in front of the webcam."""

from __future__ import annotations

from typing import Any

from echo.tools.base import Tool, ToolContext


class ReadText(Tool):
    name = "read_text"
    description = "Read aloud the text the user is holding in front of the webcam (OCR)."

    async def run(self, ctx: ToolContext, **_: Any) -> str:
        if ctx.vision is None:
            return "I don't have access to a camera."
        return await ctx.vision.read_text()
