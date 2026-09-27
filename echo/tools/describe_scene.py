"""'What do you see?' — describe the webcam view with the local vision model."""

from __future__ import annotations

from typing import Any

from echo.tools.base import Tool, ToolContext


class DescribeScene(Tool):
    name = "describe_scene"
    description = "Look through the webcam and describe what is visible, or answer a question about it."
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Optional question about the scene."}
        },
    }

    async def run(self, ctx: ToolContext, question: str | None = None, **_: Any) -> str:
        if ctx.vision is None:
            return "I don't have access to a camera."
        return await ctx.vision.describe(question)
