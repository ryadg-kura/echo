from __future__ import annotations

from collections.abc import Callable
from typing import Any

from echo.tools.base import Tool, ToolContext, ToolRegistry

EXPECTED = {"open_app", "web_search", "notes", "summarize_pdf", "system_info",
            "describe_scene", "read_text"}


def test_discovers_every_builtin_tool() -> None:
    registry = ToolRegistry.discover()
    assert set(registry.names) == EXPECTED
    for schema in registry.schemas():
        assert schema["type"] == "function"
        assert schema["function"]["description"]
        assert schema["function"]["parameters"]["type"] == "object"


def test_enable_and_disable_filters() -> None:
    assert ToolRegistry.discover(enabled=["notes"]).names == ["notes"]
    assert "web_search" not in ToolRegistry.discover(disabled=["web_search"]).names


class Strict(Tool):
    name = "strict"
    description = "x"
    parameters = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        if kwargs["a"] == "boom":
            raise RuntimeError("exploded")
        return repr(sorted(kwargs))


async def test_execute_validates_and_catches(make_ctx: Callable[..., ToolContext]) -> None:
    registry = ToolRegistry([Strict()])
    ctx = make_ctx()
    assert await registry.execute("strict", {"a": "1", "junk": 2, "b": None}, ctx) == "['a']"
    assert "needs the argument(s): a" in await registry.execute("strict", {}, ctx)
    assert "Error running strict: exploded" == await registry.execute("strict", {"a": "boom"}, ctx)


async def test_config_can_require_confirmation(make_ctx: Callable[..., ToolContext]) -> None:
    registry = ToolRegistry([Strict()])
    ctx = make_ctx()
    ctx.config.tools.confirm = ["strict"]
    prompts: list[str] = []

    async def deny(prompt: str) -> bool:
        prompts.append(prompt)
        return False

    assert (await registry.execute("strict", {"a": "1"}, ctx, deny)).startswith("Cancelled")
    assert prompts == ["Should I run strict?"]


async def test_five_line_tool_example(make_ctx: Callable[..., ToolContext]) -> None:
    """The README promises a new tool in 5 lines — keep that example honest."""
    import random

    class Dice(Tool):
        name, description = "roll_dice", "Roll a six-sided die."

        async def run(self, ctx: ToolContext, **kw: Any) -> str:
            return f"You rolled a {random.randint(1, 6)}."

    result = await ToolRegistry([Dice()]).execute("roll_dice", {}, make_ctx())
    assert result.startswith("You rolled a ")
