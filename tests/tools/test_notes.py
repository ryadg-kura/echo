from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from echo.core.config import expand_path
from echo.tools.base import ToolContext
from echo.tools.notes import Notes, slugify


def test_slugify() -> None:
    assert slugify("Groceries: Milk & Eggs!") == "groceries-milk-eggs"
    assert slugify("???").startswith("note-")


async def test_create_read_search_list(make_ctx: Callable[..., ToolContext]) -> None:
    ctx = make_ctx()
    notes = Notes()
    assert await notes.run(ctx, action="list") == "You don't have any notes yet."
    assert await notes.run(ctx, action="create", title="Groceries", content="milk and eggs") == (
        "Saved note 'Groceries'."
    )
    await notes.run(ctx, action="create", title="Groceries", content="bread")
    await notes.run(ctx, action="create", content="call the dentist on monday")

    directory = expand_path(ctx.config.tools.notes_dir)
    files = sorted(p.name for p in directory.glob("*.md"))
    assert files == ["call-the-dentist-on-monday.md", "groceries-2.md", "groceries.md"]
    assert (directory / "groceries.md").read_text().startswith("# Groceries\n")

    assert await notes.run(ctx, action="read", title="groceries") == "Groceries: milk and eggs"
    assert "couldn't find" in await notes.run(ctx, action="read", title="taxes")

    found = await notes.run(ctx, action="search", query="DENTIST")
    assert found.startswith("Found 1 note(s): call the dentist on monday")
    assert "No notes mention" in await notes.run(ctx, action="search", query="zebra")
    assert "Groceries" in await notes.run(ctx, action="list")


async def test_invalid_usage(make_ctx: Callable[..., ToolContext], tmp_path: Path) -> None:
    ctx = make_ctx()
    notes = Notes()
    assert await notes.run(ctx, action="create") == "What should the note say?"
    assert await notes.run(ctx, action="read") == "Which note should I read?"
    assert "Unknown notes action" in await notes.run(ctx, action="delete")


async def test_titles_cannot_escape_notes_dir(make_ctx: Callable[..., ToolContext]) -> None:
    ctx = make_ctx()
    await Notes().run(ctx, action="create", title="../../etc/passwd", content="x")
    directory = expand_path(ctx.config.tools.notes_dir)
    assert [p.name for p in directory.iterdir()] == ["etc-passwd.md"]
