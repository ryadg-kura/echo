"""Create, read, list and search local markdown notes."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from echo.core.config import expand_path
from echo.tools.base import Tool, ToolContext


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60].strip("-")
    return slug or datetime.now().strftime("note-%Y%m%d-%H%M%S")


class NoteStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def create(self, title: str, content: str) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        base = slugify(title)
        path = self.directory / f"{base}.md"
        counter = 2
        while path.exists():
            path = self.directory / f"{base}-{counter}.md"
            counter += 1
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        path.write_text(f"# {title.strip()}\n\n_{stamp}_\n\n{content.strip()}\n", encoding="utf-8")
        return path

    def find(self, title: str) -> Path | None:
        notes = self.all()
        slug = slugify(title)
        for path in notes:
            if path.stem == slug:
                return path
        return next((p for p in notes if slug in p.stem), None)

    def search(self, query: str) -> list[tuple[Path, str]]:
        needle = query.lower()
        hits: list[tuple[Path, str]] = []
        for path in self.all():
            for line in path.read_text(encoding="utf-8").splitlines():
                if needle in line.lower():
                    hits.append((path, line.strip("# ").strip()))
                    break
        return hits

    def all(self) -> list[Path]:
        if not self.directory.exists():
            return []
        return sorted(self.directory.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)


def _title_of(path: Path) -> str:
    first = path.read_text(encoding="utf-8").splitlines()[:1]
    return first[0].lstrip("# ").strip() if first else path.stem


class Notes(Tool):
    name = "notes"
    description = (
        "Manage the user's local markdown notes: create a note, read a note by title, "
        "search notes by keyword, or list recent notes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "read", "search", "list"]},
            "title": {"type": "string", "description": "Note title (create/read)."},
            "content": {"type": "string", "description": "Note body (create)."},
            "query": {"type": "string", "description": "Keyword to look for (search)."},
        },
        "required": ["action"],
    }

    async def run(
        self,
        ctx: ToolContext,
        action: str,
        title: str | None = None,
        content: str | None = None,
        query: str | None = None,
        **_: Any,
    ) -> str:
        store = NoteStore(expand_path(ctx.config.tools.notes_dir))
        return await asyncio.to_thread(self._run, store, action.lower(), title, content, query)

    @staticmethod
    def _run(
        store: NoteStore, action: str, title: str | None, content: str | None, query: str | None
    ) -> str:
        if action == "create":
            if not content and not title:
                return "What should the note say?"
            path = store.create(title or " ".join((content or "").split()[:6]), content or title or "")
            return f"Saved note '{_title_of(path)}'."
        if action == "read":
            if not title:
                return "Which note should I read?"
            found = store.find(title)
            if found is None:
                return f"I couldn't find a note called '{title}'."
            body = found.read_text(encoding="utf-8").split("\n", 1)[-1]
            body = re.sub(r"^\s*_[^_\n]+_\s*$", "", body, flags=re.MULTILINE)  # drop timestamp
            return f"{_title_of(found)}: {' '.join(body.split())}"
        if action == "search":
            if not query:
                return "What should I search your notes for?"
            hits = store.search(query)
            if not hits:
                return f"No notes mention '{query}'."
            matches = "; ".join(f"{_title_of(p)} ({line})" for p, line in hits[:5])
            return f"Found {len(hits)} note(s): {matches}"
        if action == "list":
            notes = store.all()[:10]
            if not notes:
                return "You don't have any notes yet."
            return f"Your recent notes: {', '.join(_title_of(p) for p in notes)}."
        return f"Unknown notes action '{action}'. Use create, read, search or list."
