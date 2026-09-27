"""Summarize a local PDF with the local LLM."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from echo.core.config import expand_path
from echo.llm.errors import LLMError
from echo.tools.base import Tool, ToolContext

CHUNK_CHARS = 4000
SEARCH_DIRS = (Path.home() / "Documents", Path.home() / "Downloads", Path.home() / "Desktop")

SUMMARY_SYSTEM = (
    "You summarize documents for a voice assistant. Write plain text, no markdown, "
    "at most four sentences."
)


def extract_text(path: Path, max_chars: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    total = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return re.sub(r"\s+", " ", " ".join(parts)).strip()[:max_chars]


def locate_pdf(raw: str, search_dirs: tuple[Path, ...] = SEARCH_DIRS) -> Path | None:
    path = expand_path(raw.strip().strip("'\""))
    if path.suffix.lower() != ".pdf":
        path = path.with_name(path.name + ".pdf")
    if path.is_file():
        return path
    if not path.is_absolute():
        for directory in search_dirs:
            if (candidate := directory / path).is_file():
                return candidate
    return None


def chunk(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


class SummarizePdf(Tool):
    name = "summarize_pdf"
    description = "Summarize a PDF file stored on this computer, given its path or file name."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path or file name of the PDF."},
            "focus": {"type": "string", "description": "Optional aspect to focus on."},
        },
        "required": ["path"],
    }

    async def run(self, ctx: ToolContext, path: str, focus: str | None = None, **_: Any) -> str:
        pdf = locate_pdf(path)
        if pdf is None:
            return f"I couldn't find the PDF '{path}'."
        text = await asyncio.to_thread(extract_text, pdf, ctx.config.tools.pdf_max_chars)
        if not text:
            return f"{pdf.name} has no extractable text (it may be a scanned document)."
        focus_line = f" Focus on: {focus}." if focus else ""
        try:
            partials = [
                await ctx.ask_llm(f"Summarize this excerpt.{focus_line}\n\n{part}", system=SUMMARY_SYSTEM)
                for part in chunk(text, CHUNK_CHARS)
            ]
            if len(partials) == 1:
                return f"{pdf.name}: {partials[0]}"
            combined = "\n".join(partials)
            summary = await ctx.ask_llm(
                f"Combine these partial summaries into one.{focus_line}\n\n{combined}",
                system=SUMMARY_SYSTEM,
            )
            return f"{pdf.name}: {summary}"
        except LLMError:
            preview = " ".join(re.split(r"(?<=[.!?])\s+", text)[:3])
            return f"The LLM is offline, so here is the beginning of {pdf.name}: {preview}"
