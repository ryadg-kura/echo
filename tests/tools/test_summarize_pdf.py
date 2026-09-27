from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from echo.tools import summarize_pdf
from echo.tools.base import ToolContext
from echo.tools.summarize_pdf import SummarizePdf, chunk, extract_text, locate_pdf
from tests.conftest import FakeLLM


def make_pdf(path: Path, text: str) -> Path:
    """Write a minimal one-page PDF containing ``text`` (no extra dependencies)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(bytes(out))
    return path


def test_extract_and_locate(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "report.pdf", "Quarterly revenue grew by ten percent.")
    assert extract_text(pdf, 1000) == "Quarterly revenue grew by ten percent."
    assert extract_text(pdf, 9) == "Quarterly"
    assert locate_pdf(str(pdf)) == pdf
    assert locate_pdf(str(tmp_path / "report")) == pdf  # extension optional
    assert locate_pdf("report.pdf", search_dirs=(tmp_path,)) == pdf
    assert locate_pdf("missing.pdf", search_dirs=(tmp_path,)) is None


def test_chunk() -> None:
    assert chunk("abcdefg", 3) == ["abc", "def", "g"]


async def test_summarizes_with_llm(tmp_path: Path, make_ctx: Callable[..., ToolContext],
                                   fake_llm: FakeLLM) -> None:
    pdf = make_pdf(tmp_path / "report.pdf", "Quarterly revenue grew by ten percent.")
    fake_llm.generate_reply = "Revenue grew 10%."
    result = await SummarizePdf().run(make_ctx(), path=str(pdf), focus="numbers")
    assert result == "report.pdf: Revenue grew 10%."
    assert "Quarterly revenue" in fake_llm.generate_calls[0]["prompt"]
    assert "Focus on: numbers" in fake_llm.generate_calls[0]["prompt"]


async def test_long_documents_are_map_reduced(tmp_path: Path, make_ctx: Callable[..., ToolContext],
                                              fake_llm: FakeLLM, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    pdf = make_pdf(tmp_path / "long.pdf", "word " * 50)
    monkeypatch.setattr(summarize_pdf, "CHUNK_CHARS", 100)
    await SummarizePdf().run(make_ctx(), path=str(pdf))
    assert len(fake_llm.generate_calls) == 4  # 3 chunks + 1 combine
    assert fake_llm.generate_calls[-1]["prompt"].startswith("Combine")


async def test_offline_and_missing(tmp_path: Path, make_ctx: Callable[..., ToolContext],
                                   fake_llm: FakeLLM) -> None:
    pdf = make_pdf(tmp_path / "a.pdf", "First sentence. Second one. Third. Fourth.")
    fake_llm.offline = True
    result = await SummarizePdf().run(make_ctx(), path=str(pdf))
    assert result == "The LLM is offline, so here is the beginning of a.pdf: First sentence. Second one. Third."
    assert "couldn't find" in await SummarizePdf().run(make_ctx(), path=str(tmp_path / "nope.pdf"))
