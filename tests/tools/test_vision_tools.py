from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from echo.tools.base import ToolContext
from echo.tools.describe_scene import DescribeScene
from echo.tools.read_text import ReadText
from echo.vision import service
from echo.vision.service import VisionService
from tests.conftest import FakeLLM


class FakeCamera:
    def __init__(self, frame: np.ndarray | None) -> None:
        self.frame = frame

    @property
    def available(self) -> bool:
        return self.frame is not None

    def latest_frame(self) -> np.ndarray | None:
        return self.frame


FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


async def test_describe_uses_vision_model(make_ctx: Callable[..., ToolContext], fake_llm: FakeLLM) -> None:
    ctx = make_ctx()
    ctx.vision = VisionService(FakeCamera(FRAME), fake_llm, ctx.config)
    fake_llm.generate_reply = "A dark room."
    assert await DescribeScene().run(ctx) == "A dark room."
    call = fake_llm.generate_calls[0]
    assert call["model"] == "moondream"
    assert call["images"] and isinstance(call["images"][0], str)

    await DescribeScene().run(ctx, question="Is there a cat?")
    assert fake_llm.generate_calls[1]["prompt"].startswith("Is there a cat?")


async def test_read_text_with_llm_ocr(make_ctx: Callable[..., ToolContext], fake_llm: FakeLLM) -> None:
    ctx = make_ctx()
    ctx.config.vision.ocr_engine = "llm"
    ctx.vision = VisionService(FakeCamera(FRAME), fake_llm, ctx.config)
    fake_llm.generate_reply = "EXIT\nonly"
    assert await ReadText().run(ctx) == "It says: EXIT only"
    fake_llm.generate_reply = "NO TEXT."
    assert "don't see any readable text" in await ReadText().run(ctx)


async def test_read_text_with_tesseract(monkeypatch: pytest.MonkeyPatch,
                                        make_ctx: Callable[..., ToolContext], fake_llm: FakeLLM) -> None:
    ctx = make_ctx()
    ctx.vision = VisionService(FakeCamera(FRAME), fake_llm, ctx.config)
    monkeypatch.setattr(service, "tesseract_available", lambda: True)
    monkeypatch.setattr(service, "tesseract_ocr", lambda frame: "Hello  world")
    assert await ReadText().run(ctx) == "It says: Hello world"
    assert fake_llm.generate_calls == []


async def test_no_camera(make_ctx: Callable[..., ToolContext], fake_llm: FakeLLM) -> None:
    assert await DescribeScene().run(make_ctx()) == "I don't have access to a camera."
    assert await ReadText().run(make_ctx()) == "I don't have access to a camera."
    ctx = make_ctx()
    ctx.vision = VisionService(FakeCamera(None), fake_llm, ctx.config)
    assert await DescribeScene().run(ctx) == "My camera isn't available right now."


async def test_vision_model_offline(make_ctx: Callable[..., ToolContext], fake_llm: FakeLLM) -> None:
    ctx = make_ctx()
    ctx.vision = VisionService(FakeCamera(FRAME), fake_llm, ctx.config)
    fake_llm.offline = True
    assert "can't use my vision model" in await DescribeScene().run(ctx)
