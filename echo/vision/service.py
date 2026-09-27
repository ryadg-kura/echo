"""Scene description and OCR on top of the camera and the local vision model."""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Protocol

from echo.core.config import EchoConfig
from echo.llm.errors import LLMError
from echo.llm.ollama_client import ChatBackend
from echo.vision.camera import Frame, encode_jpeg_base64

log = logging.getLogger(__name__)

DESCRIBE_PROMPT = "Describe what you see in this image in one or two short sentences."
OCR_PROMPT = (
    "Transcribe all readable text in this image exactly as written. "
    "If there is no readable text, answer: NO TEXT."
)


class FrameSource(Protocol):
    @property
    def available(self) -> bool: ...

    def latest_frame(self) -> Frame | None: ...


def tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


def tesseract_ocr(frame: Frame) -> str:
    import cv2
    import pytesseract

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return str(pytesseract.image_to_string(gray)).strip()


class VisionService:
    def __init__(self, camera: FrameSource, llm: ChatBackend, config: EchoConfig) -> None:
        self.camera = camera
        self.llm = llm
        self.config = config

    def _frame(self) -> Frame | None:
        if not self.camera.available:
            return None
        return self.camera.latest_frame()

    async def describe(self, question: str | None = None) -> str:
        frame = self._frame()
        if frame is None:
            return "My camera isn't available right now."
        image = await asyncio.to_thread(encode_jpeg_base64, frame)
        prompt = f"{question.strip()} Answer briefly." if question else DESCRIBE_PROMPT
        try:
            answer = await self.llm.generate(self.config.llm.vision_model, prompt, images=[image])
        except LLMError as exc:
            return f"I can't use my vision model: {exc}."
        return answer or "I couldn't make out anything."

    async def read_text(self) -> str:
        frame = self._frame()
        if frame is None:
            return "My camera isn't available right now."
        engine = self.config.vision.ocr_engine
        if engine == "tesseract" or (engine == "auto" and tesseract_available()):
            text = await asyncio.to_thread(tesseract_ocr, frame)
        else:
            image = await asyncio.to_thread(encode_jpeg_base64, frame)
            try:
                text = await self.llm.generate(
                    self.config.llm.vision_model, OCR_PROMPT, images=[image]
                )
            except LLMError as exc:
                return f"I can't read text without OCR: {exc}."
        text = " ".join(text.split())
        if not text or text.upper().startswith("NO TEXT"):
            return "I don't see any readable text. Try holding it closer and steady."
        return f"It says: {text}"
