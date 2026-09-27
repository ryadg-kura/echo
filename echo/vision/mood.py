"""Facial mood detection behind a pluggable interface.

The default detector is a placeholder that always reports "neutral". To plug in MoodLens
(or any model), implement ``MoodDetector`` and point ``mood.provider`` in config.yaml to
``"your_package.module:YourDetector"``.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from echo.core.config import MoodConfig
from echo.core.events import EventBus
from echo.core.module import Module, ModuleState, ModuleUnavailable
from echo.vision.camera import CameraModule, Frame

log = logging.getLogger(__name__)

EMOTIONS = ("neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted")


@dataclass(frozen=True)
class MoodReading:
    emotion: str
    confidence: float


@runtime_checkable
class MoodDetector(Protocol):
    def detect(self, frame: Frame) -> MoodReading | None:
        """Return the dominant emotion of the main face, or ``None`` if no face is visible."""
        ...


class PlaceholderMoodDetector:
    """Stand-in until MoodLens is plugged in."""

    def detect(self, frame: Frame) -> MoodReading | None:
        return MoodReading("neutral", 1.0)


def load_detector(provider: str) -> MoodDetector:
    if provider in ("", "placeholder"):
        return PlaceholderMoodDetector()
    module_name, _, class_name = provider.partition(":")
    if not class_name:
        raise ValueError(f"mood.provider must look like 'package.module:ClassName', got {provider!r}")
    detector = getattr(importlib.import_module(module_name), class_name)()
    if not isinstance(detector, MoodDetector):
        raise TypeError(f"{provider} does not implement detect(frame) -> MoodReading | None")
    return detector


class MoodModule(Module):
    name = "mood"
    depends_on = ("camera",)

    def __init__(
        self,
        bus: EventBus,
        config: MoodConfig,
        camera: CameraModule,
        detector: MoodDetector | None = None,
    ) -> None:
        super().__init__(bus, enabled=config.enabled)
        self.config = config
        self.camera = camera
        self.detector = detector
        self.current = MoodReading("neutral", 0.0)

    async def start(self) -> None:
        if not self.camera.available:
            raise ModuleUnavailable("camera unavailable (tone stays neutral)")
        if self.detector is None:
            try:
                self.detector = await asyncio.to_thread(load_detector, self.config.provider)
            except Exception as exc:
                raise ModuleUnavailable(f"mood provider failed: {exc}") from exc
        label = "placeholder" if isinstance(self.detector, PlaceholderMoodDetector) else "active"
        self.set_status(ModuleState.READY, f"neutral ({label})")
        self.spawn(self._loop())

    async def _loop(self) -> None:
        assert self.detector is not None
        while True:
            frame = self.camera.latest_frame()
            if frame is not None:
                reading = await asyncio.to_thread(self.detector.detect, frame)
                if reading:
                    self.update(reading)
            await asyncio.sleep(self.config.interval_s)

    def update(self, reading: MoodReading) -> bool:
        """Publish a ``mood`` event when the confident emotion changes."""
        if reading.confidence < self.config.min_confidence or reading.emotion == self.current.emotion:
            return False
        self.current = reading
        self.set_status(ModuleState.READY, f"{reading.emotion} ({reading.confidence:.0%})")
        self.bus.publish("mood", emotion=reading.emotion, confidence=reading.confidence)
        return True
