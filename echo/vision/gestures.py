"""MediaPipe hand tracking → open palm / fist / thumbs up events."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from echo.core.config import GestureConfig, expand_path
from echo.core.events import EventBus
from echo.core.module import Module, ModuleState, ModuleUnavailable
from echo.vision.camera import CameraModule, Frame

log = logging.getLogger(__name__)

Point = tuple[float, float]

WRIST = 0
THUMB_MCP, THUMB_TIP = 2, 4
INDEX_MCP, MIDDLE_MCP = 5, 9
# (pip, tip) for index, middle, ring, pinky
FINGERS = ((6, 8), (10, 12), (14, 16), (18, 20))


class Gesture(StrEnum):
    OPEN_PALM = "open_palm"
    FIST = "fist"
    THUMBS_UP = "thumbs_up"


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def classify_hand(points: Sequence[Point]) -> Gesture | None:
    """Classify 21 MediaPipe hand landmarks (normalized x, y; y grows downward)."""
    if len(points) != 21:
        return None
    wrist = points[WRIST]
    size = _dist(wrist, points[MIDDLE_MCP])
    if size < 1e-6:
        return None
    extended = [_dist(wrist, points[tip]) > 1.2 * _dist(wrist, points[pip]) for pip, tip in FINGERS]
    folded = [_dist(wrist, points[tip]) < 1.05 * _dist(wrist, points[pip]) for pip, tip in FINGERS]
    thumb_out = _dist(points[THUMB_TIP], points[INDEX_MCP]) > 0.7 * size
    thumb_up = thumb_out and (points[THUMB_MCP][1] - points[THUMB_TIP][1]) > 0.5 * size

    if all(extended):
        return Gesture.OPEN_PALM
    if all(folded):
        if thumb_up:
            return Gesture.THUMBS_UP
        if not thumb_out:
            return Gesture.FIST
    return None


class GestureDebouncer:
    """Emit a gesture once it has been stable for N frames, with a global cooldown."""

    def __init__(self, stable_frames: int = 4, cooldown_s: float = 1.5) -> None:
        self.stable_frames = stable_frames
        self.cooldown_s = cooldown_s
        self._candidate: Gesture | None = None
        self._count = 0
        self._fired = False  # already emitted for the current hold
        self._last_emit = -math.inf

    def update(self, gesture: Gesture | None, now: float) -> Gesture | None:
        if gesture != self._candidate:
            self._candidate, self._count, self._fired = gesture, 0, False
        self._count += 1
        if (
            gesture is not None
            and not self._fired
            and self._count >= self.stable_frames
            and now - self._last_emit >= self.cooldown_s
        ):
            self._fired = True
            self._last_emit = now
            return gesture
        return None


def ensure_model(path: Path, url: str) -> Path:
    if path.exists():
        return path
    import httpx

    log.info("downloading hand landmark model to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    tmp = path.with_suffix(".part")
    tmp.write_bytes(response.content)
    tmp.replace(path)
    return path


class HandTracker:
    """Thin wrapper around the MediaPipe Tasks HandLandmarker."""

    def __init__(self, model_path: Path) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def landmarks(self, frame: Frame) -> list[Point] | None:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.hand_landmarks:
            return None
        return [(lm.x, lm.y) for lm in result.hand_landmarks[0]]

    def close(self) -> None:
        self._landmarker.close()


class GestureModule(Module):
    name = "gestures"
    depends_on = ("camera",)

    def __init__(self, bus: EventBus, config: GestureConfig, camera: CameraModule) -> None:
        super().__init__(bus, enabled=config.enabled)
        self.config = config
        self.camera = camera
        self.debouncer = GestureDebouncer(config.stable_frames, config.cooldown_s)
        self._tracker: Any = None

    async def start(self) -> None:
        if not self.camera.available:
            raise ModuleUnavailable("camera unavailable")
        try:
            import mediapipe  # noqa: F401
        except ImportError as exc:
            raise ModuleUnavailable("mediapipe not installed") from exc
        self.set_status(ModuleState.STARTING, "loading hand model")
        try:
            model = await asyncio.to_thread(
                ensure_model, expand_path(self.config.model_path), self.config.model_url
            )
            self._tracker = await asyncio.to_thread(HandTracker, model)
        except Exception as exc:
            raise ModuleUnavailable(f"hand model unavailable: {exc}") from exc
        self.set_status(ModuleState.READY, "show ✋ ✊ 👍")
        self.spawn(self._loop())

    async def _loop(self) -> None:
        interval = 1.0 / max(self.config.fps, 1.0)
        while True:
            await asyncio.sleep(interval)
            frame = self.camera.latest_frame()
            if frame is None:
                continue
            points = await asyncio.to_thread(self._tracker.landmarks, frame)
            gesture = classify_hand(points) if points else None
            if emitted := self.debouncer.update(gesture, time.monotonic()):
                self.set_status(ModuleState.ACTIVE, f"last: {emitted.value}")
                self.bus.publish("gesture", name=emitted.value)

    async def stop(self) -> None:
        await super().stop()
        if self._tracker is not None:
            self._tracker.close()
