"""Webcam capture on a background thread; other modules read the latest frame."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import numpy as np

from echo.core.config import VisionConfig
from echo.core.events import EventBus
from echo.core.module import Module, ModuleState, ModuleUnavailable

log = logging.getLogger(__name__)

Frame = np.ndarray


class CameraModule(Module):
    name = "camera"

    def __init__(self, bus: EventBus, config: VisionConfig) -> None:
        super().__init__(bus, enabled=config.enabled)
        self.config = config
        self._capture: Any = None
        self._frame: Frame | None = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop_ref: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise ModuleUnavailable("opencv not installed") from exc
        capture = await asyncio.to_thread(self._open, cv2)
        if capture is None:
            raise ModuleUnavailable(f"no camera at index {self.config.camera_index}")
        self._capture = capture
        self._loop_ref = asyncio.get_running_loop()
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        self.set_status(ModuleState.READY, f"{self.config.width}x{self.config.height}@{self.config.fps}fps")

    def _open(self, cv2: Any) -> Any:
        capture = cv2.VideoCapture(self.config.camera_index)
        if not capture.isOpened():
            capture.release()
            return None
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        ok, _ = capture.read()
        if not ok:
            capture.release()
            return None
        return capture

    def _loop(self) -> None:
        interval = 1.0 / max(self.config.fps, 1)
        failures = 0
        while self._running.is_set():
            ok, frame = self._capture.read()
            if ok:
                failures = 0
                with self._lock:
                    self._frame = frame
            else:
                failures += 1
                if failures == 30:
                    log.warning("camera stopped delivering frames")
                    if self._loop_ref is not None:
                        self._loop_ref.call_soon_threadsafe(
                            self.set_status, ModuleState.ERROR, "no frames"
                        )
            time.sleep(interval)

    def latest_frame(self) -> Frame | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    async def stop(self) -> None:
        await super().stop()
        self._running.clear()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 2.0)
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def encode_jpeg_base64(frame: Frame, quality: int = 85) -> str:
    import base64

    import cv2

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("could not encode frame")
    return base64.b64encode(buffer.tobytes()).decode("ascii")
