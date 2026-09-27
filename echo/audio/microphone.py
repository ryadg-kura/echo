"""Async microphone stream built on sounddevice."""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any

import numpy as np

from echo.core.module import ModuleUnavailable


def probe_microphone(device: int | None = None) -> str:
    """Return the input device name or raise ``ModuleUnavailable``."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:  # OSError: PortAudio library missing
        raise ModuleUnavailable(f"audio backend missing ({exc})") from exc
    try:
        info = sd.query_devices(device, kind="input")
    except Exception as exc:
        raise ModuleUnavailable("no microphone found") from exc
    return str(info["name"])


class Microphone:
    """``async with Microphone(...) as mic: chunk = await mic.read()`` (float32 mono)."""

    def __init__(self, sample_rate: int = 16000, block_ms: int = 30, device: int | None = None):
        self.sample_rate = sample_rate
        self.blocksize = int(sample_rate * block_ms / 1000)
        self.device = device
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=500)
        self._stream: Any = None

    async def __aenter__(self) -> Microphone:
        import sounddevice as sd

        loop = asyncio.get_running_loop()

        def callback(indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
            loop.call_soon_threadsafe(self._put, indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
        )
        self._stream.start()
        return self

    def _put(self, chunk: np.ndarray) -> None:
        if self._queue.full():  # consumer is busy (e.g. transcribing): drop oldest audio
            self._queue.get_nowait()
        self._queue.put_nowait(chunk)

    async def read(self) -> np.ndarray:
        return await self._queue.get()

    def clear(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
