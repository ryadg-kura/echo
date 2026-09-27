"""Local text-to-speech: Piper (neural) → pyttsx3 (system voices) → text only."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np

from echo.core.config import TTSConfig, expand_path
from echo.core.events import Event, EventBus
from echo.core.module import Module, ModuleState, ModuleUnavailable

log = logging.getLogger(__name__)

ENGINE_ORDER = ("piper", "pyttsx3")


class Speaker(Protocol):
    def say(self, text: str) -> None:
        """Speak ``text``, blocking until playback ends."""


class PiperSpeaker:
    def __init__(self, voice_path: str) -> None:
        import sounddevice  # noqa: F401  (fail early if PortAudio is missing)
        from piper import PiperVoice

        path = expand_path(voice_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Piper voice not found at {path} "
                f"(python -m piper.download_voices en_US-lessac-medium --download-dir {path.parent})"
            )
        self._voice = PiperVoice.load(str(path))

    def say(self, text: str) -> None:
        import sounddevice as sd

        chunks = list(self._voice.synthesize(text))
        if not chunks:
            return
        audio = np.concatenate([c.audio_int16_array for c in chunks])
        sd.play(audio, chunks[0].sample_rate)
        sd.wait()


class Pyttsx3Speaker:
    def __init__(self, rate: int) -> None:
        import pyttsx3

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)

    def say(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()


def build_speaker(config: TTSConfig) -> tuple[Speaker, str]:
    """Try the configured engine first, then fall back down ``ENGINE_ORDER``."""
    if config.engine == "none":
        raise ModuleUnavailable("disabled in config")
    order = [config.engine, *[e for e in ENGINE_ORDER if e != config.engine]]
    for engine in order:
        try:
            if engine == "piper":
                return PiperSpeaker(config.piper_voice), "piper"
            if engine == "pyttsx3":
                return Pyttsx3Speaker(config.rate), "pyttsx3"
        except Exception as exc:
            log.info("TTS engine %s unavailable: %s", engine, exc)
    raise ModuleUnavailable("no TTS engine — replies are text-only")


class TTSModule(Module):
    name = "tts"

    def __init__(self, bus: EventBus, config: TTSConfig) -> None:
        super().__init__(bus, enabled=config.enabled)
        self.config = config
        # Speech engines (pyttsx3 especially) must stay on the thread that created them.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self._speaker: Speaker | None = None
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self.engine = ""

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._speaker, self.engine = await loop.run_in_executor(
            self._executor, build_speaker, self.config
        )
        self.set_status(ModuleState.READY, self.engine)
        self.bus.subscribe("assistant.reply", self._on_reply)
        self.spawn(self._worker())

    async def _on_reply(self, event: Event) -> None:
        text = str(event.get("text", "")).strip()
        if text:
            await self._queue.put(text)

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            text = await self._queue.get()
            self.set_status(ModuleState.ACTIVE, "speaking")
            self.bus.publish("tts.start")
            try:
                assert self._speaker is not None
                await loop.run_in_executor(self._executor, self._speaker.say, text)
            except Exception as exc:
                log.warning("TTS failed: %s", exc)
            finally:
                self.bus.publish("tts.end")
                self.set_status(ModuleState.READY, self.engine)

    async def stop(self) -> None:
        await super().stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
