"""Voice input: push-to-talk or wake word, transcribed by faster-whisper."""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from echo.audio.microphone import Microphone, probe_microphone
from echo.audio.stt import SpeechToText
from echo.audio.vad import UtteranceSegmenter, strip_wake_word
from echo.core.config import AudioConfig
from echo.core.events import Event, EventBus
from echo.core.module import Module, ModuleState, ModuleUnavailable

log = logging.getLogger(__name__)


class VoiceModule(Module):
    name = "voice"

    def __init__(self, bus: EventBus, config: AudioConfig, stt: SpeechToText | None = None):
        super().__init__(bus, enabled=config.enabled)
        self.config = config
        self.stt = stt or SpeechToText(config)
        self.listening = False
        self._stop_requested = asyncio.Event()
        self._speaking = False  # TTS is playing: ignore the mic so Echo doesn't hear itself
        self._armed_until = 0.0  # wake-word mode: accept a command without the wake word
        self._capture_task: asyncio.Task[None] | None = None

    @property
    def idle_detail(self) -> str:
        if self.config.mode == "wake_word":
            return f"say '{self.config.wake_words[0]}'"
        return "press Enter or ✋ to talk"

    async def start(self) -> None:
        if self.config.mode not in ("push_to_talk", "wake_word"):
            raise ModuleUnavailable(f"unknown audio.mode {self.config.mode!r}")
        device = await asyncio.to_thread(probe_microphone, self.config.device)
        self.set_status(ModuleState.STARTING, f"loading whisper {self.config.stt_model}")
        try:
            await asyncio.to_thread(self.stt.load)
        except ImportError as exc:
            raise ModuleUnavailable("faster-whisper not installed") from exc
        log.info("microphone: %s", device)
        self.bus.subscribe("listen.start", self._on_start)
        self.bus.subscribe("listen.stop", self._on_stop)
        self.bus.subscribe("listen.toggle", self._on_toggle)
        self.bus.subscribe("tts.start", self._on_tts)
        self.bus.subscribe("tts.end", self._on_tts)
        self.set_status(ModuleState.READY, self.idle_detail)
        if self.config.mode == "wake_word":
            self.spawn(self._wake_word_loop())

    # --- event handlers -------------------------------------------------------------------

    async def _on_start(self, _: Event) -> None:
        if self.config.mode == "wake_word":
            self._arm()
        elif not self.listening:
            self._capture_task = self.spawn(self._capture_once())

    async def _on_stop(self, _: Event) -> None:
        self._armed_until = 0.0
        self._stop_requested.set()

    async def _on_toggle(self, event: Event) -> None:
        if self.listening:
            await self._on_stop(event)
        else:
            await self._on_start(event)

    async def _on_tts(self, event: Event) -> None:
        self._speaking = event.topic == "tts.start"

    # --- push to talk ---------------------------------------------------------------------

    def _segmenter(self) -> UtteranceSegmenter:
        c = self.config
        return UtteranceSegmenter(
            c.sample_rate, c.silence_threshold, c.silence_duration_s, c.max_utterance_s
        )

    async def _capture_once(self) -> None:
        self.listening = True
        self._stop_requested.clear()
        self.set_status(ModuleState.ACTIVE, "listening…")
        audio: np.ndarray | None = None
        try:
            segmenter = self._segmenter()
            deadline = time.monotonic() + self.config.no_speech_timeout_s
            async with Microphone(self.config.sample_rate, device=self.config.device) as mic:
                while not self._stop_requested.is_set():
                    try:
                        chunk = await asyncio.wait_for(mic.read(), timeout=0.5)
                    except TimeoutError:
                        continue
                    if (audio := segmenter.feed(chunk)) is not None:
                        break
                    if not segmenter.started and time.monotonic() > deadline:
                        break
                else:
                    audio = segmenter.flush()  # stopped by fist / key: keep what was said
        except Exception as exc:
            log.error("microphone capture failed: %s", exc)
            self.set_status(ModuleState.ERROR, f"mic error: {exc}")
            return
        finally:
            self.listening = False
        await self._transcribe_and_publish(audio)

    async def _transcribe_and_publish(self, audio: np.ndarray | None) -> None:
        if audio is None:
            self.set_status(ModuleState.READY, self.idle_detail)
            return
        self.set_status(ModuleState.BUSY, "transcribing…")
        text = await asyncio.to_thread(self.stt.transcribe, audio)
        self.set_status(ModuleState.READY, self.idle_detail)
        if text:
            self.bus.publish("user.text", text=text, source="voice")

    # --- wake word ------------------------------------------------------------------------

    def _arm(self) -> None:
        self._armed_until = time.monotonic() + self.config.wake_window_s
        self.set_status(ModuleState.ACTIVE, "listening…")

    async def _wake_word_loop(self) -> None:
        segmenter = self._segmenter()
        async with Microphone(self.config.sample_rate, device=self.config.device) as mic:
            while True:
                chunk = await mic.read()
                armed = time.monotonic() < self._armed_until
                if not armed and self.state is ModuleState.ACTIVE:
                    self.set_status(ModuleState.READY, self.idle_detail)
                if self._speaking:
                    segmenter.reset()
                    continue
                audio = segmenter.feed(chunk)
                if audio is None:
                    continue
                text = await asyncio.to_thread(self.stt.transcribe, audio)
                mic.clear()
                command = text if armed else strip_wake_word(text, self.config.wake_words)
                if command is None:
                    continue
                if command:
                    self._armed_until = 0.0
                    self.set_status(ModuleState.READY, self.idle_detail)
                    self.bus.publish("user.text", text=command, source="voice")
                else:
                    self._arm()
                    self.bus.publish("voice.wake")
