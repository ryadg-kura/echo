"""Speech-to-text with faster-whisper (runs fully offline after the first model download)."""

from __future__ import annotations

from typing import Any

import numpy as np

from echo.core.config import AudioConfig


class SpeechToText:
    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._model: Any = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.config.stt_model,
            device=self.config.stt_device,
            compute_type=self.config.stt_compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            raise RuntimeError("SpeechToText.load() must be called first")
        segments, _ = self._model.transcribe(
            audio.astype(np.float32),
            language=self.config.language,
            beam_size=1,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
