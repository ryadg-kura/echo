"""Energy-based utterance segmentation and wake-word matching (pure, easy to test)."""

from __future__ import annotations

import re
from collections import deque

import numpy as np


def rms(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))


class UtteranceSegmenter:
    """Feed audio chunks; get back a complete utterance once the speaker pauses."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.012,
        silence_s: float = 0.9,
        max_s: float = 15.0,
        pre_roll_s: float = 0.3,
        min_speech_s: float = 0.25,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_samples = int(silence_s * sample_rate)
        self.max_samples = int(max_s * sample_rate)
        self.pre_roll_samples = int(pre_roll_s * sample_rate)
        self.min_speech_samples = int(min_speech_s * sample_rate)
        self.reset()

    def reset(self) -> None:
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_len = 0
        self._chunks: list[np.ndarray] = []
        self._length = 0
        self._speech = 0
        self._silence = 0
        self.started = False

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        loud = rms(chunk) >= self.threshold
        if not self.started:
            if loud:
                self.started = True
                self._chunks = [*self._pre_roll, chunk]
                self._length = self._pre_roll_len + len(chunk)
                self._speech = len(chunk)
                return None
            # Keep a little audio before speech starts so the first syllable isn't clipped.
            self._pre_roll.append(chunk)
            self._pre_roll_len += len(chunk)
            while self._pre_roll and self._pre_roll_len > self.pre_roll_samples:
                self._pre_roll_len -= len(self._pre_roll.popleft())
            return None
        self._chunks.append(chunk)
        self._length += len(chunk)
        if loud:
            self._speech += len(chunk)
            self._silence = 0
        else:
            self._silence += len(chunk)
        if self._silence >= self.silence_samples or self._length >= self.max_samples:
            return self.flush()
        return None

    def flush(self) -> np.ndarray | None:
        """End the current utterance now; ``None`` if it was too short to be speech."""
        audio = np.concatenate(self._chunks) if self._chunks and self.started else None
        enough = self._speech >= self.min_speech_samples
        self.reset()
        return audio if audio is not None and enough else None


def strip_wake_word(text: str, wake_words: list[str]) -> str | None:
    """Return the command after a wake word, ``""`` for a bare wake word, ``None`` if absent."""
    normalized = re.sub(r"[^\w' ]", " ", text.lower())
    normalized = " ".join(normalized.split())
    for wake in sorted(wake_words, key=len, reverse=True):
        wake = wake.lower().strip()
        if normalized == wake or normalized.startswith(wake + " "):
            return normalized[len(wake) :].strip()
    return None
