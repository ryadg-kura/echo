"""Pure logic behind the voice, gesture and mood modules."""

from __future__ import annotations

import numpy as np
import pytest

from echo.audio.vad import UtteranceSegmenter, rms, strip_wake_word
from echo.core.config import MoodConfig
from echo.core.events import Event, EventBus
from echo.vision.gestures import Gesture, GestureDebouncer, classify_hand
from echo.vision.mood import MoodModule, MoodReading, PlaceholderMoodDetector, load_detector

# --- gestures -------------------------------------------------------------------------------

WRIST = (0.5, 0.9)


def hand(fingers_extended: bool, thumb: str) -> list[tuple[float, float]]:
    """Synthetic upright hand: wrist at bottom, fingers pointing up (y decreases upward)."""
    pts = [(0.0, 0.0)] * 21
    pts[0] = WRIST
    xs = [0.40, 0.47, 0.54, 0.61]  # index, middle, ring, pinky columns
    for i, x in enumerate(xs):
        mcp, pip, dip, tip = 5 + 4 * i, 6 + 4 * i, 7 + 4 * i, 8 + 4 * i
        pts[mcp] = (x, 0.70)
        if fingers_extended:
            pts[pip], pts[dip], pts[tip] = (x, 0.60), (x, 0.52), (x, 0.45)
        else:  # curled back toward the palm
            pts[pip], pts[dip], pts[tip] = (x, 0.62), (x, 0.66), (x, 0.72)
    pts[1] = (0.40, 0.85)
    pts[2] = (0.35, 0.78)
    if thumb == "up":
        pts[3], pts[4] = (0.33, 0.62), (0.32, 0.50)
    elif thumb == "out":
        pts[3], pts[4] = (0.27, 0.74), (0.18, 0.72)
    else:  # tucked over the fingers
        pts[3], pts[4] = (0.40, 0.72), (0.45, 0.68)
    return pts


@pytest.mark.parametrize(("fingers", "thumb", "expected"), [
    (True, "out", Gesture.OPEN_PALM),
    (True, "up", Gesture.OPEN_PALM),
    (False, "tucked", Gesture.FIST),
    (False, "up", Gesture.THUMBS_UP),
    (False, "out", None),
])
def test_classify_hand(fingers: bool, thumb: str, expected: Gesture | None) -> None:
    assert classify_hand(hand(fingers, thumb)) is expected


def test_classify_rejects_bad_input() -> None:
    assert classify_hand([(0.0, 0.0)] * 5) is None
    assert classify_hand([(0.5, 0.5)] * 21) is None


def test_debouncer_needs_stable_frames_and_cooldown() -> None:
    d = GestureDebouncer(stable_frames=3, cooldown_s=1.0)
    palm = Gesture.OPEN_PALM
    assert [d.update(palm, 0.0), d.update(palm, 0.1), d.update(palm, 0.2)] == [None, None, palm]
    assert d.update(palm, 0.3) is None  # holding does not repeat
    assert [d.update(Gesture.FIST, t) for t in (0.4, 0.5, 0.6)] == [None, None, None]  # cooldown
    assert d.update(Gesture.FIST, 1.3) is Gesture.FIST  # held through the cooldown: fires once
    assert d.update(Gesture.FIST, 1.4) is None
    assert d.update(None, 2.0) is None
    assert [d.update(Gesture.THUMBS_UP, t) for t in (3.0, 3.1, 3.2)] == [None, None, Gesture.THUMBS_UP]


# --- voice activity -------------------------------------------------------------------------

SR = 1000
SILENCE = np.zeros(100, dtype=np.float32)
SPEECH = np.full(100, 0.1, dtype=np.float32)


def test_rms() -> None:
    assert rms(np.array([], dtype=np.float32)) == 0.0
    assert rms(SPEECH) == pytest.approx(0.1)


def test_segmenter_returns_utterance_after_silence() -> None:
    seg = UtteranceSegmenter(SR, threshold=0.05, silence_s=0.3, max_s=5, pre_roll_s=0.2, min_speech_s=0.2)
    outputs = [seg.feed(c) for c in [SILENCE] * 5 + [SPEECH] * 3 + [SILENCE] * 3]
    assert all(o is None for o in outputs[:-1])
    utterance = outputs[-1]
    assert utterance is not None
    assert len(utterance) == 200 + 300 + 300  # pre-roll + speech + trailing silence
    assert not seg.started


def test_segmenter_ignores_short_clicks_and_caps_length() -> None:
    seg = UtteranceSegmenter(SR, threshold=0.05, silence_s=0.2, max_s=5, min_speech_s=0.3)
    results = [seg.feed(c) for c in [SPEECH, SILENCE, SILENCE]]
    assert results == [None, None, None]  # 0.1 s click is not speech

    capped = UtteranceSegmenter(SR, threshold=0.05, silence_s=1, max_s=0.5, pre_roll_s=0)
    results = [capped.feed(SPEECH) for _ in range(5)]
    assert results[-1] is not None and len(results[-1]) == 500


def test_segmenter_flush() -> None:
    seg = UtteranceSegmenter(SR, threshold=0.05, min_speech_s=0.1, pre_roll_s=0)
    assert seg.flush() is None
    seg.feed(SPEECH)
    seg.feed(SPEECH)
    flushed = seg.flush()
    assert flushed is not None and len(flushed) == 200


@pytest.mark.parametrize(("text", "expected"), [
    ("Hey Echo, what time is it?", "what time is it"),
    ("echo open firefox", "open firefox"),
    ("Echo.", ""),
    ("hey echo", ""),
    ("echoes of the past", None),
    ("what time is it", None),
])
def test_strip_wake_word(text: str, expected: str | None) -> None:
    assert strip_wake_word(text, ["echo", "hey echo"]) == expected


# --- mood -----------------------------------------------------------------------------------

class HappyDetector:
    def detect(self, frame: np.ndarray) -> MoodReading | None:
        return MoodReading("happy", 0.9)


def test_load_detector() -> None:
    assert isinstance(load_detector("placeholder"), PlaceholderMoodDetector)
    assert isinstance(load_detector("tests.test_perception:HappyDetector"), HappyDetector)
    with pytest.raises(ValueError):
        load_detector("no_colon_here")
    with pytest.raises(TypeError):
        load_detector("pathlib:PurePath")


async def test_mood_module_publishes_changes_only() -> None:
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe("mood", seen.append)
    module = MoodModule(bus, MoodConfig(min_confidence=0.6), camera=None, detector=HappyDetector())  # type: ignore[arg-type]
    assert module.update(MoodReading("happy", 0.9))
    assert not module.update(MoodReading("happy", 0.95))  # unchanged
    assert not module.update(MoodReading("sad", 0.3))  # not confident enough
    assert module.update(MoodReading("sad", 0.8))
    await bus.drain()
    assert [e.get("emotion") for e in seen] == ["happy", "sad"]
