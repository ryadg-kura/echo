"""Typed configuration loaded from ``config.yaml`` (every key is optional)."""

from __future__ import annotations

import dataclasses
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when config.yaml contains unknown keys or malformed values."""


@dataclass
class LLMConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    vision_model: str = "moondream"
    temperature: float = 0.4
    max_tool_steps: int = 4
    context_turns: int = 6
    timeout_s: float = 120.0
    health_check_interval_s: float = 30.0


@dataclass
class AudioConfig:
    enabled: bool = True
    mode: str = "push_to_talk"  # "push_to_talk" | "wake_word"
    wake_words: list[str] = field(default_factory=lambda: ["hey echo", "echo"])
    wake_window_s: float = 8.0
    device: int | None = None
    sample_rate: int = 16000
    silence_threshold: float = 0.012
    silence_duration_s: float = 0.9
    max_utterance_s: float = 15.0
    no_speech_timeout_s: float = 6.0
    stt_model: str = "base.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    language: str | None = "en"


@dataclass
class TTSConfig:
    enabled: bool = True
    engine: str = "piper"  # "piper" | "pyttsx3" | "none"; falls back down this list
    piper_voice: str = "~/.echo/voices/en_US-lessac-medium.onnx"
    rate: int = 175


@dataclass
class VisionConfig:
    enabled: bool = True
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 15
    ocr_engine: str = "auto"  # "auto" | "tesseract" | "llm"


@dataclass
class GestureConfig:
    enabled: bool = True
    model_path: str = "~/.echo/models/hand_landmarker.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )
    fps: float = 10.0
    stable_frames: int = 4
    cooldown_s: float = 1.5


@dataclass
class MoodConfig:
    enabled: bool = True
    provider: str = "placeholder"  # or "package.module:ClassName" (e.g. MoodLens)
    interval_s: float = 3.0
    min_confidence: float = 0.5


@dataclass
class ToolsConfig:
    enabled: list[str] = field(default_factory=list)  # empty = all discovered tools
    disabled: list[str] = field(default_factory=list)
    confirm: list[str] = field(default_factory=lambda: ["open_app"])
    confirm_timeout_s: float = 12.0
    notes_dir: str = "~/.echo/notes"
    app_aliases: dict[str, str] = field(default_factory=dict)
    web_search_results: int = 5
    pdf_max_chars: int = 12000


@dataclass
class UIConfig:
    enabled: bool = True
    refresh_per_second: int = 8
    history: int = 14


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "~/.echo/logs/echo.log"


@dataclass
class EchoConfig:
    name: str = "Echo"
    llm: LLMConfig = field(default_factory=LLMConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    mood: MoodConfig = field(default_factory=MoodConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def expand_path(path: str | os.PathLike[str]) -> Path:
    """Expand ``~`` and environment variables in a configured path."""
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _build(cls: type[T], data: dict[str, Any], where: str) -> T:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown config key(s) in '{where}': {', '.join(sorted(unknown))}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            if not isinstance(value, dict):
                raise ConfigError(f"'{where}.{name}' must be a mapping")
            kwargs[name] = _build(hint, value, f"{where}.{name}")  # type: ignore[arg-type]
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any] | None) -> EchoConfig:
    return _build(EchoConfig, data or {}, "config")


def load_config(path: str | os.PathLike[str] | None = None) -> EchoConfig:
    """Load config from ``path``, ``$ECHO_CONFIG`` or ``./config.yaml``; defaults if none exist."""
    candidate = path or os.environ.get("ECHO_CONFIG") or "config.yaml"
    file = expand_path(candidate)
    if not file.exists():
        if path is not None:
            raise ConfigError(f"config file not found: {file}")
        return EchoConfig()
    with file.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{file} must contain a YAML mapping")
    return config_from_dict(data)
