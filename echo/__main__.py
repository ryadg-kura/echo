"""Entry point: ``python -m echo``."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import nullcontext

# Quiet MediaPipe / TensorFlow Lite native logging before they are imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from echo import __version__
from echo.app import run, use_dashboard
from echo.core.config import ConfigError, load_config
from echo.core.logging_setup import redirect_native_stderr, setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="echo", description="Echo — local-first multimodal assistant")
    parser.add_argument("-c", "--config", help="path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--no-ui", action="store_true", help="plain line-based terminal output")
    parser.add_argument("--text-only", action="store_true", help="disable microphone, speech and camera")
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    parser.add_argument("--version", action="version", version=f"echo {__version__}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.text_only:
        config.audio.enabled = config.tts.enabled = False
        config.vision.enabled = config.gestures.enabled = config.mood.enabled = False
    if args.no_ui:
        config.ui.enabled = False
    if args.debug:
        config.logging.level = "DEBUG"
    dashboard = use_dashboard(config)
    setup_logging(config.logging, console=not dashboard)
    native_output = redirect_native_stderr(config.logging) if dashboard else nullcontext()
    try:
        with native_output:
            asyncio.run(run(config, dashboard))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
