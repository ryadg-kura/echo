"""Logging: always to a rotating file; to the console only in plain mode."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler

from echo.core.config import LoggingConfig, expand_path

NOISY = ("httpx", "httpcore", "faster_whisper", "urllib3", "primp", "ddgs", "absl")


def setup_logging(config: LoggingConfig, console: bool) -> None:
    level = getattr(logging, config.level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    log_file = expand_path(config.file)
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        handlers.append(file_handler)
    except OSError:
        pass
    if console:
        rich_handler = RichHandler(level=logging.WARNING, show_path=False, markup=False)
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(rich_handler)
    logging.basicConfig(level=level, handlers=handlers, force=True)
    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)


@contextlib.contextmanager
def redirect_native_stderr(config: LoggingConfig) -> Iterator[None]:
    """Send C/C++ library output (MediaPipe, TFLite, ALSA) to a file.

    Native code writes straight to file descriptor 2, which would scribble over the
    full-screen dashboard. Python logging is unaffected (it goes to the log file).
    """
    target = expand_path(config.file).with_name("native.log")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.flush()
        saved = os.dup(2)
        with target.open("ab") as fh:
            os.dup2(fh.fileno(), 2)
    except OSError:
        yield
        return
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
