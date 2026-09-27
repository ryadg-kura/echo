"""Non-blocking keyboard line editor for the live dashboard (POSIX cbreak mode)."""

from __future__ import annotations

import os
import select
import sys
import threading
from collections.abc import Callable
from typing import Any


class KeyboardReader:
    """Reads keys on a thread, keeps an editable buffer, calls ``on_submit`` on Enter."""

    def __init__(self, on_submit: Callable[[str], None], on_eof: Callable[[], None]) -> None:
        self.on_submit = on_submit
        self.on_eof = on_eof
        self.buffer = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved: Any = None

    @staticmethod
    def supported() -> bool:
        return os.name == "posix" and sys.stdin.isatty()

    def start(self) -> None:
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._thread = threading.Thread(target=self._run, name="keyboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._saved is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
            self._saved = None

    def _run(self) -> None:
        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            char = os.read(fd, 1).decode("utf-8", errors="ignore")
            if char in ("\n", "\r"):
                line, self.buffer = self.buffer, ""
                self.on_submit(line)
            elif char in ("\x7f", "\b"):
                self.buffer = self.buffer[:-1]
            elif char == "\x04":  # Ctrl-D
                self.on_eof()
            elif char == "\x1b":  # Escape: clear the line
                self.buffer = ""
            elif char.isprintable():
                self.buffer += char
