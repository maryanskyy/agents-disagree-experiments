"""Prevent system sleep during long experiments (macOS caffeinate wrapper)."""

from __future__ import annotations

import atexit
from contextlib import AbstractContextManager
import platform
import subprocess
from typing import Any


class SleepInhibitor(AbstractContextManager["SleepInhibitor"]):
    """Cross-platform context manager; active implementation on macOS."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[Any] | None = None

    def start(self) -> None:
        """Start sleep inhibition process when available."""
        if platform.system() != "Darwin":
            return

        if self._process is None or self._process.poll() is not None:
            self._process = subprocess.Popen(
                ["caffeinate", "-dims"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            atexit.register(self.stop)

    def stop(self) -> None:
        """Terminate sleep inhibition process."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=2)

    def __enter__(self) -> "SleepInhibitor":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()
        return None