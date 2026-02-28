"""Token manager for GenAI Gateway authentication.

Acquires tokens via configurable command and auto-refreshes
before expiry (~20h lifetime, refreshed at 18h).

Configure via environment variables:
  GENAI_GATEWAY_URL   - base URL of the GenAI gateway
  GENAI_ORG_ID        - organization/project ID for the gateway
  GENAI_TOKEN_CMD     - shell command to obtain a bearer token
"""

from __future__ import annotations

import os
import subprocess
import threading
import time


GATEWAY_BASE_URL = os.environ.get("GENAI_GATEWAY_URL", "")
GATEWAY_ORG_ID = os.environ.get("GENAI_ORG_ID", "")
_TOKEN_CMD = os.environ.get("GENAI_TOKEN_CMD", "echo configure-GENAI_TOKEN_CMD")
TOKEN_REFRESH_SECONDS = 18 * 3600  # refresh well before 20h expiry


class TokenManager:
    """Thread-safe token cache with auto-refresh."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._acquired_at: float = 0.0
        self._lock = threading.Lock()
        self._token_version: int = 0

    def get_token(self) -> str:
        with self._lock:
            if self._token is None or self._needs_refresh():
                self._refresh()
            assert self._token is not None
            return self._token

    @property
    def token_version(self) -> int:
        with self._lock:
            return self._token_version

    def _needs_refresh(self) -> bool:
        return (time.monotonic() - self._acquired_at) > TOKEN_REFRESH_SECONDS

    def _refresh(self) -> None:
        result = subprocess.run(
            _TOKEN_CMD.split(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Token acquisition failed: {result.stderr.strip()}")

        token = result.stdout.strip().splitlines()[-1]
        if not token:
            raise RuntimeError("Token command returned empty token")

        self._token = token
        self._acquired_at = time.monotonic()
        self._token_version += 1
