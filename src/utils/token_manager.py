"""API endpoint configuration and token management.

Supports any OpenAI-compatible API endpoint (LiteLLM, vLLM, etc.)
with optional token-based authentication.

Configure via environment variables:
  API_BASE_URL    - base URL of the API endpoint (e.g. https://api.openai.com)
  API_ORG_ID      - organization/project ID (optional)
  API_TOKEN_CMD   - shell command to obtain a bearer token (optional)
"""

from __future__ import annotations

import os
import subprocess
import threading
import time


TOKEN_REFRESH_SECONDS = 18 * 3600  # refresh well before 20h expiry


def _get_env(key: str, fallback: str = "") -> str:
    """Read env var lazily (after load_dotenv has had a chance to run)."""
    return os.environ.get(key, fallback)


class _LazyEnv:
    """Descriptor that defers env-var reads until first access."""

    @property
    def API_BASE_URL(self) -> str:
        return _get_env("API_BASE_URL", _get_env("OPENAI_BASE_URL", ""))

    @property
    def API_ORG_ID(self) -> str:
        return _get_env("API_ORG_ID", "")

    @property
    def TOKEN_CMD(self) -> str:
        return _get_env("API_TOKEN_CMD", "")


_env = _LazyEnv()


def get_api_base_url() -> str:
    """Return the configured API base URL."""
    return _env.API_BASE_URL


def get_api_org_id() -> str:
    """Return the configured organization/project ID."""
    return _env.API_ORG_ID


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
        cmd = _env.TOKEN_CMD
        if not cmd:
            raise RuntimeError(
                "No API_TOKEN_CMD configured. Set this env var to a command "
                "that prints a bearer token, or use standard API keys instead."
            )
        result = subprocess.run(
            cmd.split(),
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
