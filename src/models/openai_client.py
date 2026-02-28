"""OpenAI model client adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import random
import time
from typing import Any

from ..utils.rate_limiter import AsyncRateLimiter

from .base import BaseModelClient, ModelResponse


@dataclass(slots=True)
class OpenAIClientConfig:
    """Configuration for OpenAI API calls."""

    timeout_seconds: int = 60
    rpm_limit: int = 60


class OpenAIModelClient(BaseModelClient):
    """Async wrapper around official OpenAI Python SDK."""

    _rate_limiter = AsyncRateLimiter()

    def __init__(
        self,
        model_alias: str,
        api_model: str,
        api_key: str | None = None,
        *,
        dry_run: bool = False,
        config: OpenAIClientConfig | None = None,
    ) -> None:
        super().__init__(model_alias=model_alias, api_model=api_model, dry_run=dry_run)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.config = config or OpenAIClientConfig()

        self._client: Any | None = None
        if not self.dry_run:
            if not self.api_key:
                raise ValueError("Missing OPENAI_API_KEY for non-dry run")
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError("openai package is not installed") from exc

            self._client = AsyncOpenAI(api_key=self.api_key, timeout=self.config.timeout_seconds)

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> ModelResponse:
        if self.dry_run:
            return self._mock_response(user_prompt=user_prompt)

        if self._client is None:
            raise RuntimeError("OpenAI client not initialized")

        await self._rate_limiter.acquire(key=f"openai:{self.model_alias}", rpm=self.config.rpm_limit)

        started = time.perf_counter()
        response = await self._client.responses.create(
            model=self.api_model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            temperature=temperature,
            max_output_tokens=max_tokens,
            metadata=metadata or {},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        return ModelResponse(
            text=self._extract_text(response).strip(),
            model_name=self.model_alias,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            raw={"id": getattr(response, "id", None)},
        )

    def _extract_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)

    def _mock_response(self, *, user_prompt: str) -> ModelResponse:
        started = time.perf_counter()
        seed = hash((self.model_alias, user_prompt[:80])) % 1_000_000
        rnd = random.Random(seed)
        sample = user_prompt.split()[: min(40, len(user_prompt.split()))]
        text = "[DRY-RUN:{}] {}".format(self.model_alias, " ".join(sample) or "empty prompt")
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ModelResponse(
            text=text,
            model_name=self.model_alias,
            input_tokens=max(32, len(user_prompt) // 4),
            output_tokens=max(64, len(text) // 3 + rnd.randint(0, 8)),
            latency_ms=elapsed_ms,
            raw={"dry_run": True},
        )

    async def close(self) -> None:
        if self._client is None:
            await asyncio.sleep(0)
            return

        close_fn = getattr(self._client, "close", None)
        if close_fn is not None:
            maybe_coro = close_fn()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
            return

        await asyncio.sleep(0)

