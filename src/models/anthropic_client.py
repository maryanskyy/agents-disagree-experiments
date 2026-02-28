"""Anthropic model client adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import random
import time
from typing import Any

from .base import BaseModelClient, ModelResponse


@dataclass(slots=True)
class AnthropicClientConfig:
    """Configuration for Anthropic API calls."""

    timeout_seconds: int = 60


class AnthropicModelClient(BaseModelClient):
    """Async wrapper around official anthropic SDK."""

    def __init__(
        self,
        model_alias: str,
        api_model: str,
        api_key: str | None = None,
        *,
        dry_run: bool = False,
        config: AnthropicClientConfig | None = None,
    ) -> None:
        super().__init__(model_alias=model_alias, api_model=api_model, dry_run=dry_run)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.config = config or AnthropicClientConfig()

        self._client: Any | None = None
        if not self.dry_run:
            if not self.api_key:
                raise ValueError("Missing ANTHROPIC_API_KEY for non-dry run")
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise RuntimeError("anthropic package is not installed") from exc
            self._client = AsyncAnthropic(api_key=self.api_key, timeout=self.config.timeout_seconds)

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
            raise RuntimeError("Anthropic client not initialized")

        started = time.perf_counter()
        response = await self._client.messages.create(
            model=self.api_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            metadata=metadata or {},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        text_chunks = []
        for chunk in response.content:
            if getattr(chunk, "type", None) == "text":
                text_chunks.append(chunk.text)

        usage = getattr(response, "usage", None)
        return ModelResponse(
            text="\n".join(text_chunks).strip(),
            model_name=self.model_alias,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            latency_ms=elapsed_ms,
            raw={"id": getattr(response, "id", None)},
        )

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
        # Official SDK currently does not require explicit closure.
        await asyncio.sleep(0)