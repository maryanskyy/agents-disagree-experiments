"""Google Gemini model client adapter — routed through GenAI Gateway.

Uses the OpenAI-compatible Chat Completions API exposed by the gateway
instead of the google-generativeai SDK.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from .base import BaseModelClient, ModelResponse


class GoogleModelClient(BaseModelClient):
    """Async Gemini client using OpenAI-compatible gateway endpoint."""

    def __init__(
        self,
        model_alias: str,
        api_model: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__(model_alias=model_alias, api_model=api_model, dry_run=dry_run)
        self.api_key = api_key
        self.base_url = base_url
        self.extra_headers = extra_headers or {}

        self._client: Any | None = None
        if not self.dry_run:
            if not self.api_key:
                raise ValueError("Missing API key for Google/Gemini client")
            self._init_client()

    def _init_client(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed") from exc

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": 120,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.extra_headers:
            kwargs["default_headers"] = self.extra_headers

        self._client = AsyncOpenAI(**kwargs)

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
            raise RuntimeError("Google/Gemini client not initialized")

        started = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self.api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        text = response.choices[0].message.content or "" if response.choices else ""
        usage = response.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        return ModelResponse(
            text=text.strip(),
            model_name=self.model_alias,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            raw={
                "configured_api_model": self.api_model,
            },
        )

    def _mock_response(self, *, user_prompt: str) -> ModelResponse:
        seed = hash((self.model_alias, user_prompt[:100])) % 1_000_000
        rnd = random.Random(seed)
        head = " ".join(user_prompt.split()[:50])
        text = f"[DRY-RUN:{self.model_alias}] {head}".strip()
        return ModelResponse(
            text=text,
            model_name=self.model_alias,
            input_tokens=max(28, len(user_prompt) // 4),
            output_tokens=max(56, len(text) // 3 + rnd.randint(0, 12)),
            latency_ms=1.0,
            raw={"dry_run": True, "configured_api_model": self.api_model},
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
