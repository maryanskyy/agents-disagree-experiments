"""Google Gemini model client adapter."""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any

from .base import BaseModelClient, ModelResponse


class GoogleModelClient(BaseModelClient):
    """Async wrapper around official google-generativeai SDK."""

    def __init__(
        self,
        model_alias: str,
        api_model: str,
        api_key: str | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        super().__init__(model_alias=model_alias, api_model=api_model, dry_run=dry_run)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._module: Any | None = None
        self._model: Any | None = None

        if not self.dry_run:
            if not self.api_key:
                raise ValueError("Missing GOOGLE_API_KEY for non-dry run")
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise RuntimeError("google-generativeai package is not installed") from exc

            genai.configure(api_key=self.api_key)
            self._module = genai
            self._model = genai.GenerativeModel(self.api_model)

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

        if self._model is None:
            raise RuntimeError("Google model not initialized")

        started = time.perf_counter()

        def _call() -> Any:
            return self._model.generate_content(
                contents=[
                    {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
                ],
                generation_config=self._module.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

        response = await asyncio.to_thread(_call)
        elapsed_ms = (time.perf_counter() - started) * 1000

        text = getattr(response, "text", "") or ""
        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)

        return ModelResponse(
            text=text.strip(),
            model_name=self.model_alias,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            raw={"metadata": metadata or {}},
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
            raw={"dry_run": True},
        )

    async def close(self) -> None:
        await asyncio.sleep(0)