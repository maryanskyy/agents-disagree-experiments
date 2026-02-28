"""Abstract async model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ModelResponse:
    """Normalized model response payload."""

    text: str
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: dict[str, Any]


class BaseModelClient(ABC):
    """Base class for provider-specific async model clients."""

    def __init__(self, model_alias: str, api_model: str, dry_run: bool = False) -> None:
        self.model_alias = model_alias
        self.api_model = api_model
        self.dry_run = dry_run

    @abstractmethod
    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Generate model output asynchronously."""

    async def close(self) -> None:
        """Optional resource cleanup hook."""
        return None