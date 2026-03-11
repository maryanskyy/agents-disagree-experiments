"""Consensus mechanism abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentOutput:
    """Output from a single agent invocation."""

    agent_id: str
    model_name: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConsensusResult:
    """Result of aggregating multiple agent outputs."""

    selected_text: str
    selected_agent_id: str | None
    confidence: float
    scores: dict[str, float]
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseConsensus(ABC):
    """Abstract base class for consensus implementations."""

    name: str

    @abstractmethod
    async def aggregate(
        self,
        *,
        task_type: str,
        task_prompt: str,
        rubric: list[str],
        outputs: list[AgentOutput],
        context: dict[str, Any],
    ) -> ConsensusResult:
        """Aggregate outputs into one final selection."""