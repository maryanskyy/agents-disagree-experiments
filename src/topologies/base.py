"""Topology abstraction for orchestrating multi-agent execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from src.consensus.base import AgentOutput, BaseConsensus, ConsensusResult


AgentInvoker = Callable[[int, str, str, str, float, dict[str, Any]], Awaitable[AgentOutput]]


@dataclass(slots=True)
class TopologyResult:
    """Result of topology execution and consensus."""

    outputs: list[AgentOutput]
    consensus: ConsensusResult
    rounds: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTopology(ABC):
    """Abstract topology strategy."""

    name: str

    @abstractmethod
    async def execute(
        self,
        *,
        task_type: str,
        task_prompt: str,
        rubric: list[str],
        model_assignment: list[str],
        system_prompts: list[str],
        temperature: float,
        invoke_agent: AgentInvoker,
        consensus: BaseConsensus,
        context: dict[str, Any],
    ) -> TopologyResult:
        """Run topology and return consensus result."""