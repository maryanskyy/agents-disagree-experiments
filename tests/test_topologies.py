"""Tests for topology strategies."""

from __future__ import annotations

import asyncio

from src.consensus import AgentOutput, SimpleVoteConsensus
from src.topologies import FlatTopology, HierarchicalTopology, QuorumTopology


async def _invoke_agent(idx: int, model: str, system_prompt: str, user_prompt: str, temperature: float, context: dict):
    return AgentOutput(
        agent_id=f"agent_{idx}",
        model_name=model,
        text=f"{system_prompt[:12]} | {user_prompt[:20]} | {idx}",
        input_tokens=12,
        output_tokens=24,
        latency_ms=5.0 + idx,
        metadata={"context": context},
    )


def _common_kwargs() -> dict:
    return {
        "task_type": "analytical",
        "task_prompt": "Explain the scenario",
        "rubric": ["clear"],
        "model_assignment": ["m1", "m2", "m3"],
        "system_prompts": ["s1", "s2", "s3"],
        "temperature": 0.7,
        "invoke_agent": _invoke_agent,
        "consensus": SimpleVoteConsensus(),
        "context": {"seed": 42},
    }


def test_flat_topology_executes_parallel_agents() -> None:
    """Flat topology should return one output per agent."""

    async def _run() -> None:
        topo = FlatTopology()
        result = await topo.execute(**_common_kwargs())
        assert len(result.outputs) == 3
        assert result.rounds == 1

    asyncio.run(_run())


def test_hierarchical_topology_executes_levels() -> None:
    """Hierarchical topology should report multiple levels."""

    async def _run() -> None:
        topo = HierarchicalTopology()
        result = await topo.execute(**_common_kwargs())
        assert len(result.outputs) == 3
        assert result.rounds >= 2

    asyncio.run(_run())


def test_quorum_topology_runs_revision_round() -> None:
    """Quorum topology should execute two rounds."""

    async def _run() -> None:
        topo = QuorumTopology()
        result = await topo.execute(**_common_kwargs())
        assert len(result.outputs) == 3
        assert result.rounds == 2

    asyncio.run(_run())