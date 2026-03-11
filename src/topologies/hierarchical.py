"""Hierarchical topology with local aggregation in a binary tree."""

from __future__ import annotations

import asyncio
from typing import Any

from src.consensus.base import AgentOutput

from .base import AgentInvoker, BaseConsensus, BaseTopology, TopologyResult


class HierarchicalTopology(BaseTopology):
    """Tree-based aggregation with local consensus per level."""

    name = "hierarchical"

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
        async def _call(idx: int, model: str, system_prompt: str) -> AgentOutput:
            return await invoke_agent(idx, model, system_prompt, task_prompt, temperature, context)

        leaf_outputs = list(
            await asyncio.gather(
                *[_call(i, model, system_prompts[i]) for i, model in enumerate(model_assignment)]
            )
        )

        rounds = 1
        current = leaf_outputs
        while len(current) > 1:
            next_level: list[AgentOutput] = []
            for i in range(0, len(current), 2):
                group = current[i : i + 2]
                local_consensus = await consensus.aggregate(
                    task_type=task_type,
                    task_prompt=task_prompt,
                    rubric=rubric,
                    outputs=group,
                    context=context,
                )
                representative = AgentOutput(
                    agent_id=f"cluster_{rounds}_{i//2}",
                    model_name="cluster",
                    text=local_consensus.selected_text,
                    input_tokens=sum(g.input_tokens for g in group),
                    output_tokens=sum(g.output_tokens for g in group),
                    latency_ms=max(g.latency_ms for g in group),
                    metadata={"source_agents": [g.agent_id for g in group]},
                )
                next_level.append(representative)
            current = next_level
            rounds += 1

        root_consensus = await consensus.aggregate(
            task_type=task_type,
            task_prompt=task_prompt,
            rubric=rubric,
            outputs=current,
            context=context,
        )
        return TopologyResult(
            outputs=leaf_outputs,
            consensus=root_consensus,
            rounds=rounds,
            metadata={"levels": rounds},
        )