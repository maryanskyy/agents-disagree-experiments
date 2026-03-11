"""Flat topology: all agents run in parallel, then aggregate once."""

from __future__ import annotations

import asyncio
from typing import Any

from .base import AgentInvoker, BaseConsensus, BaseTopology, TopologyResult


class FlatTopology(BaseTopology):
    """Parallel one-shot topology."""

    name = "flat"

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
        async def _call(idx: int, model: str, system_prompt: str):
            return await invoke_agent(idx, model, system_prompt, task_prompt, temperature, context)

        outputs = await asyncio.gather(
            *[_call(i, model, system_prompts[i]) for i, model in enumerate(model_assignment)]
        )

        consensus_result = await consensus.aggregate(
            task_type=task_type,
            task_prompt=task_prompt,
            rubric=rubric,
            outputs=list(outputs),
            context=context,
        )
        return TopologyResult(outputs=list(outputs), consensus=consensus_result, rounds=1)