"""Quorum topology with draft, critique, and revision rounds."""

from __future__ import annotations

import asyncio
from typing import Any

from src.consensus.base import AgentOutput

from .base import AgentInvoker, BaseConsensus, BaseTopology, TopologyResult


class QuorumTopology(BaseTopology):
    """Structured debate protocol before final consensus."""

    name = "quorum"

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
        async def _draft(idx: int, model: str, system_prompt: str) -> AgentOutput:
            return await invoke_agent(idx, model, system_prompt, task_prompt, temperature, {**context, "phase": "draft"})

        draft_outputs = list(
            await asyncio.gather(
                *[_draft(i, model, system_prompts[i]) for i, model in enumerate(model_assignment)]
            )
        )

        async def _revise(idx: int, model: str, system_prompt: str) -> AgentOutput:
            peer = draft_outputs[(idx + 1) % len(draft_outputs)]
            critique_prompt = (
                f"Original task:\n{task_prompt}\n\n"
                f"Peer draft to critique:\n{peer.text}\n\n"
                "Revise your answer after considering strengths/weaknesses in the peer draft."
            )
            return await invoke_agent(
                idx,
                model,
                system_prompt,
                critique_prompt,
                max(0.2, min(1.2, temperature - 0.1)),
                {**context, "phase": "revision"},
            )

        revised_outputs = list(
            await asyncio.gather(
                *[_revise(i, model, system_prompts[i]) for i, model in enumerate(model_assignment)]
            )
        )

        consensus_result = await consensus.aggregate(
            task_type=task_type,
            task_prompt=task_prompt,
            rubric=rubric,
            outputs=revised_outputs,
            context=context,
        )
        return TopologyResult(
            outputs=revised_outputs,
            consensus=consensus_result,
            rounds=2,
            metadata={"draft_count": len(draft_outputs), "revision_count": len(revised_outputs)},
        )