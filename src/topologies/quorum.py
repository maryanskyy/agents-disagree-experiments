"""Quorum topology with draft, cross-review, and revision rounds."""

from __future__ import annotations

import asyncio
import random
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

        rng = random.Random(int(context.get("seed", 42)))
        peer_orders: dict[int, list[AgentOutput]] = {}
        for idx in range(len(draft_outputs)):
            peers = [out for j, out in enumerate(draft_outputs) if j != idx]
            rng.shuffle(peers)
            peer_orders[idx] = peers

        async def _revise(idx: int, model: str, system_prompt: str) -> AgentOutput:
            peers = peer_orders[idx]
            peer_text = "\n\n".join(
                f"Peer {peer_idx + 1} draft:\n{peer.text}"
                for peer_idx, peer in enumerate(peers)
            )
            critique_prompt = (
                f"Original task:\n{task_prompt}\n\n"
                f"Peer drafts to review:\n{peer_text}\n\n"
                "Revise your answer after comparing all peer drafts. "
                "Preserve strong ideas, challenge weak claims, and produce your best final response."
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

        debate_rounds = [
            {
                "round": 1,
                "phase": "draft",
                "agent_outputs": [
                    {"agent_id": out.agent_id, "model_name": out.model_name, "text": out.text}
                    for out in draft_outputs
                ],
            },
            {
                "round": 2,
                "phase": "revision",
                "agent_outputs": [
                    {"agent_id": out.agent_id, "model_name": out.model_name, "text": out.text}
                    for out in revised_outputs
                ],
            },
            {
                "round": 3,
                "phase": "final_consensus",
                "agent_outputs": [
                    {
                        "agent_id": consensus_result.selected_agent_id or "consensus",
                        "model_name": "consensus",
                        "text": consensus_result.selected_text,
                    }
                ],
            },
        ]

        return TopologyResult(
            outputs=revised_outputs,
            consensus=consensus_result,
            rounds=3,
            metadata={
                "draft_count": len(draft_outputs),
                "revision_count": len(revised_outputs),
                "debate_rounds": debate_rounds,
            },
        )
