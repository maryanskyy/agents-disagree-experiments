"""Best-of-N baseline topology.

Generates N independent samples from a single-agent setup and selects the best
sample with the same judge-based consensus infrastructure used elsewhere.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import AgentInvoker, BaseConsensus, BaseTopology, TopologyResult


class BestOfNTopology(BaseTopology):
    """Cost-matched baseline: N samples from one model, pick best by judge."""

    name = "best_of_n"

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
        async def _sample(idx: int, model: str, system_prompt: str):
            prompt = (
                f"Task:\n{task_prompt}\n\n"
                f"Sampling attempt #{idx + 1}. Produce your best independent answer."
            )
            return await invoke_agent(
                idx,
                model,
                system_prompt,
                prompt,
                temperature,
                {**context, "phase": "best_of_n_sample", "sample_idx": idx + 1},
            )

        outputs = list(
            await asyncio.gather(
                *[_sample(i, model_assignment[i], system_prompts[i]) for i in range(len(model_assignment))]
            )
        )

        consensus_result = await consensus.aggregate(
            task_type=task_type,
            task_prompt=task_prompt,
            rubric=rubric,
            outputs=outputs,
            context=context,
        )

        return TopologyResult(
            outputs=outputs,
            consensus=consensus_result,
            rounds=2,
            metadata={
                "baseline_type": "best_of_n",
                "debate_rounds": [
                    {
                        "round": 1,
                        "phase": "sampling",
                        "agent_outputs": [
                            {
                                "agent_id": out.agent_id,
                                "model_name": out.model_name,
                                "text": out.text,
                            }
                            for out in outputs
                        ],
                    },
                    {
                        "round": 2,
                        "phase": "final_selection",
                        "agent_outputs": [
                            {
                                "agent_id": consensus_result.selected_agent_id or "consensus",
                                "model_name": "consensus",
                                "text": consensus_result.selected_text,
                            }
                        ],
                    },
                ],
            },
        )
