"""Pipeline topology: sequential critique-and-revise chain."""

from __future__ import annotations

from typing import Any

from .base import AgentInvoker, BaseConsensus, BaseTopology, TopologyResult


class PipelineTopology(BaseTopology):
    """Sequential topology where each agent refines the previous output."""

    name = "pipeline"

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
        outputs = []

        for idx, model in enumerate(model_assignment):
            if idx == 0:
                user_prompt = task_prompt
                phase = "pipeline_initial"
            else:
                user_prompt = (
                    f"Original task:\n{task_prompt}\n\n"
                    f"Previous draft:\n{outputs[-1].text}\n\n"
                    "Refine and improve the draft while preserving correct content. "
                    "Fix errors, strengthen reasoning, and improve clarity."
                )
                phase = "pipeline_refine"

            response = await invoke_agent(
                idx,
                model,
                system_prompts[idx],
                user_prompt,
                temperature,
                {**context, "phase": phase, "pipeline_step": idx + 1},
            )
            outputs.append(response)

        consensus_result = await consensus.aggregate(
            task_type=task_type,
            task_prompt=task_prompt,
            rubric=rubric,
            outputs=outputs,
            context=context,
        )

        rounds = [
            {
                "round": idx + 1,
                "phase": "pipeline_step",
                "agent_outputs": [
                    {
                        "agent_id": out.agent_id,
                        "model_name": out.model_name,
                        "text": out.text,
                    }
                ],
            }
            for idx, out in enumerate(outputs)
        ]
        rounds.append(
            {
                "round": len(outputs) + 1,
                "phase": "final_consensus",
                "agent_outputs": [
                    {
                        "agent_id": consensus_result.selected_agent_id or "consensus",
                        "model_name": "consensus",
                        "text": consensus_result.selected_text,
                    }
                ],
            }
        )

        return TopologyResult(
            outputs=outputs,
            consensus=consensus_result,
            rounds=len(rounds),
            metadata={"debate_rounds": rounds},
        )
