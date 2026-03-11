"""Self-revision topology: single agent generates then revises N times.

Compute-matched control for debate experiments.
The agent sees only its OWN prior output -- no multi-agent interaction.
"""

from __future__ import annotations

from typing import Any

from src.consensus.base import AgentOutput, ConsensusResult

from .base import AgentInvoker, BaseConsensus, BaseTopology, TopologyResult


class SelfRevisionTopology(BaseTopology):
    """Single agent generates, then self-revises N times."""

    name = "self_revision"

    def __init__(self, revision_rounds: int = 5):
        self.revision_rounds = revision_rounds

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
        model = model_assignment[0]
        system_prompt = system_prompts[0]

        # Initial generation
        current = await invoke_agent(
            0, model, system_prompt, task_prompt, temperature,
            {**context, "phase": "draft", "revision_round": 0},
        )

        debate_rounds = [
            {
                "round": 1,
                "phase": "draft",
                "agent_outputs": [
                    {"agent_id": current.agent_id, "model_name": current.model_name, "text": current.text}
                ],
            }
        ]

        all_outputs = [current]

        for r in range(self.revision_rounds):
            revision_prompt = (
                f"Original task:\n{task_prompt}\n\n"
                f"Your previous response:\n{current.text}\n\n"
                "Please critically review and revise your response. "
                "Improve clarity, depth, and quality. Fix any weaknesses you identify."
            )
            revised = await invoke_agent(
                0, model, system_prompt, revision_prompt,
                max(0.2, min(1.2, temperature - 0.1)),
                {**context, "phase": "revision", "revision_round": r + 1},
            )
            current = revised
            all_outputs.append(current)

            debate_rounds.append({
                "round": r + 2,
                "phase": f"self_revision_{r + 1}",
                "agent_outputs": [
                    {"agent_id": current.agent_id, "model_name": current.model_name, "text": current.text}
                ],
            })

        consensus_result = ConsensusResult(
            selected_text=current.text,
            selected_agent_id=current.agent_id,
            confidence=1.0,
            scores={current.agent_id: 1.0},
            method="self_revision_final",
            metadata={
                "revision_rounds": self.revision_rounds,
                "total_outputs": len(all_outputs),
            },
        )

        return TopologyResult(
            outputs=all_outputs,
            consensus=consensus_result,
            rounds=1 + self.revision_rounds,
            metadata={
                "topology": "self_revision",
                "revision_rounds": self.revision_rounds,
                "debate_rounds": debate_rounds,
            },
        )