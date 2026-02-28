"""Judge-based consensus powered by pairwise multi-judge panel evaluation."""

from __future__ import annotations

from .base import AgentOutput, BaseConsensus, ConsensusResult
from src.evaluation.llm_judge import JudgePanel


class JudgeBasedConsensus(BaseConsensus):
    """Consensus via external LLM judge panel with Bradley-Terry scoring."""

    name = "judge_based"

    def __init__(self, judge_panel: JudgePanel) -> None:
        self.judge_panel = judge_panel

    async def aggregate(
        self,
        *,
        task_type: str,
        task_prompt: str,
        rubric: list[str],
        outputs: list[AgentOutput],
        context: dict,
    ) -> ConsensusResult:
        if not outputs:
            raise ValueError("JudgeBasedConsensus received no outputs")

        seed = int(context.get("seed", 42))
        panel_eval = await self.judge_panel.evaluate_candidates(
            task_type=task_type,
            task_prompt=task_prompt,
            rubric=rubric,
            outputs=[o.text for o in outputs],
            seed=seed,
        )

        winner_index = panel_eval.ranking[0]
        winner = outputs[winner_index]
        scores = {outputs[idx].agent_id: float(score) for idx, score in enumerate(panel_eval.bt_scores)}

        return ConsensusResult(
            selected_text=winner.text,
            selected_agent_id=winner.agent_id,
            confidence=float(panel_eval.bt_scores[winner_index]),
            scores=scores,
            method=self.name,
            metadata={
                "judge_panel": panel_eval.to_dict(candidate_ids=[o.agent_id for o in outputs]),
            },
        )
