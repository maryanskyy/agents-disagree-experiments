"""Judge-based consensus that scores each candidate independently."""

from __future__ import annotations

from .base import AgentOutput, BaseConsensus, ConsensusResult
from src.evaluation.llm_judge import LLMJudge


class JudgeBasedConsensus(BaseConsensus):
    """Consensus via external LLM judge scoring."""

    name = "judge_based"

    def __init__(self, judge: LLMJudge) -> None:
        self.judge = judge

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
        judge_result = await self.judge.score_candidates(
            task_type=task_type,
            task_prompt=task_prompt,
            rubric=rubric,
            outputs=[o.text for o in outputs],
            seed=seed,
        )

        winner = outputs[judge_result.winner_index]
        scores = {outputs[idx].agent_id: score for idx, score in enumerate(judge_result.scores)}
        return ConsensusResult(
            selected_text=winner.text,
            selected_agent_id=winner.agent_id,
            confidence=max(judge_result.scores),
            scores=scores,
            method=self.name,
            metadata={"rationale": judge_result.rationale},
        )