"""Debate-then-vote consensus approximation."""

from __future__ import annotations

from .base import AgentOutput, BaseConsensus, ConsensusResult
from src.evaluation.metrics import jaccard_similarity


class DebateThenVoteConsensus(BaseConsensus):
    """Select output with strongest cross-agent compatibility after virtual debate."""

    name = "debate_then_vote"

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
            raise ValueError("DebateThenVoteConsensus received no outputs")

        if len(outputs) == 1:
            only = outputs[0]
            return ConsensusResult(
                selected_text=only.text,
                selected_agent_id=only.agent_id,
                confidence=1.0,
                scores={only.agent_id: 1.0},
                method=self.name,
                metadata={"debate_rounds": 0},
            )

        cohesion_scores: dict[str, float] = {}
        for candidate in outputs:
            sims = [
                jaccard_similarity(candidate.text, peer.text)
                for peer in outputs
                if peer.agent_id != candidate.agent_id
            ]
            cohesion_scores[candidate.agent_id] = sum(sims) / max(1, len(sims))

        best = max(outputs, key=lambda out: (cohesion_scores[out.agent_id], len(out.text)))
        confidence = cohesion_scores[best.agent_id]
        return ConsensusResult(
            selected_text=best.text,
            selected_agent_id=best.agent_id,
            confidence=confidence,
            scores=cohesion_scores,
            method=self.name,
            metadata={"debate_rounds": 1},
        )