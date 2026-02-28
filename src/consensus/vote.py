"""Simple majority-vote consensus."""

from __future__ import annotations

from collections import defaultdict

from .base import AgentOutput, BaseConsensus, ConsensusResult
from src.evaluation.metrics import normalize_text


class SimpleVoteConsensus(BaseConsensus):
    """Vote by normalized output equivalence with deterministic tie-breakers."""

    name = "simple_vote"

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
            raise ValueError("SimpleVoteConsensus received no outputs")

        buckets: dict[str, list[AgentOutput]] = defaultdict(list)
        for out in outputs:
            buckets[normalize_text(out.text)].append(out)

        best_key = max(
            buckets,
            key=lambda k: (
                len(buckets[k]),
                -min(o.latency_ms for o in buckets[k]),
                max(len(o.text) for o in buckets[k]),
            ),
        )
        winners = buckets[best_key]
        selected = winners[0]

        scores = {o.agent_id: float(len(buckets[normalize_text(o.text)])) for o in outputs}
        confidence = len(winners) / len(outputs)

        return ConsensusResult(
            selected_text=selected.text,
            selected_agent_id=selected.agent_id,
            confidence=confidence,
            scores=scores,
            method=self.name,
            metadata={"votes": {k: len(v) for k, v in buckets.items()}},
        )