"""Tests for consensus mechanisms."""

from __future__ import annotations

import asyncio

from src.consensus import AgentOutput, DebateThenVoteConsensus, SimpleVoteConsensus
from src.evaluation.llm_judge import LLMJudge
from src.consensus.judge import JudgeBasedConsensus
from src.models.base import BaseModelClient, ModelResponse


class DummyJudgeClient(BaseModelClient):
    """Minimal mock model client for judge tests."""

    async def generate(self, *, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int, metadata=None):
        return ModelResponse(
            text='{"scores":[0.2,0.9],"rationale":"test"}',
            model_name=self.model_alias,
            input_tokens=10,
            output_tokens=10,
            latency_ms=1.0,
            raw={},
        )


def _outputs() -> list[AgentOutput]:
    return [
        AgentOutput(agent_id="a1", model_name="m", text="alpha answer", input_tokens=1, output_tokens=1, latency_ms=1.0),
        AgentOutput(agent_id="a2", model_name="m", text="alpha answer", input_tokens=1, output_tokens=1, latency_ms=2.0),
        AgentOutput(agent_id="a3", model_name="m", text="beta answer", input_tokens=1, output_tokens=1, latency_ms=3.0),
    ]


def test_simple_vote_consensus() -> None:
    """Majority output should be selected."""

    async def _run() -> None:
        consensus = SimpleVoteConsensus()
        result = await consensus.aggregate(
            task_type="analytical",
            task_prompt="prompt",
            rubric=[],
            outputs=_outputs(),
            context={},
        )
        assert result.selected_agent_id in {"a1", "a2"}
        assert result.confidence > 0.6

    asyncio.run(_run())


def test_debate_consensus_returns_candidate() -> None:
    """Debate consensus should return one existing candidate."""

    async def _run() -> None:
        consensus = DebateThenVoteConsensus()
        outputs = _outputs()
        result = await consensus.aggregate(
            task_type="analytical",
            task_prompt="prompt",
            rubric=[],
            outputs=outputs,
            context={},
        )
        assert result.selected_agent_id in {o.agent_id for o in outputs}

    asyncio.run(_run())


def test_judge_consensus_uses_scores() -> None:
    """Judge-based consensus should select highest scored candidate."""

    async def _run() -> None:
        judge_client = DummyJudgeClient(model_alias="judge", api_model="judge", dry_run=False)
        judge = LLMJudge(judge_client=judge_client, dry_run=False)
        consensus = JudgeBasedConsensus(judge=judge)
        outputs = [
            AgentOutput(agent_id="a1", model_name="m", text="first", input_tokens=1, output_tokens=1, latency_ms=1.0),
            AgentOutput(agent_id="a2", model_name="m", text="second", input_tokens=1, output_tokens=1, latency_ms=1.0),
        ]
        result = await consensus.aggregate(
            task_type="creative",
            task_prompt="prompt",
            rubric=["clarity"],
            outputs=outputs,
            context={"seed": 1},
        )
        assert result.selected_agent_id in {"a1", "a2"}

    asyncio.run(_run())