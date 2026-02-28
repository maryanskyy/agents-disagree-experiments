"""Tests for consensus mechanisms."""

from __future__ import annotations

import asyncio

from src.consensus import AgentOutput, DebateThenVoteConsensus, SimpleVoteConsensus
from src.consensus.judge import JudgeBasedConsensus
from src.evaluation.llm_judge import JudgePanel, PairwiseJudge
from src.models.base import BaseModelClient, ModelResponse


class DummyJudgeClient(BaseModelClient):
    """Mock model client for pairwise judge tests."""

    async def generate(self, *, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int, metadata=None):
        a_start = user_prompt.find("Response A:\n")
        b_start = user_prompt.find("\n\nResponse B:\n")
        if a_start >= 0 and b_start > a_start:
            response_a = user_prompt[a_start + len("Response A:\n") : b_start]
            response_b = user_prompt[b_start + len("\n\nResponse B:\n") :]
        else:
            response_a = ""
            response_b = ""

        if "second" in response_a.lower() and "second" not in response_b.lower():
            winner = "A"
        elif "second" in response_b.lower() and "second" not in response_a.lower():
            winner = "B"
        else:
            winner = "tie"

        return ModelResponse(
            text=f'{{"winner":"{winner}","confidence":0.9,"rationale":"test"}}',
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


def test_judge_consensus_uses_pairwise_panel() -> None:
    """Judge-based consensus should select candidate preferred by panel."""

    async def _run() -> None:
        clients = [
            DummyJudgeClient(model_alias="judge1", api_model="judge", dry_run=False),
            DummyJudgeClient(model_alias="judge2", api_model="judge", dry_run=False),
            DummyJudgeClient(model_alias="judge3", api_model="judge", dry_run=False),
        ]
        panel = JudgePanel(judges=[PairwiseJudge(judge_client=c, dry_run=False) for c in clients])
        consensus = JudgeBasedConsensus(judge_panel=panel)
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
        assert result.selected_agent_id == "a2"
        assert "judge_panel" in result.metadata

    asyncio.run(_run())
