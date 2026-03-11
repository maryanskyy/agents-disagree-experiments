"""Synthesis consensus � LLM reads all agent outputs and generates a merged response.

Unlike selection-based consensus (vote, debate, judge), synthesis creates NEW text
that combines the best elements from all candidates. This breaks the Selection Ceiling:
the merged output can be BETTER than any individual candidate.
"""
from __future__ import annotations

from .base import AgentOutput, BaseConsensus, ConsensusResult


class SynthesisConsensus(BaseConsensus):
    """Synthesis via LLM that reads all outputs and generates a merged response."""

    name = "synthesis"

    def __init__(self, synthesizer_client, model_name: str, cost_recorder=None) -> None:
        self.client = synthesizer_client
        self.model_name = model_name
        self.cost_recorder = cost_recorder

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
            raise ValueError("SynthesisConsensus received no outputs")

        if len(outputs) == 1:
            only = outputs[0]
            return ConsensusResult(
                selected_text=only.text,
                selected_agent_id=only.agent_id,
                confidence=1.0,
                scores={only.agent_id: 1.0},
                method=self.name,
                metadata={"synthesis_skipped": True, "reason": "single_output"},
            )

        # Build the synthesis prompt
        rubric_text = "\n".join(f"- {r}" for r in rubric) if rubric else "general quality"
        
        candidates_text = "\n\n".join(
            f"--- Response {i+1} (by {out.agent_id}) ---\n{out.text}"
            for i, out in enumerate(outputs)
        )

        system_prompt = (
            "You are an expert editor and synthesizer. Your job is to read multiple "
            "responses to the same task and produce a single SUPERIOR response that "
            "combines the strongest elements from each.\n\n"
            "Rules:\n"
            "- Identify the best ideas, arguments, evidence, and phrasing from each response\n"
            "- Resolve contradictions by keeping the most well-supported position\n"
            "- Do NOT simply concatenate the responses\n"
            "- Do NOT mention that you are synthesizing multiple responses\n"
            "- The output should read as a single, coherent, polished response\n"
            "- The output should be BETTER than any individual input response\n"
            f"- Evaluation criteria: {rubric_text}"
        )

        user_prompt = (
            f"Task:\n{task_prompt}\n\n"
            f"Here are {len(outputs)} responses from different experts:\n\n"
            f"{candidates_text}\n\n"
            "Now write a single superior response that combines the best elements "
            "from all of the above. Produce only the final response, nothing else."
        )

        response = await self.client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=2048,
            metadata={"tag": "synthesis"},
        )

        if self.cost_recorder:
            self.cost_recorder(self.model_name, response, "synthesis")

        # Score each input by how much the synthesis drew from it
        # (approximated by word overlap � not used for quality, just metadata)
        synthesis_tokens = set(response.text.lower().split())
        contribution_scores = {}
        for out in outputs:
            out_tokens = set(out.text.lower().split())
            if synthesis_tokens:
                overlap = len(synthesis_tokens & out_tokens) / len(synthesis_tokens)
            else:
                overlap = 0.0
            contribution_scores[out.agent_id] = round(overlap, 4)

        return ConsensusResult(
            selected_text=response.text,
            selected_agent_id="synthesis",
            confidence=1.0,
            scores=contribution_scores,
            method=self.name,
            metadata={
                "synthesizer_model": self.model_name,
                "input_count": len(outputs),
                "synthesis_tokens": response.output_tokens,
                "synthesis_latency_ms": response.latency_ms,
            },
        )
