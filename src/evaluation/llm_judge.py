"""LLM-as-judge utility with blind scoring and bias mitigation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import random
from typing import Any

from src.models.base import BaseModelClient

from .metrics import evaluate_analytical, evaluate_creative


@dataclass(slots=True)
class JudgeResult:
    """Scoring result for a set of candidate outputs."""

    scores: list[float]
    winner_index: int
    rationale: str


class LLMJudge:
    """Blind evaluator that randomizes order before scoring."""

    def __init__(self, judge_client: BaseModelClient, dry_run: bool = False) -> None:
        self.judge_client = judge_client
        self.dry_run = dry_run or judge_client.dry_run

    async def score_candidates(
        self,
        *,
        task_type: str,
        task_prompt: str,
        rubric: list[str],
        outputs: list[str],
        seed: int,
    ) -> JudgeResult:
        """Return candidate scores and winning index."""
        if not outputs:
            raise ValueError("Judge received empty outputs")

        rng = random.Random(seed)
        indices = list(range(len(outputs)))
        rng.shuffle(indices)
        shuffled = [outputs[i] for i in indices]

        if self.dry_run:
            judged = self._heuristic_scores(task_type=task_type, task_prompt=task_prompt, outputs=shuffled)
            mapped = [0.0] * len(outputs)
            for local_idx, global_idx in enumerate(indices):
                mapped[global_idx] = judged[local_idx]
            winner = max(range(len(mapped)), key=lambda i: mapped[i])
            return JudgeResult(scores=mapped, winner_index=winner, rationale="heuristic dry-run")

        prompt = self._build_judge_prompt(
            task_type=task_type,
            task_prompt=task_prompt,
            rubric=rubric,
            outputs=shuffled,
        )
        response = await self.judge_client.generate(
            system_prompt="You are a strict, fair evaluator. Return JSON only.",
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=1024,
            metadata={"role": "judge"},
        )
        parsed_scores = self._parse_scores(response.text, expected=len(outputs))

        mapped_scores = [0.0] * len(outputs)
        for local_idx, global_idx in enumerate(indices):
            mapped_scores[global_idx] = parsed_scores[local_idx]

        winner = max(range(len(mapped_scores)), key=lambda i: mapped_scores[i])
        return JudgeResult(scores=mapped_scores, winner_index=winner, rationale="llm_judge")

    def _heuristic_scores(self, *, task_type: str, task_prompt: str, outputs: list[str]) -> list[float]:
        if task_type == "analytical":
            return [evaluate_analytical(task_prompt, out) for out in outputs]
        return [evaluate_creative(out) for out in outputs]

    def _build_judge_prompt(
        self,
        *,
        task_type: str,
        task_prompt: str,
        rubric: list[str],
        outputs: list[str],
    ) -> str:
        rubric_text = "\n".join(f"- {line}" for line in rubric)
        candidates = "\n\n".join(f"Candidate {idx}:\n{text}" for idx, text in enumerate(outputs))
        return (
            f"Task type: {task_type}\n"
            f"Task prompt:\n{task_prompt}\n\n"
            f"Rubric:\n{rubric_text}\n\n"
            f"Evaluate each candidate from 0.0 to 1.0.\n"
            f"Return strict JSON: {{\"scores\":[...],\"rationale\":\"...\"}}\n\n"
            f"Candidates:\n{candidates}"
        )

    def _parse_scores(self, text: str, expected: int) -> list[float]:
        try:
            payload = json.loads(text)
            scores_raw = payload["scores"]
            scores = [float(v) for v in scores_raw]
            if len(scores) != expected:
                raise ValueError("score length mismatch")
            return [min(1.0, max(0.0, s)) for s in scores]
        except Exception:
            # Conservative fallback for malformed judge output.
            return [0.5 for _ in range(expected)]