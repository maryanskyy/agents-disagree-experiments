"""Pairwise LLM-as-judge with multi-judge panel aggregation.

This module implements the production evaluation protocol required for the
experiment:
1) Pairwise A-vs-B judging (not absolute scoring)
2) Bidirectional ordering (A/B and B/A) for position-bias checks
3) Multi-judge panel support
4) Bradley-Terry aggregation for global candidate scores
5) Inter-rater reliability via Cohen's kappa
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import random
from typing import Any, Callable

import numpy as np

from src.models.base import BaseModelClient, ModelResponse

from .metrics import dev_heuristic_analytical, dev_heuristic_creative, mean


_LABEL_LEFT = "left"
_LABEL_RIGHT = "right"
_LABEL_TIE = "tie"


@dataclass(slots=True)
class PairwiseBallot:
    """Single directional ballot for one judge call."""

    ordering: str  # "ab" or "ba"
    winner_label: str  # "A" | "B" | "tie"
    confidence: float
    rationale: str


@dataclass(slots=True)
class JudgePairDecision:
    """Resolved pairwise decision for one judge after both orderings."""

    judge_model: str
    winner_index: int | None  # 0 for left candidate, 1 for right candidate, None for tie
    consistent_across_orderings: bool
    ballots: list[PairwiseBallot]
    confidence: float


@dataclass(slots=True)
class PairwiseRecord:
    """Panel-level outcome record for one candidate pair."""

    candidate_i: int
    candidate_j: int
    per_judge: dict[str, str]
    majority_winner: int | None


@dataclass(slots=True)
class PanelEvaluation:
    """Complete panel evaluation over a candidate set."""

    judges: list[str]
    bt_scores: list[float]
    ranking: list[int]
    per_judge_bt_scores: dict[str, list[float]]
    inter_rater_reliability: dict[str, Any]
    pairwise_records: list[PairwiseRecord] = field(default_factory=list)

    def to_dict(self, candidate_ids: list[str] | None = None) -> dict[str, Any]:
        """Serialize panel evaluation into JSON-compatible dict."""
        if candidate_ids is None:
            candidate_ids = [f"candidate_{idx}" for idx in range(len(self.bt_scores))]

        bt_by_id = {candidate_ids[idx]: float(score) for idx, score in enumerate(self.bt_scores)}
        per_judge_by_id = {
            judge: {candidate_ids[idx]: float(score) for idx, score in enumerate(scores)}
            for judge, scores in self.per_judge_bt_scores.items()
        }
        return {
            "judges": self.judges,
            "bt_scores": bt_by_id,
            "ranking": [candidate_ids[idx] for idx in self.ranking],
            "per_judge_bt_scores": per_judge_by_id,
            "inter_rater_reliability": self.inter_rater_reliability,
            "pairwise_records": [
                {
                    "candidate_i": candidate_ids[record.candidate_i],
                    "candidate_j": candidate_ids[record.candidate_j],
                    "per_judge": record.per_judge,
                    "majority_winner": (
                        candidate_ids[record.majority_winner] if record.majority_winner is not None else None
                    ),
                }
                for record in self.pairwise_records
            ],
        }


class PairwiseJudge:
    """LLM judge that compares two outputs with bidirectional ordering checks."""

    def __init__(
        self,
        judge_client: BaseModelClient,
        *,
        dry_run: bool = False,
        call_recorder: Callable[[str, ModelResponse, str], None] | None = None,
    ) -> None:
        self.judge_client = judge_client
        self.dry_run = dry_run or judge_client.dry_run
        self.call_recorder = call_recorder

    async def compare_pair(
        self,
        *,
        task_type: str,
        task_prompt: str,
        rubric: list[str],
        left_output: str,
        right_output: str,
        seed: int,
    ) -> JudgePairDecision:
        """Compare left vs right candidates with both orderings.

        The judge receives both A/B and B/A presentations. A consistent winner
        across both orderings is treated as robust. If orderings conflict,
        winner is set to tie (None) to avoid position-bias artifacts.
        """
        rng = random.Random(seed)
        first_order = "ab" if rng.random() < 0.5 else "ba"

        if first_order == "ab":
            first = (left_output, right_output, "ab")
            second = (right_output, left_output, "ba")
        else:
            first = (right_output, left_output, "ba")
            second = (left_output, right_output, "ab")

        ballot1, ballot2 = await asyncio.gather(
            self._single_ballot(
                task_type=task_type,
                task_prompt=task_prompt,
                rubric=rubric,
                output_a=first[0],
                output_b=first[1],
                ordering=first[2],
            ),
            self._single_ballot(
                task_type=task_type,
                task_prompt=task_prompt,
                rubric=rubric,
                output_a=second[0],
                output_b=second[1],
                ordering=second[2],
            ),
        )

        canonical_1 = self._canonical_winner(ballot1)
        canonical_2 = self._canonical_winner(ballot2)

        if canonical_1 == canonical_2:
            winner = canonical_1
            consistent = True
        elif canonical_1 is None and canonical_2 is not None:
            winner = canonical_2
            consistent = False
        elif canonical_2 is None and canonical_1 is not None:
            winner = canonical_1
            consistent = False
        else:
            winner = None
            consistent = False

        return JudgePairDecision(
            judge_model=self.judge_client.model_alias,
            winner_index=winner,
            consistent_across_orderings=consistent,
            ballots=[ballot1, ballot2],
            confidence=max(0.0, min(1.0, mean([ballot1.confidence, ballot2.confidence]))),
        )

    async def _single_ballot(
        self,
        *,
        task_type: str,
        task_prompt: str,
        rubric: list[str],
        output_a: str,
        output_b: str,
        ordering: str,
    ) -> PairwiseBallot:
        if self.dry_run:
            winner_label, confidence = self._dry_run_ballot(
                task_type=task_type,
                task_prompt=task_prompt,
                output_a=output_a,
                output_b=output_b,
            )
            return PairwiseBallot(
                ordering=ordering,
                winner_label=winner_label,
                confidence=confidence,
                rationale="dry-run heuristic",
            )

        prompt = self._build_prompt(
            task_type=task_type,
            task_prompt=task_prompt,
            rubric=rubric,
            output_a=output_a,
            output_b=output_b,
        )
        response = await self.judge_client.generate(
            system_prompt=(
                "You are a decisive blind evaluator. You MUST pick a winner. "
                "Compare Response A vs Response B on the rubric criteria and decide which is better. "
                "Do NOT say tie. One response is always at least slightly better. Find the difference. "
                'Return JSON only with keys: winner (must be "A" or "B"), confidence (0-1), rationale.'
            ),
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=1024,
            metadata={"role": "pairwise_judge", "ordering": ordering},
        )
        if self.call_recorder is not None:
            self.call_recorder(self.judge_client.model_alias, response, "pairwise_judge")

        winner_label, confidence, rationale = self._parse_ballot_json(response.text)
        return PairwiseBallot(
            ordering=ordering,
            winner_label=winner_label,
            confidence=confidence,
            rationale=rationale,
        )

    def _build_prompt(
        self,
        *,
        task_type: str,
        task_prompt: str,
        rubric: list[str],
        output_a: str,
        output_b: str,
    ) -> str:
        task_rubric = self._task_specific_rubric(task_type)
        merged_rubric = task_rubric + [line for line in rubric if line not in task_rubric]
        rubric_text = "\n".join(f"- {line}" for line in merged_rubric)
        return (
            f"Task type: {task_type}\n"
            f"Task prompt:\n{task_prompt}\n\n"
            "Evaluation rules:\n"
            "- You MUST choose either A or B as the winner. Do NOT choose tie.\n"
            "- Even if both responses seem similar, one is always at least slightly better.\n"
            "  Examine the rubric criteria carefully and find the differentiating factor.\n"
            "- Judge only quality relative to the task and rubric.\n"
            "- Ignore response length unless it harms substance.\n"
            "- Ignore formatting polish and stylistic familiarity biases.\n"
            "- You are blind to model, topology, consensus method, and agent count.\n\n"
            f"Rubric:\n{rubric_text}\n\n"
            f"Response A:\n{output_a}\n\n"
            f"Response B:\n{output_b}\n\n"
            "Return strict JSON only:\n"
            '{"winner":"A or B","confidence":0.0,"rationale":"..."}'
        )

    def _task_specific_rubric(self, task_type: str) -> list[str]:
        if task_type == "analytical":
            return [
                "Correctness: factual and logical validity",
                "Reasoning depth: clear multi-step argument quality",
                "Evidence quality: concrete support, assumptions handled explicitly",
            ]
        return [
            "Originality: novelty of ideas and voice",
            "Coherence: internal consistency and structure",
            "Craft: fluency, imagery, and stylistic control",
        ]

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences that some models wrap around JSON."""
        stripped = text.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1:]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
        return stripped.strip()

    def _parse_ballot_json(self, text: str) -> tuple[str, float, str]:
        cleaned = self._strip_markdown_fences(text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = self._extract_partial_json(cleaned)

        raw_winner = str(payload.get("winner", "tie")).strip().lower()
        if raw_winner in {"a", "response a", "candidate a"}:
            winner = "A"
        elif raw_winner in {"b", "response b", "candidate b"}:
            winner = "B"
        else:
            winner = "tie"
        confidence = float(payload.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        rationale = str(payload.get("rationale", ""))
        return winner, confidence, rationale

    @staticmethod
    def _extract_partial_json(text: str) -> dict:
        """Best-effort extraction from truncated or malformed JSON."""
        import re
        winner_match = re.search(r'"winner"\s*:\s*"([^"]*)"', text)
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
        rat_match = re.search(r'"rationale"\s*:\s*"((?:[^"\\]|\\.)*)', text)
        if winner_match:
            return {
                "winner": winner_match.group(1),
                "confidence": float(conf_match.group(1)) if conf_match else 0.5,
                "rationale": rat_match.group(1) if rat_match else "partial_parse",
            }
        return {}

    def _canonical_winner(self, ballot: PairwiseBallot) -> int | None:
        """Map ballot winner label to canonical left/right pair index.

        Returns:
          0 -> left candidate wins
          1 -> right candidate wins
          None -> tie/uncertain
        """
        label = ballot.winner_label
        if ballot.ordering == "ab":
            if label == "A":
                return 0
            if label == "B":
                return 1
            return None

        # ordering == "ba": A maps to right candidate, B maps to left candidate
        if label == "A":
            return 1
        if label == "B":
            return 0
        return None

    def _dry_run_ballot(self, *, task_type: str, task_prompt: str, output_a: str, output_b: str) -> tuple[str, float]:
        score_a = (
            dev_heuristic_analytical(task_prompt, output_a)
            if task_type == "analytical"
            else dev_heuristic_creative(output_a)
        )
        score_b = (
            dev_heuristic_analytical(task_prompt, output_b)
            if task_type == "analytical"
            else dev_heuristic_creative(output_b)
        )
        gap = abs(score_a - score_b)
        confidence = max(0.5, min(0.99, 0.5 + gap))
        if gap < 0.02:
            return "tie", confidence
        return ("A", confidence) if score_a > score_b else ("B", confidence)


class JudgePanel:
    """Multi-judge panel evaluator with Bradley-Terry aggregation."""

    def __init__(self, judges: list[PairwiseJudge]) -> None:
        if not judges:
            raise ValueError("JudgePanel requires at least one judge")
        self.judges = judges

    async def evaluate_candidates(
        self,
        *,
        task_type: str,
        task_prompt: str,
        rubric: list[str],
        outputs: list[str],
        seed: int,
    ) -> PanelEvaluation:
        """Evaluate all candidate outputs via pairwise panel judgments.

        The panel computes all unordered pair comparisons and converts outcomes
        into global candidate scores using Bradley-Terry maximum-likelihood
        estimation.
        """
        if not outputs:
            raise ValueError("JudgePanel received empty candidate list")

        judge_names = [judge.judge_client.model_alias for judge in self.judges]
        n = len(outputs)
        if n == 1:
            return PanelEvaluation(
                judges=judge_names,
                bt_scores=[1.0],
                ranking=[0],
                per_judge_bt_scores={name: [1.0] for name in judge_names},
                inter_rater_reliability={"mean_cohen_kappa": 1.0, "pairwise": {}},
                pairwise_records=[],
            )

        aggregate_wins = np.zeros((n, n), dtype=float)
        per_judge_wins: dict[str, np.ndarray] = {
            name: np.zeros((n, n), dtype=float) for name in judge_names
        }
        labels_by_judge: dict[str, list[str]] = {name: [] for name in judge_names}
        pairwise_records: list[PairwiseRecord] = []

        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                tasks = []
                for judge_idx, judge in enumerate(self.judges):
                    tasks.append(
                        judge.compare_pair(
                            task_type=task_type,
                            task_prompt=task_prompt,
                            rubric=rubric,
                            left_output=outputs[i],
                            right_output=outputs[j],
                            seed=seed + pair_idx * 997 + judge_idx * 17,
                        )
                    )
                decisions = await asyncio.gather(*tasks)
                pair_idx += 1

                per_judge_labels: dict[str, str] = {}
                majority_tally = {_LABEL_LEFT: 0, _LABEL_RIGHT: 0, _LABEL_TIE: 0}

                for decision in decisions:
                    judge_name = decision.judge_model
                    label = self._decision_to_label(decision)
                    labels_by_judge[judge_name].append(label)
                    per_judge_labels[judge_name] = label
                    majority_tally[label] += 1

                    if decision.winner_index == 0:
                        per_judge_wins[judge_name][i, j] += 1.0
                        aggregate_wins[i, j] += 1.0
                    elif decision.winner_index == 1:
                        per_judge_wins[judge_name][j, i] += 1.0
                        aggregate_wins[j, i] += 1.0
                    else:
                        per_judge_wins[judge_name][i, j] += 0.5
                        per_judge_wins[judge_name][j, i] += 0.5
                        aggregate_wins[i, j] += 0.5
                        aggregate_wins[j, i] += 0.5

                majority_winner: int | None
                if majority_tally[_LABEL_LEFT] > max(majority_tally[_LABEL_RIGHT], majority_tally[_LABEL_TIE]):
                    majority_winner = i
                elif majority_tally[_LABEL_RIGHT] > max(majority_tally[_LABEL_LEFT], majority_tally[_LABEL_TIE]):
                    majority_winner = j
                else:
                    majority_winner = None

                pairwise_records.append(
                    PairwiseRecord(
                        candidate_i=i,
                        candidate_j=j,
                        per_judge=per_judge_labels,
                        majority_winner=majority_winner,
                    )
                )

        bt_scores = bradley_terry_scores(aggregate_wins)
        ranking = sorted(range(n), key=lambda idx: bt_scores[idx], reverse=True)
        per_judge_scores = {
            name: bradley_terry_scores(wins) for name, wins in per_judge_wins.items()
        }
        inter_rater = self._inter_rater_reliability(labels_by_judge)

        return PanelEvaluation(
            judges=judge_names,
            bt_scores=bt_scores,
            ranking=ranking,
            per_judge_bt_scores=per_judge_scores,
            inter_rater_reliability=inter_rater,
            pairwise_records=pairwise_records,
        )

    def _decision_to_label(self, decision: JudgePairDecision) -> str:
        if decision.winner_index == 0:
            return _LABEL_LEFT
        if decision.winner_index == 1:
            return _LABEL_RIGHT
        return _LABEL_TIE

    def _inter_rater_reliability(self, labels_by_judge: dict[str, list[str]]) -> dict[str, Any]:
        judges = sorted(labels_by_judge.keys())
        if len(judges) < 2:
            return {"mean_cohen_kappa": 1.0, "pairwise": {}}

        kappas: list[float] = []
        pairwise: dict[str, float] = {}
        for i in range(len(judges)):
            for j in range(i + 1, len(judges)):
                a = judges[i]
                b = judges[j]
                kappa = cohen_kappa(labels_by_judge[a], labels_by_judge[b])
                kappas.append(kappa)
                pairwise[f"{a}__vs__{b}"] = kappa

        return {
            "mean_cohen_kappa": float(mean(kappas)) if kappas else 1.0,
            "pairwise": pairwise,
        }


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa for two label sequences.

    Kappa = (p_o - p_e) / (1 - p_e), where:
      - p_o is observed agreement
      - p_e is expected agreement under independent marginals
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must be same length for Cohen's kappa")
    n = len(labels_a)
    if n == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    if not categories:
        return 0.0

    observed = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / n

    probs_a: dict[str, float] = {cat: labels_a.count(cat) / n for cat in categories}
    probs_b: dict[str, float] = {cat: labels_b.count(cat) / n for cat in categories}
    expected = sum(probs_a[cat] * probs_b[cat] for cat in categories)

    if abs(1.0 - expected) < 1e-12:
        return 1.0
    return float((observed - expected) / (1.0 - expected))


def bradley_terry_scores(
    wins: np.ndarray,
    *,
    max_iter: int = 500,
    tol: float = 1e-9,
    smoothing: float = 1e-6,
) -> list[float]:
    """Estimate Bradley-Terry strengths from pairwise win matrix.

    Args:
        wins: Matrix where wins[i, j] is the number of times candidate i beat j.
        max_iter: Maximum MM iterations.
        tol: Convergence threshold on max absolute parameter change.
        smoothing: Small pseudo-count added off-diagonal for numerical stability.

    Returns:
        Normalized positive strength vector summing to 1.

    Notes:
        Uses the standard MM update for Bradley-Terry MLE:
            p_i^{new} = w_i / sum_{j!=i} n_ij / (p_i + p_j)
        where:
            w_i = total wins by item i
            n_ij = wins_ij + wins_ji
    """
    matrix = np.asarray(wins, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("wins must be a square matrix")

    n = matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    w = matrix.copy()
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i, j] += smoothing

    p = np.full(n, 1.0 / n, dtype=float)
    for _ in range(max_iter):
        prev = p.copy()
        updated = p.copy()

        for i in range(n):
            total_wins_i = float(np.sum(w[i, :]) - w[i, i])
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                nij = w[i, j] + w[j, i]
                denom += float(nij) / max(prev[i] + prev[j], 1e-12)
            updated[i] = total_wins_i / max(denom, 1e-12)

        updated = np.maximum(updated, 1e-12)
        updated = updated / np.sum(updated)
        p = updated

        if float(np.max(np.abs(p - prev))) < tol:
            break

    return [float(v) for v in p.tolist()]


# Backward-compat alias for legacy imports.
LLMJudge = PairwiseJudge
