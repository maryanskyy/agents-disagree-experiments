"""Corrected quality metrics for cross-condition comparability.

Raw Bradley-Terry (BT) scores are normalized to sum to 1.0 within each
candidate set, so the absolute value shrinks as candidate count grows.
This module computes corrected metrics that are comparable across runs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


_TIE_LABELS = {"tie", "draw", "none", "null", ""}
_LEFT_LABELS = {"left", "candidate_i", "a"}
_RIGHT_LABELS = {"right", "candidate_j", "b"}


def infer_consensus_candidate_id(
    *,
    bt_scores: dict[str, Any],
    preferred_id: str | None = None,
) -> str | None:
    """Infer the candidate id that represents the consensus output."""
    if "final_consensus" in bt_scores:
        return "final_consensus"
    if preferred_id and preferred_id in bt_scores:
        return preferred_id
    if bt_scores:
        return next(iter(bt_scores))
    return None


def compute_corrected_metrics(
    *,
    panel_payload: dict[str, Any],
    quality_score: float,
    consensus_candidate_id: str | None,
) -> dict[str, Any]:
    """Compute corrected quality metrics from panel payload and quality score."""
    bt_scores_raw = panel_payload.get("bt_scores", {}) or {}
    bt_scores: dict[str, float] = {
        str(candidate_id): float(score)
        for candidate_id, score in bt_scores_raw.items()
    }

    consensus_id = infer_consensus_candidate_id(
        bt_scores=bt_scores,
        preferred_id=consensus_candidate_id,
    )

    num_candidates = len(bt_scores)
    normalized_bt = float(quality_score) * float(num_candidates)

    competitor_scores = [
        score
        for candidate_id, score in bt_scores.items()
        if consensus_id is None or candidate_id != consensus_id
    ]
    consensus_vs_best_agent = (
        float(quality_score) >= max(competitor_scores)
        if competitor_scores
        else True
    )

    win_points = 0.0
    comparison_count = 0
    judge_points: dict[str, float] = defaultdict(float)
    judge_counts: dict[str, int] = defaultdict(int)

    for record in panel_payload.get("pairwise_records", []) or []:
        candidate_i = str(record.get("candidate_i"))
        candidate_j = str(record.get("candidate_j"))

        if consensus_id is None or (candidate_i != consensus_id and candidate_j != consensus_id):
            continue

        comparison_count += 1
        majority_winner = record.get("majority_winner")
        if majority_winner is None:
            win_points += 0.5
        elif str(majority_winner) == consensus_id:
            win_points += 1.0

        per_judge = record.get("per_judge", {}) or {}
        for judge_name, judge_vote in per_judge.items():
            judge_counts[judge_name] += 1
            judge_points[judge_name] += _judge_vote_score(
                vote=judge_vote,
                candidate_i=candidate_i,
                candidate_j=candidate_j,
                consensus_candidate_id=consensus_id,
            )

    consensus_win_rate = win_points / comparison_count if comparison_count else 0.0
    per_judge_consensus_win_rate = {
        judge_name: judge_points[judge_name] / judge_counts[judge_name]
        for judge_name in sorted(judge_counts)
        if judge_counts[judge_name] > 0
    }

    return {
        "consensus_candidate_id": consensus_id,
        "consensus_win_rate": consensus_win_rate,
        "normalized_bt_score": normalized_bt,
        "num_bt_candidates": num_candidates,
        "consensus_vs_best_agent": bool(consensus_vs_best_agent),
        "consensus_comparisons": comparison_count,
        "per_judge_consensus_win_rate": per_judge_consensus_win_rate,
    }


def _judge_vote_score(
    *,
    vote: Any,
    candidate_i: str,
    candidate_j: str,
    consensus_candidate_id: str,
) -> float:
    if vote is None:
        return 0.5

    raw_vote = str(vote).strip()
    lowered = raw_vote.lower()

    if lowered in _TIE_LABELS:
        return 0.5

    if lowered in _LEFT_LABELS:
        winner = candidate_i
    elif lowered in _RIGHT_LABELS:
        winner = candidate_j
    elif raw_vote == candidate_i or raw_vote == candidate_j:
        winner = raw_vote
    else:
        # Unknown label: treat as tie rather than forcing an arbitrary loss.
        return 0.5

    return 1.0 if winner == consensus_candidate_id else 0.0
