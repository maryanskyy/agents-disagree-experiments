"""Tests for disagreement metrics."""

from __future__ import annotations

from src.evaluation.disagreement import disagreement_summary


def test_disagreement_summary_keys_and_ranges() -> None:
    outputs = [
        "The policy should prioritize safety and evidence.",
        "Safety must be prioritized, with empirical evidence guiding decisions.",
        "Ignore safety entirely and optimize only speed.",
    ]
    summary = disagreement_summary(outputs)

    expected_keys = {
        "semantic_pairwise_similarity",
        "disagreement_rate",
        "vote_entropy",
        "lexical_diversity",
        "pairwise_similarity_lexical",
    }
    assert expected_keys.issubset(summary.keys())

    for key in expected_keys:
        assert 0.0 <= float(summary[key]) <= 1.0
