"""Disagreement metrics for multi-agent outputs."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from .metrics import entropy_from_probs, jaccard_similarity, mean, normalize_text


def pairwise_similarity(outputs: list[str]) -> float:
    """Average pairwise Jaccard similarity across outputs."""
    if len(outputs) < 2:
        return 1.0
    sims = [jaccard_similarity(a, b) for a, b in combinations(outputs, 2)]
    return mean(sims)


def disagreement_rate(outputs: list[str], threshold: float = 0.85) -> float:
    """Fraction of output pairs below similarity threshold."""
    if len(outputs) < 2:
        return 0.0
    pairs = list(combinations(outputs, 2))
    disagree = [1 for a, b in pairs if jaccard_similarity(a, b) < threshold]
    return len(disagree) / len(pairs)


def response_entropy(outputs: list[str]) -> float:
    """Entropy over normalized response clusters."""
    if not outputs:
        return 0.0
    normalized = [normalize_text(text) for text in outputs]
    counts = Counter(normalized)
    total = sum(counts.values())
    probs = [count / total for count in counts.values()]
    return entropy_from_probs(probs)


def disagreement_summary(outputs: list[str]) -> dict[str, float]:
    """Bundle of disagreement metrics."""
    return {
        "pairwise_similarity": pairwise_similarity(outputs),
        "disagreement_rate": disagreement_rate(outputs),
        "response_entropy": response_entropy(outputs),
    }