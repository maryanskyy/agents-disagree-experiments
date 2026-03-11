"""Core text metrics and dry-run-only heuristic quality proxies.

This module contains two categories of functions:
1) Utility metrics used by production code (normalization, entropy, lexical stats)
2) Development-only heuristic quality proxies used strictly in dry-run mode

The heuristic functions are intentionally preserved for mocked/local smoke tests only.
They are NOT valid research quality metrics and must never be used for production
experiment conclusions.
"""

from __future__ import annotations

import math
from difflib import SequenceMatcher
import warnings
from typing import Iterable


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for robust text comparisons."""
    return " ".join(text.lower().split())


def token_set(text: str) -> set[str]:
    """Convert text to a normalized token set."""
    return set(normalize_text(text).split())


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity on normalized token sets."""
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def sequence_similarity(a: str, b: str) -> float:
    """Character-level normalized sequence similarity."""
    return SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()


def rouge_l_approx(reference: str, candidate: str) -> float:
    """Approximate ROUGE-L via sequence similarity.

    This is intentionally lightweight and only used for diagnostics.
    """
    return sequence_similarity(reference, candidate)


def mean(values: Iterable[float]) -> float:
    """Safe arithmetic mean."""
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def entropy_from_probs(probs: list[float]) -> float:
    """Normalized Shannon entropy in [0, 1] when possible."""
    filtered = [p for p in probs if p > 0]
    if not filtered:
        return 0.0
    raw = -sum(p * math.log(p) for p in filtered)
    if len(filtered) == 1:
        return 0.0
    return raw / math.log(len(filtered))


def lexical_diversity(text: str) -> float:
    """Type-token ratio for one text."""
    tokens = normalize_text(text).split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def dev_heuristic_analytical(reference_prompt: str, candidate: str) -> float:
    """DEV-ONLY proxy for analytical quality.

    What it measures:
      - Prompt/candidate lexical overlap
      - Prompt/candidate sequence similarity
      - Relative length ratio

    Why this exists:
      - Enables deterministic dry-run smoke tests without paid API calls.

    Why it is NOT a valid quality metric:
      - It can reward prompt echoing/parroting.
      - It does not assess factual correctness, reasoning validity, or evidence quality.
    """
    overlap = jaccard_similarity(reference_prompt, candidate)
    seq = sequence_similarity(reference_prompt, candidate)
    length_penalty = min(1.0, len(candidate.split()) / max(1, len(reference_prompt.split())))
    return max(0.0, min(1.0, 0.5 * overlap + 0.4 * seq + 0.1 * length_penalty))


def dev_heuristic_creative(candidate: str) -> float:
    """DEV-ONLY proxy for creative quality.

    What it measures:
      - Lexical diversity
      - Approximate sentence-length rhythm

    Why this exists:
      - Cheap deterministic signal in dry-run testing.

    Why it is NOT a valid creativity metric:
      - It cannot evaluate originality, narrative craft, emotional resonance, or style.
    """
    tokens = normalize_text(candidate).split()
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    sentence_count = max(1, candidate.count(".") + candidate.count("!") + candidate.count("?"))
    avg_len = len(tokens) / sentence_count
    rhythm = 1.0 - min(1.0, abs(avg_len - 18.0) / 18.0)
    score = 0.65 * unique_ratio + 0.35 * rhythm
    return max(0.0, min(1.0, score))


def evaluate_analytical(reference: str, candidate: str) -> float:
    """Backward-compatible alias for the dry-run analytical heuristic.

    Production code must not call this function for research evaluation.
    """
    warnings.warn(
        "evaluate_analytical is a dry-run heuristic and must not be used as a production quality metric.",
        RuntimeWarning,
        stacklevel=2,
    )
    return dev_heuristic_analytical(reference, candidate)


def evaluate_creative(candidate: str) -> float:
    """Backward-compatible alias for the dry-run creative heuristic.

    Production code must not call this function for research evaluation.
    """
    warnings.warn(
        "evaluate_creative is a dry-run heuristic and must not be used as a production quality metric.",
        RuntimeWarning,
        stacklevel=2,
    )
    return dev_heuristic_creative(candidate)
