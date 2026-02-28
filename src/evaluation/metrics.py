"""Evaluation metrics for analytical and creative outputs."""

from __future__ import annotations

import math
from difflib import SequenceMatcher
from typing import Iterable


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for text comparisons."""
    return " ".join(text.lower().split())


def token_set(text: str) -> set[str]:
    """Convert text to lower-cased token set."""
    return set(normalize_text(text).split())


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity on token sets."""
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def sequence_similarity(a: str, b: str) -> float:
    """Character-level normalized similarity."""
    return SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()


def rouge_l_approx(reference: str, candidate: str) -> float:
    """Approximate ROUGE-L via sequence matcher ratio."""
    return sequence_similarity(reference, candidate)


def evaluate_analytical(reference: str, candidate: str) -> float:
    """Heuristic analytical score in [0, 1]."""
    overlap = jaccard_similarity(reference, candidate)
    seq = sequence_similarity(reference, candidate)
    length_penalty = min(1.0, len(candidate.split()) / max(1, len(reference.split())))
    return max(0.0, min(1.0, 0.5 * overlap + 0.4 * seq + 0.1 * length_penalty))


def evaluate_creative(candidate: str) -> float:
    """Heuristic creative quality proxy using lexical diversity and structure."""
    tokens = normalize_text(candidate).split()
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    sentence_count = max(1, candidate.count(".") + candidate.count("!") + candidate.count("?"))
    avg_len = len(tokens) / sentence_count
    rhythm = 1.0 - min(1.0, abs(avg_len - 18.0) / 18.0)
    score = 0.65 * unique_ratio + 0.35 * rhythm
    return max(0.0, min(1.0, score))


def mean(values: Iterable[float]) -> float:
    """Safe arithmetic mean."""
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def entropy_from_probs(probs: list[float]) -> float:
    """Shannon entropy with natural log, normalized to [0,1] when possible."""
    filtered = [p for p in probs if p > 0]
    if not filtered:
        return 0.0
    raw = -sum(p * math.log(p) for p in filtered)
    if len(filtered) == 1:
        return 0.0
    return raw / math.log(len(filtered))