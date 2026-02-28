"""Disagreement metrics for multi-agent outputs.

Primary disagreement metric uses semantic similarity from sentence embeddings:
    disagreement_rate = 1 - mean_pairwise_cosine_similarity

Secondary diagnostic metrics are included for analysis robustness:
- vote_entropy: entropy over semantic clusters
- lexical_diversity: mean type-token ratio over outputs
- pairwise_similarity_lexical: mean pairwise Jaccard similarity
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations
import math
from typing import Iterable

import numpy as np

from .metrics import entropy_from_probs, jaccard_similarity, lexical_diversity as text_lexical_diversity, mean, normalize_text


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_embedder(model_name: str):
    """Load and cache sentence-transformer model lazily.

    Import is deferred so lightweight unit tests and dry-run mode do not require
    the dependency unless semantic metrics are actually computed.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _pairwise_indices(n: int) -> Iterable[tuple[int, int]]:
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def pairwise_similarity_lexical(outputs: list[str]) -> float:
    """Average pairwise Jaccard similarity across outputs."""
    if len(outputs) < 2:
        return 1.0
    sims = [jaccard_similarity(a, b) for a, b in combinations(outputs, 2)]
    return mean(sims)


def semantic_pairwise_similarity(outputs: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> float:
    """Average pairwise cosine similarity using local sentence embeddings.

    This metric captures semantic agreement better than lexical overlap.
    If embeddings are unavailable, falls back to lexical similarity to keep the
    pipeline operational (with reduced semantic fidelity).
    """
    if len(outputs) < 2:
        return 1.0

    normalized = [normalize_text(text) for text in outputs]
    try:
        embedder = _load_embedder(model_name)
        embeddings = embedder.encode(normalized, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=float)
        sims = [float(np.dot(embeddings[i], embeddings[j])) for i, j in _pairwise_indices(len(outputs))]
        return float(mean(sims)) if sims else 1.0
    except Exception:
        # Conservative fallback for environments lacking sentence-transformers.
        return pairwise_similarity_lexical(outputs)


def semantic_disagreement_rate(outputs: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> float:
    """Semantic disagreement rate = 1 - mean pairwise semantic similarity."""
    similarity = semantic_pairwise_similarity(outputs, model_name=model_name)
    return float(max(0.0, min(1.0, 1.0 - similarity)))


def vote_entropy(outputs: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL, threshold: float = 0.85) -> float:
    """Entropy over semantically clustered outputs.

    Outputs are clustered by embedding cosine similarity. High entropy indicates
    diffuse support across distinct semantic solutions.
    """
    if not outputs:
        return 0.0
    if len(outputs) == 1:
        return 0.0

    normalized = [normalize_text(text) for text in outputs]
    try:
        embedder = _load_embedder(model_name)
        embeddings = embedder.encode(normalized, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=float)

        parent = list(range(len(outputs)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j in _pairwise_indices(len(outputs)):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= threshold:
                union(i, j)

        roots = [find(i) for i in range(len(outputs))]
        counts = Counter(roots)
    except Exception:
        # Fallback: exact-normalized text clusters.
        counts = Counter(normalized)

    total = sum(counts.values())
    probs = [count / total for count in counts.values()]
    return entropy_from_probs(probs)


def mean_lexical_diversity(outputs: list[str]) -> float:
    """Average type-token ratio across outputs."""
    if not outputs:
        return 0.0
    return mean(text_lexical_diversity(out) for out in outputs)


def disagreement_summary(outputs: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> dict[str, float]:
    """Bundle of semantic-first disagreement metrics.

    Keys:
      - semantic_pairwise_similarity: primary semantic agreement metric
      - disagreement_rate: primary disagreement metric (1 - semantic similarity)
      - vote_entropy: entropy over semantic clusters
      - lexical_diversity: mean lexical diversity (secondary)
      - pairwise_similarity_lexical: lexical overlap baseline (secondary)
    """
    semantic_similarity = semantic_pairwise_similarity(outputs, model_name=model_name)
    return {
        "semantic_pairwise_similarity": semantic_similarity,
        "disagreement_rate": float(max(0.0, min(1.0, 1.0 - semantic_similarity))),
        "vote_entropy": vote_entropy(outputs, model_name=model_name),
        "lexical_diversity": mean_lexical_diversity(outputs),
        "pairwise_similarity_lexical": pairwise_similarity_lexical(outputs),
    }
