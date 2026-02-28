"""Local structural quality metrics for generated text.

This module provides deterministic, non-LLM quality signals that complement
LLM-as-judge scoring.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
import logging
import math
import re
from typing import Iterable

import numpy as np

# Prevent transformers from importing TensorFlow/JAX in environments where they are incompatible.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

try:  # Optional dependency (requested in requirements)
    from textstat import textstat
except Exception:  # pragma: no cover - optional import
    textstat = None

try:  # Optional dependency (requested in requirements)
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional import
    SentenceTransformer = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

# Constants are named (no magic numbers) for reproducibility and readability.
MTLD_TTR_THRESHOLD = 0.72
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DISABLED_EMBEDDING_MODELS = {"", "none", "disabled", "__disabled__"}
MIN_STD = 1e-6
NGRAM_SIZE = 4

# Analytical-task normalization anchors used for composite z-scoring.
# Values are intentionally conservative priors and can be re-calibrated from data.
ANALYTICAL_NORMS: dict[str, tuple[float, float, str]] = {
    "mtld": (80.0, 25.0, "higher"),
    "readability_fk_grade": (13.0, 3.0, "target"),
    "coherence_mean": (0.45, 0.20, "higher"),
    "prompt_relevance": (0.55, 0.20, "higher"),
    "connective_density": (1.20, 0.60, "higher"),
    "word_count": (450.0, 220.0, "target"),
    "repetition_rate": (0.08, 0.08, "lower"),
}

CAUSAL_CONNECTIVES = {"therefore", "consequently", "because", "since"}
CONTRASTIVE_CONNECTIVES = {"however", "although", "nevertheless", "but"}
ADDITIVE_CONNECTIVES = {"furthermore", "moreover", "additionally"}
ALL_CONNECTIVES = CAUSAL_CONNECTIVES | CONTRASTIVE_CONNECTIVES | ADDITIVE_CONNECTIVES

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

_MODEL_CACHE: dict[str, SentenceTransformer | None] = {}
_EMBEDDING_CACHE: dict[tuple[str, str], np.ndarray] = {}


@dataclass(slots=True)
class StructuralMetrics:
    mtld: float
    readability_fk_grade: float
    coherence_mean: float
    prompt_relevance: float
    connective_density: float
    word_count: int
    repetition_rate: float


def _tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    parts = [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(stripped) if segment.strip()]
    return parts if parts else [stripped]


def _estimate_syllables(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0

    # Simple fallback syllable heuristic for when textstat is unavailable.
    vowel_groups = re.findall(r"[aeiouy]+", cleaned)
    syllables = len(vowel_groups)
    if cleaned.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def _fk_grade_fallback(text: str) -> float:
    words = _tokenize_words(text)
    sentences = _split_sentences(text)

    word_count = len(words)
    sentence_count = max(1, len(sentences))
    if word_count == 0:
        return 0.0

    syllable_count = sum(_estimate_syllables(word) for word in words)
    return float(0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59)


def _compute_mtld_direction(tokens: list[str], *, threshold: float = MTLD_TTR_THRESHOLD) -> float:
    if not tokens:
        return 0.0

    factors = 0.0
    token_count = 0
    types: set[str] = set()

    for token in tokens:
        token_count += 1
        types.add(token)
        ttr = len(types) / token_count
        if ttr <= threshold:
            factors += 1.0
            token_count = 0
            types.clear()

    if token_count > 0:
        ttr = len(types) / token_count
        partial = (1.0 - ttr) / max(1.0 - threshold, MIN_STD)
        factors += max(0.0, partial)

    if factors <= MIN_STD:
        return float(len(tokens))
    return float(len(tokens) / factors)


def _compute_mtld(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    if len(tokens) < 10:
        # MTLD is unstable on short text; return a conservative scaled TTR proxy.
        return float((len(set(tokens)) / max(1, len(tokens))) * len(tokens))

    forward = _compute_mtld_direction(tokens)
    backward = _compute_mtld_direction(list(reversed(tokens)))
    return float((forward + backward) / 2.0)


def _load_embedding_model(model_name: str) -> SentenceTransformer | None:
    normalized_name = model_name.strip()
    if normalized_name.lower() in DISABLED_EMBEDDING_MODELS:
        return None

    if normalized_name in _MODEL_CACHE:
        return _MODEL_CACHE[normalized_name]

    if SentenceTransformer is None:
        LOGGER.warning("sentence-transformers unavailable; embedding-based metrics will use lexical fallback.")
        _MODEL_CACHE[normalized_name] = None
        return None

    try:
        model = SentenceTransformer(normalized_name)
    except Exception as exc:  # pragma: no cover - defensive path
        LOGGER.warning("Failed to load embedding model '%s': %s", normalized_name, exc)
        model = None

    _MODEL_CACHE[normalized_name] = model
    return model


def _embed_texts(texts: Iterable[str], model_name: str) -> list[np.ndarray] | None:
    model = _load_embedding_model(model_name)
    texts_list = [text.strip() for text in texts]
    if model is None:
        return None

    missing_texts = [text for text in texts_list if (model_name, text) not in _EMBEDDING_CACHE]
    if missing_texts:
        encoded = model.encode(
            missing_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for text, vector in zip(missing_texts, encoded):
            _EMBEDDING_CACHE[(model_name, text)] = np.asarray(vector, dtype=float)

    return [_EMBEDDING_CACHE[(model_name, text)] for text in texts_list]


def _adjacent_coherence(sentences: list[str], model_name: str) -> float:
    if not sentences:
        return 0.0
    if len(sentences) == 1:
        return 1.0

    embeddings = _embed_texts(sentences, model_name)
    if embeddings is not None:
        sims: list[float] = []
        for idx in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[idx], embeddings[idx + 1]))
            sims.append(sim)
        return float(np.mean(sims)) if sims else 0.0

    # Lexical-overlap fallback if embeddings are unavailable.
    overlaps: list[float] = []
    for idx in range(len(sentences) - 1):
        left = set(_tokenize_words(sentences[idx]))
        right = set(_tokenize_words(sentences[idx + 1]))
        union = left | right
        if not union:
            overlaps.append(0.0)
        else:
            overlaps.append(len(left & right) / len(union))
    return float(np.mean(overlaps)) if overlaps else 0.0


def _prompt_relevance(prompt: str, text: str, model_name: str) -> float:
    prompt = prompt.strip()
    text = text.strip()
    if not prompt or not text:
        return 0.0

    embeddings = _embed_texts([prompt, text], model_name)
    if embeddings is not None:
        return float(np.dot(embeddings[0], embeddings[1]))

    # Lexical fallback when embeddings are not available.
    prompt_tokens = set(_tokenize_words(prompt))
    output_tokens = set(_tokenize_words(text))
    if not prompt_tokens:
        return 0.0
    return float(len(prompt_tokens & output_tokens) / len(prompt_tokens))


def _connective_density(tokens: list[str], sentence_count: int) -> float:
    if sentence_count <= 0:
        return 0.0
    connective_hits = sum(1 for token in tokens if token in ALL_CONNECTIVES)
    return float(connective_hits / sentence_count)


def _repetition_rate(tokens: list[str], ngram_size: int = NGRAM_SIZE) -> float:
    if len(tokens) < ngram_size:
        return 0.0

    ngrams = [tuple(tokens[idx : idx + ngram_size]) for idx in range(len(tokens) - ngram_size + 1)]
    counts = Counter(ngrams)
    repeated_occurrences = sum(count for count in counts.values() if count > 1)
    return float(repeated_occurrences / len(ngrams)) if ngrams else 0.0


def _safe_float(value: float) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def compute_structural_metrics(
    text: str,
    prompt: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> StructuralMetrics:
    """Compute local structural quality metrics for a single text."""
    clean_text = text or ""
    tokens = _tokenize_words(clean_text)
    sentences = _split_sentences(clean_text)

    mtld_value = _compute_mtld(tokens)

    if textstat is not None:
        try:
            readability = float(textstat.flesch_kincaid_grade(clean_text))
        except Exception:  # pragma: no cover - defensive path
            readability = _fk_grade_fallback(clean_text)
    else:
        readability = _fk_grade_fallback(clean_text)

    coherence = _adjacent_coherence(sentences, model_name)
    relevance = _prompt_relevance(prompt or "", clean_text, model_name)
    connective_density = _connective_density(tokens, len(sentences))
    repetition = _repetition_rate(tokens)

    return StructuralMetrics(
        mtld=_safe_float(mtld_value),
        readability_fk_grade=_safe_float(readability),
        coherence_mean=_safe_float(coherence),
        prompt_relevance=_safe_float(relevance),
        connective_density=_safe_float(connective_density),
        word_count=int(len(tokens)),
        repetition_rate=_safe_float(repetition),
    )


def compute_composite_score(metrics: StructuralMetrics) -> float:
    """Compute a z-normalized composite score using analytical-task priors."""
    values = {
        "mtld": float(metrics.mtld),
        "readability_fk_grade": float(metrics.readability_fk_grade),
        "coherence_mean": float(metrics.coherence_mean),
        "prompt_relevance": float(metrics.prompt_relevance),
        "connective_density": float(metrics.connective_density),
        "word_count": float(metrics.word_count),
        "repetition_rate": float(metrics.repetition_rate),
    }

    z_scores: list[float] = []
    for metric_name, value in values.items():
        mean_value, std_value, mode = ANALYTICAL_NORMS[metric_name]
        std_value = max(std_value, MIN_STD)

        if mode == "higher":
            z = (value - mean_value) / std_value
        elif mode == "lower":
            z = (mean_value - value) / std_value
        elif mode == "target":
            z = -abs(value - mean_value) / std_value
        else:  # pragma: no cover - defensive path
            raise ValueError(f"Unsupported normalization mode '{mode}' for metric '{metric_name}'")

        # Guard against extreme outliers dominating the composite.
        z_scores.append(float(max(-3.0, min(3.0, z))))

    return float(np.mean(z_scores)) if z_scores else 0.0
