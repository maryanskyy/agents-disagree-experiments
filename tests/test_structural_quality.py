"""Unit tests for structural quality metrics."""

from __future__ import annotations

from src.evaluation.structural_quality import (
    StructuralMetrics,
    compute_composite_score,
    compute_structural_metrics,
)


def test_compute_structural_metrics_basic() -> None:
    prompt = "Explain why consensus quality can improve with disagreement."
    text = (
        "Disagreement can improve consensus because independent perspectives expose hidden assumptions. "
        "However, disagreement must remain structured, since unbounded conflict harms synthesis. "
        "Therefore, teams should combine contrastive critique with explicit integration steps."
    )

    metrics = compute_structural_metrics(text=text, prompt=prompt, model_name="__disabled__")

    assert isinstance(metrics, StructuralMetrics)
    assert metrics.word_count > 0
    assert metrics.mtld > 0
    assert metrics.readability_fk_grade >= 0
    assert 0.0 <= metrics.coherence_mean <= 1.0
    assert 0.0 <= metrics.prompt_relevance <= 1.0
    assert metrics.connective_density > 0
    assert 0.0 <= metrics.repetition_rate <= 1.0


def test_composite_score_returns_float() -> None:
    metrics = StructuralMetrics(
        mtld=85.0,
        readability_fk_grade=12.5,
        coherence_mean=0.52,
        prompt_relevance=0.61,
        connective_density=1.25,
        word_count=430,
        repetition_rate=0.05,
    )
    score = compute_composite_score(metrics)
    assert isinstance(score, float)


def test_repetition_penalty_detects_looping() -> None:
    prompt = "Summarize the issue."
    repetitive = "alpha beta gamma delta " * 8
    metrics = compute_structural_metrics(text=repetitive, prompt=prompt, model_name="__disabled__")
    assert metrics.repetition_rate > 0.5
