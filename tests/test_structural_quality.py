"""Unit tests for structural quality metrics."""

from __future__ import annotations

import math

from src.evaluation.structural_quality import (
    StructuralMetrics,
    _split_sentences,
    compute_structural_descriptor,
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


def test_structural_descriptor_returns_float() -> None:
    metrics = StructuralMetrics(
        mtld=85.0,
        readability_fk_grade=12.5,
        coherence_mean=0.52,
        prompt_relevance=0.61,
        connective_density=0.15,
        word_count=430,
        repetition_rate=0.05,
    )
    score = compute_structural_descriptor(metrics, task_type="analytical")
    assert isinstance(score, float)


def test_repetition_penalty_detects_looping() -> None:
    prompt = "Summarize the issue."
    repetitive = "alpha beta gamma delta " * 8
    metrics = compute_structural_metrics(text=repetitive, prompt=prompt, model_name="__disabled__")
    assert metrics.repetition_rate > 0.5


def test_sentence_splitter_handles_abbreviations_and_decimals() -> None:
    text = "Dr. Smith measured 3.14 units. Mr. Lee agreed. Finally, they documented it."
    sentences = _split_sentences(text)
    assert len(sentences) == 3


def test_single_sentence_coherence_is_nan() -> None:
    prompt = "Describe one takeaway."
    text = "This is only one sentence."
    metrics = compute_structural_metrics(text=text, prompt=prompt, model_name="__disabled__")
    assert math.isnan(metrics.coherence_mean)


def test_structural_descriptor_skips_nan_metrics() -> None:
    metrics = StructuralMetrics(
        mtld=100.0,
        readability_fk_grade=12.0,
        coherence_mean=float("nan"),
        prompt_relevance=0.5,
        connective_density=0.1,
        word_count=350,
        repetition_rate=0.03,
    )
    descriptor = compute_structural_descriptor(metrics, task_type="analytical")
    assert math.isfinite(descriptor)


def test_task_specific_norms_change_descriptor() -> None:
    metrics = StructuralMetrics(
        mtld=140.0,
        readability_fk_grade=13.0,
        coherence_mean=0.30,
        prompt_relevance=0.55,
        connective_density=0.12,
        word_count=700,
        repetition_rate=0.04,
    )
    analytical = compute_structural_descriptor(metrics, task_type="analytical")
    creative = compute_structural_descriptor(metrics, task_type="creative")
    assert analytical != creative
