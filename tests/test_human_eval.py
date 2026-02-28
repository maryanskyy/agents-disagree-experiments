"""Tests for human evaluation sheet generation."""

from __future__ import annotations

from src.evaluation.human_eval import HumanEvalManager


def test_human_eval_sheet_creation(tmp_path) -> None:
    manager = HumanEvalManager(tmp_path / "human_eval", random_sample_rate=1.0, seed=42)
    payload = {
        "run_id": "run_abc",
        "block_id": "block4_quorum_paradox",
        "config": {"task_type": "analytical", "task_id": "a1"},
        "task": {
            "title": "Title",
            "prompt": "Prompt",
            "rubric": ["Correctness", "Reasoning"],
        },
        "outputs": [
            {"agent_id": "a1", "model_name": "m1", "text": "Output 1"},
            {"agent_id": "a2", "model_name": "m2", "text": "Output 2"},
        ],
        "evaluation": {
            "judge_panel": {
                "inter_rater_reliability": {"mean_cohen_kappa": 0.3},
                "pairwise_records": [
                    {"per_judge": {"j1": "left", "j2": "right"}}
                ],
            }
        },
    }

    decision = manager.decide(payload)
    assert decision.should_flag
    path = manager.create_sheet(payload, decision.reasons)
    assert path.exists()
