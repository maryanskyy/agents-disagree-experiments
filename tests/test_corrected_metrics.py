import pytest

from src.evaluation.corrected_metrics import compute_corrected_metrics


def test_corrected_metrics_with_final_consensus_and_ties() -> None:
    panel_payload = {
        "bt_scores": {
            "final_consensus": 0.25,
            "agent_0": 0.45,
            "agent_1": 0.30,
        },
        "pairwise_records": [
            {
                "candidate_i": "final_consensus",
                "candidate_j": "agent_0",
                "majority_winner": "agent_0",
                "per_judge": {
                    "judge_a": "right",
                    "judge_b": "tie",
                },
            },
            {
                "candidate_i": "agent_1",
                "candidate_j": "final_consensus",
                "majority_winner": None,
                "per_judge": {
                    "judge_a": "tie",
                    "judge_b": "right",
                },
            },
        ],
    }

    corrected = compute_corrected_metrics(
        panel_payload=panel_payload,
        quality_score=0.25,
        consensus_candidate_id="final_consensus",
    )

    assert corrected["num_bt_candidates"] == 3
    assert corrected["normalized_bt_score"] == 0.75
    assert corrected["consensus_win_rate"] == 0.25
    assert corrected["consensus_vs_best_agent"] is False
    assert corrected["per_judge_consensus_win_rate"] == {
        "judge_a": 0.25,
        "judge_b": 0.75,
    }


def test_corrected_metrics_with_selected_agent_as_consensus() -> None:
    panel_payload = {
        "bt_scores": {
            "agent_0": 0.40,
            "agent_1": 0.35,
            "agent_2": 0.25,
        },
        "pairwise_records": [
            {
                "candidate_i": "agent_0",
                "candidate_j": "agent_1",
                "majority_winner": "agent_0",
                "per_judge": {"judge_a": "left"},
            },
            {
                "candidate_i": "agent_0",
                "candidate_j": "agent_2",
                "majority_winner": None,
                "per_judge": {"judge_a": "tie"},
            },
        ],
    }

    corrected = compute_corrected_metrics(
        panel_payload=panel_payload,
        quality_score=0.40,
        consensus_candidate_id="agent_0",
    )

    assert corrected["consensus_candidate_id"] == "agent_0"
    assert corrected["consensus_win_rate"] == 0.75
    assert corrected["normalized_bt_score"] == pytest.approx(1.2)
    assert corrected["consensus_vs_best_agent"] is True
    assert corrected["per_judge_consensus_win_rate"] == {"judge_a": 0.75}
