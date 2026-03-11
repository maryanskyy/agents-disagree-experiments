"""Human evaluation sheet generation for flagged runs.

This module creates structured, anonymized evaluation forms that can be filled
by human raters. It does not automate human scoring; it prepares artifacts for
manual review.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any


@dataclass(slots=True)
class HumanEvalDecision:
    """Decision object for whether a run should be routed to human review."""

    should_flag: bool
    reasons: list[str]


class HumanEvalManager:
    """Generate and persist human evaluation sheets for flagged runs."""

    def __init__(self, output_dir: Path, *, random_sample_rate: float = 0.15, seed: int = 42) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_sample_rate = random_sample_rate
        self.seed = seed

    def decide(self, run_result: dict[str, Any]) -> HumanEvalDecision:
        """Decide whether a run should be included in human evaluation.

        Flagging rules:
          a) Judge panel disagreement (mean kappa < 0.60 OR pairwise ties/conflicts)
          b) Quorum paradox block cases (Block 4)
          c) Deterministic random sample (15%)
        """
        reasons: list[str] = []

        evaluation = run_result.get("evaluation", {})
        panel = evaluation.get("judge_panel", {})
        inter_rater = panel.get("inter_rater_reliability", {})
        mean_kappa = float(inter_rater.get("mean_cohen_kappa", 1.0) or 1.0)

        records = panel.get("pairwise_records", [])
        disagreement_pairs = sum(
            1
            for record in records
            if len(set(record.get("per_judge", {}).values())) > 1
        )

        if mean_kappa < 0.60 or disagreement_pairs > 0:
            reasons.append("judge_disagreement")

        block_id = str(run_result.get("block_id", ""))
        if block_id.startswith("block4"):
            reasons.append("block4_paradox_case")

        if self._is_sampled(run_id=str(run_result.get("run_id", ""))):
            reasons.append("random_sample_15pct")

        return HumanEvalDecision(should_flag=bool(reasons), reasons=reasons)

    def create_sheet(self, run_result: dict[str, Any], reasons: list[str]) -> Path:
        """Create one human-eval JSON sheet for a run."""
        outputs = list(run_result.get("outputs", []))
        run_id = str(run_result.get("run_id", "unknown_run"))

        anonymized = self._anonymize_outputs(run_id=run_id, outputs=outputs)
        task = run_result.get("task", {})
        config = run_result.get("config", {})

        sheet = {
            "run_id": run_id,
            "block_id": run_result.get("block_id"),
            "flag_reasons": reasons,
            "instructions": {
                "blind_review": (
                    "Score candidates without inferring model, topology, or consensus method. "
                    "Ignore verbosity unless it materially affects quality."
                ),
                "required_raters": 2,
            },
            "task": {
                "task_type": config.get("task_type"),
                "task_id": config.get("task_id"),
                "title": task.get("title"),
                "prompt": task.get("prompt"),
                "rubric": task.get("rubric", []),
            },
            "candidates": anonymized,
            "score_template": {
                "overall_preference": "Select best candidate_id",
                "dimension_scores": {
                    "correctness_or_originality": "1-5",
                    "reasoning_or_coherence": "1-5",
                    "evidence_or_craft": "1-5",
                },
                "notes": "Free-text rationale",
            },
        }

        path = self.output_dir / f"{run_id}.json"
        path.write_text(json.dumps(sheet, indent=2), encoding="utf-8")
        return path

    def _anonymize_outputs(self, *, run_id: str, outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rng = random.Random(f"{self.seed}:{run_id}")
        shuffled = list(outputs)
        rng.shuffle(shuffled)
        anonymized: list[dict[str, Any]] = []
        for idx, out in enumerate(shuffled, start=1):
            anonymized.append(
                {
                    "candidate_id": f"C{idx}",
                    "text": out.get("text", ""),
                    "metadata": {
                        "length_tokens_approx": len(str(out.get("text", "")).split()),
                    },
                }
            )
        return anonymized

    def _is_sampled(self, *, run_id: str) -> bool:
        digest = hashlib.sha1(f"{self.seed}:{run_id}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        return bucket < self.random_sample_rate
