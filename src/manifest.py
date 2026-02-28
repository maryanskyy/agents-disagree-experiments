"""Manifest generation and serialization utilities for experiment runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import yaml


@dataclass(slots=True)
class RunSpec:
    """Single experiment run specification."""

    id: str
    block_id: str
    task_type: str
    task_id: str
    topology: str
    consensus: str
    agent_count: int
    disagreement_level: int
    temperature: float
    prompt_strategy: str
    repetition: int
    model_assignment: list[str]
    quality_threshold: float | None = None
    posthoc_quality_thresholds: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentManifest:
    """Full collection of experiment runs."""

    generated_at: str
    seed: int
    runs: list[RunSpec]

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "generated_at": self.generated_at,
            "seed": self.seed,
            "run_count": len(self.runs),
            "runs": [asdict(run) for run in self.runs],
        }

    def save(self, path: Path) -> None:
        """Write manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ExperimentManifest":
        """Load manifest from JSON."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        runs = [RunSpec(**item) for item in raw["runs"]]
        return cls(generated_at=raw["generated_at"], seed=raw["seed"], runs=runs)


def _hash_id(parts: list[str]) -> str:
    joined = "::".join(parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"run_{digest}"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_tasks(task_file: Path) -> list[dict[str, Any]]:
    payload = _load_yaml(task_file)
    instances = payload.get("instances", [])
    if not instances:
        raise ValueError(f"No task instances found in {task_file}")
    return instances


def _expand_assignment(template_name: str, agent_count: int, templates: dict[str, dict[str, Any]]) -> list[str]:
    if template_name not in templates:
        return [template_name] * agent_count

    base_assignment = templates[template_name].get("assignment", [])
    if not base_assignment:
        raise ValueError(f"Template '{template_name}' has empty assignment.")

    if template_name == "paradox_strong_weak":
        strong = base_assignment[0]
        weak = base_assignment[-1]
        return [strong] + [weak] * max(0, agent_count - 1)

    if len(base_assignment) == 1:
        return [base_assignment[0]] * agent_count

    return [base_assignment[idx % len(base_assignment)] for idx in range(agent_count)]


def generate_manifest(
    matrix_path: Path,
    analytical_tasks_path: Path,
    creative_tasks_path: Path,
    seed: int | None = None,
) -> ExperimentManifest:
    """Generate deterministic run manifest from matrix config and task catalogs."""
    matrix = _load_yaml(matrix_path)
    seed_value = seed if seed is not None else int(matrix.get("seed", 42))
    rng = random.Random(seed_value)

    analytical_tasks = _load_tasks(analytical_tasks_path)
    creative_tasks = _load_tasks(creative_tasks_path)
    tasks_by_type = {
        "analytical": analytical_tasks,
        "creative": creative_tasks,
    }

    disagreement_cfg = matrix["disagreement_levels"]
    templates = matrix.get("agent_group_templates", {})

    runs: list[RunSpec] = []
    for block in matrix["blocks"]:
        block_id = block["id"]
        posthoc_thresholds = [float(v) for v in block.get("posthoc_quality_thresholds", [])]

        for task_type in block["task_types"]:
            task_instances = tasks_by_type[task_type]
            for task in task_instances:
                for agent_count in block["agent_counts"]:
                    for topology in block["topologies"]:
                        for consensus in block["consensus"]:
                            for disagreement_level in block["disagreement_levels"]:
                                level_config = disagreement_cfg.get(disagreement_level) or disagreement_cfg.get(str(disagreement_level))
                                if level_config is None:
                                    raise KeyError(f"Missing disagreement config for level {disagreement_level}")
                                for model_spec in block["models"]:
                                    assignment = _expand_assignment(model_spec, agent_count, templates)
                                    for rep in range(1, int(block["repetitions"]) + 1):
                                        part_list = [
                                            block_id,
                                            task_type,
                                            task["id"],
                                            topology,
                                            consensus,
                                            str(agent_count),
                                            str(disagreement_level),
                                            ",".join(assignment),
                                            str(rep),
                                        ]
                                        run_id = _hash_id(part_list)
                                        runs.append(
                                            RunSpec(
                                                id=run_id,
                                                block_id=block_id,
                                                task_type=task_type,
                                                task_id=task["id"],
                                                topology=topology,
                                                consensus=consensus,
                                                agent_count=int(agent_count),
                                                disagreement_level=int(disagreement_level),
                                                temperature=float(level_config["temperature"]),
                                                prompt_strategy=str(level_config["prompt_strategy"]),
                                                repetition=rep,
                                                model_assignment=assignment,
                                                quality_threshold=None,
                                                posthoc_quality_thresholds=posthoc_thresholds,
                                                metadata={
                                                    "task_title": task.get("title", ""),
                                                    "prompt_hash": hashlib.sha1(task["prompt"].encode("utf-8")).hexdigest()[:10],
                                                },
                                            )
                                        )

    rng.shuffle(runs)
    generated_at = datetime.now(tz=timezone.utc).isoformat()
    return ExperimentManifest(generated_at=generated_at, seed=seed_value, runs=runs)
