"""Task loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class TaskInstance:
    """Single task definition loaded from YAML."""

    id: str
    task_type: str
    title: str
    prompt: str
    rubric: list[str]


def load_task_file(path: Path) -> list[TaskInstance]:
    """Load and validate a task YAML file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    task_type = raw.get("task_type")
    instances = raw.get("instances", [])

    if not task_type:
        raise ValueError(f"task_type is missing in {path}")
    if len(instances) != 8:
        raise ValueError(f"Expected 8 tasks in {path}, found {len(instances)}")

    loaded: list[TaskInstance] = []
    for item in instances:
        loaded.append(
            TaskInstance(
                id=str(item["id"]),
                task_type=str(task_type),
                title=str(item["title"]),
                prompt=str(item["prompt"]),
                rubric=[str(v) for v in item.get("rubric", [])],
            )
        )
    return loaded


def load_all_tasks(task_dir: Path) -> dict[str, dict[str, TaskInstance]]:
    """Load analytical + creative tasks and index by id."""
    catalog: dict[str, dict[str, TaskInstance]] = {}
    for name in ("analytical.yaml", "creative.yaml"):
        tasks = load_task_file(task_dir / name)
        task_type = tasks[0].task_type
        catalog[task_type] = {t.id: t for t in tasks}
    return catalog


def task_catalog_stats(catalog: dict[str, dict[str, TaskInstance]]) -> dict[str, Any]:
    """Provide quick stats for validation output."""
    return {
        "task_types": len(catalog),
        "total_tasks": sum(len(v) for v in catalog.values()),
        "per_type": {key: len(val) for key, val in catalog.items()},
    }