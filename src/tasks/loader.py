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
    """Load and validate one task YAML file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    task_type = raw.get("task_type")
    instances = raw.get("instances", [])

    if not task_type:
        raise ValueError(f"task_type is missing in {path}")
    if not instances:
        raise ValueError(f"No task instances found in {path}")

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
    """Load all task YAML files and index by task_type/task_id."""
    catalog: dict[str, dict[str, TaskInstance]] = {}
    for path in sorted(task_dir.glob("*.yaml")):
        tasks = load_task_file(path)
        task_type = tasks[0].task_type
        if task_type in catalog:
            raise ValueError(f"Duplicate task_type '{task_type}' across task files")
        by_id = {t.id: t for t in tasks}
        if len(by_id) != len(tasks):
            raise ValueError(f"Duplicate task ids detected in {path}")
        catalog[task_type] = by_id
    if not catalog:
        raise ValueError(f"No task YAML files found in {task_dir}")
    return catalog


def task_catalog_stats(catalog: dict[str, dict[str, TaskInstance]]) -> dict[str, Any]:
    """Provide quick stats for validation output."""
    return {
        "task_types": len(catalog),
        "total_tasks": sum(len(v) for v in catalog.values()),
        "per_type": {key: len(val) for key, val in catalog.items()},
    }