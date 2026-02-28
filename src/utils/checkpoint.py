"""Crash-safe checkpointing utilities based on atomic filesystem writes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
from typing import Any


@dataclass(slots=True)
class ProgressSnapshot:
    """Progress metadata persisted to progress.json."""

    timestamp_utc: str
    completed_runs: int
    pending_runs: int
    failed_runs: int
    eta_seconds: float | None
    estimated_total_cost_usd: float
    estimated_cost_by_model: dict[str, float]


class CheckpointManager:
    """Filesystem checkpoint manager with atomic writes and resume support."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def run_result_path(self, block_id: str, run_id: str) -> Path:
        """Return canonical path for a run output JSON file."""
        return self.results_dir / block_id / f"{run_id}.json"

    def save_result(self, block_id: str, run_id: str, payload: dict[str, Any]) -> Path:
        """Persist run result atomically as JSON."""
        final_path = self.run_result_path(block_id=block_id, run_id=run_id)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(final_path, payload)
        return final_path

    def save_manifest_snapshot(self, payload: dict[str, Any]) -> Path:
        """Write manifest snapshot into results directory for reproducibility."""
        path = self.results_dir / "manifest_snapshot.json"
        self._atomic_write_json(path, payload)
        return path

    def update_progress(self, snapshot: ProgressSnapshot) -> Path:
        """Atomically update progress.json."""
        path = self.results_dir / "progress.json"
        self._atomic_write_json(path, {
            "timestamp_utc": snapshot.timestamp_utc,
            "completed_runs": snapshot.completed_runs,
            "pending_runs": snapshot.pending_runs,
            "failed_runs": snapshot.failed_runs,
            "eta_seconds": snapshot.eta_seconds,
            "estimated_total_cost_usd": snapshot.estimated_total_cost_usd,
            "estimated_cost_by_model": snapshot.estimated_cost_by_model,
        })
        return path

    def load_progress(self) -> dict[str, Any]:
        """Load progress snapshot if present."""
        path = self.results_dir / "progress.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def load_completed_ids(self) -> set[str]:
        """Scan results tree and return completed run IDs."""
        completed: set[str] = set()
        if not self.results_dir.exists():
            return completed

        for path in self.results_dir.rglob("*.json"):
            if path.name in {"progress.json", "manifest_snapshot.json"}:
                continue
            completed.add(path.stem)
        return completed

    def _atomic_write_json(self, target_path: Path, payload: dict[str, Any]) -> None:
        serialized = json.dumps(payload, indent=2, sort_keys=False)
        with self._lock:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=str(target_path.parent),
                suffix=".tmp",
            ) as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, target_path)


def utc_now_iso() -> str:
    """Current UTC timestamp in ISO-8601 format."""
    return datetime.now(tz=timezone.utc).isoformat()