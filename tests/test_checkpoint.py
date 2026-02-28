"""Tests for crash-safe checkpoint behavior."""

from __future__ import annotations

from pathlib import Path

from src.utils.checkpoint import CheckpointManager, ProgressSnapshot


def test_checkpoint_save_and_resume(tmp_path: Path) -> None:
    """Saved run files should be discovered by resume scan."""
    manager = CheckpointManager(tmp_path)

    manager.save_result("block0", "run_a", {"status": "ok", "value": 1})
    manager.save_result("block1", "run_b", {"status": "ok", "value": 2})

    completed = manager.load_completed_ids()
    assert "run_a" in completed
    assert "run_b" in completed


def test_progress_atomic_write(tmp_path: Path) -> None:
    """Progress file should be written and readable."""
    manager = CheckpointManager(tmp_path)
    snapshot = ProgressSnapshot(
        timestamp_utc="2026-01-01T00:00:00Z",
        completed_runs=10,
        pending_runs=5,
        failed_runs=1,
        eta_seconds=12.5,
        estimated_total_cost_usd=1.23,
        estimated_cost_by_model={"m": 1.23},
    )
    manager.update_progress(snapshot)
    loaded = manager.load_progress()
    assert loaded["completed_runs"] == 10
    assert loaded["estimated_total_cost_usd"] == 1.23