"""Tests for phased manifest selection."""

from __future__ import annotations

from pathlib import Path

from src.manifest import generate_manifest
from scripts.run_experiments import _phase_manifest, _pilot_manifest


def test_pilot_manifest_shape() -> None:
    base_manifest = generate_manifest(
        matrix_path=Path("config/experiment_matrix.yaml"),
        analytical_tasks_path=Path("config/tasks/analytical.yaml"),
        creative_tasks_path=Path("config/tasks/creative.yaml"),
        seed=42,
    )

    pilot = _pilot_manifest(base_manifest=base_manifest, seed=42)
    assert len(pilot.runs) == 232

    block0 = [run for run in pilot.runs if run.block_id == "block0_calibration"]
    block1 = [run for run in pilot.runs if run.block_id == "block1_disagreement_dividend"]
    block4 = [run for run in pilot.runs if run.block_id == "block4_quorum_paradox"]

    assert len(block0) == 192
    assert len(block1) == 20
    assert len(block4) == 20


def test_full_phase_excludes_pilot_runs(tmp_path: Path) -> None:
    base_manifest = generate_manifest(
        matrix_path=Path("config/experiment_matrix.yaml"),
        analytical_tasks_path=Path("config/tasks/analytical.yaml"),
        creative_tasks_path=Path("config/tasks/creative.yaml"),
        seed=42,
    )

    pilot = _phase_manifest(base_manifest=base_manifest, phase="pilot", results_dir=tmp_path, seed=42)
    pilot.save(tmp_path / "pilot_manifest.json")
    pilot_ids = {run.id for run in pilot.runs}

    full = _phase_manifest(base_manifest=base_manifest, phase="full", results_dir=tmp_path, seed=42)
    full_ids = {run.id for run in full.runs}

    assert pilot_ids.isdisjoint(full_ids)
    assert len(full.runs) + len(pilot.runs) == len(base_manifest.runs)
