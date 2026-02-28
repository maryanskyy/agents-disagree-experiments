"""End-to-end dry-run test with mocked model clients."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.manifest import ExperimentManifest, generate_manifest
from src.runner import ExperimentRunner, RunnerConfig
from src.utils.logging_config import setup_logging


def test_end_to_end_dry_run(tmp_path: Path) -> None:
    """Runner should execute a small dry-run manifest and persist outputs."""

    async def _run() -> None:
        manifest = generate_manifest(
            matrix_path=Path("config/experiment_matrix.yaml"),
            analytical_tasks_path=Path("config/tasks/analytical.yaml"),
            creative_tasks_path=Path("config/tasks/creative.yaml"),
            seed=42,
        )
        tiny_manifest = ExperimentManifest(
            generated_at=manifest.generated_at,
            seed=manifest.seed,
            runs=manifest.runs[:4],
        )

        logger, _ = setup_logging(log_dir=tmp_path, name="test_dry_run")
        config = RunnerConfig(
            models_config_path=Path("config/models.yaml"),
            task_dir=Path("config/tasks"),
            results_dir=tmp_path,
            max_concurrent=2,
            dry_run=True,
            resume=False,
            seed=42,
            progress_every=2,
        )
        runner = ExperimentRunner(config=config, logger=logger)
        summary = await runner.run_manifest(tiny_manifest)
        await runner.close()

        assert summary["completed_runs"] == 4

        result_files = []
        for path in tmp_path.rglob("run_*.json"):
            if "human_eval" in path.parts:
                continue
            result_files.append(path)
        assert len(result_files) == 4

    asyncio.run(_run())
