"""Main experiment execution entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import asyncio

from dotenv import load_dotenv

from src.manifest import ExperimentManifest, generate_manifest
from src.runner import ExperimentRunner, RunnerConfig
from src.utils.keep_awake import SleepInhibitor
from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run quorum-agent experiments")
    parser.add_argument("--manifest", type=Path, default=Path("results/manifest.json"))
    parser.add_argument("--matrix", type=Path, default=Path("config/experiment_matrix.yaml"))
    parser.add_argument("--tasks-dir", type=Path, default=Path("config/tasks"))
    parser.add_argument("--models", type=Path, default=Path("config/models.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> dict:
    """Run experiments asynchronously."""
    logger, _ = setup_logging(log_dir=args.results_dir, name="run_experiments")

    if args.manifest.exists():
        manifest = ExperimentManifest.load(args.manifest)
        logger.info("Loaded existing manifest at %s", args.manifest)
    else:
        manifest = generate_manifest(
            matrix_path=args.matrix,
            analytical_tasks_path=args.tasks_dir / "analytical.yaml",
            creative_tasks_path=args.tasks_dir / "creative.yaml",
            seed=args.seed,
        )
        manifest.save(args.manifest)
        logger.info("Generated manifest at %s", args.manifest)

    config = RunnerConfig(
        models_config_path=args.models,
        task_dir=args.tasks_dir,
        results_dir=args.results_dir,
        max_concurrent=args.max_concurrent,
        dry_run=args.dry_run,
        resume=args.resume,
        seed=args.seed,
    )

    runner = ExperimentRunner(config=config, logger=logger)
    with SleepInhibitor():
        summary = await runner.run_manifest(manifest)
    await runner.close()

    logger.info("Run summary: %s", summary)
    return summary


def main() -> None:
    """Program entry point."""
    load_dotenv()
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
