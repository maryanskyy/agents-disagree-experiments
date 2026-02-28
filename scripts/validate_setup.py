"""Pre-flight validation for environment, configs, and dry-run execution."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import asyncio
import os
import platform
import shutil

from dotenv import load_dotenv

from src.manifest import ExperimentManifest, generate_manifest
from src.runner import ExperimentRunner, RunnerConfig
from src.tasks.loader import load_all_tasks, task_catalog_stats
from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Validate setup and optional dry run")
    parser.add_argument("--dry-run", action="store_true", help="Run small mocked execution")
    parser.add_argument("--results-dir", type=Path, default=Path("results/validate"))
    return parser.parse_args()


def check_python() -> str:
    """Return Python-version status message."""
    if sys.version_info < (3, 11):
        return "Warning: Python 3.11+ is recommended for production runs."
    return "Python version is compatible (3.11+)."


def check_env(dry_run: bool) -> dict[str, bool]:
    """Check API key presence."""
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY"))
    if not dry_run and (not has_anthropic or not has_google):
        raise RuntimeError("Missing API keys. Set ANTHROPIC_API_KEY and GOOGLE_API_KEY.")
    return {"anthropic_key_present": has_anthropic, "google_key_present": has_google}


async def run_smoke(results_dir: Path) -> dict:
    """Run a tiny dry-run subset to validate runner wiring."""
    manifest = generate_manifest(
        matrix_path=Path("config/experiment_matrix.yaml"),
        analytical_tasks_path=Path("config/tasks/analytical.yaml"),
        creative_tasks_path=Path("config/tasks/creative.yaml"),
        seed=42,
    )
    small_manifest = ExperimentManifest(
        generated_at=manifest.generated_at,
        seed=manifest.seed,
        runs=manifest.runs[:6],
    )

    logger, _ = setup_logging(log_dir=results_dir, name="validate_setup")
    config = RunnerConfig(
        models_config_path=Path("config/models.yaml"),
        task_dir=Path("config/tasks"),
        results_dir=results_dir,
        max_concurrent=3,
        dry_run=True,
        resume=False,
        seed=42,
    )
    runner = ExperimentRunner(config=config, logger=logger)
    summary = await runner.run_manifest(small_manifest)
    await runner.close()
    return summary


def main() -> None:
    """CLI entry point."""
    load_dotenv()
    args = parse_args()
    python_status = check_python()

    stats = task_catalog_stats(load_all_tasks(Path("config/tasks")))
    env_status = check_env(dry_run=args.dry_run)

    if args.results_dir.exists():
        shutil.rmtree(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    smoke_summary = {}
    if args.dry_run:
        smoke_summary = asyncio.run(run_smoke(args.results_dir))

    print("Validation complete")
    print(f"Platform: {platform.platform()}")
    print(python_status)
    print(f"Tasks: {stats}")
    print(f"Environment: {env_status}")
    if smoke_summary:
        print(f"Dry-run summary: {smoke_summary}")


if __name__ == "__main__":
    main()
