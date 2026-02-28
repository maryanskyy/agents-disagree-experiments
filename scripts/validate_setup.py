"""Pre-flight validation for environment, configs, and provider connectivity."""

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
from typing import Any

from dotenv import load_dotenv

from src.manifest import ExperimentManifest, generate_manifest
from src.models.catalog import load_model_catalog
from src.runner import ExperimentRunner, RunnerConfig
from src.tasks.loader import load_all_tasks, task_catalog_stats
from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Validate setup and optional dry run")
    parser.add_argument("--dry-run", action="store_true", help="Run mocked connectivity and tiny execution")
    parser.add_argument("--results-dir", type=Path, default=Path("results/validate"))
    return parser.parse_args()


def check_python() -> str:
    """Return Python-version status message."""
    if sys.version_info < (3, 11):
        return "Warning: Python 3.11+ is recommended for production runs."
    return "Python version is compatible (3.11+)."


def check_env(dry_run: bool) -> dict[str, bool]:
    """Check API key presence for all providers."""
    env_status = {
        "anthropic_key_present": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "google_key_present": bool(os.getenv("GOOGLE_API_KEY")),
    }
    if not dry_run and not all(env_status.values()):
        raise RuntimeError("Missing API keys. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, and GOOGLE_API_KEY.")
    return env_status


async def validate_model_connectivity(*, results_dir: Path, dry_run: bool) -> dict[str, Any]:
    """Verify each configured model can generate a response."""
    catalog = load_model_catalog(config_path=Path("config/models.yaml"))

    logger, _ = setup_logging(log_dir=results_dir, name="validate_connectivity")
    runner = ExperimentRunner(
        config=RunnerConfig(
            models_config_path=Path("config/models.yaml"),
            task_dir=Path("config/tasks"),
            results_dir=results_dir,
            max_concurrent=1,
            dry_run=dry_run,
            resume=False,
            seed=42,
        ),
        logger=logger,
    )

    model_status: dict[str, dict[str, Any]] = {}
    provider_status: dict[str, dict[str, Any]] = {}

    try:
        for model_alias, model_cfg in sorted(catalog.models.items()):
            provider = str(model_cfg["provider"])
            provider_bucket = provider_status.setdefault(provider, {"models": [], "ok": True, "errors": []})
            provider_bucket["models"].append(model_alias)

            try:
                client = runner._get_client(model_alias)  # internal factory reused for consistency
                response = await client.generate(
                    system_prompt="You are a connectivity test assistant. Reply with a short confirmation.",
                    user_prompt="Return the single word OK.",
                    temperature=0.0,
                    max_tokens=24,
                    metadata={"role": "validate_setup", "dry_run": dry_run},
                )
                model_status[model_alias] = {
                    "provider": provider,
                    "ok": bool(response.text.strip()),
                    "input_tokens": int(response.input_tokens),
                    "output_tokens": int(response.output_tokens),
                    "mode": "dry_run" if dry_run else "live",
                }
                if not model_status[model_alias]["ok"]:
                    raise RuntimeError("empty response")
            except Exception as exc:
                provider_bucket["ok"] = False
                provider_bucket["errors"].append(f"{model_alias}: {exc!r}")
                model_status[model_alias] = {
                    "provider": provider,
                    "ok": False,
                    "error": repr(exc),
                    "mode": "dry_run" if dry_run else "live",
                }

        all_ok = all(status.get("ok", False) for status in model_status.values())
        provider_ok = all(status.get("ok", False) for status in provider_status.values())

        if not dry_run and (not all_ok or not provider_ok):
            raise RuntimeError("Connectivity validation failed for one or more providers/models.")

        return {
            "mode": "dry_run" if dry_run else "live",
            "provider_status": provider_status,
            "models": model_status,
            "all_models_ok": all_ok,
            "all_providers_ok": provider_ok,
            "judge_pool_models": catalog.judge_pool,
            "agent_pool_models": catalog.agent_pool,
        }
    finally:
        await runner.close()


async def run_smoke(results_dir: Path) -> dict[str, Any]:
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

    connectivity = asyncio.run(validate_model_connectivity(results_dir=args.results_dir, dry_run=args.dry_run))

    smoke_summary: dict[str, Any] = {}
    if args.dry_run:
        smoke_summary = asyncio.run(run_smoke(args.results_dir))

    print("Validation complete")
    print(f"Platform: {platform.platform()}")
    print(python_status)
    print(f"Tasks: {stats}")
    print(f"Environment: {env_status}")
    print(f"Connectivity: {{'all_providers_ok': {connectivity['all_providers_ok']}, 'all_models_ok': {connectivity['all_models_ok']}}}")
    if smoke_summary:
        print(f"Dry-run summary: {smoke_summary}")


if __name__ == "__main__":
    main()
