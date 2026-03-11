#!/usr/bin/env python3
"""Run Experiment v4-scale (50-task expansion with synthesis + decoupled eval judges)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import asyncio
import json
from collections import Counter, OrderedDict
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.manifest import ExperimentManifest, RunSpec
from src.runner import ExperimentRunner, RunnerConfig
from src.utils.keep_awake import SleepInhibitor
from src.utils.logging_config import setup_logging

RUNSPECS_PATH = ROOT / "config" / "runspecs_v4_scale.json"
RESULTS_DIR = ROOT / "results" / "v4"
FIRST_BLOCK_ID = "scale_diverse_strong_judge_based"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment v4-scale")
    parser.add_argument("--runspecs", type=Path, default=RUNSPECS_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print execution plan only (no API calls)")
    parser.add_argument("--block", action="append", default=[], help="Run only block IDs/prefixes (repeatable)")
    parser.add_argument("--max-cost", type=float, default=260.0)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Auto-continue past go/no-go gate after first 42-run block",
    )
    return parser.parse_args()


def load_runspecs(path: Path, block_filters: list[str]) -> list[dict]:
    specs = json.loads(path.read_text(encoding="utf-8"))
    if not block_filters:
        return specs
    return [
        spec
        for spec in specs
        if any(str(spec.get("block_id", "")).startswith(prefix) for prefix in block_filters)
    ]


def specs_to_manifest(specs: list[dict], *, seed: int) -> ExperimentManifest:
    runs: list[RunSpec] = []
    for spec in specs:
        selector = str(spec["selector"])
        metadata = {
            "composition": spec["composition"],
            "judges": spec.get("judges", []),
            "eval_judges": spec.get("eval_judges", []),
            "selector": selector,
        }
        if selector == "synthesis":
            metadata["synthesizer_model"] = "claude-sonnet-4-6"

        runs.append(
            RunSpec(
                id=str(spec["run_id"]),
                block_id=str(spec["block_id"]),
                task_type=str(spec["task_type"]),
                task_id=str(spec["task_id"]),
                topology="flat",
                consensus=selector,
                agent_count=len(spec["team"]),
                disagreement_level=0,
                temperature=float(spec.get("temperature", 0.7)),
                prompt_strategy="standard",
                repetition=0,
                model_assignment=[str(m) for m in spec["team"]],
                quality_threshold=None,
                posthoc_quality_thresholds=[],
                metadata=metadata,
            )
        )

    return ExperimentManifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        runs=runs,
    )


def ordered_blocks(specs: list[dict]) -> OrderedDict[str, list[dict]]:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for spec in specs:
        grouped.setdefault(spec["block_id"], []).append(spec)
    return grouped


def print_summary(specs: list[dict]) -> None:
    block_counts = Counter(s["block_id"] for s in specs)
    selector_counts = Counter(s["selector"] for s in specs)
    comp_counts = Counter(s["composition"] for s in specs)

    print("\n" + "=" * 68)
    print("EXPERIMENT v4-SCALE")
    print("=" * 68)
    print(f"Total runs: {len(specs)}")
    print("\nBy block:")
    for block_id, count in block_counts.items():
        print(f"  {block_id:40s} {count:3d}")
    print(f"\nBy selector:    {dict(selector_counts)}")
    print(f"By composition: {dict(comp_counts)}")
    print("=" * 68)


async def run_block(
    runner: ExperimentRunner,
    block_id: str,
    specs: list[dict],
    *,
    seed: int,
    results_dir: Path,
) -> dict:
    manifest = specs_to_manifest(specs, seed=seed)
    manifests_dir = results_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / f"{block_id}.json").write_text(
        json.dumps(manifest.to_dict(), indent=2),
        encoding="utf-8",
    )
    return await runner.run_manifest(manifest)


async def main() -> None:
    args = parse_args()
    load_dotenv()

    specs = load_runspecs(args.runspecs, args.block)
    if not specs:
        print("No runspecs matched the selected filters.")
        return

    print_summary(specs)
    blocks = ordered_blocks(specs)

    if args.dry_run:
        print("\nDRY RUN: no API calls will be made.")
        for idx, (block_id, block_specs) in enumerate(blocks.items(), start=1):
            print(f"[{idx}] {block_id}: {len(block_specs)} runs")
        print("\nFirst 5 planned run_ids:")
        for spec in specs[:5]:
            print(f"  - {spec['run_id']}")
        return

    args.results_dir.mkdir(parents=True, exist_ok=True)
    logger, _ = setup_logging(log_dir=args.results_dir, name="experiment_v4_scale")

    config = RunnerConfig(
        models_config_path=ROOT / "config" / "models.yaml",
        task_dir=ROOT / "config" / "tasks",
        results_dir=args.results_dir,
        max_concurrent=args.max_concurrent,
        
        dry_run=False,
        resume=args.resume,
        seed=args.seed,
    )

    runner = ExperimentRunner(config=config, logger=logger)
    try:
        with SleepInhibitor():
            block_items = list(blocks.items())
            for idx, (block_id, block_specs) in enumerate(block_items):
                print(f"\nRunning block {idx + 1}/{len(block_items)}: {block_id} ({len(block_specs)} runs)")
                summary = await run_block(
                    runner,
                    block_id,
                    block_specs,
                    seed=args.seed,
                    results_dir=args.results_dir,
                )
                print(f"Block complete: {block_id} -> status={summary.get('status')} completed={summary.get('completed_runs')}")

                is_first_gate = block_id == FIRST_BLOCK_ID and idx < len(block_items) - 1
                if is_first_gate and not args.yes:
                    answer = input("\nGo/No-Go gate reached after first 42 runs. Continue to remaining blocks? [y/N]: ").strip().lower()
                    if answer not in {"y", "yes"}:
                        print("Stopping at go/no-go gate by user choice.")
                        break
    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(main())
