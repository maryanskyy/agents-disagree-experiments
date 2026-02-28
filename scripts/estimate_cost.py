"""Estimate experiment cost from manifest and pricing config."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from collections import defaultdict

import yaml

from src.manifest import ExperimentManifest, generate_manifest


DEFAULT_INPUT_TOKENS = 1200
DEFAULT_OUTPUT_TOKENS = 550


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Estimate experiment cost")
    parser.add_argument("--manifest", type=Path, default=Path("results/manifest.json"))
    parser.add_argument("--models", type=Path, default=Path("config/models.yaml"))
    parser.add_argument("--matrix", type=Path, default=Path("config/experiment_matrix.yaml"))
    parser.add_argument("--tasks-dir", type=Path, default=Path("config/tasks"))
    return parser.parse_args()


def _calls_per_run(topology: str, consensus: str, agent_count: int) -> int:
    rounds = 2 if topology == "quorum" else 1
    base = agent_count * rounds
    if consensus == "judge_based":
        base += 1
    return base


def main() -> None:
    """Entry point."""
    args = parse_args()
    if args.manifest.exists():
        manifest = ExperimentManifest.load(args.manifest)
    else:
        manifest = generate_manifest(
            matrix_path=args.matrix,
            analytical_tasks_path=args.tasks_dir / "analytical.yaml",
            creative_tasks_path=args.tasks_dir / "creative.yaml",
            seed=42,
        )

    model_cfg = yaml.safe_load(args.models.read_text(encoding="utf-8"))["models"]
    by_block = defaultdict(float)
    by_model = defaultdict(float)

    total_calls = 0
    for run in manifest.runs:
        calls = _calls_per_run(run.topology, run.consensus, run.agent_count)
        total_calls += calls
        for step in range(calls):
            model_name = run.model_assignment[step % len(run.model_assignment)]
            prices = model_cfg[model_name]["pricing_per_1m_tokens"]
            cost = (
                (DEFAULT_INPUT_TOKENS / 1_000_000) * float(prices["input"])
                + (DEFAULT_OUTPUT_TOKENS / 1_000_000) * float(prices["output"])
            )
            by_block[run.block_id] += cost
            by_model[model_name] += cost

    total_cost = sum(by_block.values())

    print(f"Estimated runs: {len(manifest.runs)}")
    print(f"Estimated API calls: {total_calls}")
    print(f"Estimated total cost (USD): ${total_cost:,.2f}")
    print("\nCost by block:")
    for block, value in sorted(by_block.items()):
        print(f"  - {block}: ${value:,.2f}")
    print("\nCost by model:")
    for model, value in sorted(by_model.items()):
        print(f"  - {model}: ${value:,.2f}")


if __name__ == "__main__":
    main()
