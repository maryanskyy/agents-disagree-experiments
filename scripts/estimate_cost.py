"""Estimate experiment cost from manifest and pricing config."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from collections import defaultdict
from math import comb

import yaml

from src.manifest import ExperimentManifest, generate_manifest


DEFAULT_AGENT_INPUT_TOKENS = 1200
DEFAULT_AGENT_OUTPUT_TOKENS = 550
DEFAULT_JUDGE_INPUT_TOKENS = 2200
DEFAULT_JUDGE_OUTPUT_TOKENS = 120


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Estimate experiment cost")
    parser.add_argument("--manifest", type=Path, default=Path("results/manifest.json"))
    parser.add_argument("--models", type=Path, default=Path("config/models.yaml"))
    parser.add_argument("--matrix", type=Path, default=Path("config/experiment_matrix.yaml"))
    parser.add_argument("--tasks-dir", type=Path, default=Path("config/tasks"))
    return parser.parse_args()


def _agent_calls_per_run(topology: str, agent_count: int) -> int:
    if topology == "quorum":
        return agent_count * 2
    return agent_count


def _judge_calls_per_run(consensus: str, agent_count: int, panel_size: int) -> tuple[int, int]:
    """Return (consensus_judge_calls, evaluation_judge_calls)."""
    if panel_size <= 0:
        return 0, 0

    if consensus == "judge_based":
        comparisons = comb(agent_count, 2) if agent_count >= 2 else 0
        return comparisons * panel_size * 2, 0

    comparisons = comb(agent_count + 1, 2) if agent_count + 1 >= 2 else 0
    return 0, comparisons * panel_size * 2


def _model_cost(model_cfg: dict, input_tokens: int, output_tokens: int) -> float:
    prices = model_cfg["pricing_per_1m_tokens"]
    return (
        (input_tokens / 1_000_000) * float(prices["input"])
        + (output_tokens / 1_000_000) * float(prices["output"])
    )


def _select_judges(models_cfg: dict, assignment: list[str]) -> list[str]:
    judge_cfg = models_cfg.get("judge_pool", {})
    panel_size = int(judge_cfg.get("panel_size", 3))
    primary = [str(m) for m in judge_cfg.get("primary_models", [])]
    reserve = [str(m) for m in judge_cfg.get("reserve_models", [])]

    pool: list[str] = []
    for model in [*primary, *reserve]:
        if model not in pool:
            pool.append(model)

    agent_set = set(assignment)
    if bool(judge_cfg.get("exclude_agent_models", True)):
        pool = [m for m in pool if m not in agent_set]

    if bool(judge_cfg.get("prefer_different_families", True)):
        model_table = models_cfg["models"]
        families = {str(model_table[m].get("family", m)) for m in assignment if m in model_table}
        preferred = [m for m in pool if str(model_table[m].get("family", m)) not in families]
        fallback = [m for m in pool if m not in preferred]
        ordered = preferred + fallback
    else:
        ordered = pool

    if len(ordered) < panel_size:
        raise ValueError(
            f"Insufficient judge models for assignment={assignment}: need {panel_size}, have {len(ordered)}"
        )
    return ordered[:panel_size]


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

    models_cfg = yaml.safe_load(args.models.read_text(encoding="utf-8"))
    model_table = models_cfg["models"]
    panel_size = int(models_cfg.get("judge_pool", {}).get("panel_size", 3))

    by_block = defaultdict(float)
    by_model = defaultdict(float)
    by_stage = defaultdict(float)

    total_agent_calls = 0
    total_judge_calls = 0

    for run in manifest.runs:
        agent_calls = _agent_calls_per_run(run.topology, run.agent_count)
        total_agent_calls += agent_calls
        for step in range(agent_calls):
            model_name = run.model_assignment[step % len(run.model_assignment)]
            cost = _model_cost(model_table[model_name], DEFAULT_AGENT_INPUT_TOKENS, DEFAULT_AGENT_OUTPUT_TOKENS)
            by_block[run.block_id] += cost
            by_model[model_name] += cost
            by_stage["agent_generation"] += cost

        consensus_calls, eval_calls = _judge_calls_per_run(run.consensus, run.agent_count, panel_size)
        judge_calls = consensus_calls + eval_calls
        total_judge_calls += judge_calls
        if judge_calls:
            judges = _select_judges(models_cfg, run.model_assignment)
            calls_per_judge = judge_calls // len(judges)
            remainder = judge_calls % len(judges)
            for idx, judge_model in enumerate(judges):
                count = calls_per_judge + (1 if idx < remainder else 0)
                cost = _model_cost(model_table[judge_model], DEFAULT_JUDGE_INPUT_TOKENS, DEFAULT_JUDGE_OUTPUT_TOKENS) * count
                by_block[run.block_id] += cost
                by_model[judge_model] += cost
                by_stage["judge_pairwise"] += cost

    total_cost = sum(by_block.values())

    print(f"Estimated runs: {len(manifest.runs)}")
    print(f"Estimated agent calls: {total_agent_calls}")
    print(f"Estimated judge calls: {total_judge_calls}")
    print(f"Estimated API calls total: {total_agent_calls + total_judge_calls}")
    print(f"Estimated total cost (USD): ${total_cost:,.2f}")

    print("\nCost by stage:")
    for stage, value in sorted(by_stage.items()):
        print(f"  - {stage}: ${value:,.2f}")

    print("\nCost by block:")
    for block, value in sorted(by_block.items()):
        print(f"  - {block}: ${value:,.2f}")

    print("\nCost by model:")
    for model, value in sorted(by_model.items()):
        print(f"  - {model}: ${value:,.2f}")


if __name__ == "__main__":
    main()
