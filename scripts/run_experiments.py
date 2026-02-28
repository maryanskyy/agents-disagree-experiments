"""Main experiment execution entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import asyncio
import json
import random
from collections import defaultdict
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from src.manifest import ExperimentManifest, RunSpec, generate_manifest
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
    parser.add_argument("--phase", choices=["pilot", "full", "all"], default="all")
    parser.add_argument("--max-cost", type=float, default=4000.0)
    return parser.parse_args()


def _sample_runs(runs: list[RunSpec], count: int, *, seed: int) -> list[RunSpec]:
    if count >= len(runs):
        return list(runs)
    rng = random.Random(seed)
    copy = list(runs)
    rng.shuffle(copy)
    return copy[:count]


def _sample_stratified(
    runs: list[RunSpec],
    *,
    group_fn,
    per_group: int,
    seed: int,
) -> list[RunSpec]:
    grouped: dict[Any, list[RunSpec]] = defaultdict(list)
    for run in runs:
        grouped[group_fn(run)].append(run)

    selected: list[RunSpec] = []
    for idx, key in enumerate(sorted(grouped)):
        selected.extend(_sample_runs(grouped[key], per_group, seed=seed + idx * 17))
    return selected


def _pilot_manifest(base_manifest: ExperimentManifest, seed: int) -> ExperimentManifest:
    by_block: dict[str, list[RunSpec]] = defaultdict(list)
    for run in base_manifest.runs:
        by_block[run.block_id].append(run)

    block0 = list(by_block.get("block0_calibration", []))
    block1 = list(by_block.get("block1_disagreement_dividend", []))
    block4 = list(by_block.get("block4_quorum_paradox", []))

    block1_selected = _sample_stratified(
        block1,
        group_fn=lambda r: r.disagreement_level,
        per_group=4,
        seed=seed + 100,
    )

    block4_n2 = [r for r in block4 if r.agent_count == 2]
    block4_n3 = [r for r in block4 if r.agent_count == 3]
    block4_selected = [
        *_sample_runs(block4_n2, 10, seed=seed + 200),
        *_sample_runs(block4_n3, 10, seed=seed + 300),
    ]

    pilot_runs = [*block0, *block1_selected, *block4_selected]
    pilot_runs = sorted(pilot_runs, key=lambda r: (r.block_id, r.id))

    return ExperimentManifest(
        generated_at=base_manifest.generated_at,
        seed=base_manifest.seed,
        runs=pilot_runs,
    )


def _load_pilot_ids(results_dir: Path) -> set[str]:
    pilot_manifest_path = results_dir / "pilot_manifest.json"
    if pilot_manifest_path.exists():
        manifest = ExperimentManifest.load(pilot_manifest_path)
        return {run.id for run in manifest.runs}

    pilot_report_path = results_dir / "pilot_report.json"
    if pilot_report_path.exists():
        payload = json.loads(pilot_report_path.read_text(encoding="utf-8"))
        return set(payload.get("completed_run_ids", []))

    return set()


def _phase_manifest(base_manifest: ExperimentManifest, phase: str, results_dir: Path, seed: int) -> ExperimentManifest:
    if phase == "all":
        return base_manifest

    if phase == "pilot":
        return _pilot_manifest(base_manifest=base_manifest, seed=seed)

    pilot_ids = _load_pilot_ids(results_dir)
    filtered_runs = [run for run in base_manifest.runs if run.id not in pilot_ids]
    return ExperimentManifest(generated_at=base_manifest.generated_at, seed=base_manifest.seed, runs=filtered_runs)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _is_disagreement_separable(level_means: dict[int, float]) -> tuple[bool, dict[str, Any]]:
    if len(level_means) < 3:
        return False, {"reason": "fewer_than_3_levels"}

    ordered = [level_means[level] for level in sorted(level_means)]
    spread = max(ordered) - min(ordered)
    adjacent_diffs = [abs(ordered[idx + 1] - ordered[idx]) for idx in range(len(ordered) - 1)]
    max_adjacent = max(adjacent_diffs) if adjacent_diffs else 0.0

    separable = spread >= 0.05 and max_adjacent >= 0.02
    return separable, {
        "spread": round(spread, 6),
        "max_adjacent_gap": round(max_adjacent, 6),
        "criteria": {"min_spread": 0.05, "min_adjacent_gap": 0.02},
    }


def _build_pilot_report(results_dir: Path, pilot_manifest: ExperimentManifest, summary: dict[str, Any]) -> dict[str, Any]:
    result_payloads: list[dict[str, Any]] = []
    completed_run_ids: list[str] = []

    for run in pilot_manifest.runs:
        path = results_dir / run.block_id / f"{run.id}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            continue
        result_payloads.append(payload)
        completed_run_ids.append(run.id)

    kappas: list[float] = []
    disagreement_by_level: dict[int, list[float]] = defaultdict(list)
    paradox_quality_n2: list[float] = []
    paradox_quality_n3: list[float] = []

    for payload in result_payloads:
        eval_data = payload.get("evaluation", {})
        panel = eval_data.get("judge_panel", {})
        reliability = panel.get("inter_rater_reliability", {})
        kappa = reliability.get("mean_cohen_kappa")
        if isinstance(kappa, (int, float)):
            kappas.append(float(kappa))

        config = payload.get("config", {})
        if payload.get("block_id") == "block1_disagreement_dividend":
            level = int(config.get("disagreement_level", 0))
            d_rate = eval_data.get("disagreement", {}).get("disagreement_rate")
            if isinstance(d_rate, (int, float)):
                disagreement_by_level[level].append(float(d_rate))

        if payload.get("block_id") == "block4_quorum_paradox":
            quality = eval_data.get("quality_score")
            if not isinstance(quality, (int, float)):
                continue
            if int(config.get("agent_count", 0)) == 2:
                paradox_quality_n2.append(float(quality))
            elif int(config.get("agent_count", 0)) == 3:
                paradox_quality_n3.append(float(quality))

    mean_kappa = _safe_mean(kappas)
    level_means = {
        level: float(mean(values))
        for level, values in sorted(disagreement_by_level.items())
        if values
    }
    separable, separability_details = _is_disagreement_separable(level_means)

    q_n2 = _safe_mean(paradox_quality_n2)
    q_n3 = _safe_mean(paradox_quality_n3)
    paradox_delta = (q_n3 - q_n2) if (q_n2 is not None and q_n3 is not None) else None

    kappa_pass = (mean_kappa is not None) and (mean_kappa > 0.4)
    go_recommended = bool(kappa_pass and separable)

    cost_snapshot = summary.get("cost", {})
    report = {
        "phase": "pilot",
        "generated_from": "scripts/run_experiments.py",
        "pilot_scope": {
            "target_runs": len(pilot_manifest.runs),
            "completed_ok_runs": len(result_payloads),
            "completed_run_ids": completed_run_ids,
            "blocks": {
                "block0_calibration": len([r for r in pilot_manifest.runs if r.block_id == "block0_calibration"]),
                "block1_disagreement_dividend": len([r for r in pilot_manifest.runs if r.block_id == "block1_disagreement_dividend"]),
                "block4_quorum_paradox": len([r for r in pilot_manifest.runs if r.block_id == "block4_quorum_paradox"]),
            },
        },
        "inter_judge_kappa": {
            "mean_cohen_kappa": mean_kappa,
            "passes_threshold_gt_0_4": kappa_pass,
        },
        "disagreement_rate_per_level": {
            "means": level_means,
            "counts": {str(k): len(v) for k, v in disagreement_by_level.items()},
            "levels_are_separable": separable,
            "separability_details": separability_details,
        },
        "preliminary_paradox_signal": {
            "quality_mean_n2": q_n2,
            "quality_mean_n3": q_n3,
            "delta_n3_minus_n2": paradox_delta,
            "observed_dip": (paradox_delta is not None and paradox_delta < 0),
        },
        "cost_so_far_usd": float(cost_snapshot.get("total_cost_usd", 0.0)),
        "go_no_go": {
            "go_recommended": go_recommended,
            "rule": "GO if mean kappa > 0.4 AND disagreement levels are separable",
            "reason": (
                "GO: reliability and level separation criteria met"
                if go_recommended
                else "NO-GO: one or more pilot gating criteria not met"
            ),
        },
    }

    report["completed_run_ids"] = completed_run_ids
    return report


async def async_main(args: argparse.Namespace) -> dict:
    """Run experiments asynchronously."""
    args.results_dir.mkdir(parents=True, exist_ok=True)
    logger, _ = setup_logging(log_dir=args.results_dir, name="run_experiments")

    base_manifest = generate_manifest(
        matrix_path=args.matrix,
        analytical_tasks_path=args.tasks_dir / "analytical.yaml",
        creative_tasks_path=args.tasks_dir / "creative.yaml",
        seed=args.seed,
    )

    phase_manifest = _phase_manifest(
        base_manifest=base_manifest,
        phase=args.phase,
        results_dir=args.results_dir,
        seed=args.seed,
    )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(phase_manifest.to_dict(), indent=2), encoding="utf-8")

    (args.results_dir / "manifest_full.json").write_text(
        json.dumps(base_manifest.to_dict(), indent=2),
        encoding="utf-8",
    )
    (args.results_dir / f"manifest_{args.phase}.json").write_text(
        json.dumps(phase_manifest.to_dict(), indent=2),
        encoding="utf-8",
    )

    if args.phase == "pilot":
        (args.results_dir / "pilot_manifest.json").write_text(
            json.dumps(phase_manifest.to_dict(), indent=2),
            encoding="utf-8",
        )

    logger.info(
        "Prepared phase='%s' manifest with %d runs (base=%d)",
        args.phase,
        len(phase_manifest.runs),
        len(base_manifest.runs),
    )

    resume = args.resume and args.phase != "all"
    if args.phase == "all" and args.resume:
        logger.info("Ignoring --resume for --phase all (full rerun from scratch requested).")

    config = RunnerConfig(
        models_config_path=args.models,
        task_dir=args.tasks_dir,
        results_dir=args.results_dir,
        max_concurrent=args.max_concurrent,
        dry_run=args.dry_run,
        resume=resume,
        seed=args.seed,
        max_cost_usd=float(args.max_cost),
    )

    runner = ExperimentRunner(config=config, logger=logger)
    with SleepInhibitor():
        summary = await runner.run_manifest(phase_manifest)
    await runner.close()

    if args.phase == "pilot":
        pilot_report = _build_pilot_report(args.results_dir, phase_manifest, summary)
        pilot_report_path = args.results_dir / "pilot_report.json"
        pilot_report_path.write_text(json.dumps(pilot_report, indent=2), encoding="utf-8")
        logger.info("Pilot report written to %s", pilot_report_path)

    logger.info("Run summary: %s", summary)
    return summary


def main() -> None:
    """Program entry point."""
    load_dotenv()
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

