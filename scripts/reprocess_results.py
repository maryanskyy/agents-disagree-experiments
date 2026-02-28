"""Reprocess existing run JSONs to add corrected quality metrics.

This script computes metrics that are comparable across runs with different
candidate counts and writes them to evaluation.corrected_metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.corrected_metrics import compute_corrected_metrics, infer_consensus_candidate_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reprocess run JSON files with corrected metrics")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("results/corrected_metrics_summary.csv"),
        help="Output CSV path for summary metrics",
    )
    return parser.parse_args()


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean_value = statistics.mean(values)
    sd_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean_value:.3f} +/- {sd_value:.3f} (n={len(values)})"


def _collect_group(rows: list[dict[str, Any]], *, block_id: str, key: str) -> dict[Any, list[float]]:
    grouped: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        if row["block_id"] != block_id:
            continue
        grouped[row[key]].append(float(row["consensus_win_rate"]))
    return grouped


def _resolve_consensus_candidate_id(payload: dict[str, Any], bt_scores: dict[str, Any]) -> str | None:
    selected_agent_id = payload.get("consensus", {}).get("selected_agent_id")
    existing_corrected = payload.get("evaluation", {}).get("corrected_metrics", {}) or {}
    preferred_id = existing_corrected.get("consensus_candidate_id") or selected_agent_id
    return infer_consensus_candidate_id(bt_scores=bt_scores, preferred_id=preferred_id)


def _process_run(path: Path) -> tuple[bool, dict[str, Any] | None, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        return False, None, False

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        return False, None, False

    panel_payload = evaluation.get("judge_panel")
    if not isinstance(panel_payload, dict):
        return False, None, False

    quality = float(evaluation.get("quality_score", 0.0))
    bt_scores = panel_payload.get("bt_scores", {}) or {}
    consensus_candidate_id = _resolve_consensus_candidate_id(payload, bt_scores)

    corrected = compute_corrected_metrics(
        panel_payload=panel_payload,
        quality_score=quality,
        consensus_candidate_id=consensus_candidate_id,
    )

    previous = evaluation.get("corrected_metrics")
    evaluation["corrected_metrics"] = corrected

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cfg = payload.get("config", {})
    row = {
        "run_id": payload.get("run_id"),
        "block_id": payload.get("block_id"),
        "agent_count": cfg.get("agent_count"),
        "topology": cfg.get("topology"),
        "disagreement_level": cfg.get("disagreement_level"),
        "task_type": cfg.get("task_type"),
        "raw_quality": quality,
        "consensus_win_rate": float(corrected["consensus_win_rate"]),
        "normalized_bt": float(corrected["normalized_bt_score"]),
        "models": "|".join(str(m) for m in cfg.get("model_assignment", [])),
    }
    return True, row, previous is not None


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir

    run_paths = sorted(results_dir.glob("block*/run_*.json"))
    if not run_paths:
        print(f"No run files found under {results_dir}/block*/run_*.json")
        return

    rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    had_previous_corrected = 0

    for path in run_paths:
        ok, row, had_previous = _process_run(path)
        if not ok:
            skipped += 1
            continue
        processed += 1
        if had_previous:
            had_previous_corrected += 1
        if row is not None:
            rows.append(row)

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "block_id",
                "agent_count",
                "topology",
                "disagreement_level",
                "task_type",
                "raw_quality",
                "consensus_win_rate",
                "normalized_bt",
                "models",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Reprocessing complete")
    print(f"Run files found: {len(run_paths)}")
    print(f"Processed (status=ok): {processed}")
    print(f"Skipped: {skipped}")
    print(f"Runs with existing corrected_metrics overwritten: {had_previous_corrected}")
    print(f"CSV summary: {args.csv_path}")

    raw_by_agent_count: dict[int, list[float]] = defaultdict(list)
    win_by_agent_count: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        agent_count = int(row["agent_count"])
        raw_by_agent_count[agent_count].append(float(row["raw_quality"]))
        win_by_agent_count[agent_count].append(float(row["consensus_win_rate"]))

    print("\nBefore vs After (all processed runs):")
    for n in sorted(raw_by_agent_count):
        raw_stats = _mean_sd(raw_by_agent_count[n])
        win_stats = _mean_sd(win_by_agent_count[n])
        print(f"n={n}: raw_quality={raw_stats} | consensus_win_rate={win_stats}")

    block4 = _collect_group(rows, block_id="block4_quorum_paradox", key="agent_count")
    print("\nBLOCK 4 (Paradox) -- Consensus Win Rate by Agent Count:")
    for n in [2, 3, 5]:
        print(f"n={n}: {_mean_sd(block4.get(n, []))}")

    block1 = _collect_group(rows, block_id="block1_disagreement_dividend", key="disagreement_level")
    print("\nBLOCK 1 (Disagreement) -- Consensus Win Rate by Level:")
    for level in [1, 2, 3, 4, 5]:
        print(f"Level {level}: {_mean_sd(block1.get(level, []))}")

    block2 = _collect_group(rows, block_id="block2_topology_comparison", key="topology")
    print("\nBLOCK 2 (Topology) -- Consensus Win Rate by Topology:")
    for topology in ["flat", "hierarchical", "quorum", "pipeline"]:
        print(f"{topology}: {_mean_sd(block2.get(topology, []))}")



if __name__ == "__main__":
    main()
