"""Compute and backfill structural quality metrics for historical run outputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.structural_quality import compute_composite_score, compute_structural_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute structural quality metrics for existing run JSON files")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("results/structural_metrics_summary.csv"),
        help="Output CSV path for per-candidate structural metrics summary",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name used for embedding-based metrics",
    )
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _summary_line(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{_mean(values):.4f} +/- {_sd(values):.4f} (n={len(values)})"


def _condition_label(cfg: dict[str, Any]) -> str:
    task_type = str(cfg.get("task_type", "unknown"))
    topology = str(cfg.get("topology", "unknown"))
    consensus = str(cfg.get("consensus", "unknown"))
    return f"{task_type}|{topology}|{consensus}"


def _row_base(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = payload.get("config", {}) or {}
    return {
        "run_id": payload.get("run_id"),
        "block_id": payload.get("block_id"),
        "task_type": cfg.get("task_type"),
        "topology": cfg.get("topology"),
        "consensus": cfg.get("consensus"),
        "agent_count": cfg.get("agent_count"),
        "disagreement_level": cfg.get("disagreement_level"),
        "condition": _condition_label(cfg),
    }


def _compute_payload_metrics(payload: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    task_prompt = str(payload.get("task", {}).get("prompt", ""))
    if not task_prompt:
        return []

    output_rows: list[dict[str, Any]] = []

    evaluation = payload.setdefault("evaluation", {})
    consensus_text = str(payload.get("consensus", {}).get("selected_text", ""))
    consensus_metrics = compute_structural_metrics(text=consensus_text, prompt=task_prompt, model_name=model_name)
    evaluation["structural_metrics"] = asdict(consensus_metrics)

    base = _row_base(payload)
    output_rows.append(
        {
            **base,
            "candidate_type": "consensus",
            "candidate_id": "final_consensus",
            **asdict(consensus_metrics),
            "composite_score": compute_composite_score(consensus_metrics),
        }
    )

    outputs = payload.get("outputs", []) or []
    for idx, output in enumerate(outputs):
        text = str(output.get("text", ""))
        agent_metrics = compute_structural_metrics(text=text, prompt=task_prompt, model_name=model_name)
        output["structural_metrics"] = asdict(agent_metrics)

        output_rows.append(
            {
                **base,
                "candidate_type": "agent",
                "candidate_id": output.get("agent_id", f"agent_{idx}"),
                **asdict(agent_metrics),
                "composite_score": compute_composite_score(agent_metrics),
            }
        )

    return output_rows


def _process_run(path: Path, model_name: str) -> tuple[bool, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        return False, []

    rows = _compute_payload_metrics(payload, model_name=model_name)
    if not rows:
        return False, []

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True, rows


def _print_group_summary(rows: list[dict[str, Any]], group_key: str) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)

    print(f"\nSummary by {group_key} (consensus outputs only):")
    for group_name in sorted(grouped):
        consensus_rows = [row for row in grouped[group_name] if row["candidate_type"] == "consensus"]
        composite_values = [float(row["composite_score"]) for row in consensus_rows]
        coherence_values = [float(row["coherence_mean"]) for row in consensus_rows]
        relevance_values = [float(row["prompt_relevance"]) for row in consensus_rows]
        repetition_values = [float(row["repetition_rate"]) for row in consensus_rows]

        print(
            f"  {group_name}: "
            f"composite={_summary_line(composite_values)} | "
            f"coherence={_summary_line(coherence_values)} | "
            f"prompt_relevance={_summary_line(relevance_values)} | "
            f"repetition={_summary_line(repetition_values)}"
        )


def main() -> None:
    args = parse_args()

    run_paths = sorted(args.results_dir.glob("block*/run_*.json"))
    if not run_paths:
        print(f"No run files found under {args.results_dir}/block*/run_*.json")
        return

    processed = 0
    skipped = 0
    all_rows: list[dict[str, Any]] = []

    for path in run_paths:
        ok, rows = _process_run(path, model_name=args.model_name)
        if ok:
            processed += 1
            all_rows.extend(rows)
        else:
            skipped += 1

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "block_id",
        "task_type",
        "topology",
        "consensus",
        "agent_count",
        "disagreement_level",
        "condition",
        "candidate_type",
        "candidate_id",
        "mtld",
        "readability_fk_grade",
        "coherence_mean",
        "prompt_relevance",
        "connective_density",
        "word_count",
        "repetition_rate",
        "composite_score",
    ]
    with args.csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("Structural metric reprocessing complete")
    print(f"Run files found: {len(run_paths)}")
    print(f"Processed (status=ok): {processed}")
    print(f"Skipped: {skipped}")
    print(f"CSV summary: {args.csv_path}")

    _print_group_summary(all_rows, "block_id")
    _print_group_summary(all_rows, "condition")

    sample = next((row for row in all_rows if row["candidate_type"] == "consensus"), None)
    if sample is not None:
        print("\nSample consensus structural metrics:")
        print(json.dumps({k: sample[k] for k in fieldnames if k in sample}, indent=2))


if __name__ == "__main__":
    main()
