"""Export experiment results into an open dataset package."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import yaml

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")

RUNS_SUMMARY_COLUMNS = [
    "run_id",
    "block_id",
    "task_id",
    "task_type",
    "topology",
    "consensus_method",
    "agent_count",
    "disagreement_level",
    "model_assignment",
    "repetition",
    "raw_bt_quality",
    "consensus_win_rate",
    "normalized_bt",
    "num_bt_candidates",
    "mtld_consensus",
    "readability_consensus",
    "coherence_consensus",
    "prompt_relevance_consensus",
    "connective_density_consensus",
    "word_count_consensus",
    "repetition_rate_consensus",
    "mean_judge_kappa",
    "timestamp",
]

AGENT_OUTPUT_COLUMNS = [
    "run_id",
    "agent_index",
    "model_name",
    "provider",
    "tier",
    "text",
    "word_count",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "mtld",
    "readability_fk_grade",
    "coherence_mean",
    "connective_density",
    "repetition_rate",
    "is_empty",
]

PAIRWISE_COLUMNS = [
    "run_id",
    "candidate_i",
    "candidate_j",
    "judge_model",
    "judge_vote",
    "ordering",
    "confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export results into reusable dataset files")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/export"))
    parser.add_argument("--tasks-dir", type=Path, default=Path("config/tasks"))
    parser.add_argument("--models-config", type=Path, default=Path("config/models.yaml"))
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text or ""))


def _parse_agent_index(agent_id: str, fallback: int) -> int:
    if not agent_id:
        return fallback
    if "_" in agent_id:
        suffix = agent_id.rsplit("_", 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    return fallback


def _format_model_assignment(models: list[Any]) -> str:
    return "|".join(str(model) for model in models)


def _load_model_snapshot(models_config_path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    raw = yaml.safe_load(models_config_path.read_text(encoding="utf-8"))

    model_lookup: dict[str, dict[str, Any]] = {}
    snapshot_rows: list[dict[str, Any]] = []

    for role_key in ("agent_pool", "judge_pool"):
        pool = raw.get(role_key, {}) or {}
        role = "agent" if role_key == "agent_pool" else "judge"
        for alias, cfg in pool.items():
            entry = {
                "name": alias,
                "role": role,
                "provider": cfg.get("provider"),
                "tier": cfg.get("tier", cfg.get("quality_tier")),
                "model_id": cfg.get("model_id", alias),
                "max_output_tokens": cfg.get("max_output_tokens"),
                "input_cost_per_1m": cfg.get("input_cost_per_1m"),
                "output_cost_per_1m": cfg.get("output_cost_per_1m"),
                "rpm_limit": cfg.get("rpm_limit", cfg.get("rpm")),
                "tpm_limit": cfg.get("tpm_limit", cfg.get("tpm")),
            }
            model_lookup[alias] = entry
            snapshot_rows.append(entry)

    return model_lookup, snapshot_rows


def _load_tasks(tasks_dir: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for file_name in ("analytical.yaml", "creative.yaml"):
        path = tasks_dir / file_name
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        task_type = str(raw.get("task_type", "unknown"))
        for item in raw.get("instances", []) or []:
            tasks.append(
                {
                    "task_id": item.get("id"),
                    "task_type": task_type,
                    "title": item.get("title"),
                    "prompt": item.get("prompt"),
                    "rubric": item.get("rubric", []),
                }
            )
    return tasks


def _extract_run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = payload.get("config", {}) or {}
    evaluation = payload.get("evaluation", {}) or {}
    corrected = evaluation.get("corrected_metrics", {}) or {}
    structural = evaluation.get("structural_metrics", {}) or {}
    panel = evaluation.get("judge_panel", {}) or {}
    irr = panel.get("inter_rater_reliability", {}) or {}

    return {
        "run_id": payload.get("run_id"),
        "block_id": payload.get("block_id"),
        "task_id": cfg.get("task_id"),
        "task_type": cfg.get("task_type"),
        "topology": cfg.get("topology"),
        "consensus_method": cfg.get("consensus"),
        "agent_count": cfg.get("agent_count"),
        "disagreement_level": cfg.get("disagreement_level"),
        "model_assignment": _format_model_assignment(cfg.get("model_assignment", []) or []),
        "repetition": cfg.get("repetition"),
        "raw_bt_quality": evaluation.get("quality_score"),
        "consensus_win_rate": corrected.get("consensus_win_rate"),
        "normalized_bt": corrected.get("normalized_bt_score"),
        "num_bt_candidates": corrected.get("num_bt_candidates"),
        "mtld_consensus": structural.get("mtld"),
        "readability_consensus": structural.get("readability_fk_grade"),
        "coherence_consensus": structural.get("coherence_mean"),
        "prompt_relevance_consensus": structural.get("prompt_relevance"),
        "connective_density_consensus": structural.get("connective_density"),
        "word_count_consensus": structural.get("word_count"),
        "repetition_rate_consensus": structural.get("repetition_rate"),
        "mean_judge_kappa": irr.get("mean_cohen_kappa"),
        "timestamp": payload.get("timestamp_utc"),
    }


def _extract_agent_rows(
    payload: dict[str, Any],
    *,
    model_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    outputs = payload.get("outputs", []) or []
    rows: list[dict[str, Any]] = []

    for fallback_index, output in enumerate(outputs):
        model_name = str(output.get("model_name", ""))
        model_meta = model_lookup.get(model_name, {})
        text = str(output.get("text", ""))
        structural = output.get("structural_metrics", {}) or {}

        word_count = _safe_int(structural.get("word_count"))
        if word_count is None:
            word_count = _token_count(text)

        stripped = text.strip()
        is_empty = stripped == "" or stripped == "[EMPTY_OUTPUT_AFTER_RETRIES]"

        rows.append(
            {
                "run_id": payload.get("run_id"),
                "agent_index": _parse_agent_index(str(output.get("agent_id", "")), fallback=fallback_index),
                "model_name": model_name,
                "provider": model_meta.get("provider"),
                "tier": model_meta.get("tier"),
                "text": text,
                "word_count": word_count,
                "input_tokens": output.get("input_tokens"),
                "output_tokens": output.get("output_tokens"),
                "latency_ms": output.get("latency_ms"),
                "mtld": structural.get("mtld"),
                "readability_fk_grade": structural.get("readability_fk_grade"),
                "coherence_mean": structural.get("coherence_mean"),
                "connective_density": structural.get("connective_density"),
                "repetition_rate": structural.get("repetition_rate"),
                "is_empty": is_empty,
            }
        )

    return rows


def _extract_pairwise_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation = payload.get("evaluation", {}) or {}
    panel = evaluation.get("judge_panel", {}) or {}
    records = panel.get("pairwise_records", []) or []

    rows: list[dict[str, Any]] = []
    for record in records:
        candidate_i = record.get("candidate_i")
        candidate_j = record.get("candidate_j")
        per_judge = record.get("per_judge", {}) or {}

        per_judge_details = record.get("per_judge_details", {}) or {}

        for judge_model, vote in per_judge.items():
            detail = per_judge_details.get(judge_model, {}) if isinstance(per_judge_details, dict) else {}
            rows.append(
                {
                    "run_id": payload.get("run_id"),
                    "candidate_i": candidate_i,
                    "candidate_j": candidate_j,
                    "judge_model": judge_model,
                    "judge_vote": vote,
                    "ordering": detail.get("ordering"),
                    "confidence": detail.get("confidence"),
                }
            )

    return rows


def _flatten_run(payload: dict[str, Any], run_summary: dict[str, Any]) -> dict[str, Any]:
    cfg = payload.get("config", {}) or {}
    task = payload.get("task", {}) or {}
    consensus = payload.get("consensus", {}) or {}
    evaluation = payload.get("evaluation", {}) or {}

    flattened = {
        **run_summary,
        "status": payload.get("status"),
        "temperature": cfg.get("temperature"),
        "prompt_strategy": cfg.get("prompt_strategy"),
        "quality_threshold": cfg.get("quality_threshold"),
        "task_title": task.get("title"),
        "task_prompt": task.get("prompt"),
        "task_rubric": task.get("rubric", []),
        "consensus_selected_agent_id": consensus.get("selected_agent_id"),
        "consensus_confidence": consensus.get("confidence"),
        "consensus_text": consensus.get("selected_text"),
        "disagreement": evaluation.get("disagreement", {}),
        "outputs": payload.get("outputs", []),
        "pairwise_records": (evaluation.get("judge_panel", {}) or {}).get("pairwise_records", []),
        "corrected_metrics": evaluation.get("corrected_metrics", {}),
    }
    return flattened


def _infer_date_range(run_summaries: list[dict[str, Any]]) -> tuple[str, str]:
    timestamps: list[datetime] = []
    for row in run_summaries:
        value = row.get("timestamp")
        if not value:
            continue
        try:
            timestamps.append(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
        except ValueError:
            continue

    if not timestamps:
        return "unknown", "unknown"

    start = min(timestamps).date().isoformat()
    end = max(timestamps).date().isoformat()
    return start, end


def _infer_total_cost_usd(results_dir: Path) -> float | None:
    cost_log = results_dir / "cost_log.jsonl"
    if not cost_log.exists():
        return None

    total_cost = 0.0
    found = False
    with cost_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = _safe_float(payload.get("total_cost_usd"))
            if value is None:
                continue
            total_cost = max(total_cost, value)
            found = True

    return total_cost if found else None


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _build_datacard(
    *,
    run_count: int,
    agent_output_count: int,
    pairwise_count: int,
    date_start: str,
    date_end: str,
    total_cost_usd: float | None,
) -> str:
    cost_text = f"${total_cost_usd:,.2f}" if total_cost_usd is not None else "not available"
    return f"""---
license: cc-by-4.0
language:
  - en
pretty_name: "Agents Disagree: Multi-Agent Consensus Experiment"
task_categories:
  - text-generation
  - text-classification
tags:
  - multi-agent
  - consensus
  - llm-evaluation
  - orchestration
---

# Dataset Card for Agents Disagree

## Dataset Summary

This dataset contains outputs and evaluations from a multi-agent LLM consensus experiment spanning 3 providers, 5 agent models, and 3 judge models.
It captures run-level configuration, per-agent text outputs, pairwise judge outcomes, and post-hoc structural characterization metrics.

- Runs: **{run_count}**
- Agent outputs: **{agent_output_count}**
- Pairwise judgments: **{pairwise_count}**

## Supported Tasks

- Multi-agent evaluation
- Consensus mechanism comparison
- LLM-as-judge analysis

## Languages

English (`en`)

## Dataset Structure

- `runs.jsonl`: One JSON object per run (self-contained run record with config, task, outputs, consensus, evaluation)
- `runs_summary.csv`: One row per run with key run metadata, BT/corrected metrics, and consensus structural metrics
- `agent_outputs.csv`: One row per agent output with model metadata, text, token/latency metadata, and structural metrics
- `pairwise_judgments.csv`: One row per pairwise comparison per judge
- `tasks.json`: Task prompts and rubrics snapshot
- `models.json`: Model configuration snapshot (agents + judges)
- `DATACARD.md`: This dataset card
- `CODEBOOK.md`: Field-level schema documentation for CSV exports

## Dataset Creation

### Methodology

The dataset was produced by a reproducible orchestration pipeline:
1. Multiple LLM agents generate candidate responses for analytical and creative tasks.
2. A consensus mechanism selects/synthesizes a final answer.
3. A 3-model judge panel performs pairwise comparisons and Bradley-Terry aggregation.
4. Structural characterization metrics are computed locally from text.

### Date Range

- Start: **{date_start}**
- End: **{date_end}**

### Cost

- Approximate total API cost (from `results/cost_log.jsonl`): **{cost_text}**

## Considerations

### Biases

- LLM-as-judge bias remains possible (style preference, model-family bias, position artifacts).
- Only two task types are included (analytical, creative), limiting broad generalization.

### Limitations

- Structural descriptor metrics are descriptive and should not be treated as direct quality proxies.
- Some outputs may be truncated by model token limits.
- Judge confidence and ordering may be unavailable in older run records.

## Citation

```bibtex
@dataset{{agents_disagree_2026,
  title={{Agents Disagree: Multi-Agent Consensus Experiment Dataset}},
  author={{Petrov, Alexei and Chen, Elena and Rivera, Marcus}},
  year={{2026}},
  publisher={{HuggingFace}},
  note={{Citation placeholder: replace with final DOI/URL}}
}}
```
"""


def _build_codebook() -> str:
    return """# CODEBOOK

This file documents all CSV fields exported by `scripts/export_dataset.py`.

## runs_summary.csv

| Column | Type | Description |
|---|---|---|
| run_id | string | Unique run identifier |
| block_id | string | Experimental block identifier |
| task_id | string | Task identifier |
| task_type | string | Task type (`analytical` or `creative`) |
| topology | string | Orchestration topology |
| consensus_method | string | Consensus mechanism used in this run |
| agent_count | int | Number of agents |
| disagreement_level | int | Experimental disagreement condition |
| model_assignment | string | Pipe-delimited model aliases assigned to agents |
| repetition | int | Repetition index within condition |
| raw_bt_quality | float | Raw Bradley-Terry score for consensus candidate |
| consensus_win_rate | float | Fraction of pairwise comparisons won by consensus |
| normalized_bt | float | Candidate-count-corrected BT score |
| num_bt_candidates | int | Number of candidates in BT pool |
| mtld_consensus | float | Consensus MTLD |
| readability_consensus | float | Consensus Flesch-Kincaid grade |
| coherence_consensus | float | Consensus adjacent-sentence coherence |
| prompt_relevance_consensus | float | Consensus prompt relevance |
| connective_density_consensus | float | Consensus discourse connective density |
| word_count_consensus | int | Consensus word count |
| repetition_rate_consensus | float | Consensus 4-gram repetition rate |
| mean_judge_kappa | float | Mean pairwise Cohen's kappa among judges |
| timestamp | string | UTC timestamp (ISO-8601) |

## agent_outputs.csv

| Column | Type | Description |
|---|---|---|
| run_id | string | Parent run identifier |
| agent_index | int | Agent index within run |
| model_name | string | Model alias used by agent |
| provider | string | Model provider (`anthropic`, `openai`, `google`, etc.) |
| tier | string | Model tier label from config |
| text | string | Raw generated text |
| word_count | int | Output word count |
| input_tokens | int | Input token count reported by client |
| output_tokens | int | Output token count reported by client |
| latency_ms | float | Generation latency in milliseconds |
| mtld | float | MTLD lexical diversity |
| readability_fk_grade | float | Flesch-Kincaid grade |
| coherence_mean | float | Adjacent-sentence coherence |
| connective_density | float | Discourse connective density |
| repetition_rate | float | 4-gram repetition rate |
| is_empty | bool | True for empty/sentinel outputs |

## pairwise_judgments.csv

| Column | Type | Description |
|---|---|---|
| run_id | string | Parent run identifier |
| candidate_i | string | First candidate id in comparison |
| candidate_j | string | Second candidate id in comparison |
| judge_model | string | Judge model alias |
| judge_vote | string | Judge vote label (`left`, `right`, `tie`, etc.) |
| ordering | string/null | Presentation ordering if available (`ab`/`ba`) |
| confidence | float/null | Judge confidence if available |
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_lookup, models_snapshot = _load_model_snapshot(args.models_config)
    tasks_snapshot = _load_tasks(args.tasks_dir)

    run_paths = sorted(args.results_dir.glob("block*/run_*.json"))
    runs_summary_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    flat_runs: list[dict[str, Any]] = []

    for run_path in run_paths:
        payload = _load_json(run_path)
        run_summary = _extract_run_summary(payload)
        runs_summary_rows.append(run_summary)

        agent_rows.extend(_extract_agent_rows(payload, model_lookup=model_lookup))
        pairwise_rows.extend(_extract_pairwise_rows(payload))
        flat_runs.append(_flatten_run(payload, run_summary))

    runs_jsonl_path = args.output_dir / "runs.jsonl"
    with runs_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in flat_runs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_csv(args.output_dir / "runs_summary.csv", runs_summary_rows, RUNS_SUMMARY_COLUMNS)
    _write_csv(args.output_dir / "agent_outputs.csv", agent_rows, AGENT_OUTPUT_COLUMNS)
    _write_csv(args.output_dir / "pairwise_judgments.csv", pairwise_rows, PAIRWISE_COLUMNS)

    (args.output_dir / "tasks.json").write_text(
        json.dumps(tasks_snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "models.json").write_text(
        json.dumps(models_snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    date_start, date_end = _infer_date_range(runs_summary_rows)
    total_cost_usd = _infer_total_cost_usd(args.results_dir)

    (args.output_dir / "DATACARD.md").write_text(
        _build_datacard(
            run_count=len(runs_summary_rows),
            agent_output_count=len(agent_rows),
            pairwise_count=len(pairwise_rows),
            date_start=date_start,
            date_end=date_end,
            total_cost_usd=total_cost_usd,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "CODEBOOK.md").write_text(_build_codebook(), encoding="utf-8")

    print("Dataset export complete")
    print(f"Runs exported: {len(runs_summary_rows)}")
    print(f"Agent outputs exported: {len(agent_rows)}")
    print(f"Pairwise judgments exported: {len(pairwise_rows)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
