#!/usr/bin/env python3
"""Decoupled Evaluation Pass: Re-score V4 outputs with independent judges.

This script reads existing V4 experiment results and re-evaluates them using
an independent judge panel (gpt-4o-mini, gemini-2.0-flash-001, glm-5) that
was NOT involved in the original selection process. This addresses the
selection-evaluation circularity concern.

Usage:
    python scripts/run_decoupled_eval.py [--block BLOCK_PREFIX] [--dry-run] [--max-concurrent 6]

Estimated cost: $30-50 for all 210 runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.evaluation.llm_judge import JudgePanel, PairwiseJudge
from src.models import OpenAIModelClient, GoogleModelClient
from src.utils.cost_tracker import CostTracker
from src.utils.token_manager import TokenManager, get_api_base_url, get_api_org_id

RESULTS_DIR = ROOT / "results" / "v4"
OUTPUT_DIR = ROOT / "results" / "v4" / "decoupled_eval"

EVAL_JUDGES = [
    {"model": "gpt-4o-mini", "provider": "openai"},
    {"model": "gemini-2.0-flash-001", "provider": "google"},
    {"model": "glm-5", "provider": "openai"},
]

BLOCKS = [
    "scale_diverse_strong_judge_based",
    "scale_homo_opus_judge_based",
    "scale_diverse_mixed_judge_based",
    "scale_diverse_strong_simple_vote",
    "scale_diverse_strong_synthesis",
]

logger = logging.getLogger("decoupled_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decoupled Evaluation Pass")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--block", action="append", default=[],
                        help="Run only specified block(s)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip results that already have eval output files")
    parser.add_argument("--max-concurrent", type=int, default=6)
    parser.add_argument("--max-cost", type=float, default=60.0,
                        help="Safety cap on total spend (default $60)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(output_dir / "decoupled_eval.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)


def create_eval_judge_panel(token_mgr: TokenManager) -> JudgePanel:
    """Create the independent evaluation judge panel."""
    base_url = get_api_base_url()
    org_id = get_api_org_id()
    token = token_mgr.get_token()
    extra_hdrs = {"OpenAI-Organization": org_id}
    openai_base = f"{base_url}/v1"

    judges: list[PairwiseJudge] = []

    for jspec in EVAL_JUDGES:
        model_name = jspec["model"]
        provider = jspec["provider"]

        if provider == "google":
            client = GoogleModelClient(
                model_alias=model_name,
                api_model=model_name,
                api_key=token,
                base_url=openai_base,
                extra_headers=extra_hdrs,
            )
        else:
            client = OpenAIModelClient(
                model_alias=model_name,
                api_model=model_name,
                api_key=token,
                base_url=openai_base,
                extra_headers=extra_hdrs,
            )

        pj = PairwiseJudge(judge_client=client)
        judges.append(pj)
        logger.info(f"Eval judge initialized: {model_name} ({provider})")

    return JudgePanel(judges=judges)


def load_result_files(results_dir: Path, block_filters: list[str]) -> list[tuple[str, Path]]:
    """Load all V4 result JSON files, optionally filtered by block."""
    files = []
    for block_dir in sorted(results_dir.iterdir()):
        if not block_dir.is_dir() or block_dir.name.startswith("."):
            continue
        if block_dir.name in ("decoupled_eval", "human_eval", "manifests"):
            continue
        if block_filters and not any(block_dir.name.startswith(b) for b in block_filters):
            continue
        for f in sorted(block_dir.glob("v4_*.json")):
            files.append((block_dir.name, f))
    return files


async def eval_single_result(
    panel: JudgePanel,
    result_path: Path,
    block_name: str,
    output_dir: Path,
    cost_tracker: CostTracker,
    semaphore: asyncio.Semaphore,
    seed: int,
) -> dict | None:
    """Re-evaluate a single result file with the independent judge panel."""
    async with semaphore:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))

            if result.get("status") != "ok":
                logger.warning(f"Skipping non-ok result: {result_path.name}")
                return None

            task = result.get("task", {})
            task_prompt = task.get("prompt", "")
            task_type = task.get("title", "unknown")
            rubric = task.get("rubric", [])

            candidates = []
            candidate_labels = []

            for i, out in enumerate(result.get("outputs", [])):
                text = out.get("text", "")
                if text and "[EMPTY_OUTPUT_AFTER_RETRIES]" not in text:
                    candidates.append(text)
                    candidate_labels.append(f"agent_{i}")

            consensus = result.get("consensus", {})
            consensus_text = consensus.get("selected_text", "")
            if consensus_text and "[EMPTY_OUTPUT_AFTER_RETRIES]" not in consensus_text:
                candidates.append(consensus_text)
                candidate_labels.append("consensus")

            if len(candidates) < 2:
                logger.warning(f"Skipping {result_path.name}: fewer than 2 candidates")
                return None

            panel_eval = await panel.evaluate_candidates(
                task_type=task_type,
                task_prompt=task_prompt,
                rubric=rubric,
                outputs=candidates,
                seed=seed,
            )

            eval_result = {
                "source_file": result_path.name,
                "block": block_name,
                "task_type": task_type,
                "task_id": task.get("title", ""),
                "run_id": result.get("run_id", ""),
                "candidate_labels": candidate_labels,
                "eval_judges": [j["model"] for j in EVAL_JUDGES],
                "bt_scores": {candidate_labels[i]: float(s) for i, s in enumerate(panel_eval.bt_scores)},
                "ranking": [candidate_labels[i] for i in panel_eval.ranking],
                "per_judge_bt_scores": {
                    judge_name: {candidate_labels[i]: float(s) for i, s in enumerate(scores)}
                    for judge_name, scores in panel_eval.per_judge_bt_scores.items()
                },
                "inter_rater_reliability": panel_eval.inter_rater_reliability,
                "pairwise_records": [
                    {
                        "candidate_i": candidate_labels[r.candidate_i],
                        "candidate_j": candidate_labels[r.candidate_j],
                        "per_judge": r.per_judge,
                    }
                    for r in (panel_eval.pairwise_records or [])
                ],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

            if "consensus" in candidate_labels:
                consensus_idx = candidate_labels.index("consensus")
                consensus_bt = panel_eval.bt_scores[consensus_idx]
                agent_bts = [panel_eval.bt_scores[i] for i in range(len(candidates)) if i != consensus_idx]
                wins = sum(1 for abt in agent_bts if consensus_bt > abt)
                eval_result["consensus_bt_score"] = float(consensus_bt)
                eval_result["consensus_rank"] = panel_eval.ranking.index(consensus_idx) + 1
                eval_result["consensus_win_rate"] = wins / len(agent_bts) if agent_bts else 0.0

            out_path = output_dir / block_name / result_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(eval_result, indent=2, default=str), encoding="utf-8")

            consensus_info = ""
            if "consensus" in candidate_labels:
                consensus_info = f" consensus_bt={eval_result['consensus_bt_score']:.3f}"
            logger.info(f"OK {block_name}/{result_path.name}{consensus_info}")
            return eval_result

        except Exception as e:
            logger.error(f"FAIL {block_name}/{result_path.name}: {e}", exc_info=True)
            return None


async def main():
    load_dotenv()
    args = parse_args()
    _setup_logging(args.output_dir)

    logger.info("=" * 60)
    logger.info("DECOUPLED EVALUATION PASS")
    logger.info(f"Results dir: {args.results_dir}")
    logger.info(f"Output dir:  {args.output_dir}")
    logger.info(f"Eval judges: {[j['model'] for j in EVAL_JUDGES]}")
    logger.info(f"Max concurrent: {args.max_concurrent}")
    logger.info(f"Resume:      {args.resume}")
    logger.info("=" * 60)

    block_filters = args.block or []
    files = load_result_files(args.results_dir, block_filters)
    logger.info(f"Found {len(files)} result files to re-evaluate")

    if args.resume:
        already_done = set()
        for block_dir in args.output_dir.iterdir() if args.output_dir.exists() else []:
            if block_dir.is_dir():
                for f in block_dir.glob("*.json"):
                    already_done.add(f.name)
        before = len(files)
        files = [(b, f) for b, f in files if f.name not in already_done]
        logger.info(f"Resume: {before - len(files)} already done, {len(files)} remaining")

    if args.dry_run:
        for block, f in files:
            print(f"  {block}/{f.name}")
        print(f"\nTotal: {len(files)} files")
        print(f"Estimated cost: ${len(files) * 0.15:.2f} - ${len(files) * 0.25:.2f}")
        return

    token_mgr = TokenManager()
    panel = create_eval_judge_panel(token_mgr)

    pricing = {
        "gpt-4o-mini": ,
        "gemini-2.0-flash-001": ,
        "glm-5": ,
    }
    cost_tracker = CostTracker(pricing=pricing, results_dir=args.output_dir)

    semaphore = asyncio.Semaphore(args.max_concurrent)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        eval_single_result(panel, fpath, block, args.output_dir, cost_tracker, semaphore, args.seed)
        for block, fpath in files
    ]

    results = await asyncio.gather(*tasks)
    successful = [r for r in results if r is not None]

    logger.info("=" * 60)
    logger.info(f"COMPLETE: {len(successful)}/{len(files)} files re-evaluated")

    block_wrs = defaultdict(list)
    for r in successful:
        if r.get("consensus_win_rate") is not None:
            block_wrs[r["block"]].append(r["consensus_win_rate"])

    logger.info("\nDecoupled Eval Win Rates (independent judges):")
    logger.info("-" * 50)
    for block in sorted(block_wrs.keys()):
        wrs = block_wrs[block]
        mean_wr = sum(wrs) / len(wrs)
        logger.info(f"  {block}: WR = {mean_wr:.3f} (n={len(wrs)})")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_judges": [j["model"] for j in EVAL_JUDGES],
        "total_files": len(files),
        "successful": len(successful),
        "failed": len(files) - len(successful),
        "per_block": {
            block: {"mean_wr": sum(wrs) / len(wrs), "n": len(wrs)}
            for block, wrs in block_wrs.items()
        },
        "cost": cost_tracker.snapshot(),
    }
    summary_path = args.output_dir / "decoupled_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
