#!/usr/bin/env python3
"""Pre-main-batch validation: checks that pilot review fixes are working.

Run this AFTER applying fixes and running 20-50 validation runs.
It checks:
  1. Judge tie rate < 40% for each judge
  2. GPT-5.2 empty output rate < 10%
  3. Consensus win rate shows variance (not flat 0.50)
  4. Multi-agent inter-rater kappa > 0.40

Usage:
    python scripts/validate_fixes.py [--results-dir results] [--min-runs 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


def load_runs(results_dir: Path) -> list[dict]:
    """Load all run JSON files from block directories."""
    runs = []
    for block_dir in sorted(results_dir.iterdir()):
        if not block_dir.is_dir() or not block_dir.name.startswith("block"):
            continue
        for f in sorted(block_dir.glob("run_*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                runs.append(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return runs


def check_tie_rates(runs: list[dict]) -> dict[str, float]:
    """Compute per-judge tie rate across all pairwise comparisons."""
    judge_ties: dict[str, int] = defaultdict(int)
    judge_total: dict[str, int] = defaultdict(int)

    multi_agent_runs = [r for r in runs if r.get("config", {}).get("agent_count", 1) > 1]
    for run in multi_agent_runs:
        records = run.get("evaluation", {}).get("judge_panel", {}).get("pairwise_records", [])
        for rec in records:
            per_judge = rec.get("per_judge", {})
            for judge, vote in per_judge.items():
                judge_total[judge] += 1
                if vote == "tie":
                    judge_ties[judge] += 1

    rates = {}
    for judge in sorted(judge_total):
        total = judge_total[judge]
        ties = judge_ties[judge]
        rates[judge] = ties / total if total > 0 else 0.0
    return rates


def check_empty_rates(runs: list[dict]) -> dict[str, tuple[int, int]]:
    """Compute per-model empty output rates."""
    model_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [empty, total]
    for run in runs:
        for out in run.get("outputs", []):
            model = out.get("model_name", "unknown")
            model_stats[model][1] += 1
            text = out.get("text", "")
            if not text or not text.strip() or text == "[EMPTY_OUTPUT_AFTER_RETRIES]":
                model_stats[model][0] += 1
    return {m: tuple(v) for m, v in model_stats.items()}


def check_win_rate_variance(runs: list[dict]) -> tuple[float, float, float]:
    """Check if consensus win rates show meaningful variance."""
    win_rates = []
    multi_agent = [r for r in runs if r.get("config", {}).get("agent_count", 1) > 1]
    for run in multi_agent:
        cm = run.get("evaluation", {}).get("corrected_metrics", {})
        wr = cm.get("consensus_win_rate")
        if wr is not None:
            win_rates.append(wr)

    if not win_rates:
        return 0.0, 0.0, 0.0

    mean_wr = sum(win_rates) / len(win_rates)
    variance = sum((x - mean_wr) ** 2 for x in win_rates) / len(win_rates)
    std_wr = variance ** 0.5
    return mean_wr, std_wr, len(win_rates)


def check_kappa(runs: list[dict]) -> tuple[float, int]:
    """Compute mean inter-rater kappa excluding Block 0."""
    kappas = []
    for run in runs:
        if run.get("block_id", "").startswith("block0"):
            continue
        k = run.get("evaluation", {}).get("judge_panel", {}).get(
            "inter_rater_reliability", {}
        ).get("mean_cohen_kappa")
        if k is not None:
            kappas.append(k)
    if not kappas:
        return 0.0, 0
    return sum(kappas) / len(kappas), len(kappas)


def main():
    parser = argparse.ArgumentParser(description="Validate post-fix experiment data")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--min-runs", type=int, default=20, help="Minimum multi-agent runs required")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory {results_dir} not found")
        sys.exit(1)

    runs = load_runs(results_dir)
    multi_agent = [r for r in runs if r.get("config", {}).get("agent_count", 1) > 1]
    print(f"Total runs loaded: {len(runs)}")
    print(f"Multi-agent runs: {len(multi_agent)}")
    print()

    if len(multi_agent) < args.min_runs:
        print(f"WARNING: Only {len(multi_agent)} multi-agent runs (need {args.min_runs})")
        print()

    # Check 1: Tie rates
    passed = True
    print("=" * 60)
    print("CHECK 1: Judge Tie Rates (target: < 30%, genuine ties allowed)")
    print("=" * 60)
    tie_rates = check_tie_rates(runs)
    for judge, rate in sorted(tie_rates.items()):
        status = "PASS" if rate < 0.30 else "FAIL"
        if rate >= 0.30:
            passed = False
        print(f"  {judge}: {rate*100:.1f}% tie rate [{status}]")
    print()

    # Check 2: Empty rates
    print("=" * 60)
    print("CHECK 2: Empty Output Rates (target: < 10%)")
    print("=" * 60)
    empty_stats = check_empty_rates(runs)
    for model, (empty, total) in sorted(empty_stats.items()):
        rate = empty / total if total > 0 else 0
        status = "PASS" if rate < 0.10 else "FAIL"
        if rate >= 0.10:
            passed = False
        print(f"  {model}: {empty}/{total} empty ({rate*100:.1f}%) [{status}]")
    print()

    # Check 3: Win rate variance
    print("=" * 60)
    print("CHECK 3: Consensus Win Rate Variance (target: std > 0.05)")
    print("=" * 60)
    mean_wr, std_wr, n_wr = check_win_rate_variance(runs)
    wr_status = "PASS" if std_wr > 0.05 else "FAIL"
    if std_wr <= 0.05:
        passed = False
    print(f"  Mean win rate: {mean_wr:.4f}")
    print(f"  Std dev: {std_wr:.4f} [{wr_status}]")
    print(f"  N runs: {int(n_wr)}")
    print()

    # Check 4: Kappa
    print("=" * 60)
    print("CHECK 4: Inter-Rater Kappa (target: > 0.40, excl Block 0)")
    print("=" * 60)
    mean_kappa, n_kappa = check_kappa(runs)
    k_status = "PASS" if mean_kappa > 0.40 else "FAIL"
    if mean_kappa <= 0.40:
        passed = False
    print(f"  Mean kappa: {mean_kappa:.4f} [{k_status}]")
    print(f"  N runs: {n_kappa}")
    print()

    # Summary
    print("=" * 60)
    if passed:
        print("OVERALL: ALL CHECKS PASSED - Safe to run main batch")
        print("=" * 60)
        print()
        print("Next step: raise --max-cost and run:")
        print("  python scripts/run_experiments.py --phase full --max-cost 1200")
    else:
        print("OVERALL: SOME CHECKS FAILED - Do NOT run main batch")
        print("=" * 60)
        print()
        print("Fix the failing checks before proceeding.")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
