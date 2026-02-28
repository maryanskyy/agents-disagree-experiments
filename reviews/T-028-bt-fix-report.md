# T-028 BT Normalization Artifact Fix Report

## Summary
Implemented and validated the BT normalization artifact fix by adding corrected quality metrics to both historical results and future run outputs.

## Changes Implemented

### 1) New reprocessing script
- **File:** `scripts/reprocess_results.py`
- Walks all `results/block*/run_*.json`
- Computes and writes `evaluation.corrected_metrics` for each completed run:
  - `consensus_win_rate`
  - `normalized_bt_score`
  - `per_judge_consensus_win_rate`
  - `consensus_vs_best_agent`
  - `num_bt_candidates`
  - `consensus_candidate_id`
  - `consensus_comparisons`
- Preserves all existing run payload content and rewrites JSON with the new metrics added.
- Emits CSV summary:
  - **Output:** `results/corrected_metrics_summary.csv`
  - Columns: `run_id, block_id, agent_count, topology, disagreement_level, task_type, raw_quality, consensus_win_rate, normalized_bt, models`
- Prints before/after summary and required block-level corrected summaries.

### 2) Runner updated for future runs
- **File:** `src/runner.py`
- `_evaluate_final_quality` now computes corrected metrics in addition to raw BT quality.
- `evaluation` payload now includes:
  - existing `quality_score`
  - new `corrected_metrics` object
- Works for both:
  - explicit final-consensus evaluation (`final_consensus` candidate)
  - judge-based consensus selection (selected agent treated as consensus candidate when `final_consensus` is absent)

### 3) Shared corrected metric utility
- **New file:** `src/evaluation/corrected_metrics.py`
- Centralizes corrected metric logic for both runtime and reprocessing script.
- Handles tie scoring and per-judge win-rate reconstruction from pairwise records.

### 4) Analysis plan updated
- **File:** `ANALYSIS_PLAN.md`
- Primary metric for cross-condition inference changed to `consensus_win_rate`.
- Updated H1-H4 mappings and exact tests to use corrected metric.
- Added explicit BT normalization artifact note and secondary use of `normalized_bt_score`.

### 5) Experiment plan updated
- **File:** `EXPERIMENT_PLAN.md`
- Added metric correction section describing:
  - BT comparability artifact
  - corrected metrics (primary + secondary)
  - no rerun requirement due reprocessable stored pairwise records

### 6) Tests
- **New file:** `tests/test_corrected_metrics.py`
- Added coverage for:
  - `final_consensus` tie/win/loss handling
  - selected-agent fallback when `final_consensus` absent

## Validation

### Pytest
Command:
```bash
python -m pytest tests/
```
Result:
- **17 passed**

### Reprocessing run
Command:
```bash
python scripts/reprocess_results.py
```
Result:
- Run files found: 400
- Processed (status=ok): 400
- Skipped: 0
- CSV written: `results/corrected_metrics_summary.csv`

## Corrected Result Snapshots (from reprocessing output)

### BLOCK 4 (Paradox) — Consensus Win Rate by Agent Count
- n=2: **0.500 ± 0.000** (n=16)
- n=3: **0.500 ± 0.000** (n=14)
- n=5: **0.512 ± 0.083** (n=8)

### BLOCK 1 (Disagreement) — Consensus Win Rate by Level
- Level 1: **0.581 ± 0.134** (n=14)
- Level 2: **0.619 ± 0.156** (n=9)
- Level 3: **0.555 ± 0.078** (n=11)
- Level 4: **0.537 ± 0.052** (n=8)
- Level 5: **0.547 ± 0.087** (n=11)

### BLOCK 2 (Topology) — Consensus Win Rate by Topology
- flat: **0.558 ± 0.150** (n=18)
- hierarchical: **0.438 ± 0.288** (n=8)
- quorum: **0.568 ± 0.144** (n=14)
- pipeline: **0.500 ± 0.000** (n=10)

## Notes
- Reprocessing is idempotent: reruns overwrite `evaluation.corrected_metrics` with deterministic recomputation.
- For runs containing `final_consensus` in BT candidates, that ID is always used for corrected metrics.
- Existing raw `quality_score` remains intact for backward compatibility and descriptive reporting.
