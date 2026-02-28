# T-024 Revision Report

**Task:** Address all committee review findings (fatal + critical issues)
**Date:** 2026-02-28
**Repo:** `C:\Users\Artem\Desktop\agents-disagree-experiments`

## Summary

All requested fixes from T-024 were implemented across evaluation, topology, matrix design, baseline controls, and documentation. The framework now uses pairwise multi-judge LLM evaluation with Bradley-Terry aggregation, semantic disagreement metrics, best-of-N baselines, confound-fixed disagreement levels, added pipeline topology, and human-eval sheet generation.

---

## Implemented Fixes

### FIX 1 (FATAL): Pairwise LLM-as-Judge + Bradley-Terry
- Rewrote `src/evaluation/llm_judge.py` to:
  - run **pairwise A/B comparisons** (not absolute scoring)
  - evaluate **both orderings** (A/B and B/A)
  - include task-specific rubrics in judge prompt
  - aggregate pairwise outcomes with **Bradley-Terry** (`bradley_terry_scores`)
- Judge prompt is blind to topology/consensus/agent count.
- Old heuristic metrics retained only as dry-run helpers in `metrics.py`.

### FIX 2 (FATAL): Multi-Judge Panel
- Added panel orchestration (`JudgePanel`) in `llm_judge.py`.
- Configured default 3-judge panel in `config/models.yaml`:
  - `gemini-2.5-pro`
  - `claude-opus-4-6`
  - `claude-sonnet-4-5` (third judge, not Flash)
- Added inter-rater reliability computation (pairwise **Cohen's kappa** + mean).
- Per-judge BT scores are now stored in run results.

### FIX 3 (FATAL): Separate Judge Pool from Agent Pool + Exclusion Logic
- `config/models.yaml` now has separate:
  - `agent_pool`
  - `judge_pool`
- Implemented per-run judge selection logic in `src/runner.py`:
  - excludes agent models from panel (`exclude_agent_models`)
  - prefers family-diverse judges (`prefer_different_families`)
  - uses reserve judges when needed.

### FIX 4 (CRITICAL): Embedding-Based Disagreement Metrics
- Replaced disagreement core with semantic metrics in `src/evaluation/disagreement.py`.
- Added local embedding model support:
  - `sentence-transformers/all-MiniLM-L6-v2`
- Implemented:
  - semantic pairwise cosine similarity
  - `disagreement_rate = 1 - mean_semantic_similarity`
  - vote entropy (semantic clustering)
  - lexical diversity (secondary)
- Added dependency: `sentence-transformers`.

### FIX 5 (CRITICAL): Best-of-N Baseline
- Added `best_of_n` topology: `src/topologies/best_of_n.py`.
- Added baseline blocks in matrix:
  - `block1_best_of_n_baseline`
  - `block4_best_of_n_baseline`
- Best-of-N runs sample N single-agent outputs and select best via judge panel.

### FIX 6 (CRITICAL): Increase Replications + Remove Threshold Multiplication
- Updated `config/experiment_matrix.yaml`:
  - Block 2 reps: **2**
  - Block 3 reps: **3**
  - Block 5 reps: **2**
- Removed redundant threshold expansion in manifest generation.
- Added post-hoc thresholds (`posthoc_quality_thresholds`) for Block 3.
- Updated `src/manifest.py` and `scripts/estimate_cost.py` accordingly.

### FIX 7 (CRITICAL): Disagreement Level Confound
- Replaced disagreement-level design in matrix with independent factor structure:
  - L1: same prompt, temp 0.3
  - L2: same prompt, temp 0.7
  - L3: perspective prompts, temp 0.3
  - L4: perspective prompts, temp 0.7
  - L5: perspective prompts + model mix framing, temp 0.9

### FIX 8 (CRITICAL): Log Intermediate Debate Rounds
- Added round logging for quorum and sequential flows.
- `src/topologies/quorum.py` now stores draft/revision/final rounds.
- Runner writes `debate_rounds` into run JSON.

### FIX 9 (IMPORTANT): Pipeline/Sequential Topology
- Added `src/topologies/pipeline.py`.
- Registered in `src/topologies/__init__.py` and runner topology map.
- Added to Blocks 2 and 5 in matrix.

### FIX 10 (IMPORTANT): Homogeneous-Strong Control in Block 4
- Added `homogeneous_opus` control condition in `block4_quorum_paradox`.
- Keeps n=2,3,5 for direct comparison with paradox_strong_weak condition.

### FIX 11 (MODERATE): Human Evaluation Infrastructure
- Added `src/evaluation/human_eval.py`.
- Runner now auto-flags and generates human-eval sheets under `results/human_eval/` for:
  - judge disagreement items,
  - block4 paradox items,
  - deterministic random 15% sample.

### FIX 12: Documentation and Dependency Updates
- Updated:
  - `EXPERIMENT_PLAN.md`
  - `README.md`
  - `requirements.txt`
  - `pyproject.toml`
  - `scripts/estimate_cost.py`

---

## Validation Runs

### 1) Tests
Command:
```bash
python -m pytest tests/
```
Result: **13 passed**.

### 2) Dry-Run Setup Validation
Command:
```bash
python scripts/validate_setup.py --dry-run
```
Result: completed successfully (no errors).

### 3) Cost Estimate
Command:
```bash
python scripts/estimate_cost.py
```
Output summary:
- Estimated runs: **3,456**
- Estimated API calls: **158,656**
  - Agent calls: 14,848
  - Judge calls: 143,808
- Estimated total cost: **$1,147.08**

---

## Notes
- Pairwise panel evaluation is now the production quality path.
- Heuristic metrics are explicitly dry-run-only.
- Run outputs now include per-judge scores, reliability metadata, and debate-round traces for mediation analysis.
