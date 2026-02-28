# T-030 Report: Structural Quality Layer + Judge Replacement

## Scope Completed
Implemented all requested T-030 deliverables in `agents-disagree-experiments`.

## 1) Structural quality module added
Created: `src/evaluation/structural_quality.py`

Implemented metrics (all local, no LLM calls):
- MTLD lexical diversity
- Flesch-Kincaid readability grade (via `textstat`, with deterministic fallback)
- Adjacent-sentence coherence (sentence-transformer cosine similarity, lexical fallback when unavailable)
- Prompt relevance (prompt/output embedding cosine similarity)
- Logical connective density (causal/contrastive/additive)
- Word count
- Repetition rate (duplicate 4-gram occurrence fraction)

Exposed API:
- `StructuralMetrics` dataclass
- `compute_structural_metrics(...)`
- `compute_composite_score(...)` (z-normalized analytical-prior composite)

## 2) Runner integration completed
Updated: `src/runner.py`

- Imports structural metric function.
- Computes structural metrics for:
  - final consensus output (`result["evaluation"]["structural_metrics"]`)
  - each individual agent output (`result["outputs"][i]["structural_metrics"]`)
- Uses `all-MiniLM-L6-v2` in normal execution.
- Uses disabled embedding mode in `dry_run` to keep tests deterministic/fast.
- Added defensive error capture (`structural_metrics_error`) if computation fails.

## 3) Reprocessing script completed
Created: `scripts/compute_structural_metrics.py`

Script behavior:
- Walks `results/block*/run_*.json`
- Computes and injects structural metrics for consensus + agent outputs
- Preserves existing run payload structure
- Writes CSV summary: `results/structural_metrics_summary.csv`
- Prints grouped summary statistics:
  - by block
  - by condition (`task_type|topology|consensus`)
- Prints one sample consensus metrics row

## 4) Gemini judge replaced with DeepSeek
Updated: `config/models.yaml`

Replaced judge entry with:
- `deepseek-v3p2`
- provider `openai` (OpenAI-compatible gateway routing)
- costs/rate limits per spec

## 5) Dependencies updated
Updated:
- `requirements.txt`
- `pyproject.toml`

Added:
- `textstat>=0.7.0`
- `lexical-diversity>=0.1.1`

## 6) Documentation updated
Updated: `EXPERIMENT_PLAN.md`

Added section:
- `## Structural Quality Validation Layer`
- Includes all 7 structural metrics and validation rationale text.

## Additional test coverage added
Created: `tests/test_structural_quality.py`

Covers:
- basic metric computation
- composite score output
- repetition penalty detection

## Validation Runs

### Pytest
Command:
- `python -m pytest tests/`

Result:
- `20 passed`

### Reprocessing script
Command:
- `python scripts/compute_structural_metrics.py`

Result:
- Run files found: `400`
- Processed: `400`
- Skipped: `0`
- CSV written: `results/structural_metrics_summary.csv`

### Sample structural metrics output
From script sample block:

```json
{
  "run_id": "run_0016b7911d3e",
  "block_id": "block0_calibration",
  "task_type": "analytical",
  "topology": "flat",
  "consensus": "simple_vote",
  "agent_count": 1,
  "disagreement_level": 1,
  "condition": "analytical|flat|simple_vote",
  "candidate_type": "consensus",
  "candidate_id": "final_consensus",
  "mtld": 35.34152892342644,
  "readability_fk_grade": 3.9715673981191237,
  "coherence_mean": 0.3684076595989007,
  "prompt_relevance": 0.6887974046368639,
  "connective_density": 0.05673758865248227,
  "word_count": 1196,
  "repetition_rate": 0.1777032690695725,
  "composite_score": -1.5181488196428055
}
```

## Files Changed
- `src/evaluation/structural_quality.py` (new)
- `src/evaluation/__init__.py`
- `src/runner.py`
- `scripts/compute_structural_metrics.py` (new)
- `config/models.yaml`
- `requirements.txt`
- `pyproject.toml`
- `EXPERIMENT_PLAN.md`
- `tests/test_structural_quality.py` (new)

## Notes
- Sentence-transformer metrics now execute with local embeddings (`all-MiniLM-L6-v2`) in non-dry runs.
- Script emits benign Windows symlink cache warning from `huggingface_hub`; computation still completes successfully.
