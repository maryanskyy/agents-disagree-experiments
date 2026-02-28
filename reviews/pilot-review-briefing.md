# Pilot Results Review Briefing — February 28, 2026

## Executive Summary

The pilot phase (Phase A) of our "When Agents Disagree" experiment has completed. **280/280 runs finished with 0 failures** in ~18.5 minutes, costing **$48.50**. The main batch is now running. Your task: review the pilot data thoroughly and identify any issues BEFORE the main batch results become our paper's evidence.

## GO/NO-GO Automated Assessment

The automated pilot report recommends **GO**:
- Inter-judge kappa: **0.934** (threshold: > 0.4) ✅
- Disagreement levels separable: **YES** (spread=0.405, threshold: > 0.05) ✅
- Preliminary paradox signal: **Q(n=2)=0.313 > Q(n=3)=0.228** (delta=-0.085) ✅

## KEY OBSERVATIONS FROM INITIAL SCAN

### 1. Block 0 Calibration — Quality Always 0.50
All 240 calibration runs show quality_score = 0.50 regardless of model or task type. Mathematically expected for Bradley-Terry with a single agent (ties with itself). BUT: Block 0 does NOT differentiate model quality.

### 2. Block 4 Paradox Signal — CONFIRMED
- n=2: mean quality = **0.3152** (SD=0.058, 11 runs)
- n=3: mean quality = **0.2301** (SD=0.047, 11 runs)
- Delta: **-0.085** (n=3 is WORSE than n=2)

### 3. Block 1 Disagreement — No Clear Pattern Yet
- Level 1: 0.357 (n=5), Level 2: 0.319 (n=4), Level 3: 0.337 (n=5), Level 4: 0.305 (n=4), Level 5: 0.379 (n=4)
- No inverted-U visible. Only 22 runs.

### 4. Judge Reliability — Variable
- Pilot report kappa=0.934 (aggregated). Sampling across blocks: mean=0.788, min=0.026, max=1.0

### 5. Quality Scores Are Low — all multi-agent runs between 0.16 and 0.50

### 6. Cost: $48.50 across 3,322 API calls

## DATA LOCATIONS
All under `C:\Users\Artem\Desktop\agents-disagree-experiments\results\`:
- `block0_calibration/` (240 runs), `block1_disagreement_dividend/` (22 runs), `block4_quorum_paradox/` (22 runs), `human_eval/` (86 runs), `pilot_report.json`, `cost_log.jsonl`, `pilot_run.log`

## REFERENCE DOCS
- `ANALYSIS_PLAN.md`, `EXPERIMENT_PLAN.md`, `EVALUATION_METHODOLOGY.md`
- Research questions: `C:\Users\Artem\.openclaw\workspace\research\agenda\research-questions.md`
- Config: `config/models.yaml`, `config/experiment_matrix.yaml`
