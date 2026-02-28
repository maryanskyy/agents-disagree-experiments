# T-025 Final Revision Report

**Task:** Final revisions from strategic reviews (pilot-first, matrix slimming, reproducibility hardening, handover readiness)
**Date:** 2026-02-28
**Repo:** `C:\Users\Artem\Desktop\agents-disagree-experiments`

---

## Summary

All requested T-025 changes were implemented. The repository now supports phased execution (`pilot -> full -> all`), has Block 5 removed, Block 4 paradox replication increased, pinned Google model versions, replaced contamination-prone tasks, pre-registered analysis documentation, MacBook handover instructions, and enforced cost guardrails.

---

## Implemented Changes

### 1) Phased execution (Pilot / Full / All)
- Updated `scripts/run_experiments.py`:
  - added `--phase {pilot,full,all}`
  - added deterministic pilot manifest generation:
    - full `block0_calibration`
    - 20 runs from `block1_disagreement_dividend`
    - 20 runs from `block4_quorum_paradox`
  - added `pilot_manifest.json` + `pilot_report.json` generation
- `pilot_report.json` now includes:
  - mean inter-judge kappa
  - disagreement rate per level + separability check
  - preliminary paradox signal (`Q_n3 - Q_n2`)
  - cost so far
  - GO/NO-GO recommendation

### 2) Cut Block 5
- Removed `block5_interaction_probe` from `config/experiment_matrix.yaml`.
- Block count reduced accordingly.

### 3) Increase Block 4 reps + controls
- `block4_quorum_paradox` reps increased **4 -> 6**.
- Homogeneous-strong control (`homogeneous_opus`) retained at n=2/3/5.

### 4) Pin Google model versions
- Updated `config/models.yaml`:
  - `gemini-2.5-pro` -> `gemini-2.5-pro-preview-05-06`
  - `gemini-2.0-flash` -> `gemini-2.0-flash-001`
  - `gemini-1.5-pro` -> `gemini-1.5-pro-001`
  - added required comment:
    - `# CRITICAL: Pinned versions prevent mid-experiment drift`
- Updated `src/models/google_client.py` to explicitly pass configured model name verbatim and record configured/response model metadata.

### 5) Replace contamination-prone tasks
- `config/tasks/analytical.yaml`
  - replaced old logicians-bar task with novel multi-constraint SOC triage deduction puzzle (`analytical_01_signal_triage_deduction`)
- `config/tasks/creative.yaml`
  - replaced old entanglement-for-child task with novel phosphorescence lighthouse explanation prompt (`creative_08_lighthouse_memory_protocol`)
- Rubric structure preserved.

### 6) Added pre-registered analysis plan
- Created `ANALYSIS_PLAN.md` with:
  - H1-H4 primary hypotheses
  - exact test per RQ
  - Holm-Bonferroni correction
  - effect-size threshold (Cohen's d >= 0.3)
  - stopping rules
  - confirmatory vs exploratory split
  - primary comparison count and adjusted alpha framing

### 7) Added MacBook handover guide
- Created `HANDOVER.md` including:
  - prerequisites
  - setup steps
  - pre-flight validation
  - pilot/full run commands
  - monitoring/resume guidance
  - troubleshooting
  - budget guardrails
  - success checklist

### 8) Cost guardrails
- Added max-cost guardrail wiring:
  - `scripts/run_experiments.py` now accepts `--max-cost` (default 1500)
  - `RunnerConfig` includes `max_cost_usd`
  - `src/runner.py` now pauses when cost cap is reached and writes warning/status to `progress.json`
  - `src/utils/checkpoint.py` progress schema extended with `status`, `warning`, `max_cost_usd`

### 9) Documentation updates
- Updated `README.md`:
  - phased execution usage
  - pilot outputs
  - max-cost behavior
- Updated `EXPERIMENT_PLAN.md`:
  - removed Block 5
  - increased Block 4 reps
  - phased approach
  - replaced tasks
  - cost estimate snapshot
  - known limitations section

### 10) Known limitations section added
- Added to `EXPERIMENT_PLAN.md`:
  - 2 providers only
  - 2 task types only
  - reduced scale / possible underpowered interactions
  - residual LLM-as-judge bias risk

---

## Additional Supporting Work

- Added `tests/test_phase_manifests.py` to verify pilot/full manifest partition behavior.

---

## Validation Results

### 1) Tests
Command:
```bash
python -m pytest tests/
```
Result: **15 passed**.

### 2) Dry-run setup validation
Command:
```bash
python scripts/validate_setup.py --dry-run
```
Result: completed successfully.

### 3) Cost estimate
Command:
```bash
python scripts/estimate_cost.py --manifest results/manifest_t025_temp.json --phase all
```
Result summary:
- Estimated runs: **2,880**
- Estimated API calls: **123,072**
- Estimated total cost: **$960.19**

---

## Final Readiness

The repository is now configured for **clone-and-run execution on a fresh MacBook Air M4** with pilot gating, hard cost limits, updated experiment matrix, and pre-registered analysis/handover documentation.
