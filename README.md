# Agents Disagree Experiments

This repository contains a resumable experiment framework for:

**When Agents Disagree: Quorum-Based Consensus and Adaptive Orchestration Topology for Multi-Agent LLM Pipelines**

It now supports **phased execution (pilot -> full)**, **cost guardrails**, pinned model versions, and pre-registered analysis artifacts.

---

## Prerequisites

- Python 3.11+
- `pip`
- Anthropic API key
- Google API key

---

## Quick Start

1. **Clone + enter repo**
   ```bash
   git clone <repo-url>
   cd agents-disagree-experiments
   ```

2. **Create venv + install**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure keys**
   ```bash
   cp .env.example .env
   # set ANTHROPIC_API_KEY and GOOGLE_API_KEY
   ```

4. **Pre-flight validation**
   ```bash
   python scripts/validate_setup.py --dry-run
   ```

---

## Run Modes (Phased Execution)

### Phase A: Pilot

Runs:
- full `block0_calibration`
- 20 runs from `block1_disagreement_dividend`
- 20 runs from `block4_quorum_paradox`

Command:
```bash
python scripts/run_experiments.py --phase pilot --resume --max-cost 1500
```

Outputs:
- `results/pilot_manifest.json`
- `results/pilot_report.json` with:
  - inter-judge kappa
  - disagreement rate per level
  - preliminary paradox signal (Q at n=2 vs n=3)
  - cost so far
  - GO/NO-GO recommendation

### Phase B: Full

Runs all remaining conditions except pilot-completed runs.

```bash
python scripts/run_experiments.py --phase full --resume --max-cost 1500
```

### Phase C: All (fresh rerun)

Runs entire matrix from scratch (ignores resume).

```bash
python scripts/run_experiments.py --phase all --max-cost 1500
```

---

## Cost Guardrail (Required)

`run_experiments.py` includes:

- `--max-cost` (default: **1500 USD**)
- If cost reaches the guardrail, execution pauses and writes a warning to:
  - `results/progress.json`

Monitor files during run:
- `results/progress.json`
- `results/cost_log.jsonl`
- `results/run_experiments.log`

---

## Cost Estimation

Estimate expected spend before launching:

```bash
python scripts/estimate_cost.py --phase pilot
python scripts/estimate_cost.py --phase full
python scripts/estimate_cost.py --phase all
```

---

## Revisions Included in This Final Plan

- Block 5 interaction probe removed (redundant with Block 2)
- Block 4 paradox replication increased to **6 reps**
- Homogeneous-strong controls retained in Block 4 (`homogeneous_opus`, n=2/3/5)
- Google models pinned to fixed versions in `config/models.yaml`
- Contamination-prone tasks replaced in analytical + creative sets
- Added `ANALYSIS_PLAN.md` (pre-registered statistics)
- Added `HANDOVER.md` (clone-and-run operator guide)

---

## Test

```bash
python -m pytest tests/
```

---

## Post-Run Analysis

```bash
python scripts/analyze_results.py --results-dir results --out-dir results/analysis
```

Primary outputs:
- `results/analysis/run_metrics.csv`
- `results/analysis/summary_table.csv`
- figures (`.pdf`, `.png`)
