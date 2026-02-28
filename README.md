# Agents Disagree Experiments

This repository contains the resumable experiment framework for:

**When Agents Disagree: Quorum-Based Consensus and Adaptive Orchestration Topology for Multi-Agent LLM Pipelines**

The current configuration uses **3 providers** (Anthropic, OpenAI, Google), with a **5-model agent pool** and **3-model judge panel** with zero model overlap between pools.

---

## Prerequisites

- Python 3.11+
- `pip`
- Anthropic API key
- OpenAI API key
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
   # set ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
   ```

4. **Pre-flight validation**
   ```bash
   python scripts/validate_setup.py --dry-run
   ```

---

## Model Lineup (Current)

### Agent Pool (5)
- `claude-opus-4-6` (Anthropic, strong)
- `gpt-5.2` (OpenAI, strong)
- `gemini-2.5-pro` (Google, strong; pinned API model)
- `claude-haiku-4-5` (Anthropic, weak/fast)
- `gemini-2.5-flash` (Google, weak/fast)

### Judge Pool (3, non-overlapping)
- `claude-sonnet-4-6` (Anthropic)
- `gpt-4o` (OpenAI)
- `gemini-3.1-pro-preview` (Google)

Judge models are used only for evaluation; they are excluded from agent assignments by config + runtime checks.

---

## Run Modes (Phased Execution)

### Phase A: Pilot
Runs:
- full `block0_calibration`
- 20 sampled runs from `block1_disagreement_dividend`
- 20 sampled runs from `block4_quorum_paradox`

```bash
python scripts/run_experiments.py --phase pilot --resume --max-cost 4000
```

### Phase B: Full
Runs all remaining conditions except pilot-completed runs.

```bash
python scripts/run_experiments.py --phase full --resume --max-cost 4000
```

### Phase C: All (fresh rerun)
Runs full matrix from scratch.

```bash
python scripts/run_experiments.py --phase all --max-cost 4000
```

---

## Cost Guardrail

`run_experiments.py` includes:

- `--max-cost` (default: **4000 USD**)
- Automatic pause when cumulative tracked spend reaches the guardrail
- Status emitted to `results/progress.json`

Monitor during run:
- `results/progress.json`
- `results/cost_log.jsonl`
- `results/run_experiments.log`

Estimate before launch:
```bash
python scripts/estimate_cost.py --phase pilot
python scripts/estimate_cost.py --phase full
python scripts/estimate_cost.py --phase all
```

---

## Validation + Tests

```bash
python -m pytest tests/
python scripts/validate_setup.py --dry-run
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

