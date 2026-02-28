# Agents Disagree Experiments

This repository contains a resumable experiment framework for evaluating multi-agent LLM orchestration strategies from the paper **"When Agents Disagree: Quorum-Based Consensus and Adaptive Orchestration Topology for Multi-Agent LLM Pipelines."**

## What's New (Committee Revision)

The framework now implements the critical methodology fixes:

- **Pairwise LLM-as-judge** (A vs B) with **bidirectional ordering** checks
- **Multi-judge panel** (3 judges) with **Bradley-Terry aggregation**
- **Inter-rater reliability** reporting (Cohen's kappa)
- **Separate `agent_pool` and `judge_pool`** with per-run judge exclusion logic
- **Semantic disagreement metrics** using `sentence-transformers/all-MiniLM-L6-v2`
- **Best-of-N baseline topology** (`best_of_n`) for cost-matched controls
- **Pipeline topology** (`pipeline`) added to topology comparisons
- **Debate round logging** (`debate_rounds`) for quorum/debate analyses
- **Human evaluation sheet generation** under `results/human_eval/`

## Quick Start

1. **Clone and enter repo**
   ```bash
   git clone <repo-url>
   cd agents-disagree-experiments
   ```
2. **Create Python environment (3.11+) and install**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Configure API keys**
   ```bash
   cp .env.example .env
   # fill ANTHROPIC_API_KEY and GOOGLE_API_KEY
   ```
4. **Validate setup**
   ```bash
   python scripts/validate_setup.py --dry-run
   ```
5. **Generate manifest**
   ```bash
   python scripts/generate_manifest.py
   ```
6. **Run experiments (resumable)**
   ```bash
   python scripts/run_experiments.py --resume --max-concurrent 10
   ```

## Architecture Overview

```text
config/experiment_matrix.yaml + config/tasks/*
                  |
                  v
           src/manifest.py
                  |
                  v
             src/runner.py
                  |
      +-----------+------------+
      |                        |
      v                        v
  topologies               consensus
(flat/hier/quorum/      (vote/debate/
 pipeline/best_of_n)      judge_panel)
      |                        |
      +-----------+------------+
                  v
          evaluation modules
  (pairwise judges + BT + kappa +
   semantic disagreement + human_eval)
                  |
                  v
        results/{block}/{run}.json
```

## Key Evaluation Behavior

- Pairwise comparisons are done in both orderings for position-bias checks.
- Judge panel scores are aggregated with Bradley-Terry.
- `evaluation.judge_panel` stores:
  - global BT scores
  - per-judge BT scores
  - inter-rater reliability
  - pairwise records
- `evaluation.disagreement` stores semantic disagreement plus secondary lexical metrics.

## Human Evaluation Infrastructure

Runs are auto-flagged to `results/human_eval/` when:

1. judges disagree (low kappa or pair disagreements),
2. run belongs to Block 4 paradox probes,
3. deterministic random 15% sample.

Each sheet includes anonymized candidate outputs, task prompt/rubric, and structured score fields.

## Monitoring While Running

- `results/progress.json`: completed/pending/failed, ETA, cumulative cost
- `results/cost_log.jsonl`: per-call token and cost records
- `results/run_experiments.log`: runtime logs

## Cost Estimation Before Launch

Run:
```bash
python scripts/estimate_cost.py
```

The estimator includes both agent-generation calls and judge pairwise calls.

## Analyze Results After Completion

Run:
```bash
python scripts/analyze_results.py --results-dir results --out-dir results/analysis
```

Outputs:
- `results/analysis/run_metrics.csv`
- `results/analysis/summary_table.csv`
- `results/analysis/*.pdf` and `*.png` figures

## Tests

```bash
python -m pytest tests/
```

## Reproducibility Notes

- Manifest generation is deterministic (`--seed`).
- Each run stores full config metadata, outputs, consensus details, judge panel details, and disagreement metrics.
- Persistence is filesystem-only with atomic JSON writes.
