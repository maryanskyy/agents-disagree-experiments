# Agents Disagree Experiments

This repository contains a production-oriented, resumable experiment framework for evaluating quorum-based multi-agent LLM orchestration strategies from the paper **"When Agents Disagree: Quorum-Based Consensus and Adaptive Orchestration Topology for Multi-Agent LLM Pipelines."** It is designed for a 24-hour MacBook Air M4 run with async parallelism, crash-safe checkpointing, deterministic manifests, and filesystem-only persistence (no database).

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
                        +-----------------------------+
                        |  config/experiment_matrix   |
                        +-------------+---------------+
                                      |
                                      v
+----------------+         +----------+----------+         +----------------+
| config/tasks/* | ------> | src/manifest.py     | ----->  | results/manifest|
+----------------+         +----------+----------+         +----------------+
                                      |
                                      v
                        +-------------+--------------+
                        | src/runner.py (async core) |
                        +------+------------+--------+
                               |            |
             +-----------------+            +-----------------+
             v                                              v
+-------------------------------+             +-------------------------------+
| topologies (flat/hier/quorum) |             | consensus (vote/debate/judge) |
+-------------------------------+             +-------------------------------+
             |                                              |
             +-----------------+----------------------------+
                               v
                   +-----------+-----------+
                   | model adapters        |
                   | anthropic / google    |
                   +-----------+-----------+
                               v
                   +-----------+-----------------------------+
                   | checkpoint + cost tracker + progress    |
                   | results/{block}/{run_id}.json (atomic)  |
                   +------------------------------------------+
```

## Resume and Recovery

- Every run writes to `results/<block>/<run_id>.json` using **atomic temp-file + rename**.
- Progress snapshots are written to `results/progress.json` every N runs.
- Resume mode (`--resume`) scans completed run IDs and skips finished runs.
- Crashes/sleep interruptions are recoverable without database replay.
- `src/utils/keep_awake.py` uses `caffeinate -dims` on macOS to prevent sleep during long runs.

## Monitoring While Running

- Watch `results/progress.json` for:
  - completed vs pending runs
  - failed runs
  - ETA seconds
  - estimated cost totals by model
- Inspect `results/cost_log.jsonl` for incremental token/cost entries.
- Read `results/run_experiments.log` for execution timeline and errors.

## Cost Estimation Before Launch

Run:
```bash
python scripts/estimate_cost.py
```
The script reports projected run count, API calls, block-level cost, and model-level cost using pricing in `config/models.yaml`.

## Analyze Results After Completion

Run:
```bash
python scripts/analyze_results.py --results-dir results --out-dir results/analysis
```
Outputs:
- `results/analysis/run_metrics.csv`
- `results/analysis/summary_table.csv`
- `results/analysis/*.pdf` and `*.png` figures

## Project Layout

See the repository tree in the project brief; code is organized into:
- `src/` core runtime + adapters
- `scripts/` operational entry points
- `config/` models, tasks, matrix
- `tests/` regression + dry-run integration

## Contributing / Extending

- Add new models in `config/models.yaml` plus adapter support if provider changes.
- Add task sets under `config/tasks/` (keep IDs stable for reproducibility).
- Add new topologies in `src/topologies/` by implementing `BaseTopology`.
- Add new consensus methods in `src/consensus/` by implementing `BaseConsensus`.
- Always include tests and run:
  ```bash
  python -m pytest tests/
  ```

## Reproducibility Notes

- The framework is deterministic at manifest level (`--seed`).
- Each run stores full config metadata with task ID, topology, consensus, model assignment, and disagreement level.
- No external DB dependencies: all recoverability is from files under `results/`.