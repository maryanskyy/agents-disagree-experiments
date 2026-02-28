# HANDOVER.md

This guide is for the operator running the final experiment on a **MacBook Air M4**.

---

## 1) Prerequisites

- macOS with terminal access
- Python **3.11+**
- `pip`
- Git
- API keys:
  - `ANTHROPIC_API_KEY`
  - `OPENAI_API_KEY`
  - `GOOGLE_API_KEY`

Optional but recommended:
- `tmux` (protect long runs from terminal disconnects)
- charger connected + sleep disabled while running

---

## 2) Step-by-Step Setup

```bash
git clone <repo-url>
cd agents-disagree-experiments

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY + OPENAI_API_KEY + GOOGLE_API_KEY
```

---

## 3) Pre-flight Validation (Mandatory)

```bash
python scripts/validate_setup.py --dry-run
```

This validates:
- Python/runtime compatibility
- task catalog integrity
- environment key checks
- model connectivity wiring (mocked in dry-run)
- tiny end-to-end dry-run execution

If this fails, do **not** start paid runs.

---

## 4) Model Configuration Snapshot

### Agent pool (5)
- `claude-opus-4-6` (Anthropic, strong)
- `gpt-5.2` (OpenAI, strong)
- `gemini-2.5-pro` (Google, strong)
- `claude-haiku-4-5` (Anthropic, weak/fast)
- `gemini-2.5-flash` (Google, weak/fast)

### Judge pool (3; no overlap with agents)
- `claude-sonnet-4-6` (Anthropic)
- `gpt-4o` (OpenAI)
- `gemini-3.1-pro-preview` (Google)

---

## 5) Phase A (Pilot)

Run:

```bash
python scripts/run_experiments.py --phase pilot --resume --max-cost 4000
```

Pilot executes:
- full Block 0
- 20 sampled runs from Block 1
- 20 sampled runs from Block 4

Check after completion:
- `results/pilot_report.json`

GO/NO-GO criteria:
- GO if:
  - mean inter-judge kappa > 0.4
  - disagreement levels are separable
- NO-GO otherwise (revisit judge rubric/task setup before full run)

---

## 6) Phase B (Full)

Run remaining experiment (excluding pilot-completed runs):

```bash
python scripts/run_experiments.py --phase full --resume --max-cost 4000
```

Expected runtime:
- depends on API throughput and retries; typically many hours.

Monitoring:
- `tail -f results/run_experiments.log`
- inspect `results/progress.json` for completed/pending/ETA/cost
- inspect `results/cost_log.jsonl` for per-call spend

---

## 7) During the Run

### Check progress
- `results/progress.json` is the primary live status file.
- `status` may switch to `paused_max_cost` if guardrail is hit.

### If errors occur
1. Read latest stack trace in `results/run_experiments.log`.
2. Fix issue (API key, network, dependency).
3. Resume safely:
   ```bash
   python scripts/run_experiments.py --phase full --resume --max-cost 4000
   ```

### If laptop sleeps/crashes
- Resume with same command + `--resume`.
- Completed run files are checkpointed on disk.

---

## 8) After Completion

### Sharing results back (to main repo / Windows machine)

Results are in `results/` and are gitignored. To share them back:

```bash
# 1. Run analysis first (generates CSV, tables, figures)
python scripts/analyze_results.py --results-dir results --out-dir results/analysis

# 2. Push results to a dedicated branch (does not affect main)
git checkout -b results/$(date +%Y%m%d)
git add -f results/
git commit -m "Experiment results $(date +%Y-%m-%d)"
git push -u origin HEAD
```

Then on your main machine:
```bash
git fetch origin results/YYYYMMDD
git checkout results/YYYYMMDD
# results/ is now populated
```

1. Run analysis:
   ```bash
   python scripts/analyze_results.py --results-dir results --out-dir results/analysis
   ```

2. Verify outputs:
   - `results/analysis/run_metrics.csv`
   - `results/analysis/summary_table.csv`
   - `results/analysis/*.pdf`
   - `results/analysis/*.png`

3. Review pilot + final summaries:
   - `results/pilot_report.json`
   - `results/progress.json`

---

## 9) Troubleshooting

### A) API key/auth errors
- Symptom: authentication/permission exceptions.
- Fix: verify `.env` keys and active shell environment.

### B) Rate limits / quota spikes
- Symptom: repeated retry/backoff messages.
- Fix:
  - wait and resume later
  - reduce concurrency:
    ```bash
    python scripts/run_experiments.py --phase full --resume --max-concurrent 6
    ```

### C) Network instability
- Symptom: timeout/connection errors.
- Fix: stabilize network, then rerun with `--resume`.

### D) Dependency issues
- Symptom: import/module errors.
- Fix:
  ```bash
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

---

## 10) Budget Guardrails

Default hard limit:
- `--max-cost 4000`

Behavior:
- when cumulative cost reaches limit, runner pauses and writes warning to `results/progress.json`.

Recommended practice:
- start with tighter cap for pilot if desired (example: `--max-cost 500`)
- monitor `results/cost_log.jsonl` and `results/progress.json`

Cost estimation commands:
```bash
python scripts/estimate_cost.py --phase pilot
python scripts/estimate_cost.py --phase full
python scripts/estimate_cost.py --phase all
```

---

## 11) What Success Looks Like

Minimum viable successful run includes:

1. Pilot completed + `results/pilot_report.json` generated.
2. Full phase completed (or intentionally paused by guardrail).
3. `results/progress.json` shows near-zero pending runs (unless paused intentionally).
4. Run JSON outputs exist under each block directory.
5. Analysis artifacts generated under `results/analysis/`.

If all above are present, repository is in clone-and-run operational state.

