# HANDOVER.md

This guide is for the operator running the experiment. Updated 2026-02-28 with
GenAI Gateway integration and pilot review fixes.

---

## 1) Prerequisites

- Linux/macOS with terminal access
- Python **3.11+**
- `pip`, Git
- **GenAI Gateway access** (replaces individual API keys):
  - Endpoint: OpenAI-compatible gateway serving all providers
  - Auth: Token command (configurable via `GENAI_TOKEN_CMD` env var)
  - Org header required for all requests
- `tmux` (**required** for long runs — protects against terminal/laptop disconnects)

---

## 2) Step-by-Step Setup

```bash
git clone https://github.com/maryanskyy/agents-disagree-experiments.git
cd agents-disagree-experiments

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Configure gateway access in `.env`:
```bash
export GENAI_GATEWAY_URL="<your-gateway-url>"
export GENAI_ORG_ID="<your-org-id>"
export GENAI_TOKEN_CMD="<command-to-get-bearer-token>"
```

Authentication is handled automatically by the `TokenManager` which acquires
and refreshes tokens before expiry.

---

## 3) Pre-flight Validation (Mandatory)

```bash
python scripts/validate_setup.py --dry-run
```

This validates:
- Python/runtime compatibility
- Task catalog integrity
- Gateway token acquisition (replaces API key checks)
- Model connectivity via GenAI Gateway
- Tiny end-to-end dry-run execution

If this fails, do **not** start paid runs.

---

## 4) Model Configuration Snapshot

All models are accessed through the unified GenAI Gateway using the OpenAI Chat
Completions API (`/v1/chat/completions`), regardless of provider.

### Agent pool (5)
- `claude-opus-4-6` (Anthropic, strong, max_output_tokens=4096)
- `gpt-5.2` (OpenAI, strong, max_output_tokens=4096)
- `gemini-2.5-pro` (Google, strong, max_output_tokens=4096)
- `claude-haiku-4-5` (Anthropic, weak/fast, max_output_tokens=2048)
- `gemini-2.5-flash` (Google, weak/fast, max_output_tokens=2048)

### Judge pool (3; no overlap with agents)
- `claude-sonnet-4-6` (Anthropic)
- `gpt-4o` (OpenAI)
- `gemini-3-pro-preview` (Google) — replaced `gemini-3.1-pro-preview` due to position bias

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

## After Experiment Completes

1. Reprocess all results:
   ```bash
   python scripts/reprocess_results.py
   python scripts/compute_structural_metrics.py
   ```

2. Export dataset:
   ```bash
   python scripts/export_dataset.py --results-dir results --output-dir results/export
   ```

3. Upload to GitHub:
   ```bash
   tar -czf results/full_results_final.tar.gz results/block*/
   git add results/export/ results/full_results_final.tar.gz
   git commit -m "Final experiment results + dataset export"
   git push origin master
   ```

4. Upload to HuggingFace:
   ```bash
   # Using huggingface_hub CLI
   huggingface-cli upload maryanskyy/agents-disagree results/export/ --repo-type dataset
   ```
## 9) Troubleshooting

### A) Token / auth errors
- Symptom: 401 or "you are not allowed to access this endpoint".
- Fix: Run your token command manually to verify it works. Token auto-refreshes
  every 18h via `TokenManager`, but network access must be active.
- Note: All models (including Anthropic) route through the OpenAI-compatible
  `/v1/chat/completions` endpoint. The Anthropic-style `/v1/messages` endpoint
  is restricted for some org IDs.

### B) Rate limits / quota spikes
- Symptom: repeated retry/backoff messages.
- Fix:
  - wait and resume later
  - reduce concurrency:
    ```bash
    python scripts/run_experiments.py --phase full --resume --max-concurrent 6
    ```
- Rate limiting is enforced at two levels:
  - **Per-model RPM** (configured in `config/models.yaml`)
  - **Global 150 RPM cap** across all models (in `src/utils/rate_limiter.py`)

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

---

## 12) Pilot Review Findings & Fixes Applied (T-029)

The pilot phase (280 runs, $48.50) was reviewed by 5 specialist agents. Four
critical and three moderate issues were identified and resolved before resuming
the main batch.

### Critical Fixes

| # | Issue | Root Cause | Fix | Status |
|---|-------|-----------|-----|--------|
| 1 | BT normalization artifact | Raw BT scores sum to 1.0 across candidates; more agents = mechanically lower score | Added `consensus_win_rate` and `normalized_bt_score` in `src/evaluation/corrected_metrics.py` | Fixed (commit `51c6d74`) |
| 2 | GPT-5.2 empty outputs (17-35%) | Model returns empty text despite consuming tokens | Retry loop (3 attempts, 2s delay) in `src/runner.py` invoke_agent; sentinel `[EMPTY_OUTPUT_AFTER_RETRIES]` for filtering | Fixed |
| 3 | Judge tie bias (GPT-4o 88%, Gemini 98.5%) | Weak forced-choice prompt; Gemini had position bias (always picks "A") | Forced-choice system prompt ("Do NOT say tie"), increased judge temp to 0.1, robust JSON parser with markdown fence stripping and partial-JSON extraction. Swapped `gemini-3.1-pro-preview` → `gemini-3-pro-preview` (no position bias) | Fixed |
| 4 | Agent max_tokens too low (2048) | Creative outputs truncated mid-sentence | Strong models → 4096, weak models → 2048 in `config/models.yaml` | Fixed |

### Moderate Fixes (Analysis-Only)

| # | Issue | Resolution |
|---|-------|-----------|
| 5 | Non-monotonic disagreement levels | Use OBSERVED disagreement_rate as independent variable, not level number |
| 6 | Deterministic outputs at low temp | Tag runs where all outputs are identical; report separately |
| 7 | Kappa inflated by Block 0 | Report kappa excluding Block 0 single-agent runs |

### Validation Results (post-fix)

- GPT-5.2 empty rate: **0%** (was 17-35%) — PASS
- Claude Sonnet tie rate: **0%** (was ~50%) — PASS
- GPT-4o tie rate: **0%** (was 88%) — PASS
- Gemini 3 Pro tie rate: **0%** (was 98.5% for gemini-3.1) — PASS
- Corrected metrics present: **100%** of runs — PASS
- All bidirectional consistency checks pass for all 3 judges

### Current Experiment State

- **Pilot**: 280/280 complete, GO recommended (kappa=0.934)
- **Main batch**: 421/8408 runs complete (paused for fixes), 0 failures
- **Total cost so far**: ~$128.52
- **Checkpoint**: `results/progress.json` — resume with `--resume` flag
- **To resume**: `python scripts/run_experiments.py --phase full --resume --max-cost 1500`

### Key Architecture Decisions

1. **Unified Gateway**: All 3 providers (Anthropic, OpenAI, Google) routed
   through a single OpenAI-compatible gateway endpoint. No individual API
   keys needed. Tokens auto-refresh via `TokenManager`.
2. **Rate Limiting**: Conservative per-model RPMs (30-35 for strong models) +
   global 150 RPM cap to avoid alerting the GenAI prod team.
3. **Crash Resilience**: `tmux` session + per-run JSON checkpointing + `--resume`
   flag enables seamless recovery from any interruption.

### Files Modified (from original repo)

| File | Change |
|------|--------|
| `src/utils/token_manager.py` | NEW — Gateway token lifecycle management |
| `src/models/anthropic_client.py` | Rewritten to use OpenAI SDK via gateway |
| `src/models/openai_client.py` | Adapted for gateway (chat completions API) |
| `src/models/google_client.py` | Rewritten to use OpenAI SDK via gateway |
| `src/runner.py` | Gateway integration, token refresh, empty output retry |
| `src/evaluation/llm_judge.py` | Forced-choice prompt, markdown fence parser, partial JSON extraction, temp 0.1, max_tokens 1024 |
| `src/utils/rate_limiter.py` | Added global 150 RPM cap |
| `config/models.yaml` | max_output_tokens, conservative RPMs, judge model swap |
| `scripts/validate_setup.py` | Gateway token validation instead of API keys |


---

## PRE-MAIN-BATCH VALIDATION CHECKLIST

**MANDATORY: Run this before starting the main batch.**

After applying fixes (commit 1451798+) and running 20-50 validation runs:

```bash
# 1. Run validation script on your post-fix results
python scripts/validate_fixes.py --results-dir results --min-runs 20

# Expected output: ALL CHECKS PASSED
# If any check FAILS, do NOT proceed.
```

### What the script checks:

| Check | Target | Why |
|-------|--------|-----|
| Judge tie rate | < 40% per judge | Fix 3 forced A/B decisions. Old: GPT-4o 93% ties |
| GPT-5.2 empty rate | < 10% | Fix 2 added retry logic. Old: 35-45% empty |
| Win rate variance | std > 0.05 | Fixes should produce meaningful quality differentiation |
| Inter-rater kappa | > 0.40 (excl Block 0) | Old kappa inflated by Block 0 trivial ties |

### If validation FAILS:

- **Tie rate still high**: Check judge system prompt in `src/evaluation/llm_judge.py`. The prompt should say "You MUST pick a winner. Do NOT say tie."
- **Empty rate still high**: Check retry logic in `src/runner.py` (~line 295). Ensure retries fire on empty text.
- **Win rate flat at 0.50**: The BT correction may not be computing correctly. Check `src/evaluation/corrected_metrics.py`.
- **Kappa low**: Judges may still be unreliable. Consider replacing the weakest judge model.

### Starting the main batch:

```bash
# 2. Clear pre-fix results if mixing old/new data
#    (or run from a fresh results directory)

# 3. Raise max-cost for full batch
python scripts/run_experiments.py --phase full --max-cost 1200

# 4. Monitor progress
cat results/progress.json | python -m json.tool
```

### After main batch completes:

```bash
# 5. Reprocess all results with corrected metrics
python scripts/reprocess_results.py

# 6. Run validation again on full data
python scripts/validate_fixes.py --results-dir results --min-runs 100

# 7. Compress and upload
tar -czf results/full_results_final.tar.gz results/block*/
git add results/full_results_final.tar.gz results/progress.json
git commit -m "Final experiment results"
git push origin master
```

---

## IMPORTANT: Separate Pre-Fix and Post-Fix Data

The pilot (280 runs) and partial batch (400 runs) were collected BEFORE the fixes.
Those results have known issues (high tie rate, empty outputs, uncorrected BT scores).

**Options:**
- **Option A (recommended):** Start the main batch fresh. Delete old block*/ directories
  in results/ (keep the .tar.gz archives as backups). Run `--phase all --max-cost 1200`.
- **Option B:** Keep old data but tag it. The `reprocess_results.py` script adds
  `corrected_metrics` to all runs. Post-fix runs can be identified by timestamp
  (after the fix commit) and by having lower tie rates.

