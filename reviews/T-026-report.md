# T-026 Report — Multi-Provider Model Selection Update

Date: 2026-02-28
Repo: `C:\Users\Artem\Desktop\agents-disagree-experiments`

## Scope Completed

Implemented the full T-026 upgrade from 2-provider to 3-provider experiment coverage, including OpenAI integration and documentation refresh.

### 1) OpenAI provider integration
- Added `src/models/openai_client.py` using the official `openai` SDK (`AsyncOpenAI`) with async generation.
- Supports both configured OpenAI models:
  - `gpt-5.2` (agent)
  - `gpt-4o` (judge)
- Reads key from `OPENAI_API_KEY`.
- Includes built-in RPM limiter per OpenAI model key.

### 2) Unified model config normalization
- Added `src/models/catalog.py` to normalize model schemas.
- Supports new pool schema (`agent_pool` + `judge_pool`) and legacy schema.
- Enforces **no overlap** between agent and judge pools.

### 3) Runner updates
- Updated `src/runner.py` to:
  - load normalized model catalog
  - use factory-style client creation by provider (`anthropic`, `google`, `openai`)
  - instantiate `OpenAIModelClient` when provider=`openai`
  - select judges from judge pool with agent exclusion
  - validate that run agent assignments use only agent-pool models
- Cost tracking now consumes pricing for all configured models (agents + judges).

### 4) Config updates
- Replaced `config/models.yaml` with requested new pools:
  - Agent pool: Opus, GPT-5.2, Gemini-2.5-Pro, Haiku-4-5, Gemini-2.5-Flash
  - Judge pool: Sonnet-4-6, GPT-4o, Gemini-3.1-Pro-Preview

### 5) Experiment matrix updates
- Updated `config/experiment_matrix.yaml` to new model lineup and block logic:
  - Block 0: all 5 models solo, 16 tasks, 3 reps
  - Block 1: homogeneous strong + heterogeneous + high-disagreement mixes
  - Block 2: heterogeneous cross-provider + homogeneous strong comparisons
  - Block 3: counts `{1,2,3,5}` with full-pool coverage
  - Block 4: required paradox/control configs, 6 reps
  - Best-of-N baselines use `gemini-2.5-pro`
- Added generic `lead_plus_rest` template composition support.

### 6) Validation and tooling updates
- Updated `scripts/validate_setup.py`:
  - checks 3 API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`)
  - validates connectivity for **every configured model** (dry-run/live modes)
  - still runs tiny dry-run smoke execution
- Updated `scripts/estimate_cost.py`:
  - uses normalized model catalog
  - robustly regenerates manifest if stale manifest references removed model names

### 7) Dependency/env updates
- Added `openai>=1.0.0` to `requirements.txt` and `pyproject.toml`.
- Updated `.env.example` with all 3 keys.

### 8) Documentation updates
- Updated:
  - `README.md`
  - `EXPERIMENT_PLAN.md`
  - `HANDOVER.md`
  - `ANALYSIS_PLAN.md`
- Removed old 2-provider limitation and replaced known limitations accordingly.
- Added provider-factor sensitivity note to analysis plan.

### 9) Manifest ID collision fix
- Updated `src/manifest.py` run-id hashing inputs to include `model_spec`.
- Prevents accidental collisions when different templates produce the same assignment for some `agent_count` (important for phased run separation correctness).

---

## Verification Results

### A) Unit tests
Command:
```bash
python -m pytest tests/
```
Result:
- **15 passed, 0 failed**

### B) Setup validation (dry-run)
Command:
```bash
python scripts/validate_setup.py --dry-run
```
Result:
- Completed successfully
- Connectivity checks passed in dry-run mode for all providers/models
- Tiny manifest smoke run completed

### C) Cost estimation
Command:
```bash
python scripts/estimate_cost.py
python scripts/estimate_cost.py --phase pilot
python scripts/estimate_cost.py --phase full
```

Observed totals:
- **All:** 8,688 runs, **$3,553.25**
- **Pilot:** 280 runs, **$31.13**
- **Full:** 8,408 runs, **$3,522.12**

---

## Notes for Operator / PI

- Current default run guardrail was raised to `--max-cost 4000` to align with new estimated full cost.
- Judge pool now spans Anthropic + OpenAI + Google with no model overlap versus agents, directly addressing the prior reviewer concern.
