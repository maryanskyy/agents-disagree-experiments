# ACTION ITEMS — Pre-Resume Fixes (T-029)

**Date**: 2026-02-28
**Status**: Main batch PAUSED after 400/8408 runs ($80 spent)
**Context**: Pilot review by 5 agents found 4 critical + 4 moderate issues
**Goal**: Fix all critical issues, then resume main batch

---

## CRITICAL FIX 1: BT Normalization Artifact ✅ ALREADY FIXED

The `quality_score` (raw BT) cannot be compared across different agent counts because BT
scores sum to 1.0 across all candidates (consensus + N agents). More agents = mechanically
lower score.

**Already done (commit `51c6d74`):**
- `src/evaluation/corrected_metrics.py` — computes `consensus_win_rate` and `normalized_bt_score`
- `src/runner.py` — future runs include `evaluation.corrected_metrics` automatically
- `scripts/reprocess_results.py` — reprocesses existing runs
- `ANALYSIS_PLAN.md` — updated to use `consensus_win_rate` as primary metric

**What you need to do:**
1. `git pull origin master`
2. `python scripts/reprocess_results.py` — reprocess your existing 400 runs
3. Check `results/corrected_metrics_summary.csv` for the corrected data

---

## CRITICAL FIX 2: GPT-5.2 Empty Output Bug 🔴 NEEDS CODE

**Problem**: GPT-5.2 produces empty text (`""`) 17-35% of the time despite consuming 2048
output tokens. This corrupts BT scoring (all ties), inflates kappa, and makes cross-model
comparisons invalid.

**Where to fix**: `src/runner.py`, in the agent generation loop (~line 280-300)

**What to do**:
Add retry logic after each agent generation call. If the response text is empty or
whitespace-only, retry up to 3 times with a 2-second delay. If still empty after 3 retries,
log a warning and mark the output as `"[EMPTY_OUTPUT_AFTER_RETRIES]"` so it can be filtered
in analysis.

```python
# After: response = await agent_client.generate(...)
# Add:
MAX_RETRIES = 3
for retry_attempt in range(MAX_RETRIES):
    if response.text and response.text.strip():
        break
    logger.warning(
        f"Empty output from {agent_client.model_alias} on attempt {retry_attempt + 1}/{MAX_RETRIES}, "
        f"run={run_spec.id}"
    )
    await asyncio.sleep(2)
    response = await agent_client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        metadata=metadata,
    )
if not response.text or not response.text.strip():
    logger.error(f"Empty output after {MAX_RETRIES} retries: {agent_client.model_alias}, run={run_spec.id}")
    response = ModelResponse(
        text="[EMPTY_OUTPUT_AFTER_RETRIES]",
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        model_name=response.model_name,
    )
```

Also add to analysis: filter runs where ANY agent output contains `[EMPTY_OUTPUT_AFTER_RETRIES]`.

---

## CRITICAL FIX 3: Judge Tie Bias 🔴 NEEDS CODE

**Problem**: GPT-4o votes "tie" 92.8% of the time. Gemini 3.1 Pro votes "tie" 84.5%.
Only Claude Sonnet actually differentiates (49.2% non-tie). The 3-judge panel is
effectively a 1-judge system.

**Where to fix**: `src/evaluation/llm_judge.py`, method `_build_prompt()` (~line 198)

**What to do**: Strengthen the discrimination prompt. Replace the current evaluation rules with:

```python
"Evaluation rules:\n"
"- You MUST choose a winner (A or B). Only choose 'tie' if the responses are truly\n"
"  indistinguishable in quality after careful analysis.\n"
"- Ties should occur in fewer than 20% of your judgments. If both seem similar,\n"
"  look harder at the rubric criteria to find meaningful differences.\n"
"- Judge only quality relative to the task and rubric.\n"
"- Ignore response length unless it harms substance.\n"
"- Ignore formatting polish and stylistic familiarity biases.\n"
"- You are blind to model, topology, consensus method, and agent count.\n\n"
```

Also: change judge temperature from `0.0` to `0.1` in `_single_ballot()` (~line 224):
```python
temperature=0.1,  # was 0.0; slight noise helps break tie bias
```

**Validation**: After fix, run 20 pilot runs and check that tie rate drops below 40% for
all judges.

---

## CRITICAL FIX 4: Increase Agent max_tokens 🔴 NEEDS CONFIG

**Problem**: Agent output is capped at 2048 tokens (the default in `src/runner.py` line 283).
Creative responses are truncated mid-sentence. Gemini models report high token counts but
produce very short text (250-440 chars), suggesting a mismatch.

**Where to fix**: `config/models.yaml`

**What to do**: Add `max_output_tokens` to each agent model config:

```yaml
# For strong models (Opus, GPT-5.2, Gemini 2.5 Pro):
max_output_tokens: 4096

# For weak models (Haiku, Flash):
max_output_tokens: 2048
```

Example — add to each agent_pool entry:
```yaml
  claude-opus-4-6:
    provider: anthropic
    model_id: claude-opus-4-6
    tier: strong
    max_output_tokens: 4096    # ADD THIS
    input_cost_per_1m: 15.0
    ...
```

---

## MODERATE FIX 5: Non-Monotonic Disagreement Levels 🟡 ANALYSIS ONLY

**Problem**: Disagreement level 4 (0.199) produces LESS disagreement than level 1 (0.300).
The 5-level manipulation doesn't form an ordered scale.

**What to do**: In the analysis, use OBSERVED disagreement rate (from
`evaluation.disagreement.disagreement_rate`) as the independent variable instead of the
manipulated level number. This is already a continuous variable and doesn't assume
monotonicity. Update `ANALYSIS_PLAN.md` H1 test to:
```
H1: Regress consensus_win_rate on OBSERVED disagreement_rate (not level)
```

---

## MODERATE FIX 6: Deterministic Outputs at Low Temperature 🟡 ANALYSIS ONLY

**Problem**: Claude Opus at temp=0.3 with identical prompts produces byte-for-byte identical
text across agents. These runs are effectively single-agent with duplicated API calls.

**What to do**: In the analysis, tag runs where all agent outputs are identical
(`semantic_pairwise_similarity == 1.0`) and report them separately. They are valid data
points (they show that homogeneous configs with identical prompts produce zero disagreement)
but should not be counted as evidence of multi-agent effects.

---

## MODERATE FIX 7: Kappa Reporting 🟡 ANALYSIS ONLY

**Problem**: Aggregate kappa is inflated by Block 0 single-agent runs (trivial agreement).

**What to do**: Always report kappa EXCLUDING Block 0. The corrected multi-agent kappa
is ~0.54-0.56. After fixing the judge tie bias (Fix 3), re-measure — it should improve
since judges will make more discriminating decisions.

---

## EXECUTION ORDER

```
1. git pull origin master                    # Get BT fix (already committed)
2. Apply Fix 2 (GPT-5.2 retry logic)        # Code change in runner.py
3. Apply Fix 3 (judge tie bias prompt)       # Code change in llm_judge.py
4. Apply Fix 4 (max_tokens in models.yaml)   # Config change
5. python -m pytest tests/                   # Verify nothing broken
6. python scripts/reprocess_results.py       # Reprocess existing 400 runs
7. Run 20 validation runs:
   python scripts/run_experiments.py --phase pilot --max-cost 30
8. Check validation results:
   - GPT-5.2 empty rate < 5% (was 17-35%)
   - Judge tie rate < 40% for ALL judges (was 85-93% for GPT-4o/Gemini)
   - corrected_metrics present in new runs
9. If validation passes → resume main batch:
   python scripts/run_experiments.py --phase full --max-cost 1200
10. If validation fails → report issues, do NOT proceed
```

---

## FILES CHANGED BY BT FIX (already committed)

- `src/evaluation/corrected_metrics.py` (NEW)
- `src/runner.py` (UPDATED — adds corrected_metrics to evaluation output)
- `scripts/reprocess_results.py` (NEW)
- `tests/test_corrected_metrics.py` (NEW)
- `ANALYSIS_PLAN.md` (UPDATED)
- `EXPERIMENT_PLAN.md` (UPDATED)
- `reviews/T-028-bt-fix-report.md` (NEW)

## FILES THAT NEED CHANGES (Fixes 2-4)

- `src/runner.py` — Add GPT-5.2 retry logic (Fix 2)
- `src/evaluation/llm_judge.py` — Strengthen discrimination prompt + raise temp (Fix 3)
- `config/models.yaml` — Add max_output_tokens per model (Fix 4)

## REVIEW REPORTS (for reference)

All in `reviews/`:
- `pilot-analysis-theoretical.md` — Elena (PhD-1): BT artifact analysis
- `pilot-analysis-statistical.md` — Marcus (PhD-2): Effect sizes, power analysis
- `pilot-review-mas.md` — Dr. Tanaka: MAS perspective, paradox validation
- `pilot-review-nlp.md` — Dr. Sharma: Judge bias, evaluation methodology
- `pilot-review-emp.md` — Dr. Okonkwo: Data integrity, reproducibility

## SUCCESS CRITERIA

After fixes, the corrected pilot data should show:
- `consensus_win_rate` varies meaningfully across conditions (not flat 0.50)
- Judge tie rate < 40% for all 3 judges
- GPT-5.2 empty output rate < 5%
- Multi-agent kappa > 0.50 (ideally > 0.60)
- No truncated creative responses
