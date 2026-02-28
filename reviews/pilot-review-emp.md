# Pilot Review — Empirical Methodology & Statistics

**Reviewer:** Dr. James Okonkwo (Empirical Methodology)  
**Date:** 2026-02-28  
**Phase:** Pilot (280 runs, Phase A)  
**Document:** T-027e  

---

## Verdict: MAJOR ISSUES

Three critical issues must be resolved before the main-batch data can be considered valid evidence. The most serious — a measurement confound in the Block 4 paradox metric — would invalidate the headline finding of this paper if carried forward unchanged.

---

## 1. Experimental Integrity

### 1.1 Temporal Ordering — ADEQUATE

Runs were **interleaved across blocks**, not executed sequentially. Block 1 timestamps span 21:21–21:35 UTC; Block 4 spans 21:25–21:38 UTC. These ranges overlap substantially, confirming that the experiment runner dispatched conditions in a pseudo-randomized or round-robin order. This eliminates the most obvious temporal confound (all Block 4 runs occurring during a late-night API performance trough, for example).

### 1.2 Latency Patterns — MODEL-DEPENDENT, NOT BLOCK-DEPENDENT

Latency variation is driven by model identity and output length, not by block or temporal position:

| Model | Typical Latency Range |
|---|---|
| claude-opus-4-6 | 56–65 seconds (long-form drafts) |
| gpt-5.2 | 15–55 seconds (length-dependent) |
| gemini-2.5-pro | 3–26 seconds |
| Judge models | 3–15 seconds |

No systematic drift was observed across the 18.5-minute execution window. I found no evidence of API throttling or degradation affecting later-scheduled blocks.

### 1.3 Run Count Discrepancy — MINOR

The pilot report declares `block1=20, block4=20` (total=280). The actual directories contain **22 files each**. The four extra runs (two per block) carry the latest timestamps and appear to be early main-batch runs that completed before the pilot report was generated. The pilot report's statistical summaries (paradox means, disagreement rates) were computed on **22** runs each despite claiming 20, introducing a minor internal inconsistency in the pilot report.

**Recommendation:** Clarify the pilot/main-batch boundary. Tag runs explicitly with `phase: pilot` vs `phase: main` in the run JSON.

---

## 2. Reproducibility Check

### 2.1 Deterministic Outputs at dl=1 — CRITICAL

At disagreement level 1 (identical prompts, temperature=0.3), agents produce **byte-for-byte identical text**:

- **run_ccb11924d67f** (Block 1, n=5, claude-opus-4-6 x 5): All five agents returned the same 7,126-character response — title, formatting, every character identical.
- **run_f71343ba7971** (Block 1, n=2, claude-opus-4-6 x 2): String equality confirmed. Both outputs: 4,550 chars, 1,107 tokens.

This is API-level determinism (or response caching) at low temperature. It means **dl=1 conditions are not multi-agent at all** — they are single-agent runs with duplicated API calls. Any comparison that treats dl=1 as "low disagreement multi-agent" is methodologically invalid; it is "zero disagreement, zero diversity."

**Impact:** Undermines H1 (disagreement dividend) at the left anchor of the disagreement spectrum. The inverted-U would need its low-disagreement anchor to come from actual multi-agent systems with low-but-nonzero disagreement, not from cloned outputs.

**Recommendation:** Either (a) raise the temperature floor for dl=1 to ensure stochastic variation (e.g., temp >= 0.7), or (b) relabel dl=1 as "single-agent baseline" and exclude it from the quadratic regression, or (c) add a per-agent seed/system-prompt perturbation to guarantee independence even at low temperature.

### 2.2 Within-Configuration Variation — INCONCLUSIVE

With only 1–2 repetitions per configuration in the pilot, there is insufficient data to assess whether same-config runs yield appropriately noisy-but-centered results. The main batch (6 reps per condition in Block 4, variable in others) should provide the necessary data.

---

## 3. Statistical Power Assessment

### 3.1 Block 4 "Paradox" Effect Size — ARTIFACTUAL (CRITICAL)

The briefing reports d ~ 1.55, which would be an implausibly large effect by any social/behavioral science standard. **The effect is almost entirely a measurement artifact.**

#### The BT Denominator Confound

The quality metric (`quality_score`) is the Bradley-Terry score of the consensus output, computed from pairwise comparisons among **all candidates** (individual agent outputs + final consensus). The number of candidates varies with agent count:

| Condition | Agents | BT Candidates | Uniform BT Score |
|---|---|---|---|
| n=2 | 2 agents + consensus | 3 | 1/3 = 0.333 |
| n=3 | 3 agents + consensus | 4 | 1/4 = 0.250 |

Even if the consensus output is **equally good** relative to individual agents in both conditions, its BT score will be mechanically lower with more candidates.

#### Quantitative Decomposition

I computed the deviation of each run's consensus BT score from its condition-specific uniform baseline:

| Condition | Mean BT Score | Uniform Baseline | Mean Deviation |
|---|---|---|---|
| n=2 (11 runs) | 0.315 | 0.333 | **-0.018** |
| n=3 (11 runs) | 0.230 | 0.250 | **-0.020** |

The deviations are virtually identical (-0.018 vs -0.020). The consensus performs about equally poorly relative to individual agents in both conditions. The "paradox" decomposes as:

- **Raw difference:** 0.315 - 0.230 = 0.085
- **Expected from denominator alone:** 0.333 - 0.250 = 0.083
- **Residual after correction:** 0.085 - 0.083 = **0.002**

The residual effect is d ~ 0.02 — **essentially zero**. The headline "paradox" finding is 98% denominator artifact, 2% noise.

**Impact:** This is the most critical finding in this review. If H3 (quorum paradox) is tested using raw BT quality_score as the dependent variable, it will ALWAYS show Q(n=3) < Q(n=2) regardless of whether an actual quality difference exists, because the metric is confounded with condition.

**Recommendation:** Three options, in order of preference:

1. **Use deviation from uniform** as the dependent variable: `quality_adjusted = quality_score - 1/(n_agents+1)`. This removes the denominator bias while preserving the relative strength of consensus.
2. **Compare consensus output against a fixed external reference** (e.g., Block 0 calibration outputs) rather than against same-run individual agents, so the number of BT candidates is constant across conditions.
3. **Use an absolute quality metric** (e.g., rubric-anchored judge scoring on a fixed 1–5 scale) rather than a relative ranking metric.

### 3.2 Block 1 — Underpowered (Expected)

With 4–5 observations per disagreement level, there is no hope of detecting an inverted-U in the pilot. The observed pattern (0.357, 0.319, 0.337, 0.305, 0.379) shows no systematic shape. This is expected and appropriate for a pilot — the purpose is infrastructure validation, not hypothesis testing.

The main batch allocates 2,400 runs to Block 1. Assuming ~480 runs per disagreement level (5 levels), this provides adequate power for quadratic regression (rule of thumb: >=20 per coefficient x 3 coefficients = 60 minimum; 480 per level is generous). **However**, the dl=1 determinism issue (section 2.1) means the leftmost anchor produces zero-variance data, which will distort the quadratic fit.

### 3.3 Main Batch Sample Sizes — Adequate If Metric Is Fixed

| Block | Planned Runs | Per-Cell Size | Assessment |
|---|---|---|---|
| Block 1 (H1) | 2,400 | ~480/level | Adequate for quadratic regression |
| Block 2 (H4) | 3,072 | Varies by cell | Likely adequate for 2-way ANOVA |
| Block 3 (H2) | 1,152 | ~288/count | Adequate for logistic regression |
| Block 4 (H3) | 1,152 | ~576/condition | Adequate **if metric is fixed** |

Power is moot if the metric is confounded (section 3.1) or if empty-text bugs (section 6.1) persist.

---

## 4. Previous Fatal Flaw Resolution

### 4.1 Evaluation Metrics — PARTIALLY RESOLVED

I previously flagged: *"Evaluation metrics measure text statistics, not quality."*

The revision implemented:
- Pairwise LLM-as-judge with 3 independent judge models (Claude Sonnet, GPT-4o, Gemini 3.1 Pro)
- Three providers represented in the judge panel, with zero model overlap with agent pool
- Bidirectional pairwise comparisons (position randomization)
- Bradley-Terry aggregation from pairwise win/loss/tie records
- Human review flagging (86 runs flagged for manual inspection)

**However**, the BT scores are not differentiating conditions meaningfully. Quality scores cluster at a small set of exact fractions (0.167, 0.250, 0.333, 0.400, 0.500) because:
- Many pairwise comparisons result in "tie" (especially for GPT-4o and Gemini 3.1 Pro)
- With few candidates per run (3–6), BT converges to coarse discrete values
- Claude Sonnet is the only judge that consistently discriminates, but its signal is outvoted by the two "all-tie" judges

Evidence: In run_040a9b089fee (Block 1), GPT-4o and Gemini both assigned uniform 0.250 to all 4 candidates (all ties). Only Claude Sonnet differentiated (agent_1 = 0.997, others ~ 0.001). The final BT scores are dominated by the two non-discriminating judges.

**The evaluation is no longer measuring text statistics, which resolves my original fatal flaw. But the replacement metric has its own significant problems: low resolution, BT denominator confound, and judge passivity.** I upgrade my assessment from "Fatal Flaw" to "Major Issue — Metric Needs Refinement."

### 4.2 Block Powering — IMPROVED

I previously flagged Blocks 2/3/5 as underpowered. Block 5 was cut. Block 4 received 6 repetitions per condition (1,152 runs total). Block 2 got 3,072 runs. These are substantial improvements. Power is adequate for the planned tests **contingent on** the metric issues being resolved.

### 4.3 Aggregate Kappa Is Misleading

The pilot report claims mean inter-judge kappa = **0.934**. This is technically correct but highly misleading:

- Block 0 contributes 240 calibration runs, each with a single agent compared against itself — all ties — kappa = 1.0 by construction.
- These 240 perfect-kappa runs dominate the aggregate (240/280 = 86% of runs).
- Actual Block 1/4 kappa ranges: **0.026 to 1.0**, median ~ 0.333.
- Kappa = 1.0 in Blocks 1/4 occurs when all judges say "tie" (often on runs with empty text) — this is not meaningful agreement, it is shared inability to discriminate.

**Recommendation:** Report kappa separately by block, and exclude Block 0 from the aggregate. The meaningful statistic is Block 1/4 kappa among runs with non-degenerate outputs, which is considerably lower than 0.934.

---

## 5. Cost Efficiency

### 5.1 Pilot Cost — Internally Inconsistent

- **Pilot report says:** $48.50 for 280 runs
- **Cost log cumulative tracker (last line):** $12.36
- **Cost log sum (all entries):** ~$60.87 (includes main-batch runs that started after pilot)

The cumulative `total_cost_usd` tracker in the cost log **resets on process resume**, causing it to under-report. The pilot report's $48.50 is plausible but cannot be verified from the cost log alone. The cost tracker bug needs to be fixed for reliable budget monitoring during the main batch.

### 5.2 Extrapolation to Full Batch

- Pilot: ~$48.50 for 280 runs ~ $0.173/run (average across blocks)
- Full experiment: 8,688 runs x $0.173/run ~ $1,503
- But: Experiment plan estimates $3,553 for the full experiment
- The discrepancy is expected: Block 0 calibration (cheap single-agent runs) drives down the pilot average; main-batch blocks are more expensive per run (more agents, multi-round debate, more judge comparisons)

**Assessment:** The cost estimate of ~$3,500 appears reasonable. The pilot cost-per-run for Blocks 1/4 specifically ($14.24/22 ~ $0.65 for Block 1; $16.47/22 ~ $0.75 for Block 4) extrapolates to:
- Block 1: 2,400 x $0.65 = ~$1,560
- Block 4: 1,152 x $0.75 = ~$864
These are in the right ballpark.

### 5.3 Judge/Agent Cost Ratio — APPROPRIATE

Observed: 43.6% judges, 56.4% agents. This is reasonable for a design that uses 6 pairwise judge calls per candidate pair (3 judges x 2 orderings). The judge cost is dominated by Claude Sonnet ($11.29), which is also the only judge that consistently discriminates — money well spent.

### 5.4 Wasted API Calls

- **GPT-5.2 empty-text calls:** 49 agents across Blocks 1 and 4 produced 0-length text but consumed 2048 tokens each. At GPT-5.2 pricing (~$0.06/1K output tokens), this wastes approximately 49 x 2048 x $0.06/1000 ~ **$6.02** in the pilot alone.
- **No explicit retry/error entries** appear in the cost log — the framework treats empty text as a valid response, not an error. This means the bug is silent.
- **Gemini token inflation:** 23+ agents report 2044 output_tokens but produce only 249–454 characters of text. The cost is based on token count, not text length, so Gemini's "thinking tokens" are being paid for but not used.

---

## 6. Data Quality

### 6.1 GPT-5.2 Empty Text Bug — CRITICAL

**31 out of 174** Block 1 agent outputs (17.8%) and **18 out of 114** Block 4 agent outputs (15.8%) have `text: ""` (zero-length) despite `output_tokens: 2048`. **Every single empty-text instance is from the `gpt-5.2` model.** This is a systematic API response parsing bug, not random failure.

Affected runs include cases where:
- ALL agents produced empty text (e.g., run_bdb623aefffe: 3 agents, run_ff479dcc53a6: 5 agents, run_020d7a1bc18f: 2 agents)
- Some agents are empty while others are not (e.g., run_64611cf5a84a: agent_1 empty, agent_0 has 8,246 chars)

**Downstream consequences:**
- Evaluation proceeds on empty text → judges call everything "tie" → BT scores = uniform → inflated kappa
- Consensus "selects" the one non-empty agent's output (or an empty output if all are empty)
- Quality scores for these runs are meaningless
- The affected runs contaminate aggregate statistics

**Recommendation:** (a) Fix the GPT-5.2 response parser immediately, (b) exclude all runs with any empty-text agents from pilot analysis, (c) re-run affected conditions, (d) add an output-length validator that flags and retries empty responses.

### 6.2 Gemini Token–Text Mismatch — MAJOR

23 instances in Block 1 of `gemini-2.5-pro` agents reporting 2044 output_tokens but only 249–454 characters of actual text. Example: run_2d2ab668bdcb agent_2 has 2044 tokens but 404 characters (~3 tokens per character would imply ~130 tokens, not 2044).

This suggests Gemini is counting internal "thinking" or "reasoning" tokens in its response metadata, but the text field only contains the visible response. The pairwise judges evaluate the visible text, but cost accounting uses the inflated token count.

**Impact:** (a) Cost overestimation for Gemini runs, (b) any analysis using output_tokens as a proxy for output length will be distorted, (c) these short responses may receive lower quality scores not because the model is worse but because the response was truncated or abbreviated.

### 6.3 Identical Outputs — See Section 2.1

Five agents producing byte-identical text (run_ccb11924d67f) is not a caching bug per se — it is the expected behavior of a deterministic API at low temperature. But it does mean these runs do not represent independent multi-agent deliberation.

### 6.4 Output Truncation

Many agents hit the 2048-token output limit (particularly gpt-5.2 and claude-opus-4-6). In Block 4, runs with long creative tasks (board game design, value conflict dialogue) frequently show all agents at exactly 2048 tokens — the responses were truncated mid-sentence. This ceiling effect compresses quality variation: truncated outputs may be penalized not for poor reasoning but for incomplete rendering.

**Recommendation:** Increase `max_output_tokens` to 4096+ for creative tasks, or add a completion check that retries if the output appears truncated.

---

## 7. Summary Table

| Category | Issue | Severity | Blocks Affected |
|---|---|---|---|
| BT denominator confound | Quality metric incomparable across n | **Critical** | Block 4 (H3) |
| GPT-5.2 empty text | 17–18% of agents produce 0-length text | **Critical** | Blocks 1, 4 |
| Deterministic dl=1 | Identical prompts → identical outputs | **Critical** | Block 1 (H1) |
| Gemini token inflation | 2044 tokens but 250–450 chars of text | **Major** | Block 1 |
| Aggregate kappa inflated | 0.934 driven by Block 0 ties | **Major** | Report-level |
| Quality score discreteness | BT clusters at 0.167, 0.250, 0.333 etc. | **Minor** | All multi-agent |
| Run count discrepancy | 22 in directory vs 20 in report | **Minor** | Blocks 1, 4 |
| Cost tracker bug | Cumulative total resets on resume | **Minor** | Monitoring |
| Output truncation | 2048-token ceiling compresses variation | **Minor** | Blocks 1, 4 |

---

## 8. Recommendations (Prioritized)

1. **Fix the BT denominator confound before any further analysis.** Use `quality_adjusted = quality_score - 1/(n_agents+1)` as the dependent variable for H3, or switch to a fixed-reference evaluation where the number of BT candidates is constant.

2. **Fix the GPT-5.2 empty text bug immediately.** Add response validation (reject and retry if `text.length == 0`). Exclude all contaminated runs from analysis. Re-run affected conditions.

3. **Address dl=1 determinism.** Either raise the temperature floor, add per-agent prompt perturbations, or reclassify dl=1 as single-agent baseline.

4. **Report kappa by block**, excluding Block 0 from the multi-agent reliability assessment.

5. **Investigate Gemini token counting.** Determine whether the inflated token count reflects genuine API behavior (thinking tokens) or a parsing bug. Adjust cost accounting accordingly.

6. **Increase max_output_tokens** for long-form tasks to prevent ceiling effects.

7. **Fix the cost tracker** to prevent cumulative total from resetting on process resume.

---

## 9. Assessment of GO/NO-GO Decision

The automated GO decision was based on:
- Kappa > 0.4 (but inflated — see section 4.3)
- Disagreement levels separable (based on only 4 obs/level)
- Paradox signal detected (but artifactual — see section 3.1)

**My assessment:** The GO was premature. The kappa criterion was met trivially (Block 0 inflation). The paradox signal is a measurement artifact. The GPT-5.2 empty-text bug contaminates ~17% of multi-agent data.

However, the infrastructure fundamentally works: runs execute, judges evaluate, data is captured. The problems are all fixable without redesigning the experiment. I recommend a **conditional GO**: proceed with the main batch **after** fixing the GPT-5.2 parser bug and confirming the BT metric correction, then re-validate the paradox signal on the first ~50 Block 4 main-batch runs with the corrected metric.

---

**Overall Verdict: MAJOR ISSUES**

The pilot demonstrates that the experimental infrastructure is functional but reveals three critical measurement problems that would invalidate headline findings if carried forward. All three are fixable. The experiment design is sound; the implementation needs targeted repairs.

*— Dr. James Okonkwo, February 28, 2026*
