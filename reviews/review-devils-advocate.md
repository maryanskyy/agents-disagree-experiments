# Devil's Advocate Review: What Could Go Catastrophically Wrong?

**Reviewer:** Dr. Priya Sharma (NLP/LLM Evaluation Specialist)  
**Date:** 2026-02-28  
**Target:** "When Agents Disagree" — $1,147 experiment plan  
**Stance:** Most hostile reviewer this paper could draw

---

## Overall Risk Level: HIGH

This experiment is well-designed relative to the average multi-agent paper. It addresses known evaluation pitfalls (position bias, self-preference, verbosity) with documented mitigations. That said, several structural risks remain that could render the entire $1,147 investment worthless. The risks are manageable, but some require action *before* the run starts.

---

## Top 5 Ways This Experiment Produces Worthless Results

*Ranked by probability × severity.*

### 1. Evaluation Circularity Creates a False Disagreement Dividend (Probability: 40%)

**The problem.** The judge pool and agent pool draw from the same two model families (Anthropic and Google). The primary judges are `gemini-2.5-pro`, `claude-opus-4-6`, and `claude-sonnet-4-5`. The agent pool is `claude-opus-4-6`, `gemini-2.5-pro`, and `gemini-2.0-flash`. Per-run exclusion prevents a model from judging its own outputs, but the *reserve* judges (`claude-haiku-4`, `gemini-1.5-pro`) are still from the same two families. There are **zero judges from outside the Anthropic/Google ecosystem.**

When all three agents are used (heterogeneous_strong_mix), the judge pool must exclude all three primary judges. The fallback panel becomes `claude-sonnet-4-5`, `claude-haiku-4`, `gemini-1.5-pro` — still entirely Anthropic + Google. Family-level stylistic preferences are not controlled.

**Why this is catastrophic for the Disagreement Dividend specifically:** Multi-agent debate/consensus iteratively refines outputs, making them smoother, more hedged, and more "LLM-like" (Fang et al. 2026 documented this exact bias). LLM judges from the same family ecosystem will systematically prefer these polished consensus outputs over raw single-agent outputs. This could create an *artifactual* Disagreement Dividend that disappears under human evaluation.

The evaluation methodology document (Section 1.2, Bias #4) explicitly identifies this threat and recommends human validation to detect it. But human validation is not budgeted, staffed, or scheduled.

**Risk level:** HIGH — this is the single most likely way to produce a false positive on H1.

**Mitigation:**
- Add at least one judge from a different ecosystem (GPT-4o, Llama 3.1 70B, or Mistral Large). GPT-4o is the standard in the literature and its absence is conspicuous.
- If no third ecosystem is feasible: increase human validation to 30% for Block 1 specifically and report LLM-vs-human agreement broken out by condition (single-agent vs. multi-agent).
- Run the length-controlled analysis described in the eval methodology. If the quality advantage of multi-agent outputs disappears when controlling for length, the Dividend is likely a verbosity artifact.

---

### 2. Null Result on All Four Hypotheses Due to Insufficient Power and Model Similarity (Probability: 30%)

**The problem.** The experiment plan mentions power analysis as a requirement (Evaluation Methodology, Phase 0, Step 0.5) but **no power analysis has been conducted.** Effect sizes are completely unknown. The experiment is flying blind on statistical power.

Specific underpowered conditions:
- **Block 2 (Topology Comparison):** 2 repetitions × 8 tasks = 16 observations per cell. With 48 cells in the factorial (4 topologies × 3 consensus × 2 agent counts × 2 task types), you need ~96 independent tests. At α=0.05 with Bonferroni correction, you need p < 0.0005 per test. With 16 observations per cell, you can only detect effects of Cohen's d > 1.2. If the topology effect is moderate (d ≈ 0.5), you will miss it entirely.
- **Block 5 (Interaction Probe):** Same structure, same problem. The interaction effect (topology × consensus) is typically smaller than main effects. You're hunting for a second-order effect with first-order sample sizes.
- **Block 1 (Disagreement Dividend):** 2 repetitions across 5 disagreement levels × 3 agent counts × 2 task types = 60 cells. Even aggregating across agent counts, the inverted-U curve needs enough resolution to detect a peak. With 2 reps per cell, noise will drown the signal.

Meanwhile, model similarity is a real concern. Claude Opus and Gemini 2.5 Pro are both frontier models — they may produce near-identical quality outputs on well-defined analytical tasks. If so, "disagreement" between them is just surface-level stylistic variation with no quality-relevant signal. The Disagreement Dividend requires that disagreement actually reflects *complementary* reasoning, not just paraphrase diversity.

**Risk level:** HIGH — this is a $1,147 experiment that may produce confidence intervals too wide to distinguish any effect from noise.

**Mitigation:**
- Run a pilot: 50 runs from Block 1 and Block 4 before committing the full budget. Estimate effect sizes from the pilot. If d < 0.3, increase repetitions or reduce the factorial.
- Aggregate across tasks within type for primary analyses (don't slice by individual task until secondary analysis).
- Use mixed-effects models (task as random effect) instead of per-cell t-tests. This pools variance across tasks and dramatically increases effective power.
- Explicitly pre-register the analysis plan including multiple comparison corrections (Bonferroni-Holm or FDR control).

---

### 3. Task Contamination Confounds Analytical Results (Probability: 50% for at least one task)

**The problem.** Several of the 16 hand-curated tasks are well-known problems or close paraphrases of common benchmark items.

**Definite contamination risks:**
- `analytical_01_logicians_bar`: The "three logicians in a bar" puzzle is one of the most famous logic puzzles on the internet. Every frontier LLM has seen this exact problem and its solution hundreds of times in training data. Performance on this task measures *recall*, not reasoning. This task is **compromised** and should be removed or replaced.
- `analytical_03_premise_consequence`: Modus ponens chain reasoning on formal premises is a standard textbook exercise. The specific content (E2E encryption → keyword filtering) is novel, but the reasoning pattern is trivially handled by any LLM trained on logic exercises.
- `analytical_06_constraint_satisfaction`: Constraint satisfaction scheduling with 5 variables and 6 constraints is a standard CSP textbook exercise. LLMs have seen hundreds of these.
- `creative_08_entanglement_for_child`: "Explain quantum entanglement to a child" is perhaps the single most common science communication prompt in LLM training data.

**Why this matters for the experiment specifically:** Contaminated tasks will show artificially LOW disagreement (all models retrieve the same memorized solution) and artificially HIGH quality (the memorized solution is correct). This compresses the disagreement range and biases against finding the Disagreement Dividend on analytical tasks. Worse, it could create a spurious analytical-vs-creative difference that's actually a contamination-vs-no-contamination difference.

**Risk level:** MEDIUM-HIGH — at least 2 of 8 analytical tasks are severely compromised, and at least 1 creative task is contaminated. With only 8 tasks per type, losing 2-3 tasks to contamination leaves too few for robust inference.

**Mitigation:**
- **Replace** `analytical_01_logicians_bar` with a novel logic puzzle that has the same structure but different surface content. (E.g., change the setting, add a twist, use a less-known variant.)
- **Replace** `creative_08_entanglement_for_child` with a less commonly prompted science explanation (e.g., explain chirality, or explain why the sky is dark at night despite infinite stars).
- For remaining tasks: run each model solo on each task and inspect whether all three models produce near-identical reasoning chains. If they do, the task is likely contaminated.
- Add a contamination analysis section to the paper: compute inter-model similarity on solo runs and flag tasks where solo outputs converge suspiciously.

---

### 4. API Instability + Version Drift Invalidates Cross-Model Comparisons (Probability: 25%)

**The problem.** The experiment runs for ~24 hours on a laptop with live API calls to Anthropic and Google. Two structural risks:

**a) Model version pinning.** Anthropic models are properly version-pinned (`claude-opus-4-1-20250805`, `claude-sonnet-4-5-20250929`). But Google models are **NOT pinned**: `gemini-2.5-pro`, `gemini-2.0-flash`, `gemini-1.5-pro` use the floating alias. If Google updates any of these mid-experiment — which happens without warning — runs before and after the update come from *different models*. This is a silent, undetectable confound.

The experiment uses Gemini models as both agents and judges. A mid-experiment Gemini update would affect:
- Agent outputs (quality, style, length)
- Judge evaluations (preferences, calibration)
- Both simultaneously, compounding the confound

**b) Outage risk.** A 24-hour run on two providers guarantees exposure to at least rate limit fluctuations. The retry logic (max_retries=5 with backoff) handles transient failures, but a sustained outage (>1 hour) could:
- Stall the experiment and extend runtime to 48+ hours
- Cause the semaphore to saturate with blocked coroutines
- Result in partial blocks (some conditions completed, others not)
- If the laptop sleeps/hibernates, lose in-flight work

**c) Rate limit changes.** Google periodically adjusts Gemini rate limits. If the configured RPM (60 for gemini-2.5-pro) exceeds the actual limit, every call triggers rate-limiting and retries, ballooning cost and runtime.

**Risk level:** MEDIUM — the Google version pinning gap is a real methodological issue that a reviewer will catch.

**Mitigation:**
- **Pin Google model versions** using dated snapshots (e.g., `gemini-2.5-pro-preview-05-06` or whatever the latest stable version identifier is). Check the Google AI API documentation for exact version strings.
- Log the model version returned in each API response and verify consistency post-experiment.
- Run on a machine with guaranteed uptime (cloud VM, not a laptop). A `t3.medium` on AWS for 48 hours costs ~$2.
- Implement graceful pause/resume in the runner (the checkpoint system already supports this — just ensure the laptop power settings don't interfere).
- Add a pre-flight rate limit check: make 5 rapid calls to each provider and verify RPM capacity before committing.

---

### 5. The "Only Two Providers" Problem Makes Results Non-Generalizable (Probability: 80% that a reviewer raises this; ~40% that it's fatal to acceptance)

**The problem.** The experiment uses 3 models from 2 providers (Anthropic, Google). The claims are about *multi-agent orchestration topology* — a general architectural principle. But the evidence is entirely from Claude × Gemini interactions.

A reviewer will ask: *"How do you know the Disagreement Dividend isn't just a Claude-Gemini complementarity effect? If you replaced Gemini with GPT-4o, or with Llama 3.1, would the same topologies still be optimal?"*

This is devastating because the paper's central claim (RQ4) is that **topology matters more than model capability**. But with only 2 provider families, you cannot distinguish:
- "Topology matters" (the general claim)
- "Mixing Anthropic and Google models creates unique interaction effects" (a provider-specific finding)
- "Claude is good at X, Gemini is good at Y, and combining them helps" (a capability complementarity finding)

The `paradox_strong_weak` condition (opus + flash) compounds this: Gemini Flash is a deliberately cheap/fast model. The "paradox" of weak agents hurting quality may just be "bad model pollutes good model's output" — which is unsurprising and uninteresting.

**Risk level:** HIGH — this is the single most predictable reviewer objection and the hardest to mitigate post-hoc.

**Mitigation:**
- **Best mitigation:** Add at least one model from a third provider (GPT-4o-mini, Llama 3.1 70B via Together API, or Mistral Large). Even a reduced factorial with the third provider (e.g., only Block 1 and Block 4) would transform the generalizability argument.
- **Minimum mitigation:** Frame the claims carefully. "We demonstrate the Disagreement Dividend in Claude × Gemini configurations" is publishable. "Topology matters more than model capability" is NOT defensible with 2 providers.
- Add a limitations section explicitly acknowledging the provider diversity gap and proposing follow-up work.
- Compute within-family and across-family effects separately. If the Disagreement Dividend only appears in cross-family conditions, that's a different (but still interesting) finding.

---

## Top 3 Ways to WASTE the $1,147

### Waste Mode 1: Run the Full Experiment Without a Pilot, Discover Effect Sizes Are Too Small

**What happens:** You spend $1,147 on 3,456 runs. Analysis reveals that 95% confidence intervals overlap for all major comparisons. The topology differences exist but are d ≈ 0.2 — too small to detect with 2-4 reps per cell. The Disagreement Dividend curve is noisy but flat-ish. You have a pile of data that "trends in the right direction" but nothing statistically defensible. 

**Probability:** 25-30%

**Prevention:** Spend $100 on a 300-run pilot (Block 0 + partial Block 1 + partial Block 4). Estimate effect sizes. Recalibrate sample sizes before committing the remaining $1,000.

---

### Waste Mode 2: Google Updates gemini-2.5-pro Mid-Experiment, Invalidating Half the Data

**What happens:** Runs 1-1,500 use Gemini 2.5 Pro v1. Google silently updates the alias. Runs 1,501-3,456 use Gemini 2.5 Pro v2. The quality distribution shifts. Cross-block comparisons are invalid. You can't tell if topology effects are real or version effects.

**Probability:** 10-15% (Google has done this before)

**Prevention:** Pin model versions. Log response headers. Run a consistency check comparing first-half vs second-half solo calibration scores.

---

### Waste Mode 3: LLM Judges Agree With Each Other But Not With Humans, Invalidating All Quality Scores

**What happens:** You run the experiment, find beautiful results — the inverted-U curve, the Quorum Paradox, topology effects. Then you do human validation (or a reviewer demands it). The human raters disagree with the LLM judges on 40%+ of comparisons. The pattern: LLM judges systematically prefer multi-agent consensus outputs (which are longer, more hedged, more "LLM-like"), but humans prefer the more direct single-agent outputs. Every headline result is a verbosity/style artifact.

**Probability:** 15-20%

**Prevention:** Run human validation on the pilot data FIRST. If human-LLM agreement < 0.6 on pairwise comparisons, redesign the judge prompt or add explicit length-normalization before the full run.

---

## Detailed Risk Register

### Risk 1: Evaluation Circularity / LLM-as-Judge Bias
| | |
|---|---|
| **Risk Level** | HIGH |
| **Probability** | 40% |
| **Impact** | False positive on Disagreement Dividend (H1); paper retracted or publicly challenged |
| **Mitigation** | Add GPT-4o or open-weight judge; increase human validation to 30% for Block 1; run length-controlled analysis |
| **Residual Risk** | Medium (even with diverse judges, shared LLM biases may persist) |

### Risk 2: Insufficient Statistical Power
| | |
|---|---|
| **Risk Level** | HIGH |
| **Probability** | 30% |
| **Impact** | No significant results on any hypothesis; $1,147 wasted |
| **Mitigation** | Pilot study; mixed-effects models; pre-registered analysis plan with correction method |
| **Residual Risk** | Low (pilot provides early warning) |

### Risk 3: Task Contamination
| | |
|---|---|
| **Risk Level** | MEDIUM-HIGH |
| **Probability** | 50% (for ≥1 task) |
| **Impact** | Biased disagreement measurement on analytical tasks; spurious task-type differences |
| **Mitigation** | Replace 2-3 contaminated tasks; run solo contamination screen; add contamination analysis to paper |
| **Residual Risk** | Low (replaceable before experiment starts) |

### Risk 4: Google Model Version Drift
| | |
|---|---|
| **Risk Level** | MEDIUM |
| **Probability** | 15% |
| **Impact** | Silent confound invalidating cross-condition comparisons involving Gemini models |
| **Mitigation** | Pin version strings; log response model metadata; consistency checks |
| **Residual Risk** | Low (fully mitigable) |

### Risk 5: Provider Diversity Gap (2 Providers)
| | |
|---|---|
| **Risk Level** | HIGH |
| **Probability** | 80% a reviewer raises it |
| **Impact** | Results framed as non-generalizable; claims about topology dominance rejected |
| **Mitigation** | Add third provider even in limited conditions; reframe claims appropriately |
| **Residual Risk** | Medium (framing helps, but the fundamental gap remains) |

### Risk 6: Multiple Comparison Inflation (Type I Error)
| | |
|---|---|
| **Risk Level** | MEDIUM |
| **Probability** | 35% |
| **Impact** | "Significant" results that don't replicate; embarrassing correction |
| **Mitigation** | Pre-register analysis plan; use Bonferroni-Holm or FDR correction; report effect sizes with CIs, not just p-values |
| **Residual Risk** | Low (standard statistical practice) |

### Risk 7: Human Evaluation Not Executed
| | |
|---|---|
| **Risk Level** | HIGH |
| **Probability** | 60% (probability it doesn't happen, not that it's needed) |
| **Impact** | Paper rejected at any top venue; eval methodology document itself requires 15-25% human validation |
| **Mitigation** | Budget human evaluation NOW; recruit raters; design evaluation sheets (already partially done) |
| **Residual Risk** | Low (it's a logistics problem, not a conceptual one) |

### Risk 8: Cost Overrun
| | |
|---|---|
| **Risk Level** | LOW-MEDIUM |
| **Probability** | 20% |
| **Impact** | Budget exceeds $2K+; experiment paused mid-block |
| **Mitigation** | Cost tracking is already implemented; set hard budget cap with auto-pause; run cheapest blocks first |
| **Residual Risk** | Low (cost tracker provides real-time visibility) |

### Risk 9: Laptop/Infrastructure Failure
| | |
|---|---|
| **Risk Level** | LOW |
| **Probability** | 10% |
| **Impact** | Lost progress; partial data requiring re-runs |
| **Mitigation** | Checkpoint system already implemented; use cloud VM; ensure power settings prevent sleep |
| **Residual Risk** | Very Low |

### Risk 10: "Strong + Weak" Paradox Is Trivially Expected
| | |
|---|---|
| **Risk Level** | MEDIUM |
| **Probability** | 50% a reviewer dismisses it |
| **Impact** | RQ4 / H4 rejected as obvious ("of course adding a bad model hurts") |
| **Mitigation** | The homogeneous-strong control (opus × opus × opus) helps. But the paper needs to show the paradox occurs even in STRONG mixtures, not just strong+weak. Consider adding a homogeneous_gemini_pro condition at n=2,3,5 alongside the paradox conditions. |
| **Residual Risk** | Medium (the current framing is vulnerable) |

---

## The Killer Questions Reviewers Will Ask

### Q1: "You claim topology matters more than model capability, but you only tested 3 models from 2 providers."
**Prepared answer needed:** Acknowledge limitation. Present within-family vs. across-family effect decomposition. Frame as "first demonstration of topology effects in heterogeneous LLM ensembles" rather than universal claim.

### Q2: "Your LLM judges share the same training biases as your agents. How do you know the Disagreement Dividend isn't a self-reinforcing evaluation artifact?"
**Prepared answer needed:** Human validation data. Length-controlled analysis. Per-judge agreement rates broken out by condition. If you don't have human validation data, you cannot answer this question.

### Q3: "With 48+ factorial cells and 2 repetitions each, how many comparisons did you test? What's your family-wise error rate?"
**Prepared answer needed:** Pre-registered analysis plan with explicit correction method. Report adjusted p-values. Lead with effect sizes and confidence intervals.

### Q4: "The three logicians task is in every LLM's training data. How do you know your analytical results aren't just measuring retrieval?"
**Prepared answer needed:** Remove or replace the task. Run solo contamination analysis. Report per-task results transparently.

### Q5: "You planned 15-25% human validation in your methodology but report 0% in the paper. Why?"
**Prepared answer needed:** You can't answer this. Do the human evaluation.

### Q6: "Your 'paradox' is just 'adding Gemini Flash to Claude Opus hurts quality.' Is that surprising to anyone?"
**Prepared answer needed:** Show it happens even in strong-only mixtures. Show it's topology-dependent (quorum yes, pipeline no). Show the mechanism (conformity pressure, diversity collapse) via debate round analysis.

---

## Multiple Comparison Burden Analysis

| Block | Primary Comparisons | Cells | Correction Needed |
|---|---|---|---|
| Block 1 (Disagreement Dividend) | 5 levels × 3 agent counts × 2 task types = 30 means; pairwise = ~435 comparisons | 30 | FDR or trend test |
| Block 2 (Topology) | 4 topologies × 3 consensus × 2 counts × 2 types = 48 cells | 48 | ANOVA + post-hoc Tukey HSD |
| Block 3 (MVQ) | 4 agent counts × 4 thresholds × 2 types = 32 curves | 32 | Threshold is post-hoc; FDR |
| Block 4 (Paradox) | 3 agent counts × 2 model configs × 2 types = 12 cells + baselines | 12 | Paired tests with Holm correction |
| Block 5 (Interaction) | 4 × 3 × 2 × 2 = 48 cells; interaction test = 4 × 3 = 12 interaction terms | 48 | ANOVA with interaction terms |
| **Total unique statistical tests** | **~150 primary comparisons** | | **At α=0.05, expect ~7.5 false positives by chance** |

Without correction: ~7-8 "significant" results expected by pure chance. This is enough to construct a plausible-looking but false narrative. **Pre-registration and correction are mandatory.**

---

## Go/No-Go Recommendation

### Verdict: CONDITIONAL GO

The experiment design is fundamentally sound. The evaluation methodology is among the most rigorous I've seen for a multi-agent study. The risks are real but mostly mitigable with pre-experiment actions that cost little time or money.

### Required Before Starting (No-Go Without These):

1. **Pin Google model versions.** This takes 5 minutes and eliminates a silent confound. No excuse not to do it.

2. **Replace 2 contaminated tasks.** Remove `analytical_01_logicians_bar` and `creative_08_entanglement_for_child`. Replace with novel tasks of similar difficulty. This takes 1-2 hours.

3. **Pre-register the analysis plan.** Write down: which comparisons are primary, which are exploratory, what correction method will be used, what effect size constitutes "meaningful." OSF pre-registration is free.

4. **Run a $100-150 pilot.** 200-300 runs from Block 0 + Block 1 + Block 4. Estimate effect sizes. Verify judge-human agreement on 30 pairs. Verify cost estimates are accurate. If effect sizes are tiny, restructure before spending $1,000.

### Strongly Recommended (Paper Quality):

5. **Add one judge from a third ecosystem** (GPT-4o or open-weight). This addresses the circularity concern and is the single highest-ROI improvement.

6. **Budget and schedule human evaluation.** 75-100 person-hours based on the eval methodology. Recruit 2-3 raters now. Without this, the paper is unpublishable at any top venue.

7. **Add one model from a third provider** to at least Block 1 and Block 4. Even GPT-4o-mini in a reduced condition set would transform the generalizability argument.

### Acceptable Residual Risks:

- Null result on H3 (MVQ saturation may not be detectable at small n). This is fine — report as "inconclusive."
- Moderate effect sizes on H2 (topology). Even small effects with proper CIs are publishable if the methodology is sound.
- Creative task evaluation noise. LLM judges are known to be unreliable here. Higher human validation rate (25%) is already planned.

### The Bottom Line:

The $1,147 is not wasted *if and only if* the pre-experiment actions above are completed. Without them, the most likely outcome is a pile of data that's suggestive but not defensible — the worst possible outcome for a paper that needs to establish a new empirical phenomenon. The pilot alone could save $900 if it reveals the experiment needs restructuring.

Spend the first $150 proving the experiment works. Then spend the remaining $1,000 running it properly.
