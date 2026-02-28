# Alternative Strategy Review: Is There a Better Way?

**Reviewer:** Dr. James Okonkwo (Empirical Methodology)
**Date:** 2026-02-28
**Budget:** $1,147 | **Time:** 24 hours | **Hardware:** MacBook M4

---

## Current Design Verdict: **Acceptable — Run with Tweaks (Pilot First)**

The experiment is well-structured and addresses all four research questions. However, it spreads $1,147 across too many conditions, creating a "mile wide, inch deep" risk. Two specific blocks (Block 5 interaction and the best-of-N baselines) consume $559 (49% of budget) while providing evidence that is either redundant or achievable more cheaply. A phased pilot-then-commit strategy would reduce the risk of wasting the full budget on a miscalibrated evaluation pipeline.

---

## 1. Budget Optimization: Evidence-per-Dollar Analysis

### Block-by-Block ROI

| Block | Cost | Runs | Primary RQ | Evidence Value | Verdict |
|-------|------|------|-----------|---------------|---------|
| Block 0 Calibration | $19 | 192 | All | Essential baseline; grounds all comparisons | **Must run** |
| Block 3 MVQ | $48 | 192 | RQ2 | Direct MVQ curves; cheap because small factorial | **Must run** |
| Block 1 Disagreement | ~$150 | 480 | RQ1 | The paper's signature figure (Q(d) curve) | **Must run** |
| Block 4 Paradox | ~$165 | 384 | RQ3 | Most citable finding if confirmed | **Must run** |
| Block 1 Best-of-N | ~$150 | 480 | RQ1 | Important control, but Opus is expensive here | Optimize |
| Block 4 Best-of-N | ~$83 | 192 | RQ3 | Important control for paradox interpretation | Optimize |
| Block 2 Topology | $252 | 768 | RQ2/RQ4 | 4x3 factorial; answers topology question | Keep but reduce |
| Block 5 Interaction | $279 | 768 | RQ4 | Overlaps heavily with Block 2 | **Cut or merge** |

### Key Observation: Block 5 Is Largely Redundant

Block 5 tests 4 topologies x 3 consensus x 2 agent counts x 2 task types at disagreement level 2. Block 2 tests the same 4 x 3 x 2 x 2 at disagreement level 3. They are nearly identical factorials at adjacent disagreement levels. **Merging them** — running the Block 2 design at disagreement levels {2, 3} with 2 reps each — would provide the same interaction evidence for ~$252 instead of $531. This alone saves $279.

### Judge Cost Dominance

A critical but underappreciated issue: **143,808 of 158,656 total API calls (91%) are judge calls**, not agent calls. With a 3-judge panel evaluating each pair in both orderings (A/B + B/A), every pairwise comparison generates 6 judge calls. The evaluation methodology is sound, but judge costs likely dominate the budget far more than model selection for agents.

**Recommendation:** Verify the cost split between agent and judge calls. If judge costs exceed 60% of total, consider:
- Using 2 judges instead of 3 for non-critical blocks (0, 3)
- Reducing pairwise comparisons via smart tournament seeding rather than exhaustive comparison
- Using a cheaper judge model (e.g., GPT-4o-mini) as the third panelist

---

## 2. Alternative Experiment Designs

### A. Simpler A/B Test Design
**Verdict: Not recommended for RQ1, useful for RQ3**

The inverted-U curve (RQ1) requires 5+ points on the disagreement axis — an A/B test can't map a curve. However, for RQ3 (Quorum Paradox), a focused A/B design (n=2 vs n=3 vs n=5, within-subjects) with higher replication (8-10 reps instead of 4) would give more statistical power for detecting the paradox dip. The current 4-rep design may be underpowered for a subtle effect.

### B. Bayesian Adaptive Design
**Verdict: Appealing but impractical for this timeline**

A Bayesian adaptive approach (e.g., Thompson sampling across conditions) could allocate budget to promising cells as data arrives. This would be ideal for Block 2/5 topology exploration. However:
- Implementation adds ~4-6 hours to a 24-hour window
- Harder to pre-register (reviewers may suspect post-hoc optimization)
- Requires real-time analysis between batches
- Gains are largest when the condition space is large and many cells are uninteresting — but with only 4 topologies x 3 consensus, the space isn't large enough to justify the complexity

**Use instead:** A simple two-phase approach (Phase 1: 1 rep everywhere; Phase 2: 2 more reps on interesting cells).

### C. Tournament-Style Design
**Verdict: Good complement for topology comparison, not a replacement**

Round-robin tournaments between topologies (each pair competes head-to-head on the same tasks) would produce clean pairwise rankings with Bradley-Terry aggregation. This is essentially what the evaluation methodology already does via pairwise LLM-judge comparison. The factorial design is the right structure for data collection; the tournament framing applies at the analysis stage.

### D. Recommended Hybrid Design

**Core insight: Separate the "curve-mapping" experiments from the "factorial-exploration" experiments.**

- **Curve-mapping blocks** (Blocks 1, 3, 4): These trace specific functional forms (Q(d), MVQ curves, paradox dips). They need controlled, pre-specified conditions with enough reps for confidence intervals. Keep factorial structure, increase reps.
- **Exploration blocks** (Blocks 2, 5): These explore a large condition space (topology x consensus). Use a two-phase adaptive approach: Phase 1 screens all cells (1 rep), Phase 2 adds reps to interesting cells.

---

## 3. Phased Approach: Strongly Recommended

### Phase 0: Pilot ($120-150, ~4 hours)

**What the pilot tests:**

1. **Judge reliability** — Run 30 pairwise comparisons with the 3-judge panel. Compute inter-judge Cohen's kappa. If kappa < 0.40, the entire experiment is in jeopardy and you need to redesign rubrics before spending $900+.

2. **Disagreement level separation** — Run Block 0 calibration ($19) + a mini Block 1 (5 disagreement levels x 2 task types x n=3 only x 1 rep = 10 runs, ~$15). Verify that disagreement levels actually produce measurably different disagreement rates. If levels 2 and 3 produce identical disagreement, the confound-fix design has a gap.

3. **Paradox existence check** — Run 2 task types x {n=2, n=3} x 2 model configs (paradox + homogeneous) x 2 reps = 16 runs (~$25). If there's zero signal of a quality dip at n=3, you may want to reallocate Block 4 budget.

4. **Pipeline integration test** — End-to-end test of data collection, evaluation, and analysis pipeline on real data. Uncover bugs before the main run.

5. **Effect size estimation** — Use pilot data for post-hoc power analysis to right-size replication counts for the main run.

**Pilot cost breakdown:**

| Component | Cost |
|-----------|------|
| Block 0 full calibration | $19 |
| Mini Block 1 (10 runs) | ~$15 |
| Mini Block 4 (16 runs) | ~$25 |
| Judge reliability test (30 pairs x 6 judge calls) | ~$30 |
| Pipeline overhead / debugging | ~$30 |
| **Pilot total** | **~$120-150** |

**Go/No-go criteria after pilot:**
- Judge kappa > 0.50: Proceed
- Disagreement levels produce distinct d values: Proceed
- At least directional paradox signal: Proceed with Block 4
- Judge kappa < 0.40: **Stop**, fix rubrics, re-pilot
- No disagreement level separation: Redesign levels
- No paradox signal at all: Reduce Block 4, reallocate to Block 1

### Phase 1: Main Run ($700-800, hours 4-18)

Run in priority order so highest-value data is collected first:

| Priority | Block | Reps | Cost | Time |
|----------|-------|------|------|------|
| 1 | Block 1 Disagreement | 3 | ~$225 | 3h |
| 2 | Block 4 Paradox | 5 | ~$205 | 3h |
| 3 | Block 3 MVQ | 3 | $48 | 1h |
| 4 | Block 1 Best-of-N | 1 | ~$75 | 1.5h |
| 5 | Block 2 Topology (1 rep) | 1 | ~$126 | 2h |
| **Subtotal** | | | **~$679** | **10.5h** |

### Phase 2: Targeted Follow-up ($150-250, hours 18-22)

Informed by Phase 1 preliminary analysis:
- Add reps to near-significant Block 4 cells
- Add 1 more rep to Block 2 for conditions showing interesting topology effects
- Run targeted follow-ups on any unexpected findings
- Generate human evaluation sheets for key claims

### Phase 3: Analysis and Writeup (hours 22-24)

- Full statistical analysis
- Generate all figures
- Preliminary paper sections

---

## 4. Model Cost Optimization

### Where Opus Money Goes

Claude Opus appears in:
- **Block 0** (1/3 of calibration): Tiny cost, must keep for baseline
- **Block 1** (heterogeneous_strong_mix): 1/3 of each agent group — necessary for heterogeneity
- **Block 1 best-of-N** (homogeneous_opus): **100% Opus** — this is expensive
- **Block 2** (heterogeneous_strong_mix): 1/3 of each group
- **Block 3** (heterogeneous_strong_mix): 1/3 of each group
- **Block 4 paradox** (paradox_strong_weak): Opus + Flash — necessary for strong/weak design
- **Block 4 homogeneous controls** (homogeneous_opus): **100% Opus** — expensive
- **Block 4 best-of-N** (homogeneous_opus): **100% Opus** — expensive
- **Block 5** (heterogeneous_strong_mix): 1/3 of each group

### Optimization Suggestions

**Suggestion 1: Replace Opus with Sonnet in best-of-N baselines (saves ~$100-150)**

The best-of-N baselines exist to answer: "Is multi-agent better than just generating N outputs from one strong model and picking the best?" The "strong model" doesn't have to be the *most* expensive model — it needs to be the strongest model in the agent pool. If you replace Opus with Claude Sonnet (3.5 or 4) in best-of-N baselines, the cost-matching argument gets slightly weaker but the control is still meaningful. Alternatively, keep Opus for best-of-N but reduce to 1 rep (since best-of-N has lower variance than multi-agent).

**Suggestion 2: Replace homogeneous_opus controls in Block 4 with homogeneous_pro (saves ~$50-80)**

The Block 4 homogeneous controls test whether the paradox disappears with equally-capable agents. Gemini 2.5 Pro is a reasonable "strong model" alternative. The paradox contrast (strong+weak vs homogeneous-strong) is preserved whether "strong" = Opus or "strong" = Pro.

**Suggestion 3: Reduce judge panel from 3 to 2 for non-critical blocks (saves ~$100-150)**

For Block 0 (calibration) and Block 3 (MVQ), a 2-judge panel with position randomization may suffice. Reserve the full 3-judge panel for Blocks 1, 4 (where the key claims live) and Block 2 (topology comparison). This cuts ~33% of judge costs on 2 blocks.

**Suggestion 4: Use a cheaper third judge (saves ~$50-100)**

If the 3-judge panel uses three frontier models, the third could be replaced with a capable but cheaper model (GPT-4o-mini or Gemini Flash as a "diversity" judge). The BT-sigma aggregation will downweight an unreliable judge automatically. This provides the multi-family diversity requirement while cutting cost.

### Model Optimization Summary

| Change | Estimated Savings | Risk |
|--------|------------------|------|
| Sonnet in best-of-N baselines | $100-150 | Slightly weaker cost-match argument |
| Pro instead of Opus in Block 4 controls | $50-80 | Minor; "strong" is relative |
| 2-judge panel for Blocks 0, 3 | $100-150 | Lower reliability detection |
| Cheaper third judge | $50-100 | BT-sigma compensates |
| **Total potential savings** | **$300-480** | |

**Could we use Sonnet instead of Opus everywhere?** No — and here's why. The paradox_strong_weak design *requires* a capability gap between agents. Opus is the "strong" anchor; Flash is the "weak" one. Replacing Opus with Sonnet narrows the capability gap and may wash out the paradox effect. For RQ3, Opus is essential. For RQ1 (heterogeneous mix), having Opus in the mix creates more interesting disagreement dynamics. The savings from eliminating Opus entirely (~$300) would come at the cost of weakening two research questions.

---

## 5. Scale vs. Depth Tradeoff

### Current: 3,456 runs across 4 RQs (broad)
### Alternative: ~2,000 runs on RQ1 + RQ3 (deep)

**Arguments for focusing on RQ1 + RQ3:**
- These are the most novel, citable findings
- The inverted-U curve (RQ1) and the paradox (RQ3) are "surprising" results that attract citations
- More reps = narrower confidence intervals = more convincing evidence
- Simpler paper = easier to write, review, and publish

**Arguments against dropping RQ2 + RQ4:**
- Block 3 (MVQ) costs only $48 and answers a distinct question — no reason to cut it
- The decision matrix (RQ4) is the paper's most *practical* contribution — practitioners need it
- A 4-RQ paper with supporting formal framework is positioned for a top venue; a 2-RQ paper competes at a lower tier
- The research questions were carefully adjudicated by the PI from two independent proposals

### Recommendation: Focus, Don't Amputate

Keep all 4 RQs but redistribute budget:

| Block | Current Reps | Recommended Reps | Reasoning |
|-------|-------------|-----------------|-----------|
| Block 0 | 4 | 4 | Already minimal |
| Block 1 disagreement | 2 | 3 | Core finding; needs tight CIs |
| Block 1 best-of-N | 2 | 1 | Control; lower variance |
| Block 2 topology | 2 | 1 (Phase 1) + 1 targeted | Screen then focus |
| Block 3 MVQ | 3 | 3 | Already lean and cheap |
| Block 4 paradox | 4 | 5-6 | Key claim; needs power |
| Block 4 best-of-N | 4 | 2 | Control; reduce |
| Block 5 interaction | 2 | 0 (merge into Block 2) | Redundant |

This shifts ~$300 from exploration (Blocks 2, 5) and controls (best-of-N) toward the core findings (Blocks 1, 4) while keeping all RQs alive.

---

## 6. Open-Source / Local Model Alternative

### M4 Capabilities for Inference

An M4 MacBook can run 7B-8B parameter models at roughly 30-60 tokens/second via llama.cpp or MLX. For a typical task response (~500 tokens), that's 8-17 seconds per generation. For 192 calibration runs, that's ~25-55 minutes — feasible.

### Where Local Models Add Value

**Good use cases:**
1. **Block 0 calibration baseline** — Add a local 8B model (e.g., Llama 3.1 8B, Mistral 7B) as a 4th calibration model. Cost: $0. Benefit: establishes a floor capability level and extends the capability range.
2. **Block 4 "weak" agent** — Replace or supplement Gemini Flash with a local 8B model for the paradox_strong_weak condition. This creates a *larger* capability gap, potentially making the paradox more visible.
3. **Supplementary replication** — Run additional reps of Blocks 1/4 with a local model substituted for Flash. Free reps increase N.

**Bad use cases:**
1. **Primary heterogeneous mix** — A 7B model in a mix with Opus and Gemini Pro creates such extreme capability asymmetry that the "disagreement" is more like noise vs. signal.
2. **Judge model** — Local 8B models lack the reasoning depth for reliable evaluation. Do not use as judges.
3. **Creative tasks** — 7B models produce noticeably lower-quality creative outputs, which could floor-effect the quality scores.

### Practical Constraint: Time

With 3,456 runs and ~15 seconds per agent generation, replacing one API model with a local model saves money but costs ~14 hours of local compute (assuming 1 generation per run). This is tight in a 24-hour window where you also need time for evaluation and analysis. Local models should supplement, not replace.

### Recommendation

Add a local 8B model to Block 0 calibration (free, 30 min compute) and optionally to Block 4 as an even-weaker agent (amplifies the paradox signal). Do not rely on local models for primary experiment blocks.

---

## 7. Worst-Case Salvage: What If Everything Is Null?

### Null Result Scenarios and Reframing

| RQ | Null Result | Reframing |
|----|-------------|-----------|
| RQ1 | Q(d) is flat or monotonically decreasing | "The Disagreement Tax: Multi-agent disagreement does not improve generative quality" — important negative finding that saves practitioners money |
| RQ2 | MVQ curves show no threshold behavior | "Diminishing returns, not threshold effects, characterize multi-agent scaling" — still useful |
| RQ3 | No paradox; quality monotonically increases | "Good news: more agents never hurts" — reassuring for system designers |
| RQ4 | No interaction effects; topology doesn't matter | "Topology-agnostic consensus: practitioners can use any topology" — simplifies deployment |

### Is a Null-Results Paper Publishable?

**Yes, with caveats.** A negative results paper is publishable if:

1. **The experiment had adequate power to detect meaningful effects.** This is where the current design is at risk — some blocks have only 2 reps. If CIs are wide, "null result" becomes "inconclusive," which is much harder to publish. This is another argument for the pilot-first approach: ensure you have the power to actually reject the hypotheses.

2. **The methodology is sound and well-documented.** The evaluation methodology (EVALUATION_METHODOLOGY.md) is exceptional — arguably publishable on its own. Even with null findings on RQ1-4, the evaluation framework (pairwise BT, bias mitigation, multi-judge panel with BT-sigma) is a methodological contribution.

3. **The results are pre-registered.** If hypotheses and analysis plans are committed before data collection, null results are credible. If they're post-hoc, reviewers will suspect you simply ran the wrong experiment.

### Salvage Value Assessment

| Component | Value Even If Null | Venue |
|-----------|-------------------|-------|
| Evaluation methodology | High (standalone contribution) | NeurIPS Datasets and Benchmarks |
| Cost-quality tradeoff data | Medium (useful empirical reference) | Workshop paper |
| Negative results on all 4 RQs | Medium-High (saves community effort) | EMNLP Findings, negative results workshop |
| Formal framework (MVQ bound, etc.) | Medium (theoretical interest persists) | Supports formal contributions regardless |
| Open-source experiment harness | Medium (community resource) | GitHub / accompanying code paper |

**Bottom line:** Total loss risk is low. The worst case is a workshop paper + methodology contribution. The evaluation framework alone may be worth the investment.

---

## $500 Minimum Viable Experiment Design

### Design Philosophy
Prioritize RQ1 (Disagreement Dividend) and RQ3 (Quorum Paradox) with enough statistical power to detect medium effects (Cohen's d >= 0.5).

### Block Structure

| Block | Modification | Runs | Est. Cost |
|-------|-------------|------|-----------|
| Block 0 Calibration | Keep as-is | 192 | $19 |
| Block 1 Disagreement | Keep as-is, 2 reps | 480 | ~$150 |
| Block 1 Best-of-N | **Cut** — replace with 1-rep Sonnet best-of-N | 240 | ~$40 |
| Block 2 Topology | **1 rep only**, merge disagreement levels 2+3 | 384 | ~$126 |
| Block 3 MVQ | Keep as-is, 3 reps | 192 | $48 |
| Block 4 Paradox | **3 reps** (down from 4), drop homogeneous controls | 192 | ~$80 |
| Block 4 Best-of-N | **Cut** | 0 | $0 |
| Block 5 Interaction | **Cut** (merged into Block 2) | 0 | $0 |
| **Total** | | **~1,680** | **~$463** |

Remaining ~$37 reserved for: judge reliability pilot, debugging, and a few targeted follow-up runs.

### What You Lose
- Homogeneous-strong controls for paradox (weaker but not fatal — Block 0 provides single-model baselines)
- Best-of-N baseline for Block 4 (can argue cost-matching from Block 1 best-of-N data)
- Interaction probe depth (1 rep limits interaction effect detection)
- Some statistical power on Block 4 (3 reps instead of 4)

### What You Keep
- Full Q(d) curve for RQ1 (signature figure)
- MVQ curves for RQ2 (cheapest block)
- Paradox detection for RQ3 (adequate power for medium effects)
- Topology comparison for RQ4 (screening level)

---

## Recommended Changes (Ranked by Impact-per-Effort)

| Rank | Change | Impact | Effort | Savings |
|------|--------|--------|--------|---------|
| 1 | **Pilot first** ($150 to validate assumptions) | Very High — prevents catastrophic waste | Low — 4 hours | Risk mitigation |
| 2 | **Cut Block 5** (merge into Block 2 at 2 disagreement levels) | High — eliminates redundant $279 | None — fewer conditions to run | $279 |
| 3 | **Increase Block 4 reps from 4 to 5-6** | High — paradox is the key claim | Low — just more runs | -$40 to -$80 |
| 4 | **Reduce best-of-N baselines** (1 rep, Sonnet instead of Opus) | Medium — saves $100+ | Low | $100-150 |
| 5 | **Cheaper third judge** (GPT-4o-mini or Flash) | Medium — saves $50-100 on judge calls | Low | $50-100 |
| 6 | **Run blocks in priority order** (1 then 4 then 3 then 2) | Medium — ensures core data collected even if time runs out | None | Time insurance |
| 7 | **Add local 8B model to Block 0** (free calibration point) | Low-Medium — extends capability range | Low — 30 min compute | $0 |
| 8 | **Increase Block 1 reps from 2 to 3** | Medium — tighter CIs on signature figure | Low | -$75 |

---

## Final Recommendation: **Pilot First, Then Run with Tweaks**

The current design is scientifically sound but strategically suboptimal in three ways:

1. **No pilot phase** — You're betting $1,147 on untested assumptions (judge reliability, disagreement level separation, paradox existence). A $150 pilot takes 4 hours and could save $1,000 in wasted compute.

2. **Budget spread too thin** — Block 5 ($279) is redundant with Block 2, and best-of-N baselines use expensive Opus when Sonnet would suffice. Reallocating ~$350 from low-ROI blocks to high-ROI blocks (more reps on Blocks 1 and 4) would dramatically improve statistical power for the paper's core claims.

3. **Reps misallocated** — Block 4 (paradox) is the most extraordinary claim and has only 4 reps. Block 2 (topology comparison) is the most conventional finding and has 2 reps x 2 blocks (effectively 4). The paradox needs more evidence, not the topology comparison.

**Concrete action plan:**
1. Run pilot ($150, 4 hours)
2. Analyze go/no-go criteria
3. Run core blocks with adjusted reps (Blocks 1, 4, 3 first; Block 2 second)
4. Cut Block 5 entirely; merge interesting conditions into Block 2
5. Use savings ($279 + ~$150 from best-of-N optimization) for more reps on core findings
6. Reserve $100-150 for targeted follow-up after preliminary analysis

**Expected outcome with tweaks:**
- Same 4 RQs answered
- ~40% more statistical power on RQ1 and RQ3
- Reduced risk of wasted budget via pilot validation
- Total cost: $1,000-1,100 (saving $50-150 while improving power)
