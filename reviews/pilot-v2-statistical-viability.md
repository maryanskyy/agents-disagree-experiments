# Pilot V2 Statistical Viability Analysis

**Author:** Marcus Rivera (PhD-2)  
**Date:** 2026-02-28  
**Data:** 28 multi-agent deepseek-panel runs (blocks 1 & 4), 189 pairwise comparisons

---

## Executive Summary

**Bottom line: We can proceed, but we need ~300 runs per condition (not 200) to reliably detect moderate effects (10pp), and we should either replace GPT-4o or switch to rubric-scoring to boost discrimination. The current three-judge panel has a usable overall discrimination rate of 39.3% at the run level, but this masks a strong judge: Sonnet discriminates 67.9% of the time. The panel is being dragged down by GPT-4o (25.0%).**

---

## 1. Inter-Judge Agreement

### 1.1 Cohen's Kappa

| Judge Pair | κ (all comparisons) | κ (non-tie only) | n (non-tie) |
|---|---|---|---|
| Sonnet vs DeepSeek | 0.434 | 0.078 | 110 |
| Sonnet vs GPT-4o | 0.222 | 0.041 | 107 |
| DeepSeek vs GPT-4o | 0.262 | -0.059 | 78 |

**Interpretation:**

- The "all comparisons" kappa values (0.22–0.43) are inflated by massive agreement on ties. When two judges both say "tie" on 60–85% of comparisons, their raw agreement rate looks decent, but it's almost entirely driven by shared inability to discriminate.
- When we filter to comparisons where at least one judge discriminated, kappa drops to near-zero or negative. This means judges are essentially **independent** in their discrimination decisions — when one judge sees a difference, the others usually don't.
- The negative kappa for DeepSeek vs GPT-4o (−0.059) means they're slightly worse than chance agreement when both actually discriminate. But this is based on very few GPT-4o discrimination events.

### 1.2 Conditional Agreement: When a Judge Discriminates, Do Others Agree?

**When Sonnet discriminates (n=105 pairwise comparisons):**

| Other Judge | Agrees | Ties | Disagrees |
|---|---|---|---|
| DeepSeek | 46 (43.8%) | 41 (39.0%) | 18 (17.1%) |
| GPT-4o | 24 (22.9%) | 79 (75.2%) | 2 (1.9%) |

**When DeepSeek discriminates (n=69):**

| Other Judge | Agrees | Ties | Disagrees |
|---|---|---|---|
| Sonnet | 46 (66.7%) | 5 (7.2%) | 18 (26.1%) |
| GPT-4o | 19 (27.5%) | 50 (72.5%) | 0 (0.0%) |

**When GPT-4o discriminates (n=28):**

| Other Judge | Agrees | Ties | Disagrees |
|---|---|---|---|
| Sonnet | 24 (85.7%) | 2 (7.1%) | 2 (7.1%) |
| DeepSeek | 19 (67.9%) | 9 (32.1%) | 0 (0.0%) |

**Key patterns:**

1. **GPT-4o is a timid follower.** When GPT-4o *does* discriminate (only 28 times out of 189), it almost always agrees with the other judges (85.7% with Sonnet, 67.9% with DeepSeek) and literally **never disagrees with DeepSeek**. GPT-4o adds zero independent signal — it only votes when the answer is already obvious.

2. **Sonnet is the driver.** Sonnet discriminates most often (105/189 = 55.6%), and when it does, DeepSeek agrees 43.8% of the time with a meaningful 17.1% disagreement rate. This is the only pair showing genuine independent judgment.

3. **DeepSeek is a useful second opinion.** When DeepSeek discriminates, Sonnet usually has an opinion too (only 7.2% of the time does Sonnet tie when DeepSeek discriminates). The 26.1% disagreement rate between them is healthy — they are not clones.

4. **The asymmetry is revealing.** When Sonnet discriminates, GPT-4o ties 75.2% of the time. GPT-4o is essentially blind to quality differences that Sonnet and DeepSeek can see.

---

## 2. Effective Sample Size

### 2.1 Overall Discrimination Rate

| Metric | Value |
|---|---|
| Runs with win_rate ≠ 0.50 | 11/28 (39.3%) |
| Per-judge: Sonnet | 19/28 (67.9%) |
| Per-judge: DeepSeek | 16/28 (57.1%) |
| Per-judge: GPT-4o | 7/28 (25.0%) |

### 2.2 Effective Observations for Main Batch

| Planned N/condition | n_eff (at 39.3%) | Sufficient for... |
|---|---|---|
| 100 | 39 | Barely: bootstrap CI only |
| 150 | 59 | Marginal: Wilcoxon works but low power |
| 200 | 79 | Adequate for standard tests, borderline power |
| 250 | 98 | Good |
| 300 | 118 | Comfortable for all tests |

**For standard nonparametric tests (Wilcoxon rank-sum, Mann-Whitney U):** The normal approximation requires n ≥ 30 per group minimum. With n_eff = 79 (200 runs), we meet this threshold, but power will be low for small effects. For bootstrap CIs with 1000 resamples, 50+ effective observations is recommended — we just clear that bar at N=150.

**Critical caveat:** The 39.3% rate is based on *majority vote* across the three-judge panel. The per-judge rates are higher (Sonnet: 67.9%, DeepSeek: 57.1%). We can extract more signal by analyzing per-judge win rates separately and aggregating statistically, rather than collapsing to a single majority-vote outcome per run.

---

## 3. Power Simulation

### 3.1 Three-Judge Panel (Current Setup)

Simulation parameters:
- Discrimination rates: Sonnet=67.9%, DeepSeek=57.1%, GPT-4o=25.0%
- Each judge independently decides to discriminate, then votes
- Majority vote determines run outcome
- Mann-Whitney U test, α=0.05, two-sided
- 3,000 simulations per cell

| N/condition | Δ=0.10 | Δ=0.15 | Δ=0.20 |
|---|---|---|---|
| 50 | 0.200 | 0.384 | 0.609 |
| 100 | 0.361 | 0.677 | 0.889 |
| **150** | **0.499** | **0.839** | **0.977** |
| **200** | **0.620** | **0.930** | **0.997** |
| 250 | 0.716 | 0.972 | 0.999 |
| **300** | **0.798** | **0.987** | **1.000** |
| 400 | 0.893 | 0.998 | 1.000 |
| 500 | 0.941 | 1.000 | 1.000 |

### 3.2 Sonnet-Only Judge

| N/condition | Δ=0.10 | Δ=0.15 | Δ=0.20 |
|---|---|---|---|
| 50 | 0.132 | 0.230 | 0.376 |
| 100 | 0.213 | 0.417 | 0.642 |
| 200 | 0.382 | 0.699 | 0.921 |
| 300 | 0.517 | 0.865 | 0.986 |
| 500 | 0.734 | 0.978 | 1.000 |

### 3.3 Interpretation

**For Δ=0.10 (small but meaningful effect):**
- Three-judge panel needs **N=300/condition** for 80% power
- Sonnet-only needs N>500/condition (unacceptable)
- The panel helps because majority vote across three partially-independent judges reduces noise

**For Δ=0.15 (moderate effect — our realistic target):**
- Three-judge panel needs **N=150/condition** for ~84% power
- At N=200/condition, power is 93% — very good
- Sonnet-only needs N=300/condition for 87% power

**For Δ=0.20 (large effect):**
- Three-judge panel needs only **N=100/condition** for 89% power
- Easily detectable at planned N=200

**Key insight:** The three-judge panel actually outperforms Sonnet-only despite GPT-4o being deadweight. This is because DeepSeek provides genuine independent signal, and majority vote aggregation reduces variance. GPT-4o's ties are neutral (they default to 0.5), not harmful.

---

## 4. Rubric-Scoring Alternative

### 4.1 Single Judge Tie Probability (Uniform Scores 1–5)

| Criteria | Tie Probability | Discrimination Rate |
|---|---|---|
| 1 | 0.199 | 0.801 |
| 2 | 0.136 | 0.864 |
| **3** | **0.112** | **0.888** |
| 4 | 0.098 | 0.902 |
| 5 | 0.087 | 0.913 |
| 6 | 0.081 | 0.920 |

### 4.2 Score Range Effect (3 Criteria)

| Range | Tie Probability | Discrimination Rate |
|---|---|---|
| 1–3 | 0.194 | 0.807 |
| **1–5** | **0.112** | **0.888** |
| 1–7 | 0.080 | 0.920 |
| 1–10 | 0.055 | 0.945 |

### 4.3 Effect of Score Correlation Between Candidates

When candidates are similar (correlated quality), tie rates increase:

| Correlation | Tie Probability | Discrimination Rate |
|---|---|---|
| 0.0 (independent) | 0.200 | 0.800 |
| 0.3 | 0.259 | 0.741 |
| **0.5** | **0.343** | **0.657** |
| 0.7 | 0.653 | 0.347 |
| 0.9 | 1.000 | 0.000 |

### 4.4 Three-Judge Rubric Panel

| Configuration | Panel Tie Prob | Panel Disc Rate |
|---|---|---|
| 3 criteria, 1–5 scale | 0.134 | **0.866** |
| 4 criteria, 1–5 scale | 0.120 | **0.880** |

### 4.5 Interpretation

**Theoretical best case (uncorrelated scores):** Rubric scoring with 3 criteria on a 1–5 scale gives 88.8% discrimination per judge, and 86.6% for a three-judge panel. This is dramatically better than the current pairwise approach (39.3% panel discrimination).

**Realistic case (moderate correlation ~0.5):** Even with r=0.5 correlation between candidate scores, single-judge discrimination is still 65.7% — comparable to Sonnet's current 67.9% pairwise discrimination, and much better than DeepSeek (57.1%) and GPT-4o (25.0%).

**However, there are important caveats:**
1. The correlation between candidates in our experiment may be high (0.5–0.7), since multi-agent outputs on the same task may be genuinely similar in quality.
2. Rubric scoring introduces a different measurement construct — it measures absolute quality per criterion, not holistic preference.
3. LLM judges may show systematic biases in rubric scores (e.g., central tendency, always scoring 3–4) that reduce effective range below the theoretical 1–5.
4. Rubric design requires careful criterion selection; poorly chosen criteria may not capture the dimensions we care about.

---

## 5. Vote Pattern Analysis

The most common vote patterns across the 189 pairwise comparisons:

| Sonnet | DeepSeek | GPT-4o | Count | % |
|---|---|---|---|---|
| tie | tie | tie | 77 | 40.7% |
| left | left | tie | 23 | 12.2% |
| left | tie | tie | 20 | 10.6% |
| left | left | left | 16 | 8.5% |
| right | tie | tie | 14 | 7.4% |
| left | right | tie | 9 | 4.8% |
| Other patterns | | | 30 | 15.8% |

**Key observations:**
- 40.7% of comparisons are unanimous ties — no signal at all.
- When there IS discrimination, Sonnet is almost always involved (only 5 comparisons have discrimination without Sonnet participating).
- Agreement patterns (left-left-tie, left-left-left) are 3× more common than disagreement patterns (left-right-tie), suggesting when judges discriminate, they tend to agree on direction.

---

## 6. Recommendation

### Option A: Proceed with Current Panel (Viable Fallback)

**Expected discrimination rate:** 39.3% (run-level, majority vote)

**Required sample size for 80% power:**

| Target Δ | N per condition |
|---|---|
| 0.10 | 300 |
| 0.15 | 150 |
| 0.20 | 100 |

**Modifications needed:**
1. **Increase planned runs to 300 per condition** (up from 200) to ensure 80% power for Δ=0.10.
2. **Analyze per-judge win rates separately** in addition to majority vote. Sonnet and DeepSeek provide independent signal; reporting both gives more nuanced results.
3. **Keep GPT-4o** — it doesn't hurt (ties are neutral), it provides rare but high-confidence votes, and removing it saves cost but doesn't improve power significantly.

**Cost estimate:** 300 runs × ~4 conditions × 3 judges = 3,600 judge calls.

### Option B: Replace GPT-4o with a Better Discriminator

**Expected improvement:** If we replace GPT-4o with a judge that matches Sonnet's discrimination rate (67.9%), the panel discrimination rate would rise from 39.3% to roughly 50–55%. This would reduce required N by ~20%.

**Risk:** We don't know which judge will discriminate well without piloting. Another pilot round costs time.

### Option C: Switch to Rubric Scoring

**Expected discrimination rate:** ~86–88% (three-judge panel, theoretical)

**Realistic discrimination rate (with ~0.5 score correlation):** ~65–70%

**Required sample size for 80% power:** Roughly 150–180 per condition for Δ=0.10 (compared to 300 with current approach). This nearly halves required runs.

**Tradeoffs:**
- (+) Much higher discrimination rate
- (+) Richer data (criterion-level scores enable deeper analysis)
- (+) Natural fit for ablation analysis (which criteria drive topology differences?)
- (−) Requires designing a rubric (1–2 days of work)
- (−) Different measurement construct; not directly comparable to prior pairwise pilot
- (−) Risk of central tendency bias in LLM scoring

### Option D: Hybrid — Rubric Scoring with Pairwise Decision (★ Recommended)

**Approach:** Each judge scores both candidates on 3–4 criteria (1–5 scale), then the higher total wins. This gives us rubric data AND a pairwise outcome.

**Expected discrimination rate:** Same as Option C (~86–88% theoretical, ~65–70% realistic)

**This is my actual recommendation.** It combines the benefits of rubric scoring (high discrimination, rich data) with pairwise outcomes (compatible with current analysis pipeline). The rubric data is a bonus for analysis, not a requirement for the main test.

---

## 7. Final Verdict

| | Current Panel | Replace GPT-4o | Rubric Scoring | Hybrid (★) |
|---|---|---|---|---|
| Panel disc. rate | 39.3% | ~50–55% | ~65–88% | ~65–88% |
| N/cond for 80% power (Δ=0.10) | 300 | ~240 | ~150–180 | ~150–180 |
| N/cond for 80% power (Δ=0.15) | 150 | ~120 | ~80–100 | ~80–100 |
| Implementation cost | None | Pilot needed | Rubric design | Rubric design |
| Data richness | Low | Low | High | Highest |

**Recommendation: Option D (Hybrid Rubric + Pairwise)**

If rubric design is too slow, fall back to **Option A with N=300/condition**. The current panel works — it's just inefficient. With 300 runs per condition, we have 80% power for a 10pp effect and 99% power for a 15pp effect. That's publishable.

---

*Analysis generated from 28 deepseek-panel pilot runs (20 block1, 8 block4) with 189 pairwise comparisons. All simulations used 3,000 iterations with fixed random seed (42) for reproducibility.*
