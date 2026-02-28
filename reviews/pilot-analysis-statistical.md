# Pilot Statistical Analysis — "When Agents Disagree"

**Author:** Marcus Rivera (PhD-2, Empirical Track)  
**Date:** February 28, 2026  
**Scope:** 280 pilot runs (Block 0: 240, Block 1: 22, Block 4: 22)  
**Task:** T-027b — Rigorous statistical analysis of pilot data before main batch results become paper evidence.

---

## Executive Summary

The pilot data show a **large and statistically robust paradox signal** in Block 4 (Cohen’s d = 1.62, well above our pre-registered threshold of 0.3), but **no detectable disagreement dividend signal** in Block 1 (R² = 0.001 for the manipulated variable). Judge reliability is adequate overall (mean κ = 0.93) but this is inflated by Block 0’s 240 single-agent runs where κ is trivially 1.0. In the multi-agent blocks where it matters, mean κ drops to **0.54–0.58**, with 24 runs below 0.4. Quality scores are **heavily discretized** by the Bradley-Terry scoring system, which limits the granularity of our statistical tests. Several findings demand attention before the full analysis.

**Key concerns:**
1. The d = 1.62 effect size is implausibly large — likely inflated by small sample + score discretization. Expect substantial shrinkage in the main batch.
2. Block 1 has zero signal on the manipulated disagreement level, but a confounded correlation with observed disagreement rate (R² = 0.46) that is entirely explained by agent count.
3. Judge agreement is genuinely problematic for multi-agent runs — most of the “high kappa” in the aggregate comes from trivially-agreeing single-agent calibration runs.

---

## 1. Effect Size Estimation

### 1.1 Block 4: Quorum Paradox (H3)

**Design:** n=2 vs n=3 agents, quorum topology, 11 runs per condition.

| Statistic | n=2 | n=3 |
|-----------|-----|-----|
| N | 11 | 11 |
| Mean quality | 0.3152 | 0.2301 |
| SD | 0.0575 | 0.0469 |
| Min | 0.2500 | 0.1643 |
| Max | 0.4000 | 0.2902 |

**Effect size:**
- Delta (n=2 − n=3) = **0.0851**
- Pooled SD = **0.0525**
- **Cohen’s d = 1.62** (exceeds threshold of d ≥ 0.3)
- Welch’s t = **3.80**, df = 19.2
- 95% CI for delta: **[0.038, 0.132]** (entirely above zero)

**By task type:**

| Task Type | n(n=2) | n(n=3) | M(n=2) | M(n=3) | d |
|-----------|--------|--------|--------|--------|---|
| Analytical | 3 | 6 | 0.328 | 0.214 | 2.15 |
| Creative | 8 | 5 | 0.310 | 0.249 | 1.14 |

**Paired analysis** (only 2 tasks have data at both n=2 and n=3):
- creative_05: diff = +0.083
- creative_07: diff = +0.036
- Mean paired diff = 0.059, paired d_z = 1.75

**⚠️ Critical caveat:** The d = 1.62 is almost certainly inflated. With n=11 per group, sampling error is enormous — the 95% CI for d itself is roughly [0.6, 2.6]. The true population effect is likely much smaller. Additionally, quality scores cluster at a handful of discrete values (0.25, 0.333, 0.400), so the “continuous” assumption underlying Cohen’s d is questionable. **Expect shrinkage toward d ≈ 0.5–0.8 in the main batch** based on typical pilot-to-main attenuation ratios.

**Verdict:** The paradox direction is confirmed and the effect clears our threshold even pessimistically. The main batch will have ample power (see §2).

### 1.2 Block 1: Disagreement Dividend (H1)

**Design:** Disagreement levels 1–5, 22 runs total (4–5 per level), mixed agent counts (2, 3, 5) and model compositions.

| Level | N | Mean Q | SD |
|-------|---|--------|-----|
| 1 | 5 | 0.357 | 0.130 |
| 2 | 4 | 0.319 | 0.137 |
| 3 | 5 | 0.337 | 0.074 |
| 4 | 4 | 0.305 | 0.078 |
| 5 | 4 | 0.379 | 0.103 |

**Linear regression (Q on disagreement_level):**
- Q = 0.334 + 0.002 × level
- R = 0.030, **R² = 0.0009** (essentially zero)
- t(β₁) = 0.13 — nowhere near significant

**Quadratic regression (Q = a + b·d + c·d²):**
- Q = 0.412 − 0.066d + 0.012d²
- **β₂ = +0.012 (POSITIVE)** — this is a U-shape, not the pre-registered inverted-U
- R²(quad) = 0.042 — negligible improvement
- F-test for quadratic term: F = 0.80 — not significant

**No inverted-U detected.** The manipulated disagreement level has essentially zero predictive power over quality. The pattern, if anything, is flat with noise.

**Surprising finding — actual disagreement rate:** When using the *observed* disagreement_rate (not the manipulated level), R² jumps to **0.459**, with a positive slope (β = +0.194). This means higher observed disagreement → higher quality. However, this is almost certainly a **confound with agent count**:

| Agent Count | N | Mean Quality |
|-------------|---|-------------|
| 2 | 7 | 0.410 |
| 3 | 9 | 0.363 |
| 5 | 6 | 0.225 |

The disagreement_rate varies chaotically within each level (range 0.00 to 1.00 at every level), and quality strongly decreases with agent count. Since agent count was not held constant across disagreement levels, the regression of quality on disagreement_rate is confounded. The observed correlation is likely spurious or mediating through an uncontrolled variable.

**Verdict:** No signal for the disagreement dividend in pilot data. But with n=4–5 per cell, statistical power was near zero for anything but a massive effect. The main batch, with proper factorial control of agent count × disagreement level, is essential.

---

## 2. Power Analysis

### 2.1 Block 4: Quorum Paradox

Using the pilot estimates (d = 1.62, pooled SD = 0.053):

| N per group | Power (α=0.05, one-sided) | Power (α=0.005, Holm-corrected) |
|-------------|---------------------------|----------------------------------|
| 5 | ≥0.80 | — |
| 9 | — | ≥0.80 |
| 20 | ≈1.00 | ≈1.00 |
| 50 | ≈1.00 | ≈1.00 |
| 100 | ≈1.00 | ≈1.00 |

Even if the true effect shrinks by 60% (to d ≈ 0.65), n=50 per group would give power > 0.95 at α=0.005.

**Minimum detectable effect (MDE) for 80% power:**

| N per group | MDE d (α=0.05) | MDE d (α=0.005) |
|-------------|----------------|------------------|
| 50 | 0.50 | 0.68 |
| 100 | 0.35 | 0.48 |
| 150 | 0.29 | 0.39 |
| 200 | 0.25 | 0.34 |

**Recommendation:** With n≥100 per group, we can detect effects as small as d = 0.35 at uncorrected α, or d = 0.48 at the Holm-corrected threshold. Given the pilot d = 1.62, even with 70% shrinkage the main batch will be well-powered. **The main batch is adequate for H3.**

### 2.2 Block 1: Disagreement Dividend

The pilot R² = 0.001 (linear) yields Cohen’s f² = 0.001. At this effect size, we’d need **N ≈ 6,841 runs** for 80% power — far beyond our main batch capacity.

However, this is likely because:
1. The pilot mixed agent counts across disagreement levels (confound)
2. Only 22 runs total (4–5 per cell) — essentially no power
3. The main batch presumably controls for agent count within each disagreement level

**If the true effect is f² = 0.05 (moderate):** N ≈ 100 would suffice.  
**If the true effect is f² = 0.02 (small-to-moderate):** N ≈ 315 would suffice.

**Recommendation:** The pilot provides essentially no information about H1’s effect size due to confounding and tiny N. The main batch should treat this as an exploratory hypothesis unless the controlled comparisons yield stronger signals. Do not rely on the pilot for Block 1 power estimates.

---

## 3. Distribution Analysis

### 3.1 Quality Score Distributions

**Block 0 (calibration):** All 240 runs yield Q = 0.500 exactly. This is a mathematical artifact of Bradley-Terry scoring with a single agent (ties with itself). **Block 0 provides zero variance** and cannot differentiate model quality.

**Block 1 (disagreement dividend):**
- n=22, mean=0.340, median=0.350, SD=0.100
- Range: [0.167, 0.500]
- Skewness: −0.16 (roughly symmetric)
- Kurtosis: −0.91 (slightly platykurtic)
- **No extreme violations of normality** at this sample size

**Block 4 (quorum paradox):**
- All: n=22, mean=0.273, median=0.250, SD=0.067
- n=2: mean=0.315, SD=0.058; n=3: mean=0.230, SD=0.047
- Skewness: 0.21 (near symmetric); Kurtosis: −0.35

### 3.2 Score Discretization Problem

**This is the most important distributional finding.** Quality scores in multi-agent runs cluster at a small number of discrete values:

| Score | Count (B4) |
|-------|-----------|
| 0.164 | 1 |
| 0.167 | 2 |
| 0.206 | 1 |
| 0.248 | 2 |
| 0.250 | 7 |
| 0.290 | 2 |
| 0.333 | 5 |
| 0.400 | 2 |

Across Block 1 + Block 4 combined (44 runs), there are only **19 unique quality score values**. This discretization is inherent to the Bradley-Terry pairwise comparison method with 3 judges and a small number of candidates.

**Implications:**
- Standard parametric tests assume continuous distributions. The discretized scores violate this.
- For the full analysis, **non-parametric tests (Wilcoxon signed-rank, Mann-Whitney U) should be the primary tests**, with parametric tests as sensitivity checks.
- Consider ordinal logistic regression as an alternative to OLS for the quality outcome.

### 3.3 Heteroscedasticity

- Block 4: Var(n=2)/Var(n=3) = **1.50** — within acceptable bounds (0.5–2.0)
- Welch’s t-test (which we’re using) is already robust to moderate heteroscedasticity

### 3.4 Outliers

No outliers detected by the 1.5×IQR rule in any condition. The bounded nature of BT scores (approximately [0.16, 0.50] for multi-agent runs) naturally constrains the range.

---

## 4. Judge Agreement Deep Dive

### 4.1 Overall Picture

| Statistic | Value |
|-----------|-------|
| Overall mean κ | 0.931 |
| Overall median κ | 1.000 |
| SD | 0.209 |
| Min | 0.026 |
| Max | 1.000 |

**But this is misleading.** The distribution is bimodal:

| Kappa Range | Count | Fraction |
|-------------|-------|----------|
| <0.20 | 5 | 1.8% |
| 0.20–0.40 | 19 | 6.7% |
| 0.40–0.60 | 3 | 1.1% |
| 0.60–0.80 | 2 | 0.7% |
| 0.80–1.00 | 0 | 0.0% |
| 1.00 (exact) | 255 | 89.8% |

**255 of 284 runs have κ = 1.0 — all from Block 0** (single agent → all comparisons are ties → perfect agreement by definition). The remaining 44 multi-agent runs show much lower agreement.

### 4.2 Kappa by Block (the real story)

| Block | Mean κ | SD | Min | N |
|-------|--------|-----|-----|---|
| Block 0 (calibration) | **1.000** | 0.000 | 1.000 | 240 |
| Block 1 (disagreement) | **0.538** | 0.362 | 0.026 | 22 |
| Block 4 (paradox) | **0.576** | 0.328 | 0.333 | 22 |

**The aggregate κ = 0.93 is almost entirely driven by Block 0.** The multi-agent blocks where judge agreement actually matters have mean κ ≈ 0.55, with substantial runs below 0.40.

### 4.3 Kappa by Agent Count

| Agent Count | Mean κ | SD | Min | N |
|-------------|--------|-----|-----|---|
| 1 | 1.000 | 0.000 | 1.000 | 240 |
| 2 | 0.667 | 0.343 | 0.333 | 18 |
| 3 | 0.481 | 0.303 | 0.026 | 20 |
| 5 | 0.479 | 0.427 | 0.056 | 6 |

**Clear pattern:** More agents → lower judge agreement. This is mechanically expected — with more candidates to compare, judges disagree more on rankings. But it also means our most complex conditions (the ones testing our most interesting hypotheses) have the least reliable measurements.

### 4.4 Low-Kappa Runs

24 runs have κ < 0.40 (all from Blocks 1 and 4):
- **5 runs with κ < 0.20**: run_b3fd86d00449 (κ=0.026), run_6a1b7c0db4df (κ=0.056), run_4d73854046c1 (κ=0.076), run_8a4663fc197b (κ=0.167), run_a983d0428138 (κ=0.173)
- All 5 extreme cases are from Block 1 (disagreement dividend)
- Most low-κ runs involve 3+ agents, where more pairwise comparisons create more opportunities for disagreement

**No clear pattern by task type:** Analytical κ = 0.927, Creative κ = 0.936 — essentially identical (but again inflated by Block 0).

### 4.5 Judge Systematic Bias

Mean BT score (for selected output) by judge:

| Judge | Mean | SD | Median |
|-------|------|-----|--------|
| claude-sonnet-4-6 | 0.471 | 0.100 | 0.500 |
| gemini-3.1-pro-preview | 0.470 | 0.079 | 0.500 |
| gpt-4o | 0.467 | 0.083 | 0.500 |

**No systematic bias detected.** All three judges have nearly identical mean and median scores for the consensus output.

**Pairwise agreement rates:**

| Judge Pair | Agreement Rate |
|------------|---------------|
| gpt-4o vs gemini-3.1-pro-preview | **0.952** |
| claude-sonnet-4-6 vs gemini-3.1-pro-preview | 0.819 |
| claude-sonnet-4-6 vs gpt-4o | 0.774 |

**Notable finding:** GPT-4o and Gemini agree 95% of the time, but Claude-Sonnet agrees with either at only 77–82%. When judges disagree, it’s almost always claude-sonnet-4-6 that breaks from the other two. This suggests claude-sonnet-4-6 applies different quality criteria — it might be more discriminating or using different rubric interpretation. Worth investigating in the main batch whether this represents genuine quality differentiation or systematic bias.

---

## 5. Confound Check: Model Composition in Block 4

### 5.1 The Concern

In Block 4, model composition varies with agent count. At n=2, we see pairs like (claude-opus, claude-opus), (gpt-5.2, gpt-5.2), (gpt-5.2, gemini-flash). At n=3, we see trios like (claude-opus×3), (gpt-5.2×3), (gpt-5.2, gemini-flash×2). If model composition independently affects quality, the paradox could be an artifact.

### 5.2 Homogeneous Configs (Same Model)

| Condition | N | Mean Q | SD |
|-----------|---|--------|-----|
| Homogeneous n=2 | 6 | 0.317 | 0.058 |
| Homogeneous n=3 | 8 | 0.228 | 0.045 |
| Delta | — | **+0.088** | — |

**Paradox holds: YES** — For homogeneous teams (same model at both n=2 and n=3), adding a third copy of the same model still decreases quality.

### 5.3 Heterogeneous Configs (Mixed Models)

| Condition | N | Mean Q | SD |
|-----------|---|--------|-----|
| Heterogeneous n=2 | 5 | 0.313 | 0.064 |
| Heterogeneous n=3 | 3 | 0.235 | 0.063 |
| Delta | — | **+0.078** | — |

**Paradox holds: YES** — For heterogeneous teams, the effect direction is the same.

### 5.4 By Specific Model Configuration

| Configuration | n=2 Mean Q | n=3 Mean Q |
|--------------|-----------|-----------|
| claude-opus only | 0.292 (n=4) | 0.225 (n=5) |
| gpt-5.2 only | 0.367 (n=2) | 0.234 (n=3) |
| gpt-5.2 + gemini-flash | 0.328 (n=3) | 0.269 (n=2) |
| claude-opus + claude-haiku | 0.292 (n=2) | 0.167 (n=1) |

Every model configuration where we have both n=2 and n=3 data shows the paradox direction (n=2 > n=3).

### 5.5 Verdict

**The paradox is NOT explained by model composition confounding.** It persists across:
- Homogeneous configs (Δ = 0.088)
- Heterogeneous configs (Δ = 0.078)
- Every specific model composition tested

The effect sizes are similar across composition types, suggesting the paradox is driven by agent count itself, not by which models are added. However, sample sizes are tiny (1–5 per configuration), so the main batch needs to test this with proper factorial control.

---

## 6. Multiple Comparisons

### 6.1 Primary Comparison Family

Per the ANALYSIS_PLAN.md, m = **10 primary comparisons**:

| # | Test | Description |
|---|------|-------------|
| 1–2 | H1 (β₂ < 0) | Inverted-U, analytical + creative |
| 3–4 | H2 (γ₁ > 0) | Quality threshold attainment, analytical + creative |
| 5–6 | H3 (Q₂ > Q₃) | Paradox, analytical + creative |
| 7–8 | H3 control | Homogeneous-strong, analytical + creative |
| 9–10 | H4 (interaction) | Topology × consensus, analytical + creative |

### 6.2 Holm-Bonferroni Thresholds

| Step | Adjusted α |
|------|-----------|
| 1 (most stringent) | 0.005 |
| 2 | 0.0056 |
| 3 | 0.0063 |
| … | … |
| 10 (least stringent) | 0.050 |

### 6.3 Will Pilot Effects Survive Correction?

**H3 (paradox): LIKELY YES.** With d = 1.62 in pilot and planned N ≫ 50 per group, even after shrinkage to d ≈ 0.6–0.8, the p-value should be well below 0.005. At n=100/group with d=0.6, power at α=0.005 exceeds 0.95.

**H1 (disagreement dividend): UNLIKELY.** The pilot shows R² = 0.001 and β₂ in the wrong direction. Unless the main batch’s controlled design reveals a hidden signal, this will not survive correction. We should prepare to report this as a null result.

**H2, H4: UNKNOWN.** These hypotheses aren’t directly testable in the pilot data (no topology variation in Block 4, no threshold analysis set up). Power depends on the main batch’s effect sizes.

### 6.4 Recommendation

Given that H1 is currently showing no signal and H3 is showing a massive signal, there’s a risk of a “one-hit paper” where only the paradox result is significant. To mitigate:
1. Ensure the main batch has sufficient runs for H1 with agent count properly controlled
2. Consider pre-registering a revision that reduces m (e.g., drop H1 from confirmatory if the main batch’s initial results look as flat as the pilot)
3. Focus exploratory analyses on understanding *why* the paradox occurs (mechanism analysis from debate traces)

---

## 7. Additional Findings and Concerns

### 7.1 Block 0 Is Uninformative

All Block 0 runs produce Q = 0.500 regardless of model or task. This is mathematically necessary for BT scoring with a single candidate (the only comparison is with itself → tie → BT score = 0.5). Block 0 serves its purpose as a calibration check (confirming the scoring system works) but provides:
- No information about model quality differences
- No variance to analyze
- Inflated aggregate kappa statistics

### 7.2 Quality Scores Are Compressed

All multi-agent quality scores fall in [0.164, 0.500]. The theoretical maximum of 1.0 is never approached. This is because BT scores from pairwise comparisons with only 3–4 candidates and 3 judges are inherently bounded near 0.25–0.50. This compression affects:
- **Effect size interpretation**: A delta of 0.085 on a scale effectively spanning [0.16, 0.50] (range = 0.34) represents about 25% of the effective range — much larger than it appears on the [0,1] scale.
- **Statistical tests**: The bounded, discrete nature of scores makes non-parametric tests more appropriate.

### 7.3 Agent Count Is Confounded with Quality in Block 1

In Block 1, agent count (2, 3, 5) varies across disagreement levels and is strongly negatively correlated with quality:
- n=2: mean Q = 0.410
- n=3: mean Q = 0.363
- n=5: mean Q = 0.225

This confound makes the Block 1 pilot data uninterpretable for the disagreement dividend hypothesis. The main batch MUST hold agent count constant within disagreement-level comparisons, or include agent count as a covariate.

---

## 8. Recommendations for Full Analysis

### 8.1 Statistical Methods

1. **Primary tests should be non-parametric** (Mann-Whitney U for Block 4, Spearman/Kendall for Block 1) due to score discretization. Report parametric tests as sensitivity checks.
2. **Use mixed-effects models** with task as a random effect to account for task-level clustering.
3. **For Block 1**: Use agent count as a covariate or stratify by agent count before testing the disagreement effect.
4. **Report both raw and adjusted p-values** for all 10 primary comparisons.
5. **Report Cohen’s d with 95% CI** — the point estimate alone is misleading at small N.

### 8.2 Judge Reliability

6. **Report separate kappa statistics for multi-agent runs** (exclude Block 0 from aggregate kappa).
7. **Flag and sensitivity-check runs with κ < 0.40.** Currently 24/44 multi-agent runs (55%) fall below this threshold — this is a genuine concern.
8. **Investigate claude-sonnet-4-6’s divergence** from the gpt-4o/gemini pair. If one judge systematically differs, consider a 2-of-3 majority rule or weighted BT scoring.

### 8.3 Effect Size Expectations

9. **Expect the paradox d to shrink** from 1.62 to approximately 0.5–0.8 in the main batch. Plan interpretations accordingly.
10. **For Block 1**, do not assume any detectable effect. The pilot provides no basis for an effect size estimate of the disagreement dividend.

### 8.4 Design Concerns

11. **Block 4 paired structure is sparse**: Only 2 of 16 task_ids have data at both n=2 and n=3 in the pilot. The main batch must ensure every task appears at both agent counts for within-task paired analysis (per H3 specification).
12. **Quality score compression**: Consider supplementing BT scores with direct rubric scores (1–5 scale per criterion) to get a richer quality distribution.

---

## Summary Table

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Block 4 Cohen’s d | **1.622** | ≥ 0.3 | ✅ PASS |
| Block 4 delta (n=2 − n=3) | +0.085 | — | Positive direction |
| Block 4 95% CI | [0.038, 0.132] | Excludes 0 | ✅ |
| Block 4 Welch’s t | 3.80 | — | — |
| Block 1 R² (linear on level) | **0.001** | — | ❌ No signal |
| Block 1 β₂ (quadratic) | **+0.012** | < 0 | ❌ Wrong direction |
| Block 1 R² (on actual rate) | 0.459 | — | ⚠️ Confounded |
| Mean κ (all runs) | 0.931 | > 0.4 | ✅ (inflated) |
| Mean κ (multi-agent only) | **0.557** | > 0.4 | ⚠️ Marginal |
| Min κ | 0.026 | — | ❌ Problematic |
| Runs with κ < 0.40 | 24/44 (55%) | — | ❌ |
| Paradox holds (homogeneous) | YES (Δ=0.088) | — | ✅ |
| Paradox holds (heterogeneous) | YES (Δ=0.078) | — | ✅ |
| N needed/group (80% pwr, α=.05) | 5 | — | Main batch is ample |
| N needed/group (80% pwr, α=.005) | 9 | — | Main batch is ample |

---

## Bottom Line

**The paradox (H3) is the strongest result in this experiment.** It’s robust across model compositions, task types, and survives every check. The main batch will easily confirm or adjust it.

**The disagreement dividend (H1) is in trouble.** Zero signal in pilot, wrong direction on the quadratic term, and the only apparent signal (R² = 0.46 on observed disagreement rate) is a confound with agent count. We need to be intellectually honest: if the main batch also shows no inverted-U, we report the null. A null result for H1 combined with a strong H3 is actually a compelling and surprising story — “more agents hurt, and disagreement doesn’t help” — that’s potentially more interesting than what we pre-registered.

**Judge reliability needs attention.** The aggregate κ = 0.93 is a vanity statistic. The real number — κ = 0.56 for multi-agent runs, with 55% below 0.40 — suggests our evaluation methodology has meaningful noise. This doesn’t invalidate the Block 4 result (which has a large enough effect to shine through noise), but it could bury any smaller effects in Blocks 1 and 2.
