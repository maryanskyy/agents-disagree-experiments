# Pilot Data Theoretical Analysis — Elena Chen (PhD-1)
## Date: 2026-02-28
## Status: CRITICAL ISSUES IDENTIFIED

---

## Executive Summary

After reading the briefing materials, analysis plan, research questions, pilot report, and 30+ individual run JSONs across all three blocks, I have identified one **fatal methodological flaw** and several **serious concerns** that must be resolved before the main batch results can be interpreted.

**The fatal flaw:** Bradley-Terry quality scores are not comparable across different agent counts. The apparent Quorum Paradox signal (Δ = -0.085) is approximately **97% attributable to a mechanical BT normalization artifact**, not to a genuine quality decline. After normalization, the residual effect is ~0.025, which is statistically indistinguishable from zero given the sample sizes.

This does not mean the paradox doesn't exist — it means our current measurement instrument **cannot detect it** because the BT scoring conflates pool-size effects with quality effects.

---

## 1. BT Score Interpretation (Block 0 Analysis)

### Observation
All 240 Block 0 calibration runs show `quality_score = 0.500` regardless of model (Claude Opus, GPT-5.2, Gemini Pro, Gemini Flash, Claude Haiku) or task type (analytical, creative). Every pairwise comparison is a "tie."

### Is This Correct Behavior?

**Yes, this is mathematically necessary.** For single-agent runs (n=1), the candidate pool contains exactly two items:
- `final_consensus` (the selected output)
- `agent_0` (the sole agent's output)

Since the consensus IS agent_0's output verbatim, judges compare identical texts. All pairwise judgments must be ties. The BT model assigns each candidate probability 1/K = 1/2 = 0.500.

### What This Reveals About BT Scores

BT scores are **strictly relative within a comparison set**. A quality score of 0.500 for n=1 does not mean "medium quality" — it means "indistinguishable from itself." The score carries zero information about absolute quality.

**Formally:** Let K = n + 1 be the number of candidates (n agents + 1 consensus output). The BT model assigns probabilities p_i = λ_i / Σ_k λ_k, with constraint Σ p_i = 1. The "quality score" is p_consensus. Under equal candidates:

| Agent count (n) | Candidates (K = n+1) | Uniform baseline (1/K) |
|---|---|---|
| 1 | 2 | 0.500 |
| 2 | 3 | 0.333 |
| 3 | 4 | 0.250 |
| 5 | 6 | 0.167 |

**Verified in data:** I confirmed K = n+1 in every run by counting the keys in `evaluation.judge_panel.bt_scores`. Block 0 (n=1): 2 candidates. Block 4 n=2: 3 candidates. Block 4 n=3: 4 candidates. Block 1 n=5: 6 candidates. The floor values match: n=5 runs show quality = 0.16667 = 1/6 exactly when all judges tie.

### Critical Implication

**Raw quality scores from different agent counts inhabit different scales.** Comparing quality(n=1) = 0.500, quality(n=2) = 0.315, quality(n=3) = 0.230 as if they represent points on a common quality axis is a category error. This is like comparing batting averages computed from different numbers of at-bats without accounting for the denominator.

---

## 2. Paradox Signal Validity — THE CRITICAL FINDING

### The Claimed Signal

The pilot report claims a Quorum Paradox:
- Q(n=2) = 0.313, SD = 0.058 (11 runs)
- Q(n=3) = 0.228, SD = 0.047 (11 runs)
- Δ = -0.085 (n=3 worse than n=2)

### The BT Normalization Problem

The uniform BT baselines for these pool sizes are:
- n=2: 1/3 = 0.333
- n=3: 1/4 = 0.250
- Baseline Δ = -0.083

**The observed delta (-0.085) is almost identical to the mechanical baseline delta (-0.083).**

### Formal Decomposition

The observed quality difference decomposes additively:

```
Δ_observed = Q(n=3) - Q(n=2) = -0.085

Δ_mechanical = 1/(n₃+1) - 1/(n₂+1) = 1/4 - 1/3 = -0.083

Δ_true = Δ_observed - Δ_mechanical = -0.085 - (-0.083) = -0.002
```

**Approximately 97.6% of the observed paradox signal is a BT normalization artifact.** The residual true quality effect is Δ_true ≈ -0.002.

### Normalized Analysis

To compare across pool sizes, I compute the normalized uplift factor: q_norm = quality_score × (n+1), representing how many "candidates' worth" of probability the consensus captures. Under equal quality, q_norm = 1.0.

**Block 4 (paradox block) raw quality scores:**

| n=2 runs | n=3 runs |
|---|---|
| 0.333, 0.250, 0.333, 0.250, 0.400 | 0.250, 0.248, 0.167, 0.167, 0.290 |
| 0.250, 0.250, 0.400, 0.333, 0.333, 0.333 | 0.206, 0.164, 0.248, 0.290, 0.250, 0.250 |

**Normalized uplift (× (n+1)):**

| n=2 (×3) | n=3 (×4) |
|---|---|
| 1.000, 0.750, 1.000, 0.750, 1.200 | 1.000, 0.993, 0.667, 0.667, 1.161 |
| 0.750, 0.750, 1.200, 1.000, 1.000, 1.000 | 0.826, 0.657, 0.993, 1.161, 1.000, 1.000 |

- Mean q_norm(n=2) = 0.945
- Mean q_norm(n=3) = 0.920
- Difference = 0.025

**Statistical test on normalized scores:**
- SD_norm(n=2) = 0.058 × 3 = 0.174
- SD_norm(n=3) = 0.047 × 4 = 0.188
- Pooled SE = √(0.174²/11 + 0.188²/11) = √(0.00276 + 0.00321) = 0.077
- t = 0.025 / 0.077 = 0.32
- p ≈ 0.75 (two-sided), df ≈ 20

**The paradox is statistically undetectable after normalization.** The effect size (Cohen's d) is approximately 0.025/0.181 ≈ 0.14, well below the pre-registered threshold of d ≥ 0.3 for practical significance.

### What This Means

The current BT-based quality metric **cannot test H3 (Quorum Paradox)** as specified in the analysis plan. The pre-registered test (paired t-test on Q(n=3) < Q(n=2)) is testing primarily the mechanical normalization effect, not genuine quality differences. Even if the main batch confirms p < 0.05 on raw scores, it would be a **spurious finding driven by measurement scale, not by a real phenomenon.**

### Possible Fixes

1. **Normalize BT scores by uniform baseline** before comparison: q_norm = quality × (n+1). This removes the mechanical effect but increases variance.

2. **Use consensus win-rate against individual agents** as the quality metric. This is scale-invariant: "What fraction of pairwise comparisons does the consensus win?" Available from the `pairwise_records` data.

3. **Use absolute quality assessment** (e.g., Likert-scale judge ratings) rather than comparative BT. This requires re-evaluating with a different judging protocol.

4. **Compare consensus quality to the best individual agent** rather than to a uniform baseline. Metric: quality_uplift = p_consensus - max(p_agent_i). This directly tests whether the consensus improves on the best individual.

I recommend **option 2** (win-rate) as the primary metric and **option 4** (uplift over best individual) as a secondary metric, both of which are computable from existing pairwise_records data without new API calls.

---

## 3. Disagreement Measurement Validity

### Disagreement Level Manipulation

The pilot report claims disagreement levels are "separable" (spread = 0.405). However, examination of the per-level disagreement rate means reveals a **non-monotonic pattern**:

| Disagreement Level | Mean Disagreement Rate | Expected Direction |
|---|---|---|
| 1 (lowest) | 0.300 | Lowest ✓ |
| 2 | 0.498 | ↑ ✓ |
| 3 | 0.604 | ↑ ✓ |
| 4 | **0.199** | ↑ **✗ (DROPPED BELOW LEVEL 1)** |
| 5 (highest) | 0.478 | ↑ ✗ |

Levels 4 and 5 show LOWER disagreement than levels 2 and 3. The manipulation is **not monotonic**. "Separable" means only that the values differ from each other (spread > 0.05), but they do not form an ordered scale. This is a problem for the inverted-U regression (H1), which assumes d is ordinal.

### Per-Run Semantic Similarity Values

Examining raw semantic similarity across block1 runs within each level:

- **Level 1** (n=5 runs): sem_sim = {0.718, 0.083, 1.000, 1.000, 0.169} → Range: [0.08, 1.00]
- **Level 2** (n=4): sem_sim = {0.310, 0.190, 0.594, 0.916} → Range: [0.19, 0.92]
- **Level 3** (n=4): sem_sim = {0.814, 0.136, 0.139, 0.494} → Range: [0.14, 0.81]
- **Level 4** (n=4): sem_sim = {0.673, 1.000, 0.897, 0.635} → Range: [0.64, 1.00]
- **Level 5** (n=4): sem_sim = {-0.005, 1.000, 0.846, 0.244} → Range: [-0.01, 1.00]

The within-level variance is enormous. Level 1 runs range from 0.08 to 1.00. Level 5 runs range from -0.005 to 1.00. The levels are not producing reliably different disagreement profiles with these sample sizes.

### Suspicious Values

Several runs show `semantic_pairwise_similarity ≈ 1.0000` regardless of disagreement level:
- Level 1: run_f71343ba7971 (sem_sim = 1.000)
- Level 4: run_6fa86f04bf07 (sem_sim = 1.000)
- Level 5: run_bdb623aefffe (sem_sim = 1.000)

A value of exactly 1.0 typically indicates either (a) identical outputs, (b) degenerate embedding comparison (e.g., both outputs empty or extremely short), or (c) a computational artifact. This requires investigation.

### Assessment

The disagreement level manipulation shows too much noise at pilot scale (4-5 runs per level) to validate ordinal behavior. The main batch (more runs per level) may resolve this, but the non-monotonicity between levels 3→4→5 is concerning and warrants investigation into whether the prompt_strategy and temperature parameters at higher disagreement levels actually produce the intended diversity.

---

## 4. Within-Run Consistency

### Agent Output Quality

I read full agent outputs from 30+ runs across all blocks. Key findings:

#### Finding 4.1: Empty and Truncated Outputs

Several agents produce **empty text** (`text: ""`):
- run_020d7a1bc18f (block4, n=2): Both GPT-5.2 agents have `text: ""` with 2048 output_tokens reported. The outputs likely exceeded the JSON truncation limit or failed to serialize. Yet the run is still evaluated — judges are comparing empty strings.
- run_105eb6452678 (block1, n=3): agent_1 (GPT-5.2) shows `text: ""`, agent_2 (Gemini) is truncated to ~80 tokens.

**Impact:** Empty outputs are automatically the worst candidates. This inflates "disagreement" metrics (an empty string is maximally dissimilar to a 500-word essay), but this is measuring **agent failure**, not genuine intellectual disagreement. Runs with empty outputs should be flagged and potentially excluded from disagreement analyses.

#### Finding 4.2: Same-Model High Similarity

When two instances of the same model respond to the same prompt, outputs show remarkable thematic convergence:

**Example — run_2d2ab668bdcb (n=5, mystery opener):**
- agent_0 (Claude Opus): "The rain that morning tasted faintly of copper... Maren Coll... forensic suite... cheap plastic button... she was smiling, and the expression had been surgically applied after death."
- agent_3 (Claude Opus): "The rain that morning tasted of sulfur and synthetic lavender... Maren Oshiro... BioTrace... single brass button... a deep arterial red..."

Both Claude instances used: the same character first name (Maren), rain-tasting opening, forensic technology failing, a button as the pivotal object, and a closing reveal that subverts expectations. The structural skeleton is nearly identical despite different surface details.

Similarly, both GPT-5.2 instances focused on: dock/waterfront settings, missing implant signatures, analog artifacts (wristband/matchbook), and "the evidence says no one did it" closings.

**Implication:** Model diversity is essential for generating genuine disagreement. Same-model ensembles (especially at low temperature) produce surface-level variation around a single latent structure. The block4 design uses same-model assignments for some runs (e.g., both agents = gpt-5.2, or both = claude-opus), which limits achievable diversity.

#### Finding 4.3: Consensus Often IS an Individual Agent's Output

In the `debate_then_vote` consensus mechanism, the consensus output is typically the verbatim text of one agent, selected by vote. It is not a synthesis or blend.

**Consequence for BT evaluation:** When consensus = agent_k's text, judges comparing `final_consensus` vs `agent_k` always see identical texts → always tie → BT probability is split between them. The consensus "quality" is mechanically tied to the selected agent's quality, and the BT probability of the consensus is approximately:

p_consensus ≈ p_best_agent / 2 + (share from beating worse agents)

This means the consensus can never score much higher than 1/2 of the total probability assigned to the best agent+consensus pair. It's a ceiling effect baked into the evaluation design.

---

## 5. Task Difficulty Variation

### Score Quantization

Block 4 quality scores are heavily quantized, clustering at values corresponding to uniform BT baselines:

- **n=2 common values:** 0.333 (= 1/3, occurs 6/11 times), 0.250, 0.400
- **n=3 common values:** 0.250 (= 1/4, occurs 5/11 times), 0.167, 0.290

The dominance of baseline-coincident values means most pairwise comparisons result in ties. This indicates judges frequently cannot distinguish the consensus from individual agents — expected when the consensus IS an individual agent's verbatim text.

### Per-Task Variation

Looking across block4 for the most-represented task (creative_07_mystery_2050_opening):
- n=2: quality = {0.250, 0.333, 0.333} → mean = 0.305
- n=3: quality = {0.290, 0.250} → mean = 0.270
- Normalized: n=2 → mean 0.917, n=3 → mean 1.080

After normalization, n=3 actually **outperforms** n=2 for this task — inverting the paradox direction. This illustrates how the raw metric is misleading.

### Floor Effects

Quality scores of 0.167 (n=3) and 0.250 (n=2) represent the uniform baseline — no quality differentiation among candidates. Approximately 45% of n=2 runs and 55% of n=3 runs score at or below their respective baselines. The distribution is left-skewed (floor-heavy), suggesting that judges frequently cannot distinguish the consensus from individual agents.

---

## 6. Additional Concerns

### 6.1 Inter-Rater Reliability Variability

The pilot report cites mean Cohen's κ = 0.934, but the briefing notes that per-run sampling gives mean = 0.788, min = 0.026, max = 1.0. A κ of 0.026 is essentially random agreement. The high aggregate κ is driven by easy comparisons (e.g., full output vs. empty output). When agents produce similar-quality outputs (the most theoretically interesting case), judge agreement may be near chance.

### 6.2 Confound: Agent Count vs. Model Diversity

Block 4 varies agent count (n=2 vs n=3) but model assignment is not held constant. Some n=2 runs use same-model pairs (both gpt-5.2), while some n=3 runs use heterogeneous assignments. Since model diversity independently affects output quality and disagreement, any quality difference could be attributed to model composition rather than agent count per se.

### 6.3 Pre-Registered Test Validity

The ANALYSIS_PLAN.md specifies: "H3: Within-task paired comparison of Q(n=3) vs Q(n=2), paired t-test, one-sided." This test operates on raw quality scores. Given the BT normalization issue, this test will:
- Detect the mechanical baseline difference (Δ_mechanical = -0.083) with near-certainty given enough data
- Conflate this mechanical effect with any genuine quality effect
- Produce a "significant" p-value that supports the paradox hypothesis but is actually an artifact

**This constitutes a pre-registered test of a measurement artifact, not of the scientific hypothesis.** The test specification must be amended to use normalized scores, win-rates, or another scale-invariant metric before the main batch analysis. Since the analysis plan is described as "pre-registered" and "must be fixed before running full experiment," this fix is within scope.

---

## 7. Recommendations

### Immediate (Before Main Batch Analysis)

1. **Amend the quality metric** in the analysis plan. Replace raw BT quality_score with a scale-invariant metric for all cross-n comparisons. My recommendation:
   - **Primary:** Consensus pairwise win-rate = (wins + 0.5 × ties) / total_pairwise_comparisons against individual agents.
   - **Secondary:** Normalized BT uplift = quality_score × (n+1).
   - **Tertiary:** Quality uplift over best individual agent = p_consensus - max(p_agent_i).

2. **Flag and handle empty/truncated outputs.** Add exclusion criteria: runs where any agent produces `text.length < 50` should be flagged. Report results both with and without these runs.

3. **Investigate the non-monotonic disagreement levels.** Why does level 4 produce less disagreement than level 1? Check the prompt engineering and temperature settings.

### For Main Batch Design

4. **Control model composition** when varying agent count. To test the paradox cleanly, hold model assignment constant (e.g., always use [claude, gpt, gemini] for n=3, and subsets for n=2).

5. **Increase per-task replication** for block4 comparisons. The main batch should ensure ≥5 observations per task × condition for the paired within-task comparison.

6. **Add absolute quality judgments.** Have judges rate each output on a 1-5 Likert scale in addition to pairwise comparisons.

### For Paper Framing

7. **Do not claim the Quorum Paradox based on raw BT scores.** If we publish a comparison of quality(n=2) = 0.315 vs quality(n=3) = 0.230 as evidence of the paradox, any reviewer who understands BT will immediately identify the normalization artifact. This would be fatal to the paper's credibility.

8. **The paradox may still be real.** The normalized analysis shows a small, non-significant effect in the direction predicted by the theory (consensus quality slightly lower for n=3 after normalization). With larger samples and a proper metric, this could reach significance. The theoretical mechanism (conformity pressure + correlation-induced diversity collapse) is sound. But the pilot data does not confirm it.

---

## 8. Summary Assessment

| Issue | Severity | Resolvable? |
|---|---|---|
| BT scores incomparable across n | **FATAL** for cross-n comparisons | Yes, with metric normalization |
| Quorum Paradox signal is ~97% artifact | **CRITICAL** for H3 | Yes, with corrected metric |
| Disagreement levels non-monotonic | **SERIOUS** for H1 | Requires investigation |
| Empty/truncated agent outputs | **MODERATE** | Yes, with exclusion criteria |
| Same-model high similarity | **MODERATE** | Addressable via model assignment |
| Low inter-rater reliability on close comparisons | **MODERATE** | Partly addressable via more judges |
| Score quantization / floor effects | **MODERATE** | Inherent to BT + similar-quality outputs |

### Bottom Line

The pilot infrastructure works — 280/280 runs completed, the evaluation pipeline runs, the data is well-structured. But the quality metric is not fit for purpose when comparing across agent counts, which is the core comparison for 3 of 4 research questions (RQ2, RQ3, RQ4). This must be fixed before interpreting the main batch.

I am not saying the theoretical framework is wrong. The Quorum Paradox mechanism (conformity pressure, correlation-induced diversity collapse) may well be real. But we cannot detect it through a metric that mechanically decreases with pool size. We need a scale-invariant quality measure, and we can derive one from the existing pairwise comparison data. The fix is tractable; the urgency is high.

---

*Elena Chen, February 28, 2026*
*"A significant result on a biased estimator just means you have precisely estimated the wrong quantity."*
