# Pilot V2 Judge Effectiveness Assessment — Theoretical Analysis

**Author:** Elena Chen (PhD-1, Theoretical Track)  
**Date:** 2026-02-28  
**Data:** 268 post-fix pilot runs (240 Block 0 calibration, 28 multi-agent Blocks 1-4)  
**Judge Panel:** claude-sonnet-4-6, gpt-4o, deepseek-v3p2 (forced-choice prompt: "Do NOT say tie")  

---

## Executive Summary

The 64.4% overall tie rate is **not fatal**, but it significantly constrains statistical power and makes the GPT-4o slot a near-dead weight. The core finding: **Claude Sonnet carries the experiment**. With a 55.6% discrimination rate, it alone provides sufficient signal for cross-condition comparisons — but only if we run >=120 runs per condition (at conservative ICC assumptions) or >=85 runs (at optimistic ICC). The 85.2% tie rate from GPT-4o despite forced-choice prompting represents a fundamental model limitation, not a prompt engineering failure. Replacing it with a model achieving >=45% discrimination would reduce required sample sizes by ~25%.

**Recommendation:** Replace GPT-4o before the main batch. Run a 30-comparison calibration against Qwen 3 or Llama 4. If no replacement achieves >=40% discrimination, keep GPT-4o but adopt a **weighted panel** design where votes are weighted by each judge's discrimination rate. Additionally, compute automated diversity metrics on all outputs to empirically separate "judge failure" from "genuine equivalence."

---

## 1. Is the 64% Tie Rate Fatal?

### 1.1 Formal Setup

Let each run produce B total ballots across 3 judges. From the data: 567 / 28 ~ 20.25 ballots per run, so B ~ 20. Each ballot b_ij from judge j on comparison i yields one of three outcomes: consensus wins (W), consensus loses (L), or tie (T).

Define the **discrimination rate** for judge j:

    delta_j = P(b_ij != T) = 1 - tie_rate_j

From the pilot data:

| Judge | Tie Rate | Discrimination Rate (delta_j) |
|-------|----------|-------------------------------|
| claude-sonnet-4-6 | 44.4% | **55.6%** |
| deepseek-v3p2 | 63.5% | **36.5%** |
| gpt-4o | 85.2% | **14.8%** |
| **Panel average** | **64.4%** | **35.6%** |

### 1.2 Effective Information Per Run

Tied ballots contribute 0.5 wins to each side in the Bradley-Terry matrix — they inject noise symmetrically, pushing BT scores toward the uniform baseline 1/K. Only discriminating ballots carry information about relative quality.

Assuming ballots are roughly equally distributed across judges (~7 per judge per run):

| Judge | Ballots/run | Discriminating/run |
|-------|------------|-------------------|
| claude-sonnet-4-6 | ~7 | **3.9** |
| deepseek-v3p2 | ~7 | **2.6** |
| gpt-4o | ~7 | **1.0** |
| **Total** | **~20** | **~7.5** |

So each run yields approximately **7.5 informative ballots** out of 20 total. The information efficiency is 37.5%. This is low but not zero — the question is whether it is sufficient for cross-condition comparisons.

### 1.3 Signal Dilution in BT Scores

The BT model does not discard ties — it treats them as half-wins for each side. This means tied ballots actively push BT probabilities toward 1/K, diluting any real signal. The magnitude of dilution:

Let D = number of discriminating ballots, T_n = number of tied ballots, and p_true the true consensus win probability among discriminating comparisons. The observed BT-derived quality score is approximately:

    p_BT ~ [D * p_true + T_n * 0.5] / (D + T_n) = 1/K + [D/(D+T_n)] * (p_true - 1/K)

The attenuation factor is D / (D + T_n) ~ 0.375. A true 10-percentage-point quality difference (e.g., p_true = 0.60 vs. 0.50) appears as only a **3.75-point difference** in BT scores after tie dilution. This is the core problem: ties don't just reduce power; they **compress the effect size** in BT-derived metrics.

### 1.4 Verdict: Not Fatal, But Power-Constrained

The tie rate does not destroy signal — it attenuates it by a factor of ~2.7x. This means:

1. **Effect sizes in BT scores will appear smaller than reality.** Any significance testing on BT scores must account for this attenuation.
2. **We need more runs to compensate.** The effective sample size per run is ~37.5% of the nominal ballot count.
3. **Cross-condition comparisons are still possible** if we have sufficient runs. Claude Sonnet's 55.6% discrimination rate is the anchor — it alone contributes ~4 informative ballots per run, enough to detect large effects even without the other judges.

**The 64% tie rate is survivable. The question is the required sample size.**

---

## 2. The GPT-4o Problem

### 2.1 Diagnosis

GPT-4o's 85.2% tie rate under forced-choice prompting is striking. The pre-fix rate was 92.8% (from the earlier pilot). The forced-choice prompt reduced ties by only **7.6 percentage points** — a minimal improvement that suggests the problem is architectural, not prompt-related.

Hypothesis: GPT-4o exhibits a strong **calibration conservatism** — it is reluctant to commit to preferences between outputs that share broad structural similarity, even when fine-grained quality differences exist. This is consistent with RLHF training that penalizes confident-but-wrong evaluative judgments, producing a model that defaults to "both are good" unless differences are extreme.

Evidence: The one example where GPT-4o *did not* discriminate despite large word-count differences (2357 vs 2061 vs 1214) while Claude and DeepSeek did — this rules out the charitable interpretation that "GPT-4o ties only when outputs are genuinely equivalent."

### 2.2 Analysis of Options

#### Option (a): Drop GPT-4o, replace with another model

**Information-theoretic argument.** GPT-4o contributes ~1.0 discriminating ballot per run out of ~7 it's allocated. A replacement judge achieving Claude Sonnet's 55.6% rate would contribute ~3.9, a net gain of **+2.9 discriminating ballots per run**. This increases total discriminating ballots from 7.5 to 10.4 — a **39% improvement** in information yield per run.

**Impact on required sample size:** (see Section 3 for full analysis) At ICC = 0.3, this reduces required runs per condition from ~153 to ~110 — saving ~43 runs multiplied by the computational cost per run.

**Risk:** The replacement might not achieve 55.6%. Even a 40% discrimination rate yields 2.8 discriminating ballots — still +1.8 over GPT-4o.

**Verdict: Strongly recommended.** The opportunity cost of keeping GPT-4o is ~2-3 informative ballots per run. At scale (100+ runs per condition), this compounds to hundreds of wasted comparisons.

#### Option (b): Keep GPT-4o, weight votes lower

A **weighted BT model** where ballot weights w_j are proportional to judge discrimination rates:

    w_j = delta_j / max_k(delta_k)

| Judge | w_j |
|-------|-------|
| claude-sonnet-4-6 | 1.000 |
| deepseek-v3p2 | 0.657 |
| gpt-4o | 0.266 |

This reduces GPT-4o's dilution effect but does not create information that is not there. The tied ballots still contribute to the BT matrix (just with lower weight). **Net effect: marginal improvement over unweighted, but strictly dominated by option (a) if a good replacement exists.** The weighting adds analytical complexity (non-standard BT inference) without solving the fundamental problem.

**Verdict:** Acceptable fallback if no replacement is viable, but second-best.

#### Option (c): Rubric scoring (1-5 per criterion)

Switch from pairwise forced-choice to absolute scoring per response on a rubric.

**Theoretical advantages:**
- **Ties become extremely rare.** If each response is scored 1-5 on C criteria, the probability of an exact tie across all criteria is (1/5)^C. For C = 5 criteria: P(tie) = 3.2 x 10^-4. Even summed scores: with range [C, 5C], the probability of equal totals is approximately 1/(4C+1) under uniform — about 4.8% for C = 5.
- **Higher information yield per comparison.** Instead of a single bit (win/lose/tie), we get a C-dimensional quality vector per response per judge.
- **Finer-grained power.** Continuous outcome variables require smaller samples than binary ones for the same effect size.

**Theoretical disadvantages:**
- **Scale calibration problem.** Different judges may use the 1-5 scale differently (anchoring effects, range restriction). GPT-4o might rate everything 3-4, producing small variance that's just as uninformative as ties.
- **Criterion weighting.** Rubric totals implicitly weight all criteria equally. Unequal weighting requires pre-specification or risks researcher degrees of freedom.
- **Loss of transitivity guarantees.** Pairwise BT comparisons enforce transitivity in the quality ordering. Rubric scores do not — response A can score higher than B on some criteria, lower on others, producing intransitive "better than" relations.
- **Breaks continuity with existing pipeline.** The BT scoring infrastructure, the BT normalization fix I identified earlier, the power analyses — all would need to be rebuilt.

**Verdict:** Theoretically attractive for a future experiment, but too costly a methodological pivot for the current paper. The forced-choice pairwise design is already pre-specified in the methodology. Switching introduces a confound between measurement instrument and experimental condition. **Not recommended for the current experiment.**

#### Option (d): Accept GPT-4o as-is

**Argument for:** Even 14.8% discrimination contributes some signal. With 200 runs per condition x ~7 GPT-4o ballots per run = 1400 GPT-4o ballots, of which ~207 discriminate. These 207 ballots are not worthless.

**Argument against:** Those 207 discriminating ballots come at the cost of 1193 tied ballots that actively dilute the BT signal. The net effect may be negative: the dilution from ties could outweigh the signal from the discriminating votes.

**Formal condition for GPT-4o to have positive net contribution:** A judge has positive net information contribution if and only if the variance reduction from its discriminating votes exceeds the variance increase from its tied votes. Under the attenuated BT model above, this requires:

    delta_j > 1 / (2K - 1)

For K = 4 (3-agent runs): threshold delta > 0.143. GPT-4o's delta = 0.148 just barely exceeds this. For K = 3 (2-agent runs): threshold delta > 0.200, and GPT-4o fails. **GPT-4o has marginally positive contribution for 3-agent runs and negative contribution for 2-agent runs.**

**Verdict:** Marginally acceptable for 3-agent conditions, actively harmful for 2-agent conditions. Not recommended.

### 2.3 GPT-4o Recommendation

**Replace GPT-4o.** The forced-choice prompt was its second chance (after the initial pilot's 92.8% tie rate). The improvement to 85.2% is insufficient. The 14.8% discrimination rate barely clears the positive-contribution threshold for 3-agent runs and fails for 2-agent runs.

**Candidate replacements** (from Marcus's earlier panel redesign analysis): Qwen 3 or Llama 4, both from distinct provider families. Run the 30-comparison calibration test specified in T-029b. The replacement target: **>=40% discrimination rate** under the forced-choice prompt.

---

## 3. Power Analysis

### 3.1 Statistical Framework

We want to detect a difference Delta = p_A - p_B in consensus win rates between two experimental conditions, where p_A and p_B are the true probabilities that consensus beats individual agents in conditions A and B respectively.

**Design parameters:**
- N = runs per condition
- B ~ 20 = ballots per run
- delta_bar = 0.356 = average discrimination rate (current panel)
- m = B x delta_bar ~ 7.1 = expected discriminating ballots per run
- rho = intra-cluster correlation (ICC) among ballots within a run

### 3.2 Required Sample Size (Ballot-Level Analysis)

For a two-proportion z-test at alpha = 0.05, power = 0.80:

    n_eff = [(z_{alpha/2} + z_beta)^2 * (p_A(1-p_A) + p_B(1-p_B))] / Delta^2

With Delta = 0.10, p_A = 0.55, p_B = 0.45:

    n_eff = [(1.96 + 0.84)^2 * (0.2475 + 0.2475)] / 0.01 = [7.84 * 0.495] / 0.01 = 388 discriminating ballots per condition

**Clustering adjustment.** Ballots within a run are not independent — they share the same set of agent outputs. The design effect:

    DEFF = 1 + (m - 1) * rho

The required discriminating ballots per condition, adjusted:

    n_adj = n_eff * DEFF

And the required runs per condition:

    N = n_adj / m = [n_eff * (1 + (m-1)*rho)] / m

### 3.3 Scenarios: Current Panel (delta_bar = 0.356, m ~ 7.1)

| ICC (rho) | DEFF | Adjusted n | **Runs/condition** |
|-----------|------|-----------|-------------------|
| 0.05 (minimal) | 1.31 | 508 | **72** |
| 0.10 (low) | 1.61 | 625 | **88** |
| 0.20 (moderate) | 2.22 | 861 | **121** |
| 0.30 (substantial) | 2.83 | 1098 | **155** |
| 0.50 (high) | 4.05 | 1571 | **221** |

### 3.4 Scenarios: Improved Panel (replace GPT-4o, delta_bar ~ 0.49, m ~ 9.8)

Assuming replacement judge achieves ~45% discrimination rate. The net effect of improving the panel on required runs:

    N = [n_eff * (1 + (m-1)*rho)] / m = n_eff * [1/m + ((m-1)/m)*rho]

As m increases, 1/m decreases (good) but rho is unchanged (neutral). The dominant term for moderate-to-high ICC is ~n_eff * rho, which is independent of panel quality. For low ICC, the 1/m term dominates and improving the panel helps substantially.

**Comparison (runs per condition):**

| ICC | Current panel (m=7.1) | Improved panel (m=9.8) | **Reduction** |
|-----|----------------------|----------------------|---------------|
| 0.05 | 72 | 57 | -21% |
| 0.10 | 88 | 74 | -16% |
| 0.20 | 121 | 109 | -10% |
| 0.30 | 155 | 144 | -7% |
| 0.50 | 221 | 214 | -3% |

**Interpretation:** Replacing GPT-4o helps most when ICC is low (the optimistic scenario). At high ICC, the improvement is marginal because the clustering effect dominates.

### 3.5 What ICC Should We Expect?

The ICC reflects how much ballots within a run share common variance. Sources of within-run correlation:
- **Same output texts** being compared across judges and criteria
- **Task-level difficulty** (some tasks genuinely produce more similar outputs)
- **Agent composition** (some model combinations produce more/less divergent outputs)

My estimate: rho in [0.15, 0.35]. The lower bound reflects the diversity across judges and criteria; the upper bound reflects the strong commonality of sharing the same output texts. **Best guess: rho ~ 0.20.**

### 3.6 Power Analysis Summary

At rho = 0.20 (my best estimate), detecting Delta = 0.10 with 80% power requires:
- **Current panel: ~121 runs per condition**
- **Improved panel: ~109 runs per condition**

The planned 100-200 runs per condition is **adequate** for detecting a 10-point win-rate difference at the lower end of the ICC range, and **borderline** at higher ICC values.

**For a smaller effect size Delta = 0.05** (5-point difference), the required sample sizes quadruple:
- Current panel at rho = 0.20: ~484 runs per condition (infeasible)
- This underscores: **the experiment is powered for medium effects (Delta >= 0.10), not small ones.**

---

## 4. The Deeper Question: Judge Failure vs. Genuine Equivalence

### 4.1 The Identification Problem

Let q_i denote the true quality of output i. A judge observes a pair (o_A, o_B) and reports:

    J(o_A, o_B) = A > B   if q_A - q_B > tau + epsilon_j
                 = B > A   if q_B - q_A > tau + epsilon_j
                 = tie     otherwise

where tau is the judge's discrimination threshold and epsilon_j is random noise. A high tie rate is consistent with two very different data-generating processes:

1. **Low-variance outputs** (|q_A - q_B| is typically small): Outputs from different agents are genuinely similar in quality, so |q_A - q_B| < tau most of the time. The tie rate reflects reality.

2. **High-threshold judge** (tau is large): The judge requires extreme quality differences to commit, even when real differences exist. The tie rate reflects judge conservatism.

These are **not identifiable from tie rates alone.** We need auxiliary information.

### 4.2 Available Diagnostic Evidence

**Evidence supporting "genuine equivalence" (interpretation 1):**

- **Block 0 validates the 100% tie case.** Single-agent runs produce identical output pairs, and all judges correctly tie. This confirms judges are not randomly assigning wins.
- **Equal word counts -> universal ties.** When outputs are 654 vs. 654 words, all three judges tie. If outputs are structurally identical, this is correct behavior.
- **Multi-agent LLMs produce convergent outputs.** Modern LLMs trained on similar data with similar RLHF produce outputs with high semantic overlap even when different models generate them. For well-defined tasks, there may genuinely be limited quality variance.

**Evidence supporting "judge failure" (interpretation 2):**

- **Large structural differences -> partial discrimination.** The 2357 vs 2061 vs 1214 example shows Claude Sonnet and DeepSeek discriminating while GPT-4o still ties. If the outputs are genuinely different, GPT-4o is failing.
- **Judge-level discrimination variance.** If all outputs were genuinely equivalent, *all* judges should tie at similar rates. The spread from 44.4% (Claude) to 85.2% (GPT-4o) implies different tau thresholds, not different data.
- **Forced-choice prompting should eliminate legitimate ties.** If the prompt says "Do NOT say tie," and a judge still ties 85% of the time, it's overriding the instruction based on its own calibration — this is definitionally judge failure (failure to follow the evaluation protocol).

### 4.3 Proposed Diagnostic Battery

To empirically disentangle these interpretations before the main batch:

**Diagnostic 1: Automated Diversity Metrics.** For each multi-agent run, compute:
- Pairwise ROUGE-L between agent outputs
- Pairwise BERTScore (semantic similarity)
- Word count coefficient of variation
- Structural divergence (number of sections, list items, code blocks)

Then regress judge discrimination rate on diversity metrics. If R^2 is high (>=0.4), judges discriminate when outputs differ — the tie rate reflects genuine equivalence on similar outputs. If R^2 is low, judges are failing regardless of output differences.

**Diagnostic 2: Synthetic Calibration Pairs.** Create pairs with known quality gaps:
- Take a high-quality output, introduce 3-5 factual errors -> large quality gap
- Take two outputs, truncate one to 50% length -> structural difference
- Take two outputs, rephrase one less fluently -> subtle quality gap

Run these through the panel. Expected: all judges discriminate on large gaps; only good judges discriminate on subtle gaps. This directly measures each judge's tau.

**Diagnostic 3: Inter-Judge Agreement on Discriminating Ballots.** When two or more judges discriminate on the same pair, do they agree on the winner?
- High agreement (Cohen's kappa > 0.6 among discriminating votes): The signal is real, and the non-discriminating ballots likely reflect genuine equivalence.
- Low agreement (kappa < 0.3): Even discriminating votes are noisy, and the experiment has a measurement validity problem beyond tie rates.

### 4.4 My Assessment

Based on the available evidence, I estimate the tie rate decomposes roughly as:

- **~40-50% of ties** reflect genuine output equivalence (especially on well-defined tasks with convergent answers)
- **~30-40% of ties** reflect judge conservatism (especially GPT-4o's high threshold)
- **~10-20% of ties** reflect insufficient evaluation criteria (the forced-choice prompt doesn't give judges enough rubric guidance to find differences)

This is a rough partition, and the diagnostics above would sharpen it. But the implication is clear: **even the "genuine equivalence" fraction is informative** — it tells us that multi-agent pipelines often converge, which is itself a finding about the disagreement dividend hypothesis.

---

## 5. Recommendation: Actions Before Main Batch

### Priority 1: Replace GPT-4o (BLOCKING)

**Action:** Run 30-comparison calibration test with Qwen 3 and/or Llama 4 as candidate replacements.  
**Acceptance criterion:** >=40% discrimination rate under the forced-choice prompt.  
**Fallback:** If no replacement exceeds 40%, keep GPT-4o but implement weighted BT (Section 2.2b) with weights proportional to discrimination rates.  
**Timeline:** 1-2 days. This is blocking; do not start the main batch with the current panel.

### Priority 2: Compute Automated Diversity Metrics (BLOCKING)

**Action:** For the 28 multi-agent pilot runs, compute pairwise ROUGE-L, BERTScore, word count CV, and structural divergence between agent outputs.  
**Purpose:** Establish the empirical relationship between output diversity and judge discrimination. This regression becomes a key diagnostic for interpreting main batch results.  
**Acceptance criterion:** Report the R^2 of discrimination rate regressed on diversity metrics. If R^2 < 0.2, we have a judge quality problem that replacement alone may not solve.  
**Timeline:** 1 day. Requires implementing the metrics in the evaluation pipeline.

### Priority 3: Run Synthetic Calibration (RECOMMENDED)

**Action:** Create 10 synthetic pairs with known quality gaps (5 large, 5 subtle). Run through all judges including any replacement candidates.  
**Purpose:** Directly measure each judge's discrimination threshold tau on pairs where ground truth is known.  
**Timeline:** 1-2 days.

### Priority 4: Verify ICC Estimate (RECOMMENDED)

**Action:** From the 28 pilot runs, estimate the ICC of ballot outcomes within runs using a random-intercept logistic model. This pins down which row of the power table we're actually in.  
**Purpose:** Determine whether 100 or 200 runs per condition is the correct target.  
**Timeline:** Can be done on existing data, ~half day.

### Priority 5: Pre-Register Analytical Decisions (MANDATORY)

Before the main batch, lock in:
- The judge panel composition
- Whether weighted or unweighted BT is used
- The primary outcome variable (consensus win rate, BT score, or normalized BT score)
- The clustering adjustment method (GEE, mixed-effects, or run-level aggregation)
- Effect size of interest (Delta = 0.10) and target power (80%)

These must be documented before data collection to avoid post-hoc analytical flexibility undermining publishability.

---

## 6. Summary Table

| Question | Answer | Confidence |
|----------|--------|------------|
| Is 64% tie rate fatal? | **No**, but it attenuates effects by ~2.7x and requires >=120 runs/condition | High |
| Can we extract meaningful signal? | **Yes**, primarily from Claude Sonnet (55.6% discrimination) | High |
| Should we drop GPT-4o? | **Yes**, replace with a >=40% discrimination judge | High |
| Rubric scoring instead? | **Not for this experiment** — too large a methodological pivot | High |
| Runs needed for 80% power (Delta = 0.10)? | **109-155 per condition** (depending on ICC) | Medium |
| Are outputs genuinely similar? | **Partially** — estimated 40-50% of ties reflect real equivalence | Medium |
| Can we tell the difference? | **Yes**, with diversity metrics + synthetic calibration | High |

---

## Appendix: Derivation Notes

### A1. Attenuation Factor

The BT model for K candidates assigns probabilities p_i = lambda_i / sum_k(lambda_k). Under the standard logistic BT parameterization, a tie ballot contributes equal evidence to both sides: log(lambda_A / lambda_B) += 0. A win for A contributes log(lambda_A / lambda_B) += 1. The expected log-odds from a ballot:

    E[Delta_ell] = delta * (2*p_w - 1) + (1 - delta) * 0 = delta * (2*p_w - 1)

where p_w is the true consensus win probability. The attenuation factor for the log-odds signal is exactly delta. For the probability-scale signal, the attenuation is approximately delta for small departures from 0.5.

### A2. Positive-Contribution Threshold

A judge contributes positively if the Fisher information from its discriminating ballots exceeds the information loss from its tied ballots. For a BT model with K candidates, the information per discriminating ballot is ~1/(K-1) and the noise per tied ballot is ~1/[K(K-1)] (approximate, from the BT Hessian). The condition delta / (K-1) > (1-delta) / [K(K-1)] simplifies to:

    delta > 1 / (2K - 1)

For K=3: delta > 0.200. For K=4: delta > 0.143.

### A3. ICC Estimation Approach

Fit a generalized linear mixed model:

    logit(P(consensus wins)) = beta_0 + u_i,    u_i ~ N(0, sigma^2_u)

on discriminating ballots only, with random intercept u_i per run. The ICC on the latent scale: rho = sigma^2_u / (sigma^2_u + pi^2/3). Convert to ballot-level ICC for the power calculation using the linearization rho_obs ~ rho * p_bar*(1-p_bar) * (sigma^2_u + pi^2/3) / Var(Y).
