# ANALYSIS_PLAN.md

**Status:** Pre-registered analysis plan (updated for BT normalization artifact correction).

## Metric Policy (Critical Update)

### Primary quality metric for cross-condition comparisons
- **`consensus_win_rate`** = fraction of pairwise comparisons where the consensus output beats individual-agent outputs (ties count as 0.5).

### Secondary diagnostic metric
- **`normalized_bt_score`** = `quality_score × num_bt_candidates`.

### Why this correction is required
Raw `quality_score` is a Bradley-Terry (BT) share over all candidates in each run, and BT shares sum to 1.0. As candidate count increases, the raw value is mechanically deflated (e.g., baseline ~0.50 for 2 candidates, ~0.25 for 4 candidates). Therefore, raw BT scores are **not comparable across different agent counts** and are not used as the primary metric for H1-H4.

---

## 1) Confirmatory Primary Hypotheses (H1-H4) and Exact Tests

### H1 (RQ1: Disagreement Dividend)
- **Claim:** Consensus quality as a function of disagreement level is inverted-U shaped.
- **Outcome:** `consensus_win_rate`.
- **Model:** Quadratic regression
  - `CWR = beta0 + beta1*d + beta2*d^2 + u_task + e`
  - where `d` is disagreement level proxy; `u_task` is task random effect (mixed-effects specification where available).
- **Primary test:** one-sided test `beta2 < 0`.

### H2 (RQ2: MVQ / threshold attainment)
- **Claim:** Probability of meeting quality threshold increases with n and yields an interpretable `n*`.
- **Outcome:** threshold events based on `consensus_win_rate`.
- **Model:** Logistic regression
  - `logit(P(CWR >= theta)) = gamma0 + gamma1*n + gamma2*condition + gamma3*n:condition + u_task`
- **Primary test:** `gamma1 > 0` in each task type; extract `n*` (minimum n where predicted probability crosses target).

### H3 (RQ3: Quorum Paradox)
- **Claim:** In paradox-relevant settings, consensus quality at n=3 is lower than at n=2.
- **Outcome:** `consensus_win_rate`.
- **Test:** Within-task paired comparison of `CWR(n=3)` vs `CWR(n=2)`.
  - Use paired t-test if normality of paired differences is adequate.
  - Otherwise use Wilcoxon signed-rank.
- **Direction:** one-sided (`CWR(n=3) < CWR(n=2)`).

### H4 (RQ4: Topology x Consensus interaction)
- **Claim:** Topology and consensus interact significantly.
- **Outcome:** `consensus_win_rate`.
- **Test:** Two-way ANOVA interaction F-test on consensus quality.
  - Factors: topology, consensus.
  - Conducted separately by task type for confirmatory family.

---

## 2) Multiple-Comparison Correction

All confirmatory primary tests are corrected using **Holm-Bonferroni** at family-wise `alpha = 0.05`.

Procedure:
1. Sort p-values ascending: `p_(1) <= p_(2) <= ... <= p_(m)`.
2. Compare sequentially against `alpha/(m-i+1)`.
3. Stop at first non-rejection; all larger p-values are non-significant.

---

## 3) Effect Size Thresholds (Meaningful Effects)

For primary claims, effects are considered *practically meaningful* when:
- **Cohen's d >= 0.3** (or equivalent standardized effect size).

Interpretation policy:
- Statistical significance without practical effect-size threshold is reported as "statistically detectable but small".
- Claims in abstract/conclusion require both corrected significance and meaningful effect size.

---

## 4) Stopping Rules (Pilot Gate)

Experiment stops early (NO-GO) if any of the following occurs in pilot:

1. **Judge reliability failure:** mean inter-judge kappa `< 0.30`.
2. **Disagreement manipulation failure:** disagreement levels are not separable (insufficient spread/gap in observed disagreement rates).
3. **Infrastructure instability:** persistent API failure patterns that prevent reliable completion.
4. **Budget guardrail breach:** cumulative spend exceeds configured `--max-cost` before pilot criteria are resolved.

Soft warning (requires review before proceeding):
- `0.30 <= kappa <= 0.40`.

---

## 5) Confirmatory vs Exploratory Labeling

## Confirmatory (pre-registered)
- H1-H4 tests exactly as specified above, using `consensus_win_rate` as primary quality metric.
- Primary corrected inference set (Holm-Bonferroni family).
- Core effect sizes and confidence intervals for primary outcomes.

## Exploratory (post-hoc)
- Per-task deep dives not pre-specified.
- Mediation/mechanism analyses using debate-round traces.
- Provider-factor sensitivity analysis (Anthropic/OpenAI/Google composition terms in ANOVA/GLM) to test robustness to provider identity.
- Alternative thresholds/theta sweeps beyond primary definitions.
- Secondary checks with `normalized_bt_score`.

Exploratory findings are clearly labeled "exploratory" and not used as sole support for headline claims.

---

## 6) Exact Test Mapping by RQ (Required)

- **RQ1:** Quadratic regression of `consensus_win_rate` on `d`; test `beta2 < 0`.
- **RQ2:** Logistic regression of `P(consensus_win_rate >= theta)` on `n`; extract `n*` per condition.
- **RQ3:** Paired t-test or Wilcoxon signed-rank on within-task `consensus_win_rate(n=3) < consensus_win_rate(n=2)`.
- **RQ4:** Two-way ANOVA interaction test `topology x consensus` (F-test) on `consensus_win_rate`.

---

## 7) Number of Primary Comparisons and Adjusted Alpha

Primary confirmatory family size: **m = 10** comparisons.

Defined as:
1-2. H1 (analytical, creative)
3-4. H2 (analytical, creative)
5-6. H3 paradox condition (analytical, creative)
7-8. H3 homogeneous-strong control check (analytical, creative)
9-10. H4 interaction test (analytical, creative)

Holm-Bonferroni controls FWER at `alpha = 0.05`.
- Most stringent first-step threshold: `0.05 / 10 = 0.005`.
- Later thresholds relax according to Holm ordering.

---

## Reporting Commitments

For all primary analyses:
- report test statistic, raw p, Holm-adjusted decision, effect size, and CI.
- include sample size per condition and any exclusion criteria.
- include sensitivity note if normality assumptions trigger non-parametric fallback.
- include a BT-artifact note clarifying that raw BT shares are reported descriptively only, not used for cross-agent-count inference.
