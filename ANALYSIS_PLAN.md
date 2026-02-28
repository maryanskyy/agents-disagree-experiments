# ANALYSIS_PLAN.md

**Status:** Pre-registered analysis plan (must be fixed before running full experiment).

## 1) Confirmatory Primary Hypotheses (H1-H4) and Exact Tests

### H1 (RQ1: Disagreement Dividend)
- **Claim:** Quality as a function of disagreement level is inverted-U shaped.
- **Model:** Quadratic regression
  - `Q = beta0 + beta1*d + beta2*d^2 + u_task + e`
  - where `d` is disagreement level proxy; `u_task` is task random effect (mixed-effects specification where available).
- **Primary test:** one-sided test `beta2 < 0`.

### H2 (RQ2: MVQ / threshold attainment)
- **Claim:** Probability of meeting quality threshold increases with n and yields an interpretable `n*`.
- **Model:** Logistic regression
  - `logit(P(Q >= theta)) = gamma0 + gamma1*n + gamma2*condition + gamma3*n:condition + u_task`
- **Primary test:** `gamma1 > 0` in each task type; extract `n*` (minimum n where predicted probability crosses target).

### H3 (RQ3: Quorum Paradox)
- **Claim:** In paradox-relevant settings, quality at n=3 is lower than at n=2.
- **Test:** Within-task paired comparison of `Q(n=3)` vs `Q(n=2)`.
  - Use paired t-test if normality of paired differences is adequate.
  - Otherwise use Wilcoxon signed-rank.
- **Direction:** one-sided (`Q(n=3) < Q(n=2)`).

### H4 (RQ4: Topology x Consensus interaction)
- **Claim:** Topology and consensus interact significantly.
- **Test:** Two-way ANOVA interaction F-test on quality.
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
- H1-H4 tests exactly as specified above.
- Primary corrected inference set (Holm-Bonferroni family).
- Core effect sizes and confidence intervals for primary outcomes.

## Exploratory (post-hoc)
- Per-task deep dives not pre-specified.
- Mediation/mechanism analyses using debate-round traces.
- Additional subgroup analyses by model family combinations.\n- Provider-factor sensitivity analysis (Anthropic/OpenAI/Google composition terms in ANOVA/GLM) to test whether outcomes are robust to provider identity.
- Alternative thresholds/theta sweeps beyond primary definitions.

Exploratory findings are clearly labeled "exploratory" and not used as sole support for headline claims.

---

## 6) Exact Test Mapping by RQ (Required)

- **RQ1:** Quadratic regression of `Q` on `d`; test `beta2 < 0`.
- **RQ2:** Logistic regression of `P(Q >= theta)` on `n`; extract `n*` per condition.
- **RQ3:** Paired t-test or Wilcoxon signed-rank on within-task `Q(n=3) < Q(n=2)`.
- **RQ4:** Two-way ANOVA interaction test `topology x consensus` (F-test).

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

