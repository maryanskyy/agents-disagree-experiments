# Professor Adjudication — Strategy Reviews

**Date:** 2026-03-01
**Reviewed:** Elena's theoretical review + Marcus's statistical review

---

## Where They Agree (Adopted Immediately)

1. **Two-stage strategy is sound.** Both endorse rejudge-first.
2. **RQ1 (topology) needs no additional data.** Power > 0.999 with 251 runs. Done.
3. **Pre-register analysis plan before looking at rejudge data.** Both flag this as critical.
4. **Multiple comparison correction.** Holm-Bonferroni across 4 RQs.
5. **Same-family bias is strong and novel.** Both agree it's publishable.
6. **Prepare dual framings.** Consensus paper (if RQ2 confirms) vs. topology paper (if RQ2 fails).
7. **Never compare old vs new judge scores.** Only report new-panel results.
8. **RQ2 is the biggest risk.** May not survive rejudge.

## Where They Differ (My Decisions)

### Decision 1: Group Sequential Design vs Pre-Commitment
- **Elena:** Formal group sequential design (O'Brien-Fleming alpha-spending)
- **Marcus:** Doesn't explicitly recommend sequential testing framework

**Decision:** Pre-commit approach. If Stage 1 confirms both RQs (p < 0.01 each), Stage 1 is final. If RQ2 fails, we may do Stage 2 gap-fill — but we frame any Stage 2 findings as EXPLORATORY, not confirmatory. This avoids the sequential testing complexity while being honest about the adaptive design. State explicitly in Methods.

### Decision 2: Decision Gate Thresholds
- **Elena:** Inconsistent (p<0.01 for topology, p<0.05 for disagreement) — should unify or justify
- **Marcus:** Recommends Holm-Bonferroni (effective alpha_1 = 0.0125)

**Decision:** Uniform alpha = 0.01 for both primary RQs. After Holm-Bonferroni correction across 4 RQs, effective thresholds are: 0.0125, 0.0167, 0.025, 0.05 (ordered by p-value). This satisfies both reviewers.

### Decision 3: Block 1 Analysis Approach
- **Elena:** Binned analysis, minimum 30 per bin
- **Marcus:** Continuous regression (maximizes power, N=268 for single test)

**Decision:** Primary test is continuous quadratic regression (Marcus's recommendation — more powerful). Binned visualization for figures. Both reported.

### Decision 4: Same-Family Bias Confounds
- **Elena:** Style alignment confound — need to control for structural similarity
- **Marcus:** Re-measure for gpt-5-mini specifically

**Decision:** Both. After rejudge: (1) Re-measure same-family bias for gpt-5-mini. (2) Run structural similarity control analysis. (3) If bias persists after controlling for style → genuine bias. If it disappears → style alignment. Either is publishable with different framing.

### Decision 5: Framing
- **Elena:** Narrow from "generative tasks" to "open-ended" or "subjective"
- **Marcus:** No specific recommendation

**Decision:** Adopt Elena's suggestion. "Consensus mechanisms for open-ended multi-agent generation" — more precise, harder to attack.

---

## Pre-Registration Document (Lock Before Rejudge)

### Primary Tests
- **RQ1:** Two-way ANOVA (topology x consensus) on consensus_win_rate, model as random effect. Alpha = 0.01.
- **RQ2:** Mixed-effects quadratic regression: WR ~ disagreement + disagreement^2 + task_type + (1|model_composition). Alpha = 0.01.

### Secondary Tests
- **RQ3:** One-way ANOVA on agent count (n=2,3,5). Expect null. Report power analysis.
- **RQ4:** Binomial test of same-family win rate vs 50% baseline, per judge. Alpha = 0.05 (exploratory).

### Multiple Comparison Correction
- Holm-Bonferroni across all 4 RQs.

### Decision Gate
- If RQ1 p < 0.01 AND RQ2 p < 0.01: Write paper. No Stage 2.
- If RQ1 p < 0.01 AND RQ2 p > 0.01: Stage 2 gap-fill Block 1 (Tier 1 cells per Marcus). Any Stage 2 RQ2 result is EXPLORATORY.
- If RQ1 p > 0.01: Something went wrong. Investigate.

### Robustness Checks
- Per-judge separate analyses (all 3 judges)
- DeepSeek-only results (independent anchor)
- Length bias test (win rate ~ output length)
- Structural similarity control for same-family bias
