# Experiment Plan: Quorum-Based Multi-Agent LLM Orchestration

## 1) Research Questions and Hypotheses

### RQ1 — Disagreement Dividend
**Question:** Is there a non-zero disagreement level that improves aggregate output quality versus near-identical agents?

- **H1:** Quality follows an inverted-U curve over disagreement level; moderate disagreement outperforms level-1 homogeneity.
- **Falsification criterion:** If quality is monotonic decreasing with disagreement (or flat within confidence bounds), H1 is rejected.

### RQ2 — Topology Effect
**Question:** Do flat, hierarchical, and quorum topologies differ significantly in quality-cost-latency tradeoffs?

- **H2:** Quorum topology improves quality under analytical tasks at moderate/high disagreement, but with higher token cost.
- **Falsification criterion:** If quorum does not outperform alternatives on analytical quality after cost-adjusted comparison, H2 is rejected.

### RQ3 — MVQ Behavior
**Question:** Is there a minimum viable quorum size where quality threshold attainment sharply improves?

- **H3:** Threshold attainment probability rises from n=1 to n=3 and saturates near n=5 for selected thresholds.
- **Falsification criterion:** If attainment does not improve with quorum size, or saturates at n=2 without gain at n>=3, H3 is rejected.

### RQ4 — Quorum Paradox
**Question:** Can adding weak agents to one strong agent degrade overall quality at specific n (e.g., n=3)?

- **H4:** In asymmetric (1 strong + N-1 weak) setups, a non-monotonic dip occurs around n=3 under certain consensus rules.
- **Falsification criterion:** If quality is monotonic non-decreasing as weak agents are added, H4 is rejected.

---

## 2) Experimental Design and Justification

### Task Set
- **2 task families**: analytical reasoning and creative generation.
- **8 curated instances per family** (16 total) with rubric-rich prompts.
- Rationale: capture both correctness-sensitive and open-ended generation regimes.

### Models
- `claude-opus-4-6` (strong/expensive), `gemini-2.5-pro` (strong/moderate), `gemini-2.0-flash` (fast/cheap).
- Rationale: heterogeneous capability/cost frontier is necessary to test quorum paradox and mixed-agent benefits.

### Factors
- Agent counts: {1,2,3,5}
- Topologies: {flat, hierarchical, quorum}
- Consensus: {simple_vote, debate_then_vote, judge_based}
- Disagreement levels: 1..5 via prompt diversification + temperature schedule

### Blocks
- **Block 0 (Calibration):** single-model baseline estimates of capability mean mu and pairwise correlation rho.
- **Block 1 (Disagreement Dividend):** disagreement sweep for d*.
- **Block 2 (Topology Comparison):** topology x consensus grid at fixed disagreement.
- **Block 3 (MVQ Curves):** quorum size vs threshold attainment.
- **Block 4 (Quorum Paradox):** asymmetric strong+weak scaling.
- **Block 5 (Interaction Probe):** factorial interaction for ANOVA-style decomposition.

### Operational Choices
- Async concurrency (default 10) balances throughput and rate limits.
- Filesystem-only persistence avoids DB failure modes and simplifies portability.
- Atomic writes + resume scanning guarantee crash-safe continuation.
- Judge model defaults to Gemini Flash to control scoring cost.

---

## 3) Power Analysis (Reduced-Scale)

This design prioritizes effect-size detection over fine-grained null effects. With ~1.6k-1.8k total runs:

- Main effects (topology, consensus, disagreement) are repeatedly sampled across 16 tasks and multiple agent counts.
- For medium effect sizes (Cohen-style standardized d ~0.4-0.6), cell replication across blocks is typically adequate for stable mean/rank ordering.
- Interaction detection is weaker than in full-scale studies but sufficient for screening-level inference and hypothesis triage.

Practical criterion: if bootstrap confidence intervals for key contrasts remain wide/overlapping, claims are treated as inconclusive rather than accepted.

---

## 4) Cost Estimate by Block (planning level)

Using manifest-derived call counts and config pricing (see `scripts/estimate_cost.py`), expected spend is reported per block before launch.

Planned trend (relative):
1. **Highest:** Block 5 (largest factorial surface)
2. **High:** Block 2 and Block 3
3. **Moderate:** Block 1
4. **Lower:** Block 0 and Block 4

Exact USD values are generated from current manifest + pricing table to avoid stale hard-coded estimates.

---

## 5) Expected Outcomes and Success/Failure Criteria

- **RQ1 success:** identifiable non-trivial d* where quality improves over level-1 baseline without unacceptable cost blow-up.
- **RQ2 success:** statistically and practically meaningful topology separation for at least one task family.
- **RQ3 success:** monotonic or near-monotonic threshold attainment gains from n=1 to n=5 with identifiable MVQ knee.
- **RQ4 success:** reproducible quality dip in asymmetric configurations near n=3 under at least one consensus mechanism.

Failure = inability to reproduce these signatures beyond noise bounds.

---

## 6) Limitations of Reduced-Scale Design

- Limited task corpus (16 prompts) may not generalize to all domains.
- LLM-judge scoring introduces model-dependent bias even with blind randomization.
- No human gold labels for all outputs in this phase.
- Provider-side drift across long runs can introduce temporal non-stationarity.

---

## 7) Connection to Formal Framework (MVQ Theorem and Beyond)

The empirical program maps directly onto formal claims:
- **MVQ theorem linkage:** Block 3 estimates threshold attainment as function of quorum size and disagreement, enabling empirical knee-point extraction.
- **Correlation/competence decomposition:** Block 0 supplies mu/rho ingredients used in formal quality aggregation assumptions.
- **Topology-consensus coupling:** Blocks 2 and 5 test whether orchestration graph structure changes effective aggregation behavior predicted by theory.

---

## 8) Explicit Falsification Criteria Summary

- **H1 false if** disagreement does not produce a measurable mid-range benefit.
- **H2 false if** topology differences vanish after cost-aware normalization.
- **H3 false if** quality-threshold attainment is not improved by larger quorum.
- **H4 false if** asymmetric quality is monotonic with added weak agents.

All hypotheses are evaluated with pre-registered contrast definitions from manifest factors; inconclusive evidence remains inconclusive and is not reframed as support.