# Pilot Data Review — Multi-Agent Systems Perspective

**Reviewer:** Dr. Yuki Tanaka (Multi-Agent Systems, Game Theory, Distributed Coordination)
**Date:** 2026-02-28
**Subject:** Pilot data review for "When Agents Disagree" — 280 completed runs
**Prior review:** Experiment design review (Round 1), verdict: Invest with Changes

---

## Verdict: CONDITIONAL GO

**Conditions:** The Quorum Paradox analysis must be fundamentally redesigned before the main batch results are interpreted. The quality metric is fatally confounded with panel size. Proceeding to the main batch is acceptable ONLY because the data collection is model-agnostic — the runs themselves are fine, but the analysis methodology must change before any paradox claims are made.

---

## 1. Did the Revisions Work?

**My Round 1 concern:** *"The paradox claim is about quorum size, but the experiment confounds quorum size with model composition... Include a homogeneous-strong control (e.g., 3x Opus) at n=3 to isolate the composition effect."*

**Assessment: PARTIALLY ADDRESSED — the control exists but reveals something worse.**

The pilot includes homogeneous-strong configurations in Block 4:

| Configuration | n=2 (runs) | Mean Q | n=3 (runs) | Mean Q | Delta |
|---|---|---|---|---|---|
| All-Opus | 4 | 0.292 | 5 | 0.225 | -0.067 |
| All-GPT-5.2 | 2 | 0.367 | 3 | 0.234 | -0.133 |
| Mixed (strong+weak) | 5 | 0.313 | 3 | 0.235 | -0.078 |
| **All Block 4** | **11** | **0.315** | **11** | **0.230** | **-0.085** |

The homogeneous-strong control was added as requested. The paradox appears to persist even when all agents are the same strong model. At first glance, this rules out my composition confound — the quality dip happens even without weak models dragging down the consensus.

**However, this finding led me to a much more serious problem. See section 5.**

---

## 2. Consensus Mechanism Quality

I read the full debate rounds for multiple Block 4 runs (run_426c23c5b275 being the most detailed). The `debate_then_vote` mechanism operates in 3 phases:

1. **Draft** (Round 1): Each agent independently produces a full response. Outputs are substantive and well-structured.
2. **Revision** (Round 2): Agents see each other's drafts and produce revisions. Genuine deliberation occurs — agents reference peers' arguments ("Both peer drafts identify...", "As Peer 2 correctly notes..."), incorporate complementary points, restructure their prioritization, and sometimes upgrade their analysis.
3. **Final Consensus** (Round 3): A selection mechanism picks one agent's revised output as the consensus.

**Verdict: The deliberation is genuine, but the selection is flawed.**

The revision phase shows real intellectual engagement. Agents import counterarguments, strengthen their evidence, and synthesize across perspectives. This is not rubber-stamping.

However, the final consensus step is "pick the winner" — it selects one agent's revised output wholesale. It does NOT synthesize the best elements across agents. In run_426c23c5b275 (all-Opus, n=3), the mechanism selected agent_2's output, but the judge panel preferred agents 0 and 1 (BT scores: agent_0 = 0.333, agent_1 = 0.333, final_consensus = agent_2 = 0.167). The debate produced genuine improvement, then the vote selected a suboptimal output.

This is a classic problem in voting theory: majority vote among a small panel is a noisy aggregator, especially when outputs are close in quality. The consensus mechanism is leaving value on the table.

---

## 3. Topology Implementation

**Assessment: CORRECTLY IMPLEMENTED.**

Block 4 runs all use `topology: "quorum"` with `consensus: "debate_then_vote"`. The topology metadata confirms:
- `draft_count: 3` (or 2 for n=2)
- `revision_count: 3` (or 2)
- `debate_rounds` array with explicit `round`, `phase`, and `agent_outputs` per phase

The flat topology in Block 1 shows single-round parallel generation, as expected. The quorum topology in Block 4 shows the multi-round draft-revision-consensus pipeline.

The implementation matches the claimed design. No discrepancies detected.

---

## 4. Agent Output Quality

**Assessment: SUBSTANTIVE AND HIGH-QUALITY.**

I read agent outputs across all three blocks. The analytical task (Hidden Assumption Mapping for SaaS expansion) consistently produces:
- Structured assumption tables with 10-12 assumptions across market, operational, financial, regulatory, and behavioral categories
- Specific validation/falsification criteria with quantitative thresholds (e.g., "conversion rate lifts >= 45%", "90-day retention >= 85%")
- Prioritized risk narratives with expected downside scoring
- Real-world business reasoning (not generic templates)

The creative tasks also produce substantive outputs. Individual agent responses routinely hit the 2,048-token output cap, suggesting the task prompts are well-calibrated.

There is notable **content convergence** across agents on the same task — especially when using identical prompts (disagreement_level=1). All agents identify similar assumptions (price elasticity, support cost stability, competitive response). This is expected for well-known business analysis frameworks but means the "disagreement" manipulation may not produce as much genuine divergence as intended. More in section 6.

---

## 5. Is the Paradox Real or Methodological?

**This is the critical section. My answer: (b) — the BT scoring mechanically produces the paradox as a measurement artifact.**

### The Fatal Confound

The `quality_score` is the Bradley-Terry coefficient of the consensus output, computed among a candidate pool of **n individual agent outputs + 1 consensus output = (n+1) candidates**. The BT coefficients are a probability simplex (they sum to 1.0).

This means the **null expectation** — where all candidates are equally good — is:

| Agent count (n) | Candidate pool size | BT null per candidate |
|---|---|---|
| 1 (Block 0) | 2 | 0.500 |
| 2 | 3 | 0.333 |
| 3 | 4 | 0.250 |
| 5 | 6 | 0.167 |

I verified this against actual BT scores in the data:

- **Block 0** (n=1): All 240 runs show quality = 0.500. Exactly the 2-candidate null.
- **Block 4** (n=2): run_1d3fd2a9728e shows all three candidates at exactly 0.333.
- **Block 1** (n=5): run_ccb11924d67f and run_d92d33b492b6 both show quality = 0.167. Exactly the 6-candidate null.

### The Paradox Disappears After Normalization

The observed "paradox" delta:
- n=2 mean quality: **0.315**
- n=3 mean quality: **0.230**
- Raw delta: **-0.085**

The mechanical BT null delta:
- n=2 null: **0.333**
- n=3 null: **0.250**
- Null delta: **-0.083**

**The observed delta (0.085) is almost exactly equal to the null delta (0.083).** After normalizing quality scores by their null expectations:

| Config | n=2 normalized (Q - 0.333) | n=3 normalized (Q - 0.250) | Normalized delta |
|---|---|---|---|
| All-Opus | -0.041 | -0.025 | +0.016 |
| All-GPT-5.2 | +0.034 | -0.016 | -0.050 |
| Mixed | -0.020 | -0.015 | +0.005 |
| **Overall** | **-0.018** | **-0.020** | **-0.002** |

After normalization, the delta between n=2 and n=3 is **-0.002** — essentially zero, well within noise for 22 runs.

### Cross-Validation with Block 1

Block 1 (flat topology) provides an independent check. Computing normalized deltas:

- n=2 (7 runs): mean quality 0.395, normalized = +0.062 (consensus adds value)
- n=3 (9 runs): mean quality 0.363, normalized = +0.113 (consensus adds MORE value)
- n=5 (6 runs): mean quality 0.225, normalized = +0.058 (still positive)

In Block 1, the consensus IMPROVES over individual agents at all n values, and actually improves MORE at n=3 than at n=2. The opposite of a paradox.

### Why the BT Metric Fails

The BT scoring framework evaluates the consensus output **relative to its own panel members**. As n increases:
1. More candidates divide the probability mass — scores mechanically decrease
2. The consensus is one agent's output (picked by vote) — it can never score higher than the best individual agent
3. With more candidates, the probability that the vote picks the optimal agent decreases — consensus scores regress below the mean

This is not a quality problem with multi-agent coordination. It is a measurement problem with relative scoring in variable-sized candidate pools.

### What the Data Actually Shows

The real finding, after normalization, is:

1. **Flat topology (Block 1):** Consensus adds positive value over individual agents (normalized delta approx +0.06 to +0.11). Multi-agent coordination works.
2. **Quorum topology (Block 4):** Consensus adds zero or slightly negative value (normalized delta approx -0.02). The debate-then-vote mechanism fails to select the best output.

The interesting story is **topology/mechanism quality**, not **a paradox of scale**.

---

## 6. Game-Theory Lens: Disagreement Rates

**Prediction:** With identical prompts and models (low disagreement level), agents should produce similar outputs (low disagreement). With different prompts/models (high disagreement level), outputs should diverge (high disagreement).

**Pilot data (from pilot_report.json):**

| Disagreement Level | Mean Rate | Expected Order |
|---|---|---|
| 1 | 0.300 | Lowest |
| 2 | 0.498 | ... |
| 3 | 0.604 | ... |
| 4 | 0.199 | ... |
| 5 | 0.478 | Highest |

**Assessment: NON-MONOTONIC. This is troubling.**

The ordering is: Level 4 (0.199) < Level 1 (0.300) < Level 5 (0.478) < Level 2 (0.498) < Level 3 (0.604). Level 4 has the LOWEST disagreement — lower than the "identical" condition (Level 1). Level 3 has the highest, not Level 5.

With only 4 runs per level, this could be sampling noise. But the magnitude of the inversions is concerning. Two possible explanations from a game-theoretic perspective:

1. **The disagreement manipulation is not well-calibrated.** Whatever operationalizes "level 4" may actually induce less behavioral divergence than "level 1." If levels are defined by prompt variation + model variation, the interaction effects may be non-linear.

2. **LLM agents are not strategic actors.** Game-theoretic predictions about information-driven disagreement assume agents have preferences and respond to incentives. LLMs have neither. Their "disagreement" is a function of prompt entropy, model stochasticity, and training data overlap — none of which map cleanly to information-theoretic divergence.

**Recommendation:** Before analyzing the Disagreement Dividend curve, confirm that the disagreement levels actually produce monotonically increasing behavioral divergence. If they don't, the Disagreement Dividend curve (RQ1) is uninterpretable.

---

## Additional Observations

### Block 0 Calibration Is Uninformative

All 240 Block 0 runs show quality = 0.500, regardless of model, task, or repetition. This is the BT null for a single agent. **Block 0 provides zero information about model quality baselines.** You cannot use Block 0 to say "Opus produces quality X, Flash produces quality Y" because the BT framework only measures relative quality within a candidate pool.

Block 0 was designed for "mu/rho estimation for competence/correlation decomposition" per the experiment plan, but the BT metric cannot deliver this.

### Judge Panel Reliability

The pilot reports mean Cohen's kappa = 0.934, which passes the > 0.4 threshold. However, examining individual runs reveals substantial variation:
- run_426c23c5b275: Claude Sonnet strongly preferred agents 0 and 1, while GPT-4o and Gemini saw all as ties. Pairwise kappa ranged from 0.0 to 1.0.
- Many runs show 2 of 3 judges calling ties with the third making discriminations.

The high aggregate kappa may be inflated by Block 0 (where all 240 runs produce trivial agreement). For the multi-agent blocks, judge reliability appears lower and more variable than the headline number suggests.

### The "Same Task" Problem

Every Block 4 run I examined uses variants of the same underlying task (SaaS expansion analysis for analytical, equivalent scenarios for creative). With only 2 task families, the generalizability of any finding is limited. A single outlier task could dominate the results.

---

## Summary of Required Actions Before Interpreting Main Batch

### CRITICAL (must fix before any paradox claims)

1. **Replace or normalize the quality metric.** The current BT scoring produces quality_score values that are mechanically tied to the candidate pool size. Options:
   - **(a) Fixed external baseline:** Compare the consensus output against a fixed solo-agent baseline (e.g., from Block 0) using the same judges. This makes the comparison scale-invariant.
   - **(b) Normalize by null:** Report quality as `(Q_observed - Q_null) / Q_null` where `Q_null = 1/(n+1)`. This preserves existing data but changes the analysis.
   - **(c) Absolute rubric scoring:** Have judges score each output on the task rubric (1-5 scale) independently, rather than comparing candidates pairwise.
   
   Option (a) is strongest; option (b) is the minimum-viable fix using existing data. The main batch data is still usable — this is an analysis fix, not a data collection fix.

2. **Verify disagreement level monotonicity.** Before analyzing the Disagreement Dividend curve, confirm that levels actually produce increasing divergence. If they don't, RQ1 analysis is invalid.

### IMPORTANT (should fix for paper quality)

3. **Improve the consensus selection mechanism.** The "debate_then_vote" mechanism selects one agent's output, which provably cannot exceed the best individual agent's quality in the BT framework. Consider a synthesis step where a final agent combines the best elements from all revised outputs.

4. **Report Block 0 kappa separately from multi-agent block kappa.** The aggregate kappa is misleading because Block 0 trivially inflates it.

5. **Expand task diversity.** Two task families provide minimal generalizability. Even 4-6 diverse tasks per family would substantially strengthen claims.

---

## What the Pilot Data Actually Supports

Setting aside the BT artifact, the pilot does contain real signal:

1. **Multi-agent debate produces genuine deliberation.** The revision phase shows substantive intellectual engagement. This is a positive finding about the mechanism.
2. **Flat topology consensus improves over individual agents.** After normalization, Block 1 shows positive consensus value-add, especially at n=3.
3. **Quorum topology consensus fails to improve.** After normalization, Block 4 shows zero or negative consensus value-add. The debate helps, but the vote throws away the gains.
4. **Agent outputs are high-quality.** Substantive, well-structured, and responsive to task requirements.
5. **The interesting paper is about MECHANISM quality, not a paradox of SCALE.** Why does flat consensus add value but quorum consensus doesn't? What is lost in the vote step? This is a well-motivated mechanism design question that the data can actually answer.

---

*Review prepared independently. I have no prior relationship with the authors.*

**Verdict: [CONDITIONAL GO]**

Conditions: (1) Adopt normalized or scale-invariant quality metric before interpreting paradox results. (2) Verify disagreement level ordering. The main batch run is fine to proceed — the data collection infrastructure is sound. The issue is entirely in the analysis pipeline, not the experimental runs.
