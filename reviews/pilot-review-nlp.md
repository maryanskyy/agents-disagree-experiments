# Pilot Review: Evaluation Methodology Audit

**Reviewer:** Dr. Priya Sharma (NLP/LLM Evaluation)  
**Date:** 2026-02-28  
**Scope:** 284 pilot runs (240 Block 0, 22 Block 1, 22 Block 4), 86 human_eval files, pilot_report.json  
**Previous Review:** review-nlp.md (pre-experiment methodology review)

---

## Verdict: **FUNDAMENTALLY FLAWED**

The evaluation pipeline has five independently fatal defects. Any ONE of these would require recalibration. Together, they render the pilot data scientifically unusable for the paper's claims. The main batch, if it uses the same pipeline, will produce the same flawed data at 10x the cost.

**Do not proceed with the main batch until all Critical issues below are resolved. Stop the batch if it is already running.**

---

## 1. Judge Agreement Reality Check

### The Headline Number Is Fabricated (by methodology, not by malice)

The pilot report claims **kappa = 0.934**. This number is **statistically real but scientifically meaningless**.

**How it's computed:** The aggregate kappa is calculated across ALL 284 runs. 240 of these (84.5%) are Block 0 calibration runs, where a single agent's output is compared against an identical copy of itself labeled "final_consensus." All three judges correctly recognize identical text as identical and vote "tie." This produces kappa = 1.000 for every Block 0 run. The aggregate is overwhelmed by trivial agreement on trivial comparisons.

**What the actual agreement looks like in multi-agent blocks:**

| Block | Runs | Mean kappa | Min kappa | Max kappa |
|-------|------|--------|-------|-------|
| Block 0 (calibration) | 240 | **1.000** | 1.000 | 1.000 |
| Block 1 (disagreement) | 22 | **0.537** | 0.026 | 1.000 |
| Block 4 (quorum paradox) | 22 | **0.576** | 0.333 | 1.000 |

kappa ~ 0.54 is "moderate agreement" on the Landis & Koch scale. This is below the methodology's own acceptance threshold of kappa > 0.60 for "substantial agreement." The pilot report's GO decision was based on the inflated aggregate passing the 0.4 threshold.

### Pairwise Agreement Audit (264 comparisons across Blocks 1 & 4)

| Pattern | Count | Percentage |
|---------|-------|------------|
| All 3 judges agree | 150 | 59.5% |
| 2-1 split | 113 | 39.4% |
| All different | 1 | 0.4% |

But this masks the real problem: the "agreement" is overwhelmingly agreement to call everything a tie.

---

## 2. The Two-Judge Collapse: GPT-4o and Gemini Are Not Judging

**This is the single most important finding in this review.**

Across all 264 pairwise comparisons in the multi-agent blocks:

| Judge | Tie Rate | Differentiated |
|-------|----------|---------------|
| **claude-sonnet-4-6** | 50.8% (134/264) | **49.2% (130/264)** |
| **gemini-3.1-pro-preview** | 84.5% (223/264) | 15.5% (41/264) |
| **gpt-4o** | **92.8% (245/264)** | 7.2% (19/264) |

**GPT-4o votes "tie" on 92.8% of all pairwise comparisons.** It is not functioning as a judge. It is rubber-stamping everything as equivalent. Gemini is only marginally better at 84.5% ties.

**Consequence:** The Bradley-Terry scores are effectively determined by a single judge (Claude-Sonnet). When only one judge differentiates and two don't, the "3-judge panel" provides no methodological benefit. There is no genuine inter-rater triangulation. The BT-sigma framework, which was supposed to weight judges by discriminative ability, would correctly assign near-zero weight to GPT-4o and Gemini -- producing a single-judge system with extra cost.

**Possible explanations:**
1. The evaluation prompt may be insufficiently directive for GPT-4o and Gemini, allowing them to default to "tie"
2. GPT-4o and Gemini may have stronger overrating tendencies that collapse quality differences
3. The pairwise comparison framing may interact differently with these models' training
4. The outputs being compared may genuinely be close in quality (but see Finding 3 below -- many comparisons involve empty text, which should NOT produce ties)

**The fact that GPT-4o and Gemini call "tie" between substantive text and EMPTY strings (see Section 3) indicates the judges are not reading the outputs at all, or the evaluation prompt is broken for these models.**

---

## 3. GPT-5.2 Produces Empty Output 34.7% of the Time

**This finding is catastrophic for data quality.**

Agent output completeness by model (Blocks 1 & 4, 126 total agent outputs):

| Model | Empty (0 chars) | Short (<500 chars) | Complete | Total |
|-------|-----------------|---------------------|----------|-------|
| **claude-opus-4-6** | 0 (0%) | 0 (0%) | 53 (100%) | 53 |
| **gpt-5.2** | **17 (34.7%)** | 0 (0%) | 32 (65.3%) | 49 |
| **gemini-2.5-pro** | 0 (0%) | **7 (63.6%)** | 4 (36.4%) | 11 |
| gemini-2.5-flash | 0 (0%) | 0 (0%) | 8 (100%) | 8 |
| claude-haiku-4-5 | 0 (0%) | 0 (0%) | 5 (100%) | 5 |

Additionally, Block 0 has 24/240 runs (10%) with all-empty outputs.

**What this means:**
- Over a third of GPT-5.2 outputs are literally nothing. The evaluation then compares Claude's 8,000-character response against an empty string.
- Gemini-2.5-pro is systematically truncated to ~250-440 characters on tasks requiring 800+ words. These are not complete outputs.
- Run `run_020d7a1bc18f` (Block 4): BOTH agent outputs are 0 chars, consensus text is 0 chars. All judges give ties. BT scores: all 0.333. This is a null observation being treated as valid data.
- Run `run_bdb623aefffe` (Block 1): ALL THREE gpt-5.2 agents produce 0 chars. All ties. This run contributes to the "quality at disagreement level X" calculation.

**Impact on every analysis dimension:**
1. **BT scores** are meaningless when comparing actual text against empty strings
2. **Judge agreement** appears high because it's easy to "agree" when comparing nothing vs. nothing
3. **The Quorum Paradox signal** (n=3 worse than n=2) may be entirely an artifact -- more agents = more chances of including a broken GPT-5.2 instance that produces empty output, pulling down the average quality
4. **Cross-model quality comparisons** are confounded -- Claude "wins" because it's the only model that reliably produces output

---

## 4. Self-Preference Bias: Cannot Be Assessed

The experiment design correctly identified self-preference as a confound and assigned judges from the same model families as agents. The intent was to test whether Claude-judge prefers Claude-agent outputs, etc.

**Cross-family comparison data (when one judge differentiates between agents from different families):**

| Judge Family | Picked Own Family | Picked Other | Total Opportunities | Self-Preference Rate |
|-------------|-------------------|--------------|---------------------|---------------------|
| Anthropic (claude-sonnet) | 32 | 15 | 47 | **68.1%** |
| Google (gemini-3.1-pro) | 5 | 19 | 24 | 20.8% |
| OpenAI (gpt-4o) | 1 | 11 | 12 | 8.3% |

**Superficially alarming:** Claude-Sonnet picks Anthropic agents 68.1% of the time.

**But completely confounded:** Claude agents are the only ones reliably producing complete output. Claude-Sonnet preferring Claude agents over empty GPT-5.2 outputs is not self-preference bias -- it is correct judgment. The self-preference question **cannot be answered** with this data because the output quality variable is perfectly confounded with model family.

GPT-4o's 8.3% self-preference rate is meaningless because it only differentiated 12 times total across all cross-family comparisons. Gemini's data is similarly sparse.

**To actually test self-preference, you need all agents producing substantive output.** Until the empty output problem is solved, this analysis is impossible.

---

## 5. BT Score Calibration

### Quality Scores Are Not Measuring Quality

**Block 0:** All 240 runs have quality_score = 0.500. Mathematically correct (BT with one entity = 0.5 by definition). Empirically useless. These provide zero information.

**Block 1 distribution (22 runs):**
- Range: 0.167 - 0.500
- Mean: 0.340
- The maximum possible BT score (when consensus = best agent and all judges agree) is 0.500

**Block 4 distribution (22 runs):**
- Range: 0.164 - 0.400
- Mean: 0.273

**What does a score of 0.25 vs 0.35 mean?**

In practice, a score of 0.35 means: "Claude-Sonnet slightly preferred this agent's output, while GPT-4o and Gemini said everything was a tie." A score of 0.25 means: "Claude-Sonnet also said tie, or the comparison involved one complete output vs. one empty/truncated output where even Claude-Sonnet couldn't differentiate meaningfully."

The scores are compressed because:
1. Two judges contribute almost no signal (tie bias)
2. The denominator in BT includes all candidates including empty ones
3. With n candidates, the uniform distribution is 1/n; actual scores barely deviate from this

**BT is the right framework but is being fed garbage input.** It faithfully computes rankings from judgments, but when judgments are 90% ties from 2 of 3 judges and the underlying outputs include empty strings, the rankings are meaningless.

---

## 6. Qualitative Audit: Do Judge Rankings Match Actual Quality?

### Case 1: run_105eb6452678 (Block 1, creative task)

- **agent_0 (claude-opus-4-6):** 8,894 chars. Detailed, creative traffic interventions with cost analysis, evidence reasoning, and mechanism design. Complete, substantive response.
- **agent_1 (gpt-5.2):** 0 chars. Empty.
- **agent_2 (gemini-2.5-pro):** 383 chars. Starts promisingly but is truncated mid-sentence.

**BT scores:** agent_0 = 0.456, agent_2 = 0.077, agent_1 = 0.011

**My assessment:** Rankings are directionally correct (something > truncated > nothing). But the absolute scores are meaningless -- this is not a meaningful quality comparison because two of three outputs are defective. The task is asking me to evaluate a restaurant by comparing a full meal against an empty plate and a single bite.

### Case 2: run_040a9b089fee (Block 1, analytical task)

- All three agents (all gpt-5.2) produced output for the human_eval file (3 candidates shown, each 400-600+ tokens).
- In the main run: Claude-Sonnet differentiated (saw agent_1 as best), GPT-4o and Gemini gave all ties.
- Kappa = 0.333

**My assessment of the human_eval candidates:** All three candidates (C1, C2, C3) are high quality. C1 is the most comprehensive with 6 composite conclusions and 7 non-follow conclusions. C2 is more structured with formal notation. C3 adds the valuable "denying the antecedent" observation. These ARE genuinely similar in quality. Ties are defensible here -- but then the entire run produces no useful signal for BT, and kappa drops accordingly.

**Conclusion:** In the rare cases where outputs are substantive and comparable, the judges' tie-heavy behavior may be partially justified. But this means the evaluation pipeline extracts signal only from runs where outputs are dramatically unequal (i.e., some are empty/broken). The "quality measurement" is actually measuring "which agents didn't crash."

---

## 7. Human Eval Infrastructure

### Flagging Works, But No Evaluations Have Been Completed

86 files have been flagged for human review:

| Source | Count | Notes |
|--------|-------|-------|
| random_sample_15pct | 49 | Block 0: 43, Block 1: 1, Block 4: 4, other: 2 |
| judge_disagreement | 34 | Block 1: 15, Block 4: 14, other: 5 |
| block4_paradox_case | 23 | Block 4: 22, other: 1 |

**Sampling rates by block:**
- Block 0: 43/240 = 17.9% (target 15%) -- Acceptable
- Block 1: 15/22 = 68.2% (driven by judge_disagreement) -- Acceptable
- Block 4: 22/22 = 100% (all flagged as paradox cases) -- Correct per methodology

**The good:** Flagging logic works. Sampling rates are appropriate. Block 4 gets 100% review as required.

**The bad:**
1. None of the 86 files contain actual human ratings. They have `score_template` with instructions but no `scores` field. Human evaluation has not happened.
2. 15 of the 164 candidates in human_eval files have empty or near-empty text. Human raters will be asked to compare nothing against nothing.
3. Files from blocks outside the pilot scope appear (block1_best_of_n_baseline: 2, block2_topology_comparison: 3, block4_best_of_n_baseline: 1) -- suggesting main batch contamination or scope bleed.

---

## 8. Rubric Adherence

### Rubrics Are Not Being Used

The evaluation methodology defines detailed rubrics with per-dimension scoring:
- Analytical: Correctness (0.30), Reasoning Depth (0.25), Evidence Quality (0.20), Logical Coherence (0.15), Conciseness (0.10)
- Creative: Originality (0.25), Coherence (0.20), Engagement (0.20), Craft (0.20), Task Fidelity (0.15)

**The pairwise records contain exactly four fields:** `candidate_i`, `candidate_j`, `per_judge` (left/right/tie), `majority_winner`.

There is **no per-dimension scoring, no rubric criteria evaluation, no rationale text, and no evidence that judges are using the rubric at all.** The evaluation is purely holistic pairwise comparison -- "which is better overall?" without structured criteria.

This means:
1. We cannot determine WHICH quality dimensions drive the rankings
2. We cannot test whether consensus helps reasoning but hurts creativity (a key theoretical prediction)
3. We cannot diagnose whether the Quorum Paradox operates through coherence degradation, creativity suppression, or correctness erosion
4. The rubric design effort in EVALUATION_METHODOLOGY.md is completely wasted

---

## 9. Position Randomization

### No Evidence of Bidirectional Evaluation

The methodology requires evaluating each pair in BOTH orderings to control for position bias. The pairwise records show only one evaluation per pair. There is no `position_orders` field, no consistency check, and no detection of position-sensitive items.

Given that my previous review explicitly flagged this as incomplete (Section 2.2 of review-nlp.md: "Position Randomization: Half-Implemented"), the omission is concerning. Position bias is estimated to affect 15-25% of pairwise judgments (Wang et al. 2023). With no bidirectional check, the direction and magnitude of this bias are unknown.

---

## 10. Consensus Mechanism: Selection, Not Synthesis

A factual observation: the consensus mechanism (`debate_then_vote`) does not produce new synthesized text. It selects one agent's output verbatim. The `consensus.selected_text` is character-for-character identical to the selected agent's output in every single run (44/44 checked).

This means:
- `final_consensus` quality = quality of the selected individual agent
- The BT score for `final_consensus` always equals the BT score for the selected agent
- There is no "consensus quality" distinct from "best-individual quality"
- The Disagreement Dividend, as currently measured, is: "does the debate-and-vote mechanism successfully identify the best individual output?" -- NOT "does multi-agent collaboration produce higher-quality output?"

This is not necessarily wrong, but it changes the paper's claims fundamentally. The narrative of "diverse agents synthesizing superior output through deliberation" is not what the data measures.

---

## 11. The Quorum Paradox Signal: Almost Certainly an Artifact

The pilot reports: Q(n=2) = 0.313 > Q(n=3) = 0.228, delta = -0.085.

Given:
1. GPT-5.2 produces empty output 34.7% of the time
2. With n=3 agents, the probability of at least one agent being GPT-5.2 is higher than with n=2
3. Empty outputs drag down BT scores mechanically
4. The consensus mechanism selects one agent's output -- if it accidentally selects the empty one, quality = 0 equivalent

The paradox disappears under the null hypothesis: "adding agents increases the chance of including a broken instance." This is not a paradox about multi-agent deliberation dynamics -- it is a software bug manifesting as a statistical pattern.

**To test this:** Filter to runs where ALL agents produced non-empty output (>500 chars). Recompute Q(n=2) and Q(n=3). If the paradox vanishes, it was an artifact.

---

## 12. Summary of Critical Defects

| # | Defect | Severity | Impact |
|---|--------|----------|--------|
| 1 | GPT-5.2 produces empty output 34.7% of the time | **CRITICAL** | Invalidates all cross-model comparisons, BT scores, and the quorum paradox signal |
| 2 | GPT-4o and Gemini judges don't differentiate (93%/85% tie rate) | **CRITICAL** | 3-judge panel is effectively a 1-judge system; no inter-rater triangulation |
| 3 | Aggregate kappa inflated by 84.5% trivial Block 0 comparisons | **CRITICAL** | GO/NO-GO decision was based on misleading statistic |
| 4 | No rubric-based dimension scoring | **MAJOR** | Cannot analyze which quality dimensions are affected by multi-agent setup |
| 5 | No bidirectional position evaluation | **MAJOR** | Position bias uncontrolled, estimated 15-25% of judgments affected |
| 6 | No human evaluations completed | **MAJOR** | Methodology requires 100% human validation for paradox cases; 0% done |
| 7 | Gemini-2.5-pro systematically truncated | **MAJOR** | Cross-model comparisons are confounded by output completeness |
| 8 | Consensus is selection, not synthesis | **SIGNIFICANT** | Changes the nature of claims the paper can make |
| 9 | Human eval files from non-pilot blocks | **MINOR** | Suggests batch isolation may be leaking |

---

## 13. Recommendations

### Immediate (before any further analysis)

1. **STOP the main batch** if it uses the same pipeline. The empty output problem means you're burning money on void comparisons.

2. **Diagnose and fix the GPT-5.2 empty output problem.** This could be a timeout, a rate limit, a prompt issue, or an API error being silently swallowed. Check `output_tokens` and `latency_ms` for the empty runs.

3. **Diagnose and fix the Gemini-2.5-pro truncation.** 250-440 chars on 800+ word tasks suggests either token limit misconfiguration or premature stop.

4. **Diagnose GPT-4o and Gemini judge tie bias.** Inspect the actual evaluation prompts sent to each judge. Test whether the prompt phrasing is causing these models to default to "tie." Try a more directive prompt that explicitly discourages ties when outputs differ in length by >5x.

### Before Re-running

5. **Implement rubric-based dimension scoring** alongside holistic pairwise comparison. This was in the methodology; implement it.

6. **Implement bidirectional position evaluation.** Evaluate each pair in both orderings; flag inconsistencies.

7. **Recompute kappa EXCLUDING Block 0** in all reporting. Report Block 0 kappa separately as a sanity check, not as the primary reliability metric.

8. **Filter the quality analysis to runs where ALL agents produced complete output** (>500 chars). Report filtered and unfiltered results separately.

9. **Add output completeness as a tracked metric.** Flag and quarantine runs with empty or severely truncated outputs.

### For the Paper

10. **Do not report kappa = 0.934.** Report per-block kappa with Block 0 excluded from the aggregate.

11. **Consider whether the consensus-as-selection findings are interesting enough for a different paper** than the one currently framed. "Agent-as-judge" (voting to select the best output) is a valid research direction, but it's not "consensus synthesis."

---

## 14. What I Flagged Before vs. What Actually Happened

| My Previous Concern | Status in Pilot Data |
|---|---|
| Single-judge absolute scoring -> should be pairwise BT | **FIXED.** Pairwise BT implemented. |
| Need multi-judge panel (3 models, 2 families) | **FIXED.** 3 judges from 3 families. **BUT** 2 of 3 judges don't differentiate. |
| Need embedding-based semantic similarity for disagreement | Present in data (semantic_pairwise_similarity field exists). Not audited in depth. |
| Need best-of-N baseline | Some best_of_n_baseline files exist in human_eval. Not audited. |
| Need human evaluation pipeline | **PARTIALLY FIXED.** Flagging works, sampling correct. But no evaluations done. |
| Verbosity bias uncontrolled | **STILL UNCONTROLLED.** No length normalization, no length-controlled analysis. |
| Position randomization incomplete | **STILL INCOMPLETE.** No bidirectional evaluation. |
| Rubric sensitivity analysis | **NOT DONE.** Rubrics aren't even being used by judges. |
| Heuristic fallbacks should be guarded | Not applicable to pilot (LLM judges used). |

The team fixed the most critical issue I raised (switching from absolute to pairwise BT) and added the multi-judge panel. This is commendable. But the new implementation introduced new problems (tie bias, empty outputs) that are equally severe.

---

## Closing Statement

I want to be clear: this is not a failing of the theoretical framework or the research questions. The experiment design is ambitious and well-motivated. The evaluation methodology document remains outstanding. The problems are in the execution pipeline -- specifically in API reliability (empty/truncated outputs), judge prompt engineering (tie bias), and metric aggregation (kappa inflation).

These are fixable engineering problems, not fundamental research problems. But they must be fixed before the data can support any conclusions. The pilot's purpose was exactly to catch these issues. The pilot has served its purpose.

---

*Dr. Priya Sharma*  
*Senior Research Scientist, LLM Evaluation*  
*Area Chair, ACL/EMNLP*
