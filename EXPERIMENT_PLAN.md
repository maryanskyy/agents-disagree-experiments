# Experiment Plan: Quorum-Based Multi-Agent LLM Orchestration (Revised)

## 1) Research Questions and Hypotheses

### RQ1 — Disagreement Dividend
**Question:** Is there a non-zero disagreement level that improves aggregate output quality versus near-identical agents?

- **H1:** Quality follows an inverted-U curve over disagreement level; moderate disagreement outperforms level-1 homogeneity.
- **Falsification criterion:** If quality is monotonic decreasing with disagreement (or flat within confidence bounds), H1 is rejected.

### RQ2 — Topology Effect
**Question:** Do flat, hierarchical, quorum, and pipeline topologies differ significantly in quality-cost tradeoffs?

- **H2:** Quorum/pipeline topologies improve quality on analytical tasks at moderate disagreement, with higher cost.
- **Falsification criterion:** If topology differences vanish after cost-aware normalization, H2 is rejected.

### RQ3 — MVQ Behavior
**Question:** Is there a minimum viable quorum size where quality-threshold attainment sharply improves?

- **H3:** Threshold attainment probability rises from n=1 to n=3 and saturates near n=5.
- **Falsification criterion:** If attainment does not improve with quorum size, H3 is rejected.

### RQ4 — Quorum Paradox
**Question:** Can adding weak agents to one strong agent degrade overall quality at specific n (e.g., n=3)?

- **H4:** In asymmetric (1 strong + N-1 weak) setups, a non-monotonic dip occurs around n=3.
- **Falsification criterion:** If quality is monotonic non-decreasing as weak agents are added, H4 is rejected.

---

## 2) Core Methodology Changes Implemented

1. **Pairwise LLM-as-judge** with A/B and B/A orderings.
2. **Bradley-Terry aggregation** for global quality scores.
3. **3-judge panel** with per-judge scores + inter-rater reliability (Cohen's kappa).
4. **Judge pool separated from agent pool** with per-run exclusion of agent models.
5. **Semantic disagreement metric** (`1 - mean pairwise cosine similarity`) via sentence embeddings.
6. **Best-of-N baseline** added for Block 1 and Block 4.
7. **Disagreement confound fix** (temperature and prompt factors disentangled by level design).
8. **Intermediate debate round logging** (`debate_rounds`) in run outputs.
9. **Pipeline topology** added.
10. **Homogeneous-strong controls** added to Block 4.
11. **Human eval sheet generation** for disagreement/paradox/random-sample runs.

---

## 3) Disagreement Level Design (Confound-Fixed)

- **Level 1:** same prompt, temp=0.3 (baseline)
- **Level 2:** same prompt, temp=0.7 (temperature-only)
- **Level 3:** perspective prompts, temp=0.3 (prompt-only)
- **Level 4:** perspective prompts, temp=0.7 (both factors)
- **Level 5:** perspective prompts + model-mix framing, temp=0.9 (maximum)

---

## 4) Blocks and Run Counts

Manifest counts from current matrix:

| Block | Purpose | Runs |
|---|---|---:|
| block0_calibration | Single-agent calibration | 192 |
| block1_disagreement_dividend | Multi-agent disagreement sweep | 480 |
| block1_best_of_n_baseline | Cost-matched baseline for Block 1 | 480 |
| block2_topology_comparison | 4 topologies × 3 consensus, reps=2 | 768 |
| block3_mvq_curves | MVQ curves, reps=3, thresholds post-hoc | 192 |
| block4_quorum_paradox | Asymmetric + homogeneous-strong controls | 384 |
| block4_best_of_n_baseline | Cost-matched baseline for Block 4 | 192 |
| block5_interaction_probe | Interaction probe with pipeline, reps=2 | 768 |
| **Total** |  | **3,456** |

Latest dry estimate from `python scripts/estimate_cost.py`:
- **Estimated total cost:** **$1,147.08**
- **Estimated API calls:** **158,656** (14,848 agent + 143,808 judge calls)

---

## 5) Statistical/Design Notes

- Block 2 and Block 5 replication increased to **2**.
- Block 3 replication increased to **3**.
- Block 3 threshold runs are no longer multiplied in execution; thresholds are post-hoc analyses.
- Block 4 now includes both:
  - `paradox_strong_weak`
  - `homogeneous_opus` controls at n=2,3,5.

---

## 6) Evaluation Outputs per Run

Each run now stores:

- `evaluation.quality_score` (BT score for final output)
- `evaluation.selected_per_judge_scores`
- `evaluation.judge_panel`:
  - panel models
  - global BT scores
  - per-judge BT scores
  - inter-rater reliability
  - pairwise records
- `evaluation.disagreement`:
  - semantic similarity
  - disagreement rate
  - vote entropy
  - lexical diversity
- `debate_rounds` (where applicable)
- `evaluation.human_review` flag + sheet path

---

## 7) Human Validation Workflow Support

Sheets are auto-generated under `results/human_eval/` for:

1. judge-disagreement cases,
2. all Block 4 paradox cases,
3. deterministic random 15% sample.

Each sheet contains blind/anonymized outputs, task prompt/rubric, and score fields for manual raters.

---

## 8) Success/Failure Criteria

- **RQ1 success:** non-trivial d* with better quality than level-1 baseline.
- **RQ2 success:** meaningful topology separation with cost-aware interpretation.
- **RQ3 success:** threshold-attainment curve with clear knee behavior.
- **RQ4 success:** reproducible dip in asymmetric conditions, contrasted against homogeneous-strong controls.

Inconclusive evidence is reported as inconclusive.
