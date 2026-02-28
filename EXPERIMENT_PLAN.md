# Experiment Plan (Final, T-025)

## Study

**When Agents Disagree: Quorum-Based Consensus and Adaptive Orchestration Topology for Multi-Agent LLM Pipelines**

This is the final pre-run plan for the MacBook execution handoff.

---

## 1) Research Questions and Hypotheses

### RQ1 — Disagreement Dividend
**Question:** Does moderate disagreement outperform near-homogeneous agreement?

- **H1:** Quality vs disagreement level follows an inverted-U.
- **Primary test:** Quadratic regression `Q ~ beta0 + beta1*d + beta2*d^2`; test `beta2 < 0`.

### RQ2 — Minimum Viable Quorum (MVQ)
**Question:** How does threshold-attainment change with agent count?

- **H2:** Probability of meeting quality threshold rises from n=1 to n=3 and saturates near n=5.
- **Primary test:** Logistic regression of `P(Q >= theta)` on `n`; estimate `n*` per condition.

### RQ3 — Quorum Paradox
**Question:** Can adding a third agent reduce quality in some settings?

- **H3:** Within-task quality at n=3 can dip below n=2 for paradox conditions.
- **Primary test:** Paired t-test (or Wilcoxon signed-rank if non-normal) on `Q(n=3) < Q(n=2)`.

### RQ4 — Topology x Consensus Interaction
**Question:** Do topology and consensus interact non-additively?

- **H4:** Interaction effect is significant for quality outcomes.
- **Primary test:** Two-way ANOVA interaction F-test (`topology * consensus`).

---

## 2) Final Matrix Changes Implemented

1. **Block 5 removed** (`block5_interaction_probe` deleted) as redundant with Block 2.
2. **Block 4 reps increased from 4 -> 6** for stronger paradox evidence.
3. **Homogeneous-strong control retained in Block 4** (`homogeneous_opus`, n=2/3/5).
4. **Phased execution added** (`pilot`, `full`, `all`) in `scripts/run_experiments.py`.
5. **Cost guardrail added**: `--max-cost` (default `$1500`), pause + warning in `progress.json`.
6. **Google model pins added**:
   - `gemini-2.5-pro-preview-05-06`
   - `gemini-2.0-flash-001`
   - `gemini-1.5-pro-001`
7. **Contamination-prone tasks replaced**:
   - `analytical_01_*` now novel incident-triage deduction puzzle
   - `creative_08_*` now novel phosphorescence lighthouse explanation prompt

---

## 3) Phased Execution Plan

## Phase A — Pilot (`--phase pilot`)

Runs:
- all of `block0_calibration`
- 20 runs sampled from `block1_disagreement_dividend`
- 20 runs sampled from `block4_quorum_paradox`

Produces:
- `results/pilot_manifest.json`
- `results/pilot_report.json` with:
  - mean inter-judge kappa
  - disagreement rate by level
  - preliminary paradox signal (`Q_n3 - Q_n2`)
  - cost so far
  - GO/NO-GO recommendation

GO rule:
- `kappa > 0.4` **and** disagreement levels separable.

## Phase B — Full (`--phase full`)

Runs the full matrix **excluding pilot-completed runs**.

## Phase C — All (`--phase all`)

Runs the entire matrix from scratch.

---

## 4) Block Summary (Current)

| Block | Purpose | Runs |
|---|---|---:|
| block0_calibration | Single-agent calibration | 192 |
| block1_disagreement_dividend | Multi-agent disagreement sweep | 480 |
| block1_best_of_n_baseline | Cost-matched baseline for Block 1 | 480 |
| block2_topology_comparison | Topology x consensus at controlled disagreement | 768 |
| block3_mvq_curves | MVQ threshold curves | 192 |
| block4_quorum_paradox | Paradox probes + homogeneous-strong controls (6 reps) | 576 |
| block4_best_of_n_baseline | Cost-matched baseline for Block 4 | 192 |
| **Total** |  | **2,880** |

---

## 5) Cost Estimate Snapshot

Generated with:
```bash
python scripts/estimate_cost.py --phase all
python scripts/estimate_cost.py --phase pilot
python scripts/estimate_cost.py --phase full
```

Observed estimates:

- **Phase all:** `$960.19` (2,880 runs)
- **Phase pilot:** `$31.28` (232 runs)
- **Phase full:** `$928.91` (2,648 runs)

Example CLI output (`--phase all`):

```text
Phase: all
Estimated runs: 2880
Estimated agent calls: 12288
Estimated judge calls: 110784
Estimated API calls total: 123072
Estimated total cost (USD): $960.19
```

> Note: runtime spend can exceed static estimate due to response-length variance and retries. Use `--max-cost` guardrail in live runs.

---

## 6) Output & Monitoring

During runs:
- `results/progress.json`
- `results/cost_log.jsonl`
- `results/run_experiments.log`
- `results/<block_id>/run_*.json`

Pilot-specific:
- `results/pilot_report.json`

After completion:
- `results/analysis/run_metrics.csv`
- `results/analysis/summary_table.csv`
- `results/analysis/*.pdf` and `*.png`

---

## 7) Known Limitations

1. **Only 2 model providers (Anthropic + Google):** provider-specific effects cannot be fully ruled out.
2. **Only 2 task types (analytical + creative):** findings may not generalize to summarization or code-generation tasks.
3. **Reduced scale vs ideal large-benchmark studies:** some higher-order interactions may remain underpowered.
4. **LLM-as-judge remains imperfect:** despite pairwise design, order checks, panel aggregation, and human-eval support, systematic judge biases may persist.

These limitations are acknowledged explicitly to strengthen interpretability and reviewer trust.

---

## 8) Pre-registered Analysis and Operator Handover

- Statistical preregistration details: **`ANALYSIS_PLAN.md`**
- MacBook execution SOP: **`HANDOVER.md`**
