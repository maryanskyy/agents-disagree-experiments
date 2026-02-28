# Experiment Plan (Final, T-026)

## Study

**When Agents Disagree: Quorum-Based Consensus and Adaptive Orchestration Topology for Multi-Agent LLM Pipelines**

This revision upgrades the experiment to a **3-provider setup** and removes the prior 2-provider limitation.

---

## 1) Research Questions and Hypotheses

### RQ1 — Disagreement Dividend
**Question:** Does moderate disagreement outperform near-homogeneous agreement?

- **H1:** Quality vs disagreement follows an inverted-U.
- **Primary test:** Quadratic regression `Q ~ beta0 + beta1*d + beta2*d^2`; test `beta2 < 0`.

### RQ2 — Minimum Viable Quorum (MVQ)
**Question:** How does threshold-attainment change with agent count?

- **H2:** Threshold attainment rises from n=1 to n=3 and saturates by n=5.
- **Primary test:** Logistic regression on `P(Q >= theta)` with `n` and condition terms.

### RQ3 — Quorum Paradox
**Question:** Can adding an agent reduce quality in asymmetric compositions?

- **H3:** In paradox settings, quality at n=3 can dip below n=2.
- **Primary test:** Paired t-test (or Wilcoxon fallback) on within-task deltas.

### RQ4 — Topology x Consensus Interaction
**Question:** Do topology and consensus interact non-additively?

- **H4:** Topology x consensus interaction is significant for quality.
- **Primary test:** Two-way ANOVA interaction F-test.

---

## 2) Model Configuration (Updated)

### Agent Pool (5 models, 3 providers)
- `claude-opus-4-6` (Anthropic, strong)
- `gpt-5.2` (OpenAI, strong)
- `gemini-2.5-pro` (Google, strong, pinned API model)
- `claude-haiku-4-5` (Anthropic, weak/fast)
- `gemini-2.5-flash` (Google, weak/fast)

### Judge Pool (3 models, 3 providers, no overlap with agents)
- `claude-sonnet-4-6` (Anthropic)
- `gpt-4o` (OpenAI)
- `gemini-3.1-pro-preview` (Google)

Properties now satisfied:
- 3 providers in agent pool
- 3 providers in judge pool
- zero model overlap between agent and judge pools
- strong + weak agent tiers for paradox probing

---

## 3) Matrix Changes Implemented

1. **Block 0 calibration expanded:** all 5 agent models solo on all 16 tasks, 3 reps.
2. **Block 1 disagreement dividend expanded:**
   - homogeneous strong (Opus, GPT-5.2, Gemini-Pro)
   - heterogeneous strong cross-provider mix
   - high-disagreement strong+weak composition
3. **Block 2 topology comparison:** uses heterogeneous cross-provider and homogeneous strong groups.
4. **Block 3 MVQ:** agent counts `{1,2,3,5}` with full pool coverage.
5. **Block 4 paradox:** includes
   - `paradox_strong_weak` (1 Opus + N-1 Haiku)
   - `paradox_cross_provider` (1 GPT-5.2 + N-1 Gemini-Flash)
   - `paradox_homogeneous_strong` (Opus-only control)
   - `paradox_homogeneous_cross` (GPT-5.2-only control)
   - 6 reps each
6. **Best-of-N baselines:** use `gemini-2.5-pro` (cheapest strong model).

---

## 4) Block Summary (Current)

| Block | Purpose | Runs |
|---|---|---:|
| block0_calibration | Single-agent calibration (all 5 models) | 240 |
| block1_disagreement_dividend | Disagreement sweep + composition effects | 2,400 |
| block1_best_of_n_baseline | Cost-matched baseline for Block 1 | 480 |
| block2_topology_comparison | Topology x consensus comparison | 3,072 |
| block3_mvq_curves | MVQ threshold curves | 1,152 |
| block4_quorum_paradox | Paradox probes + controls (6 reps) | 1,152 |
| block4_best_of_n_baseline | Cost-matched baseline for Block 4 | 192 |
| **Total** |  | **8,688** |

---

## 5) Cost Estimate Snapshot

Generated with:
```bash
python scripts/estimate_cost.py --phase all
python scripts/estimate_cost.py --phase pilot
python scripts/estimate_cost.py --phase full
```

Observed estimates:
- **Phase all:** `$3,553.25` (8,688 runs)
- **Phase pilot:** `$31.13` (280 runs)
- **Phase full:** `$3,522.12` (8,408 runs)

Example CLI output (`--phase all`):

```text
Phase: all
Estimated runs: 8688
Estimated agent calls: 37936
Estimated judge calls: 359328
Estimated API calls total: 397264
Estimated total cost (USD): $3,553.25
```

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

1. **Only 2 task types (analytical + creative):** findings may not transfer to broader task classes.
2. **LLM-as-judge is still imperfect:** mitigated by a 3-provider judge panel, pairwise bidirectional checks, and aggregation, but residual bias may remain.
3. **Reduced scale vs very large benchmark campaigns:** some higher-order interactions may remain underpowered despite 5 agent models.

---

## 8) Pre-registered Analysis and Handover

- Statistical preregistration details: **`ANALYSIS_PLAN.md`**
- Operator execution SOP: **`HANDOVER.md`**

