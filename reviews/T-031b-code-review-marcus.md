# T-031b: Code Review — Statistical Validity of Structural Metrics + Export Schema

**Reviewer:** Marcus Rivera (PhD-2, Empirical Track)
**Date:** 2026-02-28
**Files Reviewed:**
- `src/evaluation/structural_quality.py`
- `src/evaluation/corrected_metrics.py`
- `scripts/compute_structural_metrics.py`
- `results/structural_metrics_summary.csv` (all 400 consensus rows + 744 agent rows)
- `src/evaluation/llm_judge.py`
- Sample run JSONs from blocks 0–4

---

## STATISTICAL VALIDITY VERDICT: **NEEDS ADJUSTMENT**

The structural metrics implementation is technically correct and well-engineered. The composite score normalization is the **best available approach** (fixed priors, not data-dependent). However, the composite score has **near-zero correlation with consensus quality** (Spearman ρ = −0.002, p = 0.98), meaning it does not measure what we care about. Several individual metrics show significant task-type confounds that the single set of `ANALYTICAL_NORMS` cannot accommodate. The metrics are useful as **descriptive features** but should not be treated as a quality proxy without task-specific recalibration and explicit acknowledgment of their limitations.

---

## Part A: Statistical Validity Analysis

### A1. Composite Score Normalization: Fixed Priors vs. Data-Dependent

**Finding: GOOD DESIGN — No stationarity problem.**

The code uses `ANALYTICAL_NORMS` — fixed prior constants for each metric:

```python
ANALYTICAL_NORMS = {
    "mtld": (80.0, 25.0, "higher"),
    "readability_fk_grade": (13.0, 3.0, "target"),
    "coherence_mean": (0.45, 0.20, "higher"),
    "prompt_relevance": (0.55, 0.20, "higher"),
    "connective_density": (1.20, 0.60, "higher"),
    "word_count": (450.0, 220.0, "target"),
    "repetition_rate": (0.08, 0.08, "lower"),
}
```

These are **not** computed from the current dataset. Adding new runs will **not** change existing composite scores. This is the correct approach for reproducibility and cross-study comparability. The z-scores are pinned to fixed reference norms.

**Issues found:**

1. **Norm name is misleading.** Called `ANALYTICAL_NORMS` but applied uniformly to both analytical and creative tasks. Creative text has fundamentally different distributions (see A5). **Recommendation:** Either rename to `DEFAULT_NORMS` or implement separate `CREATIVE_NORMS`.

2. **Connective density norm is far from reality.** The prior mean is 1.20 with std 0.60, but observed data shows mean = 0.124, std = 0.140. The norm is ~8× higher than the data. This means connective_density z-scores are universally very negative (≈ −1.8), contributing a large constant drag on all composite scores. This doesn't differentiate between runs.

3. **Word count target (450) is reasonable** but the data shows bimodal behavior — short outputs (< 100 words) from truncated/empty responses and long outputs (800–1400 words) from full generations. The "target" mode penalizes both extremes, which may be appropriate but should be documented.

4. **Clipping at ±3.0** is sensible and prevents outlier domination. Good defensive practice.

**Recommendation:** Recalibrate norms from the calibration block (block0, n=240 consensus) to establish empirically grounded reference values, while keeping them fixed for all subsequent analysis. Document the calibration source.

### A2. Metric Independence: Correlation Matrix

**Pearson Correlation Matrix (n = 355 non-empty consensus outputs):**

|                    | mtld   | fk_grade | coher  | relev  | connec | wcount | repet  |
|--------------------|--------|----------|--------|--------|--------|--------|--------|
| **mtld**           | 1.000  | 0.283    | −0.327 | −0.207 | 0.025  | 0.391  | −0.263 |
| **fk_grade**       | 0.283  | 1.000    | 0.249  | 0.373  | 0.250  | 0.096  | −0.037 |
| **coherence**      | −0.327 | 0.249    | 1.000  | 0.303  | 0.071  | 0.058  | 0.263  |
| **relevance**      | −0.207 | 0.373    | 0.303  | 1.000  | 0.030  | 0.115  | 0.238  |
| **connective**     | 0.025  | 0.250    | 0.071  | 0.030  | 1.000  | 0.002  | 0.127  |
| **word_count**     | 0.391  | 0.096    | 0.058  | 0.115  | 0.002  | 1.000  | 0.230  |
| **repetition**     | −0.263 | −0.037   | 0.263  | 0.238  | 0.127  | 0.230  | 1.000  |

**Verdict: No redundancy concerns.** The highest absolute correlation is **mtld ↔ word_count at r = 0.39**, which is expected (longer texts tend to have higher lexical diversity measures, though MTLD is designed to be length-independent). No pair exceeds |r| = 0.5, let alone 0.7.

**Moderate associations worth noting:**
- `fk_grade ↔ prompt_relevance` (r = 0.37): More complex writing tends to be more on-topic — likely because longer analytical outputs hit both higher grade level and better relevance.
- `mtld ↔ coherence_mean` (r = −0.33): Higher lexical diversity correlates with lower adjacent-sentence coherence. This makes sense — vocabulary-rich text may shift topics between sentences more aggressively.
- `coherence ↔ prompt_relevance` (r = 0.30): Coherent outputs tend to stay on-topic.

**All 7 metrics are measuring sufficiently different constructs.** Equal weighting in the composite is defensible.

### A3. Metric Distributions: Normality and Implications for Z-Scores

| Metric                | Mean    | Std     | Median  | Skew   | Kurtosis | Shapiro p |
|-----------------------|---------|---------|---------|--------|----------|-----------|
| mtld                  | 138.98  | 64.87   | 133.18  | 0.53   | −0.35    | < 0.001   |
| readability_fk_grade  | 13.59   | 5.25    | 12.73   | 0.97   | 1.65     | < 0.001   |
| coherence_mean        | 0.290   | 0.111   | 0.275   | **2.48** | **12.67** | < 0.001   |
| prompt_relevance      | 0.597   | 0.173   | 0.627   | −0.61  | −0.30    | < 0.001   |
| connective_density    | 0.124   | 0.140   | 0.088   | **2.75** | **12.52** | < 0.001   |
| word_count            | 713.93  | 481.28  | 834.00  | −0.07  | **−1.58** | < 0.001   |
| repetition_rate       | 0.035   | 0.065   | 0.012   | **5.18** | **44.67** | < 0.001   |

**Verdict: Most metrics are NOT normally distributed.** Z-score interpretation is approximate at best.

**Critical concerns:**
1. **repetition_rate** is extremely right-skewed (skew = 5.18, kurtosis = 44.67). Most runs have near-zero repetition, with a heavy tail. Z-scoring this is technically valid but the scores are highly compressed in the low range where most data lives.

2. **coherence_mean** and **connective_density** are also heavily right-skewed (skew > 2.4). The coherence has a spike from the 1.0 values (single-sentence outputs that auto-score as perfect coherence).

3. **word_count** is platykurtic (kurtosis = −1.58), almost uniform between 50 and 1400. This is a flat distribution, not peaked.

4. **mtld** and **prompt_relevance** are the closest to normal (moderate skew < 1.0), making z-scores most meaningful for these.

**Recommendation:** For highly skewed metrics (repetition_rate, connective_density), consider rank-based normalization or log-transformation before z-scoring. Alternatively, acknowledge in the paper that the composite is an ordinal-level indicator, not a cardinal score.

### A4. Composite Score Validity: Correlation with Consensus Win Rate

**This is the most important finding in this review.**

| Comparison | Spearman ρ | p-value |
|------------|-----------|---------|
| **composite_score ↔ consensus_win_rate** | **−0.002** | **0.976** |
| composite_score ↔ quality_score (BT) | −0.176 | 0.001 |

**The composite structural score has ZERO correlation with consensus win rate (ρ = −0.002, p = 0.98).** This means the structural metrics composite is measuring something completely orthogonal to how well the consensus output performs against individual agent outputs in pairwise judging.

The negative correlation with BT quality_score (ρ = −0.18, p = 0.001) is small but significant and in the **wrong direction** — higher structural quality is weakly associated with *lower* BT scores.

**Individual metric correlations with consensus_win_rate:**

| Metric | Spearman ρ | p-value | Interpretation |
|--------|-----------|---------|----------------|
| mtld | **0.178** | 0.001 | More diverse vocab → slightly higher win rate |
| readability_fk_grade | 0.100 | 0.061 | Marginal, not significant |
| coherence_mean | −0.102 | 0.056 | Marginal negative (!) |
| prompt_relevance | **−0.162** | 0.002 | **Higher relevance → LOWER win rate** |
| connective_density | −0.059 | 0.268 | Not significant |
| word_count | 0.071 | 0.183 | Not significant |
| repetition_rate | −0.071 | 0.181 | Not significant |

**The paradox:** prompt_relevance is *negatively* correlated with consensus win rate. This likely reflects a confound: in multi-agent runs where consensus is formed, the consensus output may be a synthesis that's less lexically similar to the original prompt (lower cosine similarity) but actually better quality. Conversely, single-agent calibration runs (block0) produce outputs that parrot the prompt vocabulary back, inflating relevance without improving quality.

**Impact assessment:** The composite score is **not a quality proxy**. It captures surface-level structural properties (vocabulary diversity, readability, coherence patterns) that are necessary but nowhere near sufficient conditions for quality. In the paper, we should:
1. Present structural metrics as **descriptive features**, not quality indicators
2. Never use composite_score as a dependent variable for quality comparisons
3. Use consensus_win_rate as the primary quality metric
4. Report structural metrics as supplementary evidence for output characteristics (e.g., "quorum topology produces more lexically diverse outputs")

### A5. Per-Task-Type Behavior

**Analytical vs. Creative comparison (Welch t-tests with Cohen's d):**

| Metric | Analytical (n=178) | Creative (n=177) | Cohen's d | p-value |
|--------|-------------------|------------------|-----------|---------|
| mtld | 123.8 ± 64.0 | 154.2 ± 62.1 | **−0.48** | < 0.001 |
| fk_grade | 15.2 ± 5.3 | 12.0 ± 4.7 | **+0.64** | < 0.001 |
| coherence | 0.321 ± 0.101 | 0.259 ± 0.113 | **+0.58** | < 0.001 |
| relevance | 0.719 ± 0.089 | 0.475 ± 0.150 | **+1.98** | < 0.001 |
| connective | 0.119 ± 0.106 | 0.130 ± 0.167 | −0.08 | 0.466 |
| word_count | 788 ± 485 | 639 ± 465 | **+0.31** | 0.004 |
| repetition | 0.052 ± 0.066 | 0.019 ± 0.060 | **+0.52** | < 0.001 |
| **composite** | −0.503 ± 0.299 | −0.465 ± 0.302 | −0.13 | 0.234 |

**Key findings:**

1. ✅ **MTLD is higher for creative** (d = −0.48): Confirmed expectation. Creative writing uses richer vocabulary.

2. ✅ **Readability grade is higher for analytical** (d = +0.64): Analytical tasks produce more complex, higher-grade-level prose. Confirmed.

3. ⚠️ **Prompt relevance has massive task-type confound** (d = +1.98): Analytical outputs have ~72% prompt relevance vs. ~48% for creative. This is a near-2σ effect — the largest in the entire dataset. Creative tasks naturally produce outputs that diverge from prompt vocabulary (stories, poems don't echo the prompt). This means **relevance is measuring task type, not quality**.

4. ⚠️ **Repetition is higher for analytical** (d = +0.52): Analytical tasks repeat more n-grams (formulaic structure, repeated references to constraints). This doesn't indicate lower quality.

5. ✅ **Composite score shows NO significant difference** (d = −0.13, p = 0.23): The normalization happens to balance out, but this is accidental — the individual metrics differ dramatically.

**Critical problem:** Using `ANALYTICAL_NORMS` for creative tasks systematically penalizes them on relevance (because prompt_relevance is centered at 0.55 but creative outputs average 0.475) and rewards them on MTLD. The effects partially cancel, producing similar composites, but for the *wrong reasons*.

**Recommendation:** Implement separate norm tables for analytical vs. creative tasks, or weight metrics differently by task type. At minimum, document this confound prominently.

### A6. Sample Size Adequacy

**Bootstrap analysis (10,000 iterations, n = 355):**

| Statistic | Value |
|-----------|-------|
| Mean composite | −0.484 |
| 95% CI | [−0.515, −0.453] |
| CI width | 0.062 |
| Bootstrap SE | 0.016 |

**Verdict: Adequate.** The 95% CI width of 0.062 is ~20% of the composite standard deviation (0.301). This is precise enough for group-level comparisons. For detecting moderate effect sizes (d = 0.3) between experimental conditions, power analysis suggests:

- Two-group comparison: n ≈ 88 per group for 80% power at α = 0.05
- With 178 analytical and 177 creative (our largest split), we have excellent power for task-type comparisons
- Within-block comparisons (e.g., block1_disagreement_dividend, n = 53 consensus) have adequate power for detecting d ≥ 0.5

**However:** Per-block sample sizes are very uneven:
- block0_calibration: 240 consensus rows (60% of data)
- block1_disagreement_dividend: 53
- block2_topology_comparison: 50
- block4_quorum_paradox: 38
- block1_best_of_n_baseline: 7
- block3_mvq_curves: 9
- block4_best_of_n_baseline: 3

The smaller blocks (≤ 9 runs) cannot support per-block structural metric analysis at all. Block-level conclusions should be restricted to blocks with n ≥ 30.

---

## Part B: Open-Source Data Export Schema

### B1. Primary Dataset Files

| File | Format | Rows | Use Case |
|------|--------|------|----------|
| `runs.jsonl` | JSON Lines | 1 per run | Full reproducibility; contains all raw data |
| `runs_summary.csv` | CSV | 1 per run | Quick filtering, plotting, statistical analysis |
| `agent_outputs.csv` | CSV | 1 per agent output | Text-level analysis, structural metrics by agent |
| `pairwise_judgments.csv` | CSV | 1 per pairwise comparison | Judge behavior analysis, position bias studies |
| `conditions_summary.csv` | CSV | 1 per experimental condition | Aggregated stats for quick condition comparisons |

### B2. Schema: `runs.jsonl`

Each line is a self-contained JSON object:

```json
{
  "run_id": "string — unique run identifier (hex)",
  "block_id": "string — experimental block name",
  "timestamp_utc": "string — ISO 8601 timestamp",
  "status": "string — 'ok' or error status",

  "config": {
    "task_type": "string — 'analytical' | 'creative'",
    "task_id": "string — unique task identifier",
    "topology": "string — 'flat' | 'hierarchical' | 'quorum' | 'best_of_n'",
    "consensus": "string — 'simple_vote' | 'debate_then_vote' | 'judge_based'",
    "agent_count": "int — number of agents (1–5)",
    "model_assignment": ["string[] — model names assigned to each agent"],
    "disagreement_level": "int — experimental disagreement condition (1–5)",
    "temperature": "float — generation temperature",
    "prompt_strategy": "string — 'identical' | 'perspective_diversity'",
    "repetition": "int — repetition index within condition"
  },

  "task": {
    "title": "string — human-readable task title",
    "prompt": "string — full task prompt text",
    "rubric": ["string[] — evaluation rubric criteria"]
  },

  "outputs": [
    {
      "agent_id": "string — agent identifier",
      "model_name": "string — LLM model name",
      "text": "string — full generated text",
      "input_tokens": "int — prompt token count",
      "output_tokens": "int — generation token count",
      "latency_ms": "float — generation latency in milliseconds",
      "structural_metrics": {
        "mtld": "float",
        "readability_fk_grade": "float",
        "coherence_mean": "float",
        "prompt_relevance": "float",
        "connective_density": "float",
        "word_count": "int",
        "repetition_rate": "float"
      }
    }
  ],

  "consensus": {
    "method": "string — consensus mechanism used",
    "selected_agent_id": "string | null — which agent's output was selected",
    "selected_text": "string — final consensus text",
    "confidence": "float — consensus confidence score",
    "scores": {"agent_id": "float — per-agent consensus scores"}
  },

  "evaluation": {
    "quality_score": "float — BT score for consensus output",
    "structural_metrics": {
      "mtld": "float",
      "readability_fk_grade": "float",
      "coherence_mean": "float",
      "prompt_relevance": "float",
      "connective_density": "float",
      "word_count": "int",
      "repetition_rate": "float"
    },
    "composite_score": "float — z-normalized composite (descriptive, not quality proxy)",
    "corrected_metrics": {
      "consensus_candidate_id": "string",
      "consensus_win_rate": "float — fraction of pairwise comparisons won",
      "normalized_bt_score": "float — BT score × num_candidates",
      "num_bt_candidates": "int",
      "consensus_vs_best_agent": "bool",
      "consensus_comparisons": "int",
      "per_judge_consensus_win_rate": {"judge_name": "float"}
    },
    "judge_panel": {
      "judges": ["string[] — judge model names"],
      "bt_scores": {"candidate_id": "float"},
      "ranking": ["string[] — candidates ranked by BT score"],
      "inter_rater_reliability": {
        "mean_cohen_kappa": "float",
        "pairwise": {"judge_pair": "float"}
      },
      "pairwise_records": [
        {
          "candidate_i": "string",
          "candidate_j": "string",
          "per_judge": {"judge_name": "string — 'left' | 'right' | 'tie'"},
          "majority_winner": "string | null"
        }
      ]
    }
  },

  "topology": {
    "name": "string",
    "rounds": "int",
    "metadata": {}
  },
  "debate_rounds": ["object[] — debate round transcripts if applicable"]
}
```

### B3. Schema: `runs_summary.csv`

One row per run. Columns:

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | string | Unique run identifier |
| `block_id` | string | Experimental block |
| `task_type` | string | "analytical" or "creative" |
| `task_id` | string | Task identifier |
| `topology` | string | Orchestration topology |
| `consensus` | string | Consensus mechanism |
| `agent_count` | int | Number of agents |
| `disagreement_level` | int | Experimental disagreement condition |
| `temperature` | float | Generation temperature |
| `prompt_strategy` | string | Prompt variation strategy |
| `model_tier` | string | "frontier" / "mid" / "mixed" — derived from model_assignment |
| `quality_score` | float | BT score for consensus |
| `consensus_win_rate` | float | Pairwise win rate of consensus vs agents |
| `normalized_bt_score` | float | Candidate-count-adjusted BT |
| `consensus_vs_best_agent` | bool | Did consensus beat best individual? |
| `mean_cohen_kappa` | float | Inter-rater reliability |
| `mtld` | float | Consensus MTLD |
| `readability_fk_grade` | float | Consensus FK grade |
| `coherence_mean` | float | Consensus coherence |
| `prompt_relevance` | float | Consensus-to-prompt relevance |
| `connective_density` | float | Connective word density |
| `word_count` | int | Consensus word count |
| `repetition_rate` | float | 4-gram repetition rate |
| `composite_score` | float | Z-normalized composite (descriptive only) |
| `total_input_tokens` | int | Sum across all agents |
| `total_output_tokens` | int | Sum across all agents |
| `total_latency_ms` | float | Wall-clock latency |
| `num_debate_rounds` | int | Debate rounds (if applicable) |

### B4. Schema: `agent_outputs.csv`

One row per agent output:

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | string | Run identifier |
| `agent_id` | string | Agent identifier within run |
| `model_name` | string | LLM model name |
| `task_type` | string | Task type |
| `topology` | string | Topology |
| `text_length_chars` | int | Character count |
| `word_count` | int | Word count |
| `input_tokens` | int | Prompt tokens |
| `output_tokens` | int | Output tokens |
| `latency_ms` | float | Generation latency |
| `mtld` | float | MTLD score |
| `readability_fk_grade` | float | FK grade level |
| `coherence_mean` | float | Adjacent coherence |
| `prompt_relevance` | float | Prompt relevance |
| `connective_density` | float | Connective density |
| `repetition_rate` | float | Repetition rate |
| `composite_score` | float | Composite score |
| `is_consensus_source` | bool | Was this agent's output selected as consensus? |

### B5. Schema: `pairwise_judgments.csv`

One row per pairwise comparison per judge:

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | string | Run identifier |
| `judge_model` | string | Judge model name |
| `candidate_i` | string | First candidate ID |
| `candidate_j` | string | Second candidate ID |
| `ordering` | string | "ab" or "ba" — presentation order |
| `winner_label` | string | "A", "B", or "tie" |
| `confidence` | float | Judge confidence |
| `consistent_across_orderings` | bool | Same winner in both orderings? |
| `majority_winner` | string | Panel majority decision |

### B6. Schema: `conditions_summary.csv`

One row per unique experimental condition:

| Column | Type | Description |
|--------|------|-------------|
| `condition` | string | "task_type\|topology\|consensus" |
| `block_id` | string | Block |
| `task_type` | string | Task type |
| `topology` | string | Topology |
| `consensus` | string | Consensus mechanism |
| `n_runs` | int | Number of runs |
| `mean_quality_score` | float | Mean BT quality |
| `std_quality_score` | float | Std of quality |
| `mean_consensus_win_rate` | float | Mean win rate |
| `mean_composite_score` | float | Mean structural composite |
| `mean_word_count` | float | Mean word count |
| `mean_latency_ms` | float | Mean latency |

### B7. Metadata Files

#### `tasks.json`
```json
[
  {
    "task_id": "analytical_01_signal_triage_deduction",
    "task_type": "analytical",
    "title": "Multi-Constraint Incident Triage Deduction Puzzle",
    "prompt": "...",
    "rubric": ["..."],
    "difficulty_estimate": "hard",
    "skills_tested": ["deductive_reasoning", "constraint_satisfaction"],
    "source": "original"
  }
]
```

#### `models.json`
```json
[
  {
    "model_name": "claude-opus-4-6",
    "provider": "anthropic",
    "tier": "frontier",
    "context_window": 200000,
    "approx_cost_per_1k_output": 0.075,
    "release_date": "2025-06"
  }
]
```

#### `experiment_config.json`
```json
{
  "blocks": [
    {
      "block_id": "block0_calibration",
      "description": "Single-agent calibration baseline",
      "n_runs": 240,
      "conditions": {}
    }
  ],
  "randomization": {
    "task_assignment": "stratified_by_type",
    "model_assignment": "per_block_protocol",
    "seed_strategy": "deterministic_per_run"
  },
  "evaluation": {
    "judge_models": ["claude-sonnet-4-6", "gpt-4o", "gemini-3.1-pro-preview"],
    "judge_protocol": "bidirectional_pairwise",
    "aggregation": "bradley_terry_mle"
  }
}
```

---

### B8. Draft DATACARD.md

```markdown
---
license: cc-by-4.0
task_categories:
  - text-generation
  - text-classification
language:
  - en
tags:
  - multi-agent
  - llm-evaluation
  - consensus
  - orchestration
  - benchmark
pretty_name: "Agents Disagree: Multi-Agent Consensus Benchmark"
size_categories:
  - n<1K
---

# Dataset Card: Agents Disagree — Multi-Agent Consensus Benchmark

## Dataset Summary

This dataset contains **400 experimental runs** comparing multi-agent LLM orchestration
strategies for text generation quality. Each run consists of 1–5 LLM agents generating
responses to analytical or creative tasks under varying orchestration topologies
(flat, hierarchical, quorum, best-of-n), consensus mechanisms (simple vote, debate-then-vote,
judge-based), and disagreement conditions.

The dataset includes:
- **400 runs** across 7 experimental blocks
- **1,144 individual agent outputs** with full text and structural metrics
- **Pairwise LLM-as-judge evaluations** with 3-judge panels and bidirectional ordering
- **Bradley-Terry quality scores** and consensus win rates
- **7 structural quality metrics** per output (MTLD, FK grade, coherence, relevance,
  connective density, word count, repetition rate)
- **Inter-rater reliability** (Cohen's kappa) for all judge panels

The primary research question is: *When does multi-agent disagreement improve output quality,
and how should orchestration topology and consensus mechanisms be chosen?*

## Supported Tasks and Leaderboards

### Primary Tasks
1. **Multi-agent evaluation methodology research**: Testing new evaluation methods against
   our pairwise judging ground truth
2. **Orchestration strategy comparison**: Benchmarking topology × consensus × agent_count
   configurations
3. **LLM-as-judge analysis**: Studying judge agreement, position bias, and reliability

### Not Suitable For
- Pre-training or fine-tuning (text is LLM-generated, not human-written)
- Single-model benchmarking (outputs reflect multi-agent orchestration, not individual model capability)

## Languages

English only. All task prompts and outputs are in English.

## Dataset Structure

### Data Files

| File | Format | Description |
|------|--------|-------------|
| `runs.jsonl` | JSON Lines | Complete run data (1 object per run) |
| `runs_summary.csv` | CSV | One row per run with key metrics |
| `agent_outputs.csv` | CSV | One row per agent output |
| `pairwise_judgments.csv` | CSV | One row per pairwise judge comparison |
| `conditions_summary.csv` | CSV | Aggregated per-condition statistics |
| `tasks.json` | JSON | Task prompts, rubrics, and metadata |
| `models.json` | JSON | Model specifications and costs |
| `experiment_config.json` | JSON | Full experimental configuration |

### Data Splits

No formal train/test split. The experimental blocks serve as natural groupings:

| Block | Description | Runs |
|-------|-------------|------|
| `block0_calibration` | Single-agent baseline | 240 |
| `block1_best_of_n_baseline` | Best-of-N selection | ~7 |
| `block1_disagreement_dividend` | Disagreement level sweep | ~53 |
| `block2_topology_comparison` | Topology × consensus grid | ~50 |
| `block3_mvq_curves` | Model-Vote-Quality curves | ~9 |
| `block4_best_of_n_baseline` | Extended best-of-N | ~3 |
| `block4_quorum_paradox` | Quorum edge cases | ~38 |

### Key Fields in `runs_summary.csv`

- `task_type`: "analytical" (constraint satisfaction, deduction) or "creative" (game design, writing)
- `topology`: "flat", "hierarchical", "quorum", "best_of_n"
- `consensus`: "simple_vote", "debate_then_vote", "judge_based"
- `agent_count`: 1–5
- `quality_score`: Bradley-Terry normalized score (primary quality metric)
- `consensus_win_rate`: Fraction of pairwise comparisons consensus won vs agents
- `composite_score`: Z-normalized structural quality composite (descriptive, NOT a quality proxy)

## Dataset Creation

### Curation Rationale

Existing multi-agent benchmarks evaluate individual model capability but not orchestration
strategies. This dataset fills the gap by systematically varying topology, consensus mechanism,
agent count, and disagreement level while holding task difficulty constant.

### Source Data

All outputs are generated by frontier LLMs:
- **Generation models**: Claude Opus 4, GPT-5.2, Gemini 2.5 Pro
- **Judge models**: Claude Sonnet 4, GPT-4o, Gemini 3.1 Pro Preview
- **Task prompts**: 20 original tasks (10 analytical, 10 creative) designed by the research team

### Collection Process

- **Date**: February 2026
- **Method**: Automated experimental pipeline with deterministic random seeds
- **Evaluation**: Pairwise LLM-as-judge with bidirectional ordering (A/B and B/A)
  to control position bias
- **Aggregation**: Bradley-Terry maximum likelihood estimation with smoothing
- **Reliability**: 3-judge panels with Cohen's kappa inter-rater reliability

### Annotation Process

No human annotation. All quality judgments are from LLM judge panels. Inter-rater
reliability (mean Cohen's kappa) is reported per run.

## Considerations

### Social Impact

This dataset studies orchestration of AI systems, not human behavior. No personally
identifiable information is present.

### Known Biases

1. **LLM-as-judge bias**: Judge models may systematically prefer outputs from models
   of the same family. We mitigate this with a 3-model judge panel but cannot fully
   eliminate it.
2. **Position bias**: Despite bidirectional evaluation, residual position effects may
   exist. The `consistent_across_orderings` field enables filtering for robust judgments.
3. **Task selection bias**: Our 20 tasks span two types but do not cover all possible
   generation tasks (e.g., code generation, translation, summarization are absent).
4. **Model vintage bias**: Results reflect February 2026 model capabilities. Performance
   characteristics may not generalize to future or past model versions.
5. **Structural metric norms**: The composite_score uses fixed norms calibrated for
   analytical tasks. It should be treated as descriptive, not as a quality measure.

### Limitations

- **Small N for some blocks**: Blocks 1, 3, and 4 have very few runs. Statistical
  conclusions from these blocks should be treated as preliminary.
- **English only**: All tasks and outputs are in English.
- **Dry-run fallback**: The evaluation module includes a heuristic dry-run mode. If
  any runs used this fallback (identifiable by "dry-run heuristic" rationale), their
  quality scores are approximate.
- **Truncated outputs**: Some agent outputs hit the 2048-token limit, producing
  incomplete responses. These are included as-is (the structural metrics reflect the
  truncated text).
- **Composite ≠ quality**: The structural composite score (composite_score) does NOT
  correlate with consensus_win_rate (Spearman ρ ≈ 0, p = 0.98). Do not use it as a
  quality proxy.

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{petrov2026agents_disagree,
  title={Agents Disagree: Multi-Agent Consensus Benchmark Dataset},
  author={Petrov, Alexei and Chen, Elena and Rivera, Marcus},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/petrov-lab/agents-disagree},
  note={When Agents Disagree: Quorum-Based Consensus and Adaptive
        Orchestration Topology for Multi-Agent LLM Pipelines}
}
```

## Contact

For questions about the dataset, open an issue on the repository or contact the
corresponding author.
```

---

## Summary of Findings and Recommendations

### Statistical Validity Issues (Priority-Ordered)

| # | Issue | Severity | Recommendation |
|---|-------|----------|----------------|
| 1 | **Composite ≈ 0 correlation with quality** | 🔴 Critical | Do not use composite as quality proxy. Report as descriptive only. |
| 2 | **prompt_relevance negatively correlated with win rate** | 🔴 Critical | Cosine similarity to prompt ≠ quality. Consider removing from composite or documenting heavily. |
| 3 | **ANALYTICAL_NORMS applied to creative tasks** | 🟡 Major | Add CREATIVE_NORMS or rename to DEFAULT_NORMS with caveat. |
| 4 | **connective_density norm 8× higher than reality** | 🟡 Major | Recalibrate from block0 data: mean ≈ 0.12, std ≈ 0.14. |
| 5 | **Highly skewed metrics (repetition, connective)** | 🟡 Major | Consider log-transform or rank normalization before z-scoring. |
| 6 | **Empty outputs (45/400 = 11.25%)** | 🟠 Moderate | Flag in dataset; composite = −2.0 for empty outputs distorts analysis. |
| 7 | **Uneven block sizes** | 🟠 Moderate | Document; restrict per-block analysis to n ≥ 30. |

### What the Code Gets Right

1. ✅ **Fixed normalization priors** — scores are stationary and reproducible
2. ✅ **Z-score clipping at ±3** — prevents outlier domination
3. ✅ **Bidirectional pairwise judging** — excellent position bias control
4. ✅ **BT correction for candidate count** — `corrected_metrics.py` properly normalizes
5. ✅ **MTLD implementation** — bidirectional averaging, short-text fallback, standard algorithm
6. ✅ **Cohen's kappa for IRR** — proper inter-rater reliability measurement
7. ✅ **All 7 metrics are independent** — no redundancy (max |r| = 0.39)

### Export Schema Status

The schemas above are ready for implementation. Priority order:
1. `runs_summary.csv` — most used by researchers, build first
2. `runs.jsonl` — already exists as individual JSONs, just concatenate
3. `DATACARD.md` — draft provided above, ready for review
4. `pairwise_judgments.csv` — extract from judge_panel records
5. `agent_outputs.csv` — extract from outputs arrays
6. `conditions_summary.csv` — compute from runs_summary
7. Metadata files — document configuration

---

*Review complete. The structural metrics module is well-engineered but the composite score should be demoted from quality proxy to descriptive feature. The export schema is specified and ready for implementation. The DATACARD.md draft follows the HuggingFace template and includes all required sections.*
