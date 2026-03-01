# T-029b: Judge Panel Redesign & Statistical Validation Layer

**Author:** Marcus Rivera (PhD-2, Empirical Track)  
**Date:** 2026-02-28  
**Status:** Design Review + Implementation Specification  

---

## Executive Summary

Two problems, two recommendations:

1. **Judge Panel**: Replace Gemini 3 Pro Preview with DeepSeek v3p2 as the third judge, contingent on a 30-comparison calibration test. Keep GPT-4o with the fixed prompt (give it one chance to improve). If GPT-4o still votes tie >50% after the prompt fix, replace it with Qwen 3. End state: 3 judges from 3 distinct provider families, all with tie rates under 30%.

2. **Statistical Validation Layer**: A 5-metric structural quality audit computed from raw text (no LLM needed), combined via PCA into a composite score, correlated with LLM judge BT scores via Spearman's rho to validate that our judges are measuring real quality differences. Implementation: ~300 lines of new code in `src/evaluation/structural_quality.py` plus a ~100-line audit script.

---

## Part 1: Judge Panel Recommendation

### 1.1 Diagnosis: What Went Wrong

The data tells a clear story:

| Judge | Provider | Non-Tie Rate | Effective Signal |
|-------|----------|-------------|-----------------|
| Claude Sonnet 4.6 | Anthropic | 49.2% | Moderate -- discriminates ~half the time |
| GPT-4o | OpenAI | 7.2% | Nearly zero -- effectively abstains on 93% of comparisons |
| Gemini 3 Pro Preview | Google | 15.5% | Minimal -- only 1 in 6 comparisons produces a winner |

When a judge votes "tie," it contributes 0.5 wins to both candidates in the BT matrix. A judge that ties 93% of the time is essentially adding `0.5 * n_pairs` of noise to every candidate's score -- it dilutes the signal from the one judge that's actually discriminating.

**Quantifying the damage:** In a 3-candidate run (consensus + 2 agents), there are 3 pairwise comparisons. If GPT-4o ties all 3 and Gemini ties ~2.5/3, the aggregate BT matrix gets approximately 5.5 half-win injections (noise) vs. ~1.5 decisive votes (signal) from Claude Sonnet alone. The BT scores converge toward uniform (0.333 each) regardless of actual quality. This is exactly what we see in the sample run -- all three judges tied on all three pairs, producing uniform BT scores of 0.333.

### 1.2 Root Cause

The old system prompt didn't explicitly instruct judges that ties should be rare. The new prompt says: *"Reserve tie ONLY for cases where the responses are genuinely indistinguishable after careful rubric analysis. Ties should be rare, under 20 percent of your judgments."*

This should help -- but different models have different baseline willingness to commit. GPT-4o's 93% tie rate suggests a deep reluctance to discriminate, not just a prompt issue. Gemini 3 Pro Preview's 85% ties could be a model-capability issue (preview models are often more hedging).

### 1.3 Constraints

From `EVALUATION_METHODOLOGY.md` (Elena's protocol):
- **Judge models must NEVER be used as pipeline agents** -- strict model pool separation
- **At least 3 judges from at least 2 provider families** -- reduces shared-bias risk
- **No judge model appears in the agent pool** -- already enforced

Current agent pool: `claude-opus-4-6`, `gpt-5.2`, `gemini-2.5-pro`, `claude-haiku-4-5`, `gemini-2.5-flash`

Therefore, the following models are **ineligible** as judges:
- Claude Opus 4.6 (agent)
- GPT-5.2 (agent)
- Gemini 2.5 Pro (agent)
- Claude Haiku 4.5 (agent)
- Gemini 2.5 Flash (agent)

Available replacement candidates (not in agent pool):
- DeepSeek v3p2 (DeepSeek)
- Qwen 3 (Alibaba)
- Kimi k2-thinking (Moonshot)
- Llama 4 (Meta -- open-weight, typically served via API)

### 1.4 Evaluation of Replacement Options

| Model | Provider | Pros | Cons | Self-Preference Risk |
|-------|----------|------|------|---------------------|
| **DeepSeek v3p2** | DeepSeek | Strong reasoning; cost-effective; novel provider adds diversity; known for opinionated outputs | Less established as judge; may have style biases toward Chinese-originated training data | Low -- no DeepSeek agents in pool |
| **Qwen 3** | Alibaba | Very strong on reasoning benchmarks; good instruction following; multiple size options | Similar concerns about training data bias; less western-NLP community validation | Low -- no Qwen agents in pool |
| **Kimi k2-thinking** | Moonshot | Chain-of-thought reasoning; designed for deep analysis | Relatively new; less benchmarked as evaluator; "thinking" mode may be slow/expensive | Low -- no Kimi agents in pool |
| **Llama 4** | Meta | Open-weight (auditable); strong community validation; large model available | Requires API hosting; may be less opinionated than proprietary models; open models sometimes less calibrated as judges | Low -- no Llama agents in pool |

### 1.5 Recommendation: Phased Approach

#### Phase 1: Calibration Test (MANDATORY before any production runs)

Run 30 pairwise comparisons using the **fixed prompt** on a calibration set covering both task types. Test all current judges + 2 replacement candidates:

**Test panel:**
1. Claude Sonnet 4.6 (Anthropic) -- anchor, expected to stay
2. GPT-4o (OpenAI) -- testing if prompt fix resolves the tie issue
3. Gemini 3 Pro Preview (Google) -- testing if prompt fix resolves the tie issue
4. DeepSeek v3p2 (DeepSeek) -- primary replacement candidate
5. Qwen 3 (Alibaba) -- secondary replacement candidate

**Decision criteria after calibration:**

    For each judge J:
      tie_rate = count(ties) / count(total_comparisons)
      
      if tie_rate < 0.20: EXCELLENT -- keep
      if 0.20 <= tie_rate < 0.30: ACCEPTABLE -- keep with monitoring
      if 0.30 <= tie_rate < 0.50: MARGINAL -- replace if better option exists
      if tie_rate >= 0.50: UNACCEPTABLE -- must replace

#### Phase 2: Final Panel Selection

**Expected outcome** (my prediction based on model characteristics):

| Judge | Predicted Post-Fix Tie Rate | Decision |
|-------|---------------------------|----------|
| Claude Sonnet 4.6 | ~25% (down from 51%) | Keep |
| GPT-4o | ~40-50% (down from 93%) | Marginal -- prompt helps but model is inherently cautious |
| Gemini 3 Pro Preview | ~45-55% (down from 85%) | Marginal -- preview model limitations |
| DeepSeek v3p2 | ~15-25% (prediction) | Strong candidate |
| Qwen 3 | ~20-30% (prediction) | Good candidate |

**Primary recommendation (most likely outcome):**

| Slot | Judge | Provider | Rationale |
|------|-------|----------|-----------|
| Judge 1 | Claude Sonnet 4.6 | Anthropic | Proven discriminator; our anchor |
| Judge 2 | GPT-4o | OpenAI | Give it the fixed prompt; if tie rate drops below 30%, keep for industry-standard coverage |
| Judge 3 | **DeepSeek v3p2** | DeepSeek | Replaces Gemini; adds 3rd provider family; strong reasoning model |

**Fallback (if GPT-4o still >50% ties after fix):**

| Slot | Judge | Provider | Rationale |
|------|-------|----------|-----------|
| Judge 1 | Claude Sonnet 4.6 | Anthropic | Anchor |
| Judge 2 | **DeepSeek v3p2** | DeepSeek | Primary replacement |
| Judge 3 | **Qwen 3** | Alibaba | Secondary replacement; maintains 3-provider diversity |

### 1.6 Why NOT Drop to 2 Judges

Two-judge panels have a fundamental problem: **every disagreement is unresolvable**. With 2 judges:
- If they agree: clear winner (but we could get that from 1 judge)
- If they disagree: forced tie, regardless of confidence
- No majority vote mechanism: the panel adds complexity without resolving ambiguity

The marginal cost of a 3rd judge (~33% more judge API calls) is trivially small compared to the agent generation costs. Three judges give us:
- Majority vote resolution for disagreements
- Cohen's kappa computation (meaningful only with 3+ raters)
- Robustness against any single judge being unreliable

**Verdict: 3 judges is non-negotiable.**

### 1.7 Why NOT Use a Different Gemini Model

The question asks whether we should use a different Gemini model. The problem:
- Gemini 2.5 Pro is already an agent -- can't use as judge
- Gemini 2.5 Flash is already an agent -- can't use as judge
- Gemini 3 Pro Preview is the only available Gemini judge, and it's been mediocre
- Even if Google releases a non-preview Gemini 3 Pro, we'd still have only one non-overlapping option

More fundamentally: having 3 judges from 3 providers (Anthropic, OpenAI, Google) sounds good for diversity, but it's only valuable if all 3 judges actually discriminate. A Google judge that votes tie 85% of the time provides neither diversity nor signal. Better to have 3 discriminating judges from Anthropic + DeepSeek + Alibaba than to force Google representation.

### 1.8 Config Changes Required

After calibration confirms the recommendation, update `config/models.yaml`:

    judge_pool:
      claude-sonnet-4-6:
        provider: anthropic
        model_id: claude-sonnet-4-6
        input_cost_per_1m: 3.0
        output_cost_per_1m: 15.0
        rpm_limit: 35

      gpt-4o:
        provider: openai
        model_id: gpt-4o
        input_cost_per_1m: 2.50
        output_cost_per_1m: 10.0
        rpm_limit: 35

      deepseek-v3p2:               # REPLACES gemini-3-pro-preview
        provider: deepseek
        model_id: deepseek-chat
        input_cost_per_1m: 0.27
        output_cost_per_1m: 1.10
        rpm_limit: 30

Also requires adding a `DeepSeekModelClient` to `src/models/` (similar structure to existing `openai_client.py` since DeepSeek uses an OpenAI-compatible API).

---

## Part 2: Statistical Validation Layer Specification

### 2.1 Design Philosophy

The validation layer answers one question: **"Are our LLM judges measuring real quality, or are they hallucinating preferences?"**

The approach: compute text-level quality indicators that require no LLM, aggregate them into a composite score, and check whether this composite correlates with the LLM judges' Bradley-Terry rankings. If it does (rho > 0.3), the judges are at least partially tracking real quality signals. If it doesn't, we have a problem.

This is explicitly NOT a replacement for LLM judges. Structural metrics can't evaluate creativity, argument validity, or task fidelity. But they can catch cases where a judge says "Output A is clearly better" when Output A is actually incoherent, repetitive, or off-topic.

### 2.2 Metrics Specification

#### Metric 1: Lexical Diversity (TTR + MTLD)

**What it measures:** Vocabulary richness. High-quality text typically uses varied vocabulary rather than repeating the same words.

**Implementation:**
- **TTR (Type-Token Ratio):** `len(unique_tokens) / len(total_tokens)` -- simple but length-sensitive
- **MTLD (Measure of Textual Lexical Diversity):** Factor analysis approach that's length-independent. Sequentially segments text at points where TTR drops below 0.72; MTLD = mean segment length.
- **Report both; use MTLD for the composite** (more robust)

**Expected range:** TTR in [0.3, 0.9]; MTLD in [30, 200]

**Library:** `lexical-diversity` (pip install) or compute from scratch (MTLD is ~40 lines). NLTK tokenizer for preprocessing.

#### Metric 2: Readability (Flesch-Kincaid Grade Level)

**What it measures:** Syntactic complexity as a proxy for appropriate sophistication. For analytical tasks, we expect higher grade levels (complex arguments require complex sentences). For creative tasks, moderate grade levels (accessible but not simplistic).

**Implementation:**
- **Flesch-Kincaid Grade Level:** `0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59`
- Also compute **Automated Readability Index (ARI)** as corroboration

**Expected range:** FK grade level in [8, 20] for our tasks

**Library:** `textstat` (pip install textstat)

#### Metric 3: Coherence (Adjacent Sentence Similarity)

**What it measures:** How well sentences flow into each other. High coherence = the text has a logical thread. Low coherence = jumpy, disorganized, or internally contradictory.

**Implementation:**
- Split text into sentences
- Encode each sentence using `sentence-transformers` (model: `all-MiniLM-L6-v2`, already in requirements.txt)
- Compute cosine similarity between each consecutive pair of sentence embeddings
- Report: mean coherence, min coherence (weakest transition), std (uniformity)

**Expected range:** Mean coherence in [0.1, 0.8]; higher = more coherent

#### Metric 4: Prompt Coverage (Completeness)

**What it measures:** Does the response actually address the prompt? A response that's well-written but off-topic should score poorly.

**Implementation:**
- Extract key content words from the prompt (remove stopwords, keep nouns/verbs/adjectives)
- Compute keyword overlap: `|prompt_keywords intersection response_keywords| / |prompt_keywords|`
- Compute semantic coverage: embed the prompt and response, compute cosine similarity
- For rubric-based tasks: embed each rubric criterion, compute max similarity against response sentences, report mean coverage

**Expected range:** Keyword overlap in [0.1, 0.8]; semantic coverage in [0.3, 0.9]

#### Metric 5: Information Density

**What it measures:** How much unique information is packed per unit of text. Guards against verbose padding -- a response that says the same thing three ways shouldn't score higher just because it's longer.

**Implementation:**
- **Content word ratio:** proportion of non-stopword, non-function-word tokens
- **Unique concept density:** unique content lemmas per sentence
- **Repetition penalty:** 1 - (repeated n-gram fraction), where repeated = appears 3+ times

**Expected range:** Content ratio in [0.4, 0.7]; unique density in [3, 15]

### 2.3 Composite Score Construction

**Method: PCA on z-scored metrics**

PCA is preferred over arbitrary weighting because:
1. It's data-driven -- weights are determined by variance in the actual data
2. It's reproducible -- no subjective weight choices
3. The first principal component captures the shared "quality-like" dimension

**Procedure:**

1. For each output, compute the 8 core sub-metrics:
   - `mtld` (lexical diversity)
   - `flesch_kincaid_grade` (readability)
   - `mean_coherence` (coherence)
   - `keyword_overlap` (completeness -- keyword)
   - `rubric_coverage` (completeness -- rubric)
   - `content_ratio` (information density)
   - `unique_density` (information density)
   - `repetition_penalty` (information density)

2. Z-score normalize each metric across all outputs in the dataset:
   `z_i = (x_i - mean_i) / std_i`

3. Fit PCA on the z-scored matrix. Extract PC1 as the **structural quality composite**.

4. If PC1 explains less than 30% of variance, the metrics are too independent to combine meaningfully -- report individual metrics instead of composite.

5. Normalize PC1 scores to [0, 1] range via min-max scaling for interpretability.

### 2.4 Validation Protocol

#### Step 1: Compute Spearman Correlation

For each completed experiment block, compute:
- Structural composite score for each candidate output
- LLM judge BT score for each candidate output
- Spearman's rho between the two

#### Step 2: Interpret Correlation Thresholds

| Spearman rho | Interpretation | Action |
|-----------|---------------|--------|
| rho > 0.5 | **Strong agreement** -- LLM judges and structural metrics converge well | Report as validation evidence; high confidence in judge rankings |
| 0.3 < rho <= 0.5 | **Moderate agreement** -- judges are tracking real quality but also capturing dimensions structural metrics miss (expected for creative tasks) | Acceptable; investigate flagged disagreements |
| 0.1 < rho <= 0.3 | **Weak agreement** -- judges may be measuring something structural metrics can't capture, OR judges are unreliable | Investigate: examine flagged runs, expand human validation sample |
| rho <= 0.1 | **No agreement** -- serious concern | PAUSE and diagnose: either metrics or judges (or both) are broken |

**Critical nuance:** We expect rho to be **higher for analytical tasks** (where structural quality correlates more with actual quality) and **lower for creative tasks** (where originality and craft are invisible to structural metrics). Split the validation by task type:
- Analytical tasks: target rho > 0.4
- Creative tasks: target rho > 0.2 (lower bar is acceptable)

#### Step 3: Flag Disagreements

For each run, compute:
- `z_structural` = z-score of the consensus output's structural composite
- `z_bt` = z-score of the consensus output's BT score (or consensus win rate)
- `disagreement_magnitude = |z_structural - z_bt|`

**Flagging criteria:**
- `disagreement_magnitude > 2.0`: **Hard flag** -- LLM judges and structural metrics strongly disagree. This run should be human-reviewed.
- `1.5 < disagreement_magnitude <= 2.0`: **Soft flag** -- note in audit report, investigate if pattern emerges.

**What to do with flagged runs:**
1. Add to the human validation queue (Section 5 of `EVALUATION_METHODOLOGY.md`)
2. Examine whether the disagreement is systematic (e.g., judges always rate a specific model higher than structural metrics suggest -- possible self-preference)
3. If >20% of runs are hard-flagged, the validation layer has identified a problem with the judge panel -- escalate to Dr. Petrov

#### Step 4: Report as "Structural Quality Audit"

The audit report accompanies every experiment block's results. Example format:

    === STRUCTURAL QUALITY AUDIT: block1_disagreement_dividend ===
    
    Outputs analyzed: 400
    PCA variance explained by PC1: 42.3%
    
    Metric loadings on PC1:
      mtld:               0.38  (lexical diversity)
      flesch_kincaid:      0.25  (readability)
      mean_coherence:      0.41  (coherence)
      keyword_overlap:     0.33  (completeness)
      rubric_coverage:     0.44  (rubric alignment)
      content_ratio:       0.29  (density)
      unique_density:      0.35  (density)
      repetition_penalty:  0.31  (anti-repetition)
    
    Validation (structural composite vs. LLM judge BT score):
      Analytical tasks:  rho = 0.47, p < 0.001, n = 200  MODERATE
      Creative tasks:    rho = 0.28, p = 0.003, n = 200  ACCEPTABLE
    
    Flagged runs (hard disagreement, |z_diff| > 2.0): 12 / 400 (3.0%)
      -> Added to human validation queue
    
    Conclusion: LLM judges pass structural validation.

### 2.5 Handling Disagreements Between Structural Metrics and LLM Judges

Three scenarios and how to handle each:

**Scenario A: Structural metrics say output is good, LLM judges say it's bad.**
- Possible causes: output is structurally sound but factually wrong, logically flawed, or creatively boring
- Action: structural metrics are likely wrong here (they can't assess content quality). Trust the judges, but verify with human eval on flagged cases.

**Scenario B: Structural metrics say output is bad, LLM judges say it's good.**
- Possible causes: output is incoherent/repetitive/off-topic but judges are fooled by fluent surface-level writing or length
- Action: this is the dangerous case -- it may indicate judge overrating bias (Yu et al. 2026) or verbosity bias. Human review is mandatory.

**Scenario C: Both agree (whether good or bad).**
- Action: high confidence. These cases anchor the correlation and don't need additional review.

**Decision rule:** The structural validation layer does NOT override LLM judges. It serves as an **alarm system**. When it triggers, humans investigate. The final quality assessment always comes from either the LLM judges (if validated) or human raters (if judges are questioned).

### 2.6 Python Libraries Required

**New dependencies** (add to `requirements.txt`):

    textstat>=0.7.4             # Readability metrics (FK, ARI, Flesch)
    scikit-learn>=1.5.0         # PCA, StandardScaler
    nltk>=3.9.0                 # Tokenization, stopwords, sentence splitting

**Already in requirements.txt:**

    sentence-transformers>=3.0.0  # Coherence via embeddings (all-MiniLM-L6-v2)
    numpy>=2.1.0                  # Array operations
    scipy>=1.14.0                 # Spearman correlation
    pandas>=2.2.0                 # Data manipulation

**Note:** `spacy` is NOT required. We can handle tokenization with NLTK and stopword removal with a simple set. Avoiding spaCy keeps the dependency footprint small and avoids downloading large language models.

**One-time setup:**

    import nltk
    nltk.download('punkt_tab')
    nltk.download('stopwords')

---

## Part 3: Implementation Plan

### 3.1 New Files

#### `src/evaluation/structural_quality.py` (~300 lines)

The core structural quality module containing:

    class StructuralMetrics:
        """Computes all 5 metric categories for a single text."""
        def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2")
        def compute(self, text: str, prompt: str, rubric: list[str]) -> dict[str, float]
        # Returns flat dict with all 8 sub-metrics
    
    class StructuralQualityAuditor:
        """Batch validation of LLM judge decisions against structural metrics."""
        def __init__(self, metrics: StructuralMetrics)
        def audit_block(self, run_results: list[dict]) -> AuditReport
        def flag_disagreements(self, structural_scores, bt_scores, threshold=2.0) -> list[str]
        
    @dataclass
    class AuditReport:
        """Report from structural quality audit of an experiment block."""
        n_outputs: int
        pca_variance_explained: float
        metric_loadings: dict[str, float]
        spearman_rho_analytical: float
        spearman_rho_creative: float
        flagged_runs: list[str]
        summary: str
        
        def to_dict(self) -> dict
        def to_markdown(self) -> str

#### `scripts/run_structural_audit.py` (~100 lines)

Standalone script that:
1. Loads all completed run results from `results/block*/run_*.json`
2. Computes structural metrics for each output
3. Correlates with LLM judge BT scores
4. Generates the audit report
5. Writes flagged runs to a separate file for human review

Usage:

    python scripts/run_structural_audit.py --blocks block1_disagreement_dividend block2_topology_comparison
    # Outputs: results/structural_audit_report.md
    #          results/structural_audit_flagged_runs.json

#### `scripts/judge_calibration_test.py` (~150 lines)

Calibration script for the judge panel redesign:
1. Selects 30 pairwise comparisons from existing results (15 analytical, 15 creative)
2. Tests all 5 candidate judges on these pairs using the fixed prompt
3. Reports tie rates, Cohen's kappa, and consistency
4. Outputs recommendation

Usage:

    python scripts/judge_calibration_test.py --judges claude-sonnet-4-6 gpt-4o deepseek-v3p2 qwen-3
    # Outputs: results/judge_calibration_report.md

### 3.2 Modified Files

#### `src/runner.py`

Add structural metrics computation after LLM judge evaluation. In `_evaluate_final_quality()`, after computing corrected_metrics, add structural metric computation for each candidate output and store under `evaluation.structural_metrics`.

#### `config/models.yaml`

Add DeepSeek (and optionally Qwen) to judge pool after calibration confirms.

#### `requirements.txt`

Add `textstat>=0.7.4`, `scikit-learn>=1.5.0`, and `nltk>=3.9.0`.

#### `scripts/analyze_results.py`

Add structural audit summary to the analysis output:
- Include correlation table in analysis reports
- Add flagged run counts to block summaries

### 3.3 Integration with Existing Pipeline

The structural validation layer integrates at two points:

**Point 1: Runtime (online, per-run)**
- After LLM judge evaluation in `runner.py`
- Structural metrics are computed and stored in the run JSON alongside `evaluation.judge_panel`
- New key: `evaluation.structural_metrics`
- This adds ~2-5 seconds per run (embedding computation dominates)

**Point 2: Post-hoc audit (offline, per-block)**
- The `run_structural_audit.py` script processes all completed runs
- Computes PCA composite, correlations, and flags
- Generates the audit report
- Can be rerun at any time (idempotent)

**Point 3: Reprocessing (historical data)**
- The existing `scripts/reprocess_results.py` pattern can be extended to compute structural metrics for already-completed runs
- Add a `--structural` flag to reprocess_results.py that computes and writes structural metrics for historical runs

### 3.4 Cost Impact

**Judge panel redesign:**
- DeepSeek v3p2 is significantly cheaper than Gemini 3 Pro Preview (`.27/.10` per M tokens vs `.25/.0`)
- Net effect: judge costs **decrease** by ~50% for the third judge slot
- No impact on overall experiment cost (judge costs are <10% of total)

**Structural validation layer:**
- No API costs (all computation is local)
- Sentence-transformer model: ~90MB download, runs on CPU
- Per-run overhead: ~2-5 seconds (negligible vs. 30-120 second LLM judge calls)
- Total additional compute for 8,000 runs: ~6-12 hours on CPU (run once)

### 3.5 Execution Order

    1. [IMMEDIATE] Run judge calibration test (30 comparisons, 5 judges)
       -> Decision: confirm or adjust judge panel
       
    2. [BEFORE NEXT PRODUCTION RUN] Update models.yaml with new judge panel
    
    3. [PARALLEL WITH PRODUCTION] Implement structural_quality.py
       -> Unit tests first: test_structural_quality.py
       -> Integration: run on historical block0-block4 results
       
    4. [AFTER FIRST NEW BLOCK COMPLETES] Run structural audit
       -> Generate audit report
       -> Review flagged runs
       -> Calibrate PCA loadings on real data
       
    5. [ONGOING] Structural audit accompanies every block analysis

---

## Appendix A: Metric Sensitivity to Task Type

Different metrics matter differently for analytical vs. creative tasks:

| Metric | Analytical Relevance | Creative Relevance | Notes |
|--------|---------------------|-------------------|-------|
| MTLD | Medium | High | Creative text should be lexically rich |
| FK Grade | High | Medium | Analytical arguments should be sophisticated |
| Coherence | High | Medium | Arguments need logical flow; creative text can be intentionally discontinuous |
| Keyword Overlap | High | Low | Analytical text should address the prompt directly |
| Rubric Coverage | High | High | Both should address rubric criteria |
| Content Ratio | Medium | Medium | Both should avoid filler |
| Unique Density | Medium | High | Creative text needs diverse ideas |
| Repetition Penalty | High | High | Neither should be repetitive |

**Recommendation:** Compute PCA separately for analytical and creative tasks. The loadings will differ, and that's informative -- it tells us which structural dimensions matter for each task type.

## Appendix B: Known Limitations of Structural Metrics

1. **Cannot assess factual correctness.** A confidently wrong argument may score high on coherence and readability.
2. **Cannot assess creativity or originality.** A cliched but well-structured story will score higher than an experimental, boundary-pushing one.
3. **Cannot assess argument validity.** Logical fallacies expressed fluently will score well.
4. **Readability metrics are calibrated for English prose.** Technical content with formulas, code, or specialized vocabulary may score anomalously.
5. **Embedding models have biases.** `all-MiniLM-L6-v2` may rate certain styles as more coherent than others.
6. **PCA composite is relative.** Scores are only meaningful within a batch -- they cannot be compared across experiments run at different times without refitting.

These limitations are **by design**. The structural layer is a sanity check, not a quality oracle. It catches gross failures (incoherent text rated as excellent by judges) while remaining agnostic to the subtleties that only LLM judges (or humans) can assess.

## Appendix C: Calibration Test Protocol

**Materials:** 30 pairwise comparisons drawn from existing `block0_calibration` and `block1_disagreement_dividend` results, balanced:
- 15 analytical pairs, 15 creative pairs
- Include 5 "easy" pairs (clear quality gap), 15 "medium" pairs, 10 "hard" pairs (close quality)
- "Difficulty" estimated from existing Claude Sonnet 4.6 confidence scores

**Procedure:**
1. Extract the original candidate text pairs from existing run JSONs
2. Present each pair to all 5 candidate judges using the **fixed prompt**
3. Run in both orderings (AB and BA) -- 60 judge calls per model, 300 total
4. Record: winner, confidence, tie rate, ordering consistency

**Analysis:**
- Tie rate per judge (primary criterion)
- Cohen's kappa between each judge pair
- Position consistency rate (does the judge give the same answer in AB vs BA?)
- Correlation with Claude Sonnet 4.6 decisions (our anchor)

**Cost estimate:** 300 calls x ~2K input tokens x ~500 output tokens = 0.6M input + 0.15M output tokens per judge. At GPT-4o pricing: ~``. Total for 5 judges: ~``. Negligible.

---

*This document serves as the design specification for both the judge panel reconfiguration and the structural validation layer. Implementation should proceed in the order specified in Section 3.5. The calibration test (Phase 1) must complete before any production experiment runs with the new judge configuration.*
