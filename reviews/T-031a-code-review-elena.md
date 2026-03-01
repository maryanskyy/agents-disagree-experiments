# T-031a — Code Review: Structural Quality Layer + Data Export Format

**Reviewer:** Elena Chen (PhD-1, Theoretical Track)  
**Date:** 2026-02-28  
**Files reviewed:** `structural_quality.py`, `compute_structural_metrics.py`, `test_structural_quality.py`, `runner.py` (integration lines), `structural_metrics_summary.csv`, 10 sample run JSONs (5 block1, 5 block4)

---

## Code Quality Verdict: **MINOR ISSUES**

The implementation is fundamentally correct. The MTLD algorithm faithfully implements McCarthy & Jarvis (2010), the composite score avoids circularity by using fixed reference norms, and edge cases (empty/short text) are handled gracefully. However, there are several issues that could bias results or reduce metric reliability, none of which are critical blockers but several of which should be addressed before publication.

---

## Part A — Implementation Correctness

### 1. MTLD Implementation — CORRECT

**Assessment:** This is a faithful implementation of the real MTLD algorithm, not a simplified approximation.

The implementation in `_compute_mtld_direction()` (lines ~107-125) correctly:

- Iterates through tokens sequentially, tracking the running TTR
- Counts a full factor each time TTR drops to or below the threshold (0.72, matching McCarthy & Jarvis 2010)
- Computes partial factors for trailing tokens using the formula `(1.0 - ttr) / (1.0 - threshold)` — the correct interpolation
- Returns `len(tokens) / factors` — the standard MTLD definition

The wrapper `_compute_mtld()` (lines ~128-134) correctly averages forward and backward passes, which is the standard bidirectional MTLD procedure.

**Short text handling (lines 130-132):** For texts with fewer than 10 tokens, the code returns a scaled TTR proxy (`unique_types / total_tokens * total_tokens`). This is a reasonable conservative fallback, since MTLD is known to be unstable below ~50 tokens. However:

- **Minor concern:** The formula `(len(set(tokens)) / max(1, len(tokens))) * len(tokens)` simplifies to `len(set(tokens))`, i.e., just the number of unique types. This is mathematically correct but semantically odd as an "MTLD" value — it's just type count. Consider documenting this more explicitly or returning NaN/None to signal that MTLD is unreliable.

### 2. Coherence Metric — MINOR ISSUES

**Adjacent-sentence cosine similarity** (`_adjacent_coherence`, lines ~148-168) is correctly computed:

- Sentences are embedded with the SentenceTransformer model
- `normalize_embeddings=True` is set during encoding, so `np.dot()` correctly computes cosine similarity
- The mean of pairwise adjacent similarities is returned

**Issue 2a — Naive sentence splitter (lines 65-66):**

`SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")`

This regex splits on `.!?` followed by whitespace. It will produce incorrect boundaries for:
- Abbreviations: `"Dr. Smith said..."` splits into `"Dr."` and `"Smith said..."`
- Numbered lists: `"1. First point. 2. Second point."` — erroneous boundaries
- Decimal numbers: `"The score was 3.14 out of 10."` — splits at `3.`
- URLs: `"Visit https://example.com. Then click."` — splits mid-URL

**Impact:** Coherence scores will be artificially inflated or deflated depending on text style. Analytical text with numbered lists and abbreviations is disproportionately affected.

**Fix:** Use a sentence tokenizer from NLTK (`nltk.sent_tokenize`) or spaCy, or at minimum add negative lookbehind patterns for common abbreviations and numbers.

**Issue 2b — 1-sentence outputs return coherence = 1.0 (line 153):**

`if len(sentences) == 1: return 1.0`

Returning 1.0 (perfect coherence) for a single sentence is semantically misleading. A one-sentence output is not "maximally coherent" — it's simply too short to evaluate coherence. This inflates composite scores for very short outputs.

**Fix:** Return `float('nan')` or a neutral value (e.g., the norm mean from `ANALYTICAL_NORMS`, which is 0.45), and handle NaN in the composite score computation by excluding the metric. Alternatively, return 0.0 and document that coherence is undefined for single-sentence texts.

### 3. Connective Density — MINOR ISSUES

**Issue 3a — Connective list is too sparse (lines 56-58):**

The complete set has only **11 connectives**:
`{therefore, consequently, because, since, however, although, nevertheless, but, furthermore, moreover, additionally}`

Standard discourse-connective inventories (e.g., Halliday & Hasan 1976; the PDTB connective list) include 30-100+ connectives. Notably missing:
- **Causal:** thus, hence, so, accordingly, as a result, for this reason
- **Contrastive:** yet, while, whereas, on the other hand, in contrast, despite, nonetheless, still, instead
- **Additive:** also, besides, in addition, likewise, similarly
- **Temporal:** then, next, subsequently, meanwhile, afterward, finally
- **Exemplification:** for example, for instance, specifically, in particular

With only 11 words, the metric heavily underestimates connective usage. Observed values in the sample data confirm this: many runs show `connective_density = 0.0`, which is almost certainly an artifact of the sparse list rather than a genuine absence of discourse connectives.

**Issue 3b — Ambiguity of "since" and "but":**

"Since" is genuinely ambiguous (temporal: "since 2020" vs. causal: "since the data shows..."). "But" can be used in non-contrastive idioms ("all but guaranteed", "nothing but"). The code makes no attempt to disambiguate. In practice this matters less than the undersized word list (Issue 3a), since false positives are dwarfed by false negatives.

**Fix:** Expand the connective list to at least 40-50 entries covering all major discourse relation categories. Consider using the Penn Discourse Treebank (PDTB) connective inventory as a reference.

### 4. Repetition Rate — CORRECT (with notation caveat)

**Assessment:** The 4-gram duplicate fraction in `_repetition_rate()` (lines ~181-189) is correctly computed.

The implementation counts the total number of n-gram occurrences that belong to any repeated pattern (i.e., n-grams appearing more than once). This means if a 4-gram appears 3 times, all 3 occurrences contribute to the numerator. The denominator is the total number of n-gram slots.

This is a valid definition: "what fraction of the text's 4-gram slots participate in a repeated pattern." The alternative — counting only excess occurrences (count - 1) — would measure "what fraction are duplicates." Both are defensible; the current choice penalizes repetition more aggressively.

**Caveat:** The metric name "repetition_rate" could be misleading. Consider documenting the precise definition in the docstring (e.g., "fraction of 4-gram slots that belong to a pattern appearing more than once").

### 5. Composite Score — MINOR ISSUES

**Positive:** The composite score uses fixed reference norms (`ANALYTICAL_NORMS`, lines 48-56), **not** norms derived from the current dataset. This avoids the circularity problem where scores would be relative to the experimental sample rather than to an absolute reference. This is the correct design.

The z-score clamping to [-3, 3] prevents extreme outliers from dominating. The three normalization modes (higher-is-better, lower-is-better, target) are correctly implemented.

**Issue 5a — Single norm set for all task types:**

The norms are labeled "analytical-task" norms, yet they are applied to both analytical and creative tasks. Appropriate ranges differ substantially:

| Metric | Analytical | Creative |
|--------|-----------|----------|
| word_count target | ~450 | ~800-1000 (stories, dialogues) |
| readability_fk_grade target | ~13 (graduate) | ~8-10 (narrative prose) |
| mtld norm | ~80 | higher for creative vocabulary |
| coherence_mean | ~0.45 | may be lower for dialogues with scene breaks |

Using analytical norms for creative tasks systematically penalizes creative outputs for being longer and having lower FK grades than expected by the analytical norm. This confounds task type with quality.

**Fix:** Add a `task_type` parameter to `compute_composite_score()` and define separate norm sets (at minimum: `ANALYTICAL_NORMS` and `CREATIVE_NORMS`).

**Issue 5b — prompt_relevance in composite with lexical fallback:**

When embeddings are disabled (or unavailable), `prompt_relevance` falls back to word-overlap between prompt and output. This lexical proxy has very different distributional properties from embedding-based cosine similarity, yet it receives equal weight in the composite. The result is that the composite score's reliability depends on whether embeddings are available, which is an uncontrolled confound.

**Fix:** Either (a) exclude `prompt_relevance` from the composite when using lexical fallback, or (b) calibrate separate norms for the lexical fallback mode.

### 6. Edge Cases — MOSTLY HANDLED

| Edge case | Behavior | Assessment |
|-----------|----------|------------|
| Empty text ("") | All metrics -> 0.0 | Correct (verified in run_020d7a1bc18f.json) |
| Very short text (<10 words) | MTLD -> type count; other metrics compute normally | Metrics unreliable but no crash |
| Non-English text | TOKEN_PATTERN = [A-Za-z0-9']+ strips non-ASCII -> word_count = 0 for CJK text | Silent failure; returns all 0.0 |
| Text with markdown | **bold** -> tokenized as bold; # headers stripped | Acceptable |
| Text with code blocks | Code tokens treated as words; may inflate word count and deflate MTLD | Minor; not relevant for our tasks |

**Recommendation:** Add a guard in `compute_structural_metrics()` that logs a warning when `word_count < 10` after tokenization, flagging that metrics are unreliable for that sample.

### 7. Test Coverage — INSUFFICIENT

Only **3 tests** exist, covering:
1. Basic structural metrics computation with disabled embeddings
2. Composite score returns a float
3. Repetition detection on synthetic looping text

**Missing tests (ranked by importance):**

1. **Empty text** — verify all metrics return 0.0 and no exceptions
2. **Single-sentence text** — verify coherence returns 1.0 (and document that this is intentional)
3. **MTLD bidirectionality** — verify forward != backward for asymmetric text, and that average is returned
4. **Short text threshold** — verify the <10 token fallback activates correctly
5. **Connective density with connectives** — verify known connectives are detected (currently only checked indirectly)
6. **Sentence splitting edge cases** — abbreviations, numbered lists, decimals
7. **Composite score at norms** — verify that metrics at exactly the norm values produce z ~ 0
8. **Composite score extremes** — verify clamping at +/-3
9. **Prompt relevance** — both lexical fallback and embedding modes
10. **Integration test** — run compute_structural_metrics with embeddings enabled on a realistic paragraph
11. **Regression tests** — pin expected values for specific known inputs to catch accidental changes

---

## Part B — Data Export Format for Other Researchers

### 1. Current JSON Schema — ADEQUATE BUT INCOMPLETE

The run JSON files are reasonably self-contained. Each includes:
- Full config (task_type, topology, consensus, agent_count, models, temperature, etc.)
- Complete task definition (title, prompt, rubric)
- Per-agent outputs with text, token counts, latency, and structural metrics
- Consensus result (method, selected text, confidence, scores)
- Evaluation data (BT scores, corrected metrics, judge panel, disagreement metrics)
- Debate rounds with per-round agent outputs

**Missing metadata for external researchers:**

| Missing field | Why it matters |
|---------------|----------------|
| schema_version | Breaking changes in JSON structure would silently corrupt downstream parsing |
| framework_version | Reproducibility; which version of the experiment code produced this data |
| embedding_model_name | Which model computed the structural metrics embeddings |
| model_versions (API snapshot) | e.g., claude-opus-4-6-20260215 vs. claude-opus-4-6-20260301 |
| random_seed | Needed for reproducibility |
| field_descriptions or external schema | A new researcher can't know what corrected_metrics.normalized_bt_score means without reading source code |

**Structural concern:** The `outputs[].text` field is sometimes truncated (max_tokens hit). This is visible in the data (e.g., run_02530d70ac85 block1 agent outputs are 98 words with 2044 output_tokens — the text appears cut off in the JSON). This isn't a code bug but should be flagged with a `truncated: true` field.

### 2. CSV Format — GOOD, MINOR IMPROVEMENTS NEEDED

The `structural_metrics_summary.csv` is well-structured:
- Snake_case column names
- Immediately loadable in pandas (pd.read_csv(...))
- Includes both consensus and per-agent rows with candidate_type discriminator
- All numeric columns are numeric (no mixed types)

**Improvements needed:**

1. **Missing columns:** BT quality scores, judge agreement metrics, and corrected metrics are absent. A researcher wanting to correlate structural quality with LLM-judge quality must join against the raw JSONs, which defeats the purpose of a summary CSV.

2. **Missing metadata columns:** model_name (which model generated this output), temperature, prompt_strategy. These are in the JSON config but not propagated to the CSV.

3. **Column naming:** `consensus` (column) is ambiguous — it refers to the consensus method (e.g., debate_then_vote), not the consensus output. Rename to `consensus_method`.

4. **No data dictionary:** Include a companion file `structural_metrics_summary_CODEBOOK.md` that defines each column.

### 3. Proposed Standard Data Export Format

For maximum reusability, I recommend a **three-layer export**:

#### Layer 1: Run-Level JSONL (runs.jsonl)
One JSON object per line, one line per run. Each object contains:
- schema_version, run_id, block_id, timestamp
- Full config block
- Task definition (title, prompt, rubric)
- Per-agent outputs (agent_id, model_name, text, tokens, latency, truncated flag, structural_metrics)
- Consensus output (method, selected text, confidence, structural_metrics)
- Evaluation data (bt_scores, corrected_metrics, judge_panel, disagreement)

#### Layer 2: Analysis-Ready CSV (experiment_data.csv)
One row per candidate (agent output or consensus), with all metrics flattened:

run_id, block_id, task_type, task_id, topology, consensus_method, agent_count, disagreement_level, temperature, prompt_strategy, repetition, candidate_type, candidate_id, model_name, text_length_chars, word_count, input_tokens, output_tokens, latency_ms, truncated, mtld, readability_fk_grade, coherence_mean, prompt_relevance, connective_density, repetition_rate, composite_score, bt_score, normalized_bt_score, consensus_win_rate

#### Layer 3: Condition Summaries (condition_summaries.csv)
One row per experimental condition (task_type x topology x consensus x agent_count x disagreement_level), with means and standard deviations for all metrics.

### 4. Proposed DATACARD.md Outline

# Dataset Card: Multi-Agent LLM Consensus Experiment Data

## Dataset Description
- **Homepage:** [GitHub repo URL]
- **Paper:** "When Agents Disagree: Quorum-Based Consensus and Adaptive Orchestration Topology for Multi-Agent LLM Pipelines" (2026)
- **Point of Contact:** [PI email]

### Dataset Summary
Experimental data from a controlled study of multi-agent LLM pipeline configurations. Contains outputs from N runs across B experimental blocks, evaluating how consensus mechanism, agent count, model diversity, and orchestration topology affect output quality.

### Supported Tasks
- Multi-agent system evaluation
- LLM output quality analysis
- Consensus mechanism comparison
- Orchestration topology benchmarking

### Languages
English (en)

## Dataset Structure

### Data Instances
Each instance represents one experimental run with: configuration, task definition, per-agent raw outputs with structural quality metrics, consensus output with selection metadata, LLM-as-judge evaluation (Bradley-Terry scores, inter-rater reliability), and disagreement metrics.

### Data Fields
[Full field-by-field description with types and value ranges]

### Data Splits
| Block | Description | N runs | Purpose |
|-------|-------------|--------|---------|
| block0_calibration | Single-agent baselines | ... | Calibration |
| block1_disagreement_dividend | Varying disagreement levels | ... | H1 |
| block2_topology_matters | Topology comparisons | ... | H2 |
| block3_model_diversity | Homogeneous vs. mixed models | ... | H3 |
| block4_quorum_paradox | Large quorum diminishing returns | ... | H4 |

## Dataset Creation

### Curation Rationale
[Why this dataset was created, what gap it fills]

### Source Data
- Models used: Claude Opus 4, GPT-5.2, Gemini 2.5 Pro, Gemini 2.5 Flash
- Judge models: Claude Sonnet 4, GPT-4o, Gemini 3.1 Pro Preview
- API dates: [date range]
- Model version snapshots: [specific version identifiers if available]

### Collection Methodology
- Automated pipeline with framework name + version
- Each run: N agents independently respond to a prompt -> debate/revision rounds -> consensus selection -> multi-judge evaluation
- Structural quality metrics computed post-hoc using local NLP (MTLD, FK grade, coherence, connective density, repetition rate)

### Annotations
- BT scores: Derived from pairwise comparisons by 3 LLM judges
- Inter-rater reliability: Cohen's kappa computed per judge pair
- Structural metrics: Deterministic, non-LLM quality signals

## Considerations

### Social Impact
[Discussion of implications for multi-agent AI system design]

### Known Limitations
- LLM-as-judge evaluation may have systematic biases (e.g., verbosity bias, self-preference)
- Structural metrics use fixed analytical-task norms; creative task scores may be miscalibrated
- Agent outputs may be truncated at max_tokens boundary
- Connective density metric uses a limited (11-word) connective inventory
- Single run per some conditions (limited statistical power for rare conditions)

### Licensing
[License — recommend CC-BY-4.0 for academic data]

### Citation
[BibTeX for the paper]

### Contributions
[Author contributions and acknowledgments]

### 5. Reproducibility Package

| Component | Priority | Status | Notes |
|-----------|----------|--------|-------|
| Task prompts and rubrics | Critical | Present | Already embedded in run data |
| Experiment config files | Critical | Unknown | Block definitions, condition matrix — need to verify these are versioned |
| Model version identifiers | Critical | Partial | Model family names present; exact API snapshots missing |
| Random seeds | Critical | Missing | Not recorded in run JSONs |
| Framework source code | High | Present | Tag/hash the exact commit used for the experiment |
| Python environment | High | Unknown | requirements.txt or pyproject.toml with pinned versions |
| Embedding model specification | High | Implicit | all-MiniLM-L6-v2 is default but not recorded per run |
| API call logs | Nice-to-have | Missing | Raw API request/response pairs for full auditability |
| Structural metric norms justification | High | Missing | Document how ANALYTICAL_NORMS values were chosen |
| Judge prompt templates | Critical | Unknown | The exact prompts used to instruct judge models |

---

## Summary of Bugs/Issues Found

| # | Severity | File | Location | Issue | Fix |
|---|----------|------|----------|-------|-----|
| 1 | Medium | structural_quality.py | L65-66 | Naive sentence splitter mishandles abbreviations, decimals, numbered lists | Use nltk.sent_tokenize() or add negative lookbehind patterns |
| 2 | Medium | structural_quality.py | L56-58 | Connective inventory has only 11 words; vastly underestimates connective density | Expand to >=40 connectives using PDTB inventory |
| 3 | Medium | structural_quality.py | L153 | Single-sentence coherence returns 1.0 (misleading) | Return NaN or norm mean; exclude from composite |
| 4 | Medium | structural_quality.py | L48-56 | Single norm set applied to both analytical and creative tasks | Add task-type-specific norms |
| 5 | Low | structural_quality.py | L130-132 | MTLD short-text fallback simplifies to len(set(tokens)) without documentation | Document or return NaN with a flag |
| 6 | Low | structural_quality.py | L60 | TOKEN_PATTERN strips non-ASCII; silent failure on non-English text | Add warning when word_count is unexpectedly low |
| 7 | Low | structural_quality.py | L181-189 | Repetition rate definition (all occurrences vs. excess only) undocumented | Add docstring clarifying the precise definition |
| 8 | Low | compute_structural_metrics.py | CSV output | Missing columns: BT scores, model_name, temperature, prompt_strategy | Add to CSV fieldnames and row construction |
| 9 | Low | test_structural_quality.py | — | Only 3 tests; no coverage of edge cases, sentence splitting, or integration | Add >=8 additional tests (see section 7 above) |

---

## Data Export Recommendations (Ranked by Importance)

1. **[Critical] Add a schema version and field documentation** — Without this, any downstream researcher must reverse-engineer field semantics from source code. A schema.json (JSON Schema) plus CODEBOOK.md would take ~2 hours and prevent weeks of confusion.

2. **[Critical] Create a unified analysis-ready CSV** — The current structural_metrics_summary.csv is useful but incomplete. Add BT scores, model names, temperatures, and corrected metrics so researchers can perform multivariate analyses without parsing JSONs.

3. **[Critical] Write DATACARD.md** — Required for any dataset upload (HuggingFace, Zenodo, etc.). See proposed outline above.

4. **[High] Record random seeds and model version snapshots** — Currently not present in any run JSON. Without these, exact reproduction is impossible.

5. **[High] Add task-type-specific composite norms** — The current analytical-only norms systematically bias creative-task composite scores. At minimum, add separate CREATIVE_NORMS.

6. **[High] Expand the connective inventory** — The 11-word list causes most texts to register near-zero connective density, reducing the metric's discriminative power.

7. **[High] Export a condition-level summary CSV** — Aggregated means and SDs per experimental condition, ready for ANOVA/mixed-model analysis.

8. **[Medium] Add a truncated flag to outputs** — Some outputs hit max_tokens and are cut off. This should be explicitly flagged so researchers can filter or account for truncation.

9. **[Medium] Pin the embedding model in run metadata** — The default all-MiniLM-L6-v2 is implicit; if someone reruns with a different model, metrics are incomparable.

10. **[Medium] Archive judge prompt templates** — Critical for understanding potential evaluation biases.

---

*Review complete. The structural quality layer is a solid first implementation — the MTLD algorithm in particular is correctly done. The issues identified are all addressable with moderate effort. The data export format needs a documentation layer (schema + codebook + DATACARD) more than it needs structural changes.*
