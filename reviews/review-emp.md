# Statistical & Reproducibility Review — Pre-Experiment Assessment

**Reviewer:** Dr. James Okonkwo (Empirical Methodology)  
**Date:** 2026-02-28  
**Document reviewed:** Experiment framework for "When Agents Disagree"  
**Scope:** All files in `agents-disagree-experiments/` plus research questions document  

---

## Verdict: **Needs Revision — One Fatal Flaw, Multiple Critical Issues**

**Confidence:** High  

The experiment infrastructure is well-engineered (atomic checkpointing, deterministic manifests, clean async runner), but there are fundamental problems in the evaluation pipeline, statistical design, and confound control that would render $3-8K of results uninterpretable. The good news: most issues are fixable without architectural changes, and the team's own `EVALUATION_METHODOLOGY.md` already contains the blueprint for the correct approach. The code just doesn't implement it yet.

---

## 1. Fatal Issue: Evaluation Metrics Do Not Measure Quality

**Severity: FATAL — blocks all downstream analyses.**

The entire statistical program depends on `quality_score` as the primary dependent variable. Here is what the code actually computes:

```python
# runner.py, line ~155
quality_score = (
    evaluate_analytical(task.prompt, final_text)   # <-- reference = PROMPT, not gold answer
    if run_spec.task_type == "analytical"
    else evaluate_creative(final_text)
)
```

### `evaluate_analytical(reference, candidate)` — metrics.py

This function computes:
- **Jaccard token overlap** between the task prompt and the model output (weight 0.5)
- **SequenceMatcher ratio** between the prompt and the output (weight 0.4)
- **Length penalty** comparing output length to prompt length (weight 0.1)

This measures *lexical similarity between output and prompt*, not quality. An output that parrots the prompt back verbatim would score ~1.0. An excellent analytical response that uses different vocabulary would score low. This metric is directionally wrong for measuring reasoning quality.

### `evaluate_creative(candidate)` — metrics.py

This function computes:
- **Unique token ratio** (weight 0.65) — rewards lexical diversity
- **Sentence rhythm** score based on average sentence length of roughly 18 words (weight 0.35)

A random word salad with varied vocabulary and medium sentence length would outscore a beautifully crafted story that uses deliberate repetition. This metric has no construct validity for creative quality.

### The disconnect

The team has written an outstanding 30-page `EVALUATION_METHODOLOGY.md` (citing 30+ papers on LLM-as-judge, bias mitigation, Bradley-Terry aggregation, multi-panel evaluation). This document specifies:
- Pairwise comparison with position randomization
- Multi-judge panel (3 models, 2+ families)
- 15-25% human validation
- Rubric-based dimensional scoring
- BT-sigma aggregation

**None of this is implemented.** The actual evaluation is a 15-line heuristic function measuring surface-level text statistics. Every statistical test, every ANOVA, every power calculation is moot if the dependent variable doesn't measure what you claim it measures.

### Fix required

Implement the evaluation protocol from `EVALUATION_METHODOLOGY.md` — at minimum:
1. LLM-judge scoring (already partially built in `llm_judge.py`, but only used for judge-based consensus, not for final evaluation)
2. Blind evaluation with position randomization
3. Judge model must not equal pipeline agent model (currently Gemini Flash is both judge AND pipeline agent — a self-preference confound)
4. Human validation for Quorum Paradox cases

**Until this is fixed, do not spend any budget on experimental runs.**

---

## 2. Run Count Verification and Cell Size Analysis

### Computed run counts by block

I manually traced the manifest generator logic against `experiment_matrix.yaml` and the task catalogs (8 analytical + 8 creative instances):

| Block | Formula | Runs | Notes |
|-------|---------|------|-------|
| B0 (Calibration) | 2 x 8 x 1 x 1 x 1 x 1 x 3 models x 4 reps | **192** | 32 per (model x task_type) |
| B1 (Disagreement) | 2 x 8 x 3 x 1 x 1 x 5 x 1 x 2 reps | **480** | 16 per (task_type x n x d) |
| B2 (Topology) | 2 x 8 x 2 x 3 x 3 x 1 x 1 x 1 rep | **288** | **8 per cell** |
| B3 (MVQ) | 2 x 8 x 4 x 1 x 1 x 1 x 1 x 1 rep x 4 thresholds | **256** | 64 unique executions, 192 redundant |
| B4 (Paradox) | 2 x 8 x 3 x 1 x 1 x 1 x 1 x 4 reps | **192** | 32 per (task_type x n) |
| B5 (Interaction) | 2 x 8 x 2 x 3 x 3 x 1 x 1 x 1 rep | **288** | **8 per cell** |
| **Total** | | **1,696** | |

### Critical: Block 3 threshold multiplication is redundant

The `quality_thresholds: [0.60, 0.70, 0.80, 0.90]` field generates 4 separate `RunSpec` entries per condition. Each run has a different `quality_threshold` but identical `model_assignment`, `temperature`, `prompt_strategy`, `task_id`, and `repetition`. The threshold is only used in a post-hoc check:

```python
threshold_met = quality_score >= run_spec.quality_threshold
```

These 4 runs execute the same LLM calls independently (they are not deduplicated — different `run_id` hashes, different execution contexts). **This wastes 75% of Block 3's budget.** Run each condition once and apply all threshold checks during analysis.

**Fix:** Remove `quality_thresholds` from the matrix. Apply threshold analysis in `analyze_results.py`.

### Cell sizes by block — detailed

| Block | Cell definition | Obs/cell | Adequate for... |
|-------|----------------|----------|-----------------|
| B0 | model x task_type | 32 | Baseline estimation (yes) |
| B1 | task_type x n x d | 16 | Trend detection (marginal) |
| B1 | task_type x d (collapsed over n) | 48 | Inverted-U detection (yes) |
| B2 | task_type x n x topology x consensus | **8** | Nothing reliable (no) |
| B3 | task_type x n | **8** (unique) | Threshold proportions (no) |
| B4 | task_type x n | 32 | Medium effect detection (marginal) |
| B5 | task_type x n x topology x consensus | **8** | ANOVA interaction (no) |

---

## 3. Power Analysis: Per-Block Assessment

### Block 0 (Calibration) — **Adequate**
- Goal: estimate per-model capability mu and pairwise correlation rho.
- 32 observations per (model x task_type) is sufficient for stable mean estimation with bootstrap CIs.
- **Power: adequate** for descriptive statistics and baseline comparisons.

### Block 1 (Disagreement Dividend) — **Marginal**
- Goal: detect inverted-U in Q(d).
- 16 observations per (task_type x n x d) cell. Collapsing across agent counts gives 48 per (task_type x d).
- For the inverted-U test: fitting a quadratic Q(d) = B0 + B1*d + B2*d^2 requires B2 < 0 with significance. With 5 levels x 48 = 240 total observations per task_type, a quadratic regression has reasonable power for medium effect sizes (R^2 approx 0.05-0.10 detectable).
- **Risk:** The inverted-U may be weak enough that 48 per d-level is insufficient to distinguish it from monotonic decline.
- **Power: marginal.** Recommend increasing repetitions from 2 to 3 (adds 240 runs, using budget saved from Block 3 fix).

### Block 2 (Topology Comparison) — **Severely Underpowered**
- Goal: 3x3 ANOVA (topology x consensus) with agent_count as blocking factor.
- **8 observations per cell** is far below what is needed for factorial ANOVA.
- For a 3x3 design with n=8 per cell:
  - Main effect (medium, f=0.25): power approx 0.30-0.40
  - Interaction effect (medium, f=0.25): power approx 0.15-0.25
  - Conventional threshold: power >= 0.80
- **Power: inadequate** for both main effects and interactions.
- At 8 per cell, you can reliably detect only **very large** effects (f > 0.50, approximately top-30% of effect sizes in the behavioral sciences).
- **This block cannot support the claims in RQ2 or RQ4 (topology-consensus interaction).**

**Fix options:**
1. Increase repetitions to 3 (doubles to 16/cell — still marginal but detects large effects)
2. Reduce the design: test 2 topologies x 2 consensus at higher replication
3. Accept this block as exploratory/screening only and say so explicitly

### Block 3 (MVQ Curves) — **Underpowered for Threshold Analysis**
- Goal: estimate P(Q >= theta) as function of quorum size n.
- 8 unique observations per (task_type x n). Threshold attainment is binary.
- With 8 Bernoulli trials, a 95% CI for a proportion is approximately +/-35% (Clopper-Pearson). You cannot distinguish P=0.5 from P=0.85 with 8 observations.
- **Power: inadequate** for MVQ knee detection. The theoretical MVQ bound requires precise estimation of threshold attainment probabilities, which 8 observations cannot provide.
- **Fix:** Increase repetitions to at least 4 (giving 32 per cell after removing redundant thresholds). With the 192 runs saved from de-duplicating thresholds, you can afford 3 additional repetitions (total 4) and STILL save runs.

### Block 4 (Quorum Paradox) — **Marginal for Subtle Effects**
- Goal: detect Q(n=3) < Q(n=2) — a quality *dip* at specific n.
- 32 observations per (task_type x n).
- For a paired comparison (n=3 vs n=2), same task instances across conditions:
  - Cohen's d = 0.3 (small): power approx 0.38 (two-sided alpha=0.05, n=32 pairs)
  - Cohen's d = 0.5 (medium): power approx 0.72
  - Cohen's d = 0.8 (large): power approx 0.97
- The Quorum Paradox is described as "subtle" in the research questions. If d is approx 0.3, this block has only 38% power — you have a 62% chance of missing a real effect.
- **Power: marginal.** Adequate only if the effect is medium or larger (d >= 0.5).
- **Fix:** Increase repetitions from 4 to 6 (giving 48 per cell; power for d=0.3 rises to approx 0.52, for d=0.5 to approx 0.88). This adds 96 runs — affordable with Block 3 savings.

### Block 5 (Interaction Probe) — **Severely Underpowered**
- Same issue as Block 2: 8 per cell in a 3x3 factorial.
- The block's stated goal — "ANOVA-style decomposition" — requires substantially more data.
- With 8/cell, interaction effects are essentially undetectable below f=0.50.
- **Power: inadequate.**
- **Fix:** Same recommendations as Block 2. If budget is constrained, merge Blocks 2 and 5 into a single block with increased replication at a shared disagreement level.

### Power Analysis Summary Table

| Block | Target effect | Cell n | Power (alpha=0.05) | Verdict |
|-------|--------------|--------|-----------------|---------|
| B0 | Descriptive | 32 | N/A | Adequate |
| B1 | Quadratic trend | 16-48 | ~0.50-0.70 | Marginal |
| B2 | ANOVA main | 8 | ~0.30-0.40 | Underpowered |
| B2 | ANOVA interaction | 8 | ~0.15-0.25 | Severely underpowered |
| B3 | Proportion estimation | 8 | CI +/-35% | Underpowered |
| B4 | Paired comparison | 32 | 0.38-0.72 | Marginal (d-dependent) |
| B5 | ANOVA interaction | 8 | ~0.15-0.25 | Severely underpowered |

---

## 4. Critical Experimental Design Issues

### 4.1 Temperature-Prompt Strategy Confound (CRITICAL)

Disagreement levels simultaneously manipulate TWO independent variables:

| Level | Temperature | Prompt Strategy |
|-------|------------|-----------------|
| 1 | 0.3 | identical |
| 2 | 0.5 | slight_variation |
| 3 | 0.7 | perspective_diversity |
| 4 | 0.9 | perspective_plus_model_mix |
| 5 | 1.2 | adversarial_perspectives |

Temperature independently affects output quality (especially for analytical tasks at high temperature). Any observed Q(d) curve could be entirely explained by temperature degradation, not by disagreement dynamics. You cannot distinguish:
- "Moderate disagreement improves quality" (the claim)
- "Temperature 0.7 happens to produce better outputs than 0.3 or 1.2" (confound)

**Fix:** Add control conditions that vary temperature with identical prompts, and vary prompts at fixed temperature. This requires approximately 120 additional runs (2 task_types x 8 tasks x 5 temperatures x 1 rep at flat/single-agent + 2 x 8 x 5 prompt_strategies x 1 rep at fixed temp=0.7).

### 4.2 Judge Model = Pipeline Agent (CRITICAL)

`config/models.yaml` specifies:
```yaml
judge:
  model: gemini-2.0-flash
```

Gemini Flash is also a pipeline agent (in `heterogeneous_strong_mix` and as `homogeneous_flash`). The team's own `EVALUATION_METHODOLOGY.md` explicitly warns:

> "Ensure judge models are NEVER used as pipeline agents. Separate model pools."

This creates a **self-preference bias** confound: the judge may systematically favor outputs produced by itself or penalize outputs from other models.

**Fix:** Use a model not in the pipeline as judge (e.g., GPT-4o-mini, or a different Gemini variant). Alternatively, use a multi-judge panel as specified in the evaluation methodology.

### 4.3 Missing Singleton Control in Blocks 2, 3, 5

Block 2 tests agent_counts [2, 5] but not [1]. Block 5 tests [3, 5] but not [1]. Without the single-agent baseline in the same block (same disagreement level, same model mix logic), you cannot compute the *improvement* from multi-agent orchestration. Block 0 provides a baseline but at disagreement_level=1, not level 3 (Block 2) or level 2 (Block 5).

**Fix:** Add agent_count=1 to Blocks 2 and 5 (adds 2 x 8 x 1 x 3 x 3 x 1 = 144 runs to B2 and 144 to B5 — significant but necessary for interpretable contrasts). Alternatively, run a small "bridge" calibration at the matching disagreement levels.

### 4.4 No Temporal Drift Control

LLM API behavior can change between sessions (provider-side updates, load-dependent routing). The experiment plan notes this in "Limitations" but provides no mitigation. A run that takes 24 hours could face different model behavior at hour 1 vs hour 23.

**Fix:** Re-run a random 5-10% of completed runs at the end and compute a drift statistic (paired difference in quality score). If drift is significant, report it as a limitation or use block randomization.

---

## 5. Reproducibility Assessment

### What's good
- Deterministic manifest generation with fixed seed (42)
- Full config metadata stored per run (task_id, topology, consensus, model_assignment, etc.)
- Atomic write + resume scanning for crash-safe continuation
- Requirements pinned with minimum versions
- Clear README with step-by-step setup
- `.env.example` for API key management
- Manifest snapshot saved to results directory

### Reproducibility gaps

1. **LLM API non-determinism.** Even with seed=42 in the manifest and context.seed, the LLM APIs are non-deterministic at temperature > 0. The same manifest will produce different outputs on re-run. This is inherent to the problem, but the paper must acknowledge that "reproducibility" means "same experimental conditions," not "same outputs."

2. **No model version pinning.** The config specifies `api_model: claude-opus-4-1-20250805` (good — date-pinned) and `api_model: gemini-2.5-pro` (bad — not version-pinned). Gemini model behavior changes over time. Six months from now, `gemini-2.5-pro` might route to a different checkpoint.

3. **No requirements lock file.** `requirements.txt` uses `>=` minimum versions but no upper bounds or lock file. `pip install -r requirements.txt` in March 2026 vs. September 2026 may produce different dependency trees. Add a `requirements-lock.txt` or use `pip freeze` output.

4. **Task prompts not version-controlled independently.** The YAML task files are in the repo, which is good. But there is no hash or version tag on the task corpus. If someone edits a prompt and re-generates the manifest, the same `run_id` hash could map to different content (the hash includes `task["id"]` but not the full prompt text — the `metadata` field includes `prompt_hash`, but this does not affect the `run_id`).

5. **No `pyproject.toml` entry point documented.** The `pyproject.toml` exists but the README quick-start uses `python scripts/...` which works but is fragile for path resolution.

6. **No CI/CD or test coverage report.** Tests exist (`tests/`) but there is no mention of CI, and no coverage report. For reproducibility, automated test verification on clone is important.

---

## 6. Checkpoint/Recovery Robustness

### What's sound
- Atomic write pattern (`tempfile` + `os.fsync` + `os.replace`) is correct on both POSIX and Windows (Python 3.3+).
- Resume scans `.json` files and skips completed runs. Clean and simple.
- Progress snapshot provides live monitoring without database dependency.

### Edge cases and issues

1. **Failed runs are never retried on resume (by design, but undocumented).** `load_completed_ids()` treats ALL `.json` files (except `progress.json` and `manifest_snapshot.json`) as completed, including failed runs (`status: "failed"`). After exhausting `max_retries=5`, a failed run is permanently marked done. If the failure was due to a transient provider outage, the only recovery is manually deleting the `.json` file. This should be documented, and ideally there should be a `--retry-failed` flag.

2. **`threading.Lock` in async context.** `CheckpointManager` uses `threading.Lock`, but it is called from `async` coroutines in the runner. Since asyncio is single-threaded, this will not deadlock, but `os.fsync` + `os.replace` will **block the event loop** during file I/O. For small JSON files (~5KB), latency is negligible (~1ms). But if results grow large (long outputs) or if the filesystem is slow (network mount), this could become a bottleneck. Consider `asyncio.Lock` + `loop.run_in_executor` for the file operations.

3. **Cost tracker state is not recovered on resume.** `CostTracker` initializes all counters to zero. After a crash and resume, `snapshot()` reports only the cost of the resumed session, not the cumulative cost. The `cost_log.jsonl` preserves history, but `progress.json` will show inaccurate cost figures post-resume. **Fix:** On startup with `--resume`, parse `cost_log.jsonl` to reconstruct cumulative counters.

4. **No manifest integrity check on resume.** The runner saves a `manifest_snapshot.json` but does not verify that the current manifest matches the saved snapshot on resume. If someone edits `experiment_matrix.yaml` between sessions, the resumed run could mix results from different experimental designs.

---

## 7. Cost-Effectiveness Assessment

### Current budget allocation (estimated)

Using pricing from `models.yaml` and assuming ~1,500 tokens input + ~1,000 tokens output per call:

The quorum topology generates **2 API calls per agent** (draft + revision). For multi-agent runs, total calls scale as `2 x agent_count` for quorum, plus potential judge calls.

**Rough cost model per run** (quorum topology, n=3, heterogeneous_strong_mix):
- 3 draft calls: 1 x Opus ($0.09) + 1 x Gemini Pro ($0.007) + 1 x Flash ($0.0006) = approx $0.10
- 3 revision calls: same = approx $0.10
- Judge call (Flash): approx $0.001
- Total per run: approx $0.20

At ~1,700 runs (not all quorum, varying n): estimated $400-$800 for API calls alone. With Opus-heavy blocks, could reach $1,000-$2,000.

### Budget redistribution recommendations

| Action | Runs saved/added | Budget impact | Justification |
|--------|-----------------|---------------|---------------|
| De-duplicate Block 3 thresholds | -192 runs | Saves ~$40-80 | Redundant executions |
| Increase Block 3 reps to 4 | +192 runs | Costs ~$40-80 | Adequate power for MVQ |
| Increase Block 4 reps to 6 | +96 runs | Costs ~$20-40 | Better Paradox detection |
| Increase Block 1 reps to 3 | +240 runs | Costs ~$50-100 | Stronger inverted-U test |
| Add temperature control runs | +80 runs | Costs ~$16-30 | Confound control |
| Add singleton to B2/B5 | +120-288 runs | Costs ~$15-40 | Interpretable baselines |

**Net change:** approximately +340-500 additional runs, roughly +$100-250.

**Key reallocation:** Move budget from Block 3 waste (-$40-80) and Block 5 (which is underpowered regardless) into Block 4 replication and confound controls.

If total budget is constrained to $3K, consider **dropping Block 5 entirely** and redistributing its 288 runs' worth of budget into Blocks 1-4 replication. Block 5's ANOVA interaction probe is the most ambitious analysis but has the least power. Better to have four well-powered blocks than five underpowered ones.

---

## 8. What Analyses Can Actually Be Run?

Given the current design (assuming evaluation is fixed), here is an honest assessment:

### Analyses with adequate power

| Analysis | Block | Method | Status |
|----------|-------|--------|--------|
| Per-model baseline quality | B0 | Descriptive stats, bootstrap CIs | Valid |
| Pairwise model comparisons | B0 | Paired t-test / Wilcoxon, 32 pairs | Valid |
| Q(d) trend (collapsed over n) | B1 | Polynomial regression, n approx 48/level | Valid for medium+ effects |
| Q(d) per agent count | B1 | Trend test, n approx 16/cell | Marginal |
| Paradox: Q(3) vs Q(2) | B4 | Paired test, 32 pairs | Valid for d >= 0.5 |

### Analyses that are underpowered

| Analysis | Block | Method | Problem |
|----------|-------|--------|---------|
| Topology main effect | B2 | One-way ANOVA, 8/cell | Power ~0.35 for medium f |
| Consensus main effect | B2 | One-way ANOVA, 8/cell | Same |
| Topology x consensus interaction | B2, B5 | Two-way ANOVA, 8/cell | Power ~0.20 for medium f |
| MVQ threshold attainment curves | B3 | Proportion estimation, 8 trials | CI +/-35% |
| Three-way interaction (topology x consensus x task_type) | B5 | Factorial ANOVA | Essentially impossible |
| Paradox for small effects (d < 0.4) | B4 | Paired test | Power < 0.50 |

### Recommended analysis strategy

1. **Use non-parametric tests** throughout (Wilcoxon, Kruskal-Wallis) — LLM quality scores are unlikely to be normally distributed, and small samples make normality assumptions dangerous.
2. **Report effect sizes (Cohen's d, eta-squared) with bootstrap CIs** for ALL comparisons, not just p-values.
3. **Use mixed-effects models** with task instance as random effect — this accounts for the repeated-measures structure (same 16 tasks across conditions) and improves power.
4. **Apply Holm-Bonferroni correction** for all pairwise comparisons within each research question.
5. **Pre-register the analysis plan** including which comparisons are confirmatory vs. exploratory.
6. **For Blocks 2 and 5:** present results as **exploratory screening** with descriptive statistics and ranked effect sizes, not as confirmatory ANOVA. Be honest: "we observe a trend favoring topology X, but the design is underpowered for confirmatory inference."

---

## 9. Additional Design Concerns

### 9.1 Disagreement measurement uses only lexical metrics

The `disagreement.py` module computes:
- Pairwise Jaccard similarity (token overlap)
- Response entropy (exact string matching after normalization)

The `EVALUATION_METHODOLOGY.md` recommends semantic similarity via embeddings. Jaccard similarity is a poor proxy for conceptual disagreement: two responses can use entirely different words while making the same argument (Jaccard approx 0, true disagreement approx 0), or use the same words while reaching opposite conclusions (Jaccard approx 1, true disagreement = 1).

**Fix:** Add BERTScore or sentence-transformer cosine similarity as the primary disagreement metric. Keep Jaccard as a supplementary surface-level measure.

### 9.2 Quorum topology peer-review is circular, not all-to-all

In `quorum.py`, each agent revises based on ONE peer:
```python
peer = draft_outputs[(idx + 1) % len(draft_outputs)]
```

Agent 0 sees Agent 1's draft, Agent 1 sees Agent 2's, etc. This ring topology means:
- No agent sees more than one peer
- No agent's draft is seen by more than one peer
- The revision quality depends on the **order** of agents, which is deterministic by model_assignment index

This could create systematic bias in the paradox block: if the strong model (Opus, index 0) always revises based on a weak model (Flash, index 1), the revision quality is confounded with the assignment order.

**Fix:** Either randomize the peer assignment per run (using the run seed), or implement all-to-all critique where each agent sees all other drafts.

### 9.3 Prompt strategy diversity is limited

The `_build_system_prompts` function for `perspective_diversity` cycles through 5 hardcoded perspectives:
```python
perspectives = [skeptical, systems-level, pragmatic, risk-management, contrarian]
```

For agent_count=5, each agent gets a unique perspective. For agent_count=2 or 3, only a subset is used, always in the same order. This means the "disagreement" induced by perspective prompts is confounded with which specific perspectives happen to be assigned.

**Fix:** Randomize perspective assignment per run (using run seed).

### 9.4 `retry_with_backoff` catches ALL exceptions

```python
retryable_exceptions=(Exception,),
```

This retries on ANY exception, including:
- `ValueError` (bad config — should fail fast)
- `KeyError` (missing task — should fail fast)
- JSON parse errors (should fail fast)

Only transient API errors (rate limits, timeouts, 5xx) should trigger retries.

**Fix:** Narrow `retryable_exceptions` to API-specific error classes.

---

## 10. Summary of Required Actions

### Must fix before spending any budget

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| F1 | **Implement real evaluation metrics** (LLM-judge scoring, not heuristic text similarity) | Fatal — all analyses invalid without this | High (2-3 days) |
| F2 | **Separate judge model from pipeline models** | Self-preference confound | Low (config change) |
| F3 | **De-duplicate Block 3 thresholds** | Wastes 75% of block budget | Low (matrix edit) |
| F4 | **Add temperature/prompt confound controls** | Cannot interpret RQ1 results | Medium (add ~80 runs) |

### Should fix before spending budget

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| S1 | Increase Block 3 repetitions to >=4 | MVQ curves uninterpretable at 8/cell | Low (matrix edit) |
| S2 | Increase Block 4 repetitions to 6 | Paradox detection for subtle effects | Low (matrix edit) |
| S3 | Add singleton baseline to Blocks 2/5 | No interpretable multi-agent improvement metric | Medium |
| S4 | Randomize quorum peer assignment | Order confound in revision | Low (code fix) |
| S5 | Reconstruct cost tracker on resume | Inaccurate cost reporting | Low (code fix) |
| S6 | Add `--retry-failed` flag | Transient failures permanently lost | Low (code fix) |
| S7 | Implement semantic disagreement metrics | Jaccard does not equal conceptual disagreement | Medium |
| S8 | Pin Gemini model versions | Reproducibility gap | Low (config change) |

### Nice to have

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| N1 | Temporal drift check (re-run 5% at end) | Detect provider-side drift | Low |
| N2 | Manifest integrity check on resume | Prevent mixed-design results | Low |
| N3 | Narrow retry exception types | Prevent retrying permanent errors | Low |
| N4 | Add requirements lock file | Dependency reproducibility | Trivial |
| N5 | Consider dropping Block 5 for budget reallocation | Better power elsewhere | Design decision |

---

## 11. Verdict Elaboration

The experiment framework demonstrates **strong engineering** (atomic checkpointing, deterministic manifests, clean async architecture, comprehensive evaluation methodology document) but suffers from a **fatal implementation gap**: the dependent variable (quality score) is measured using heuristic text statistics that have no construct validity, while the correct evaluation protocol sits unimplemented in `EVALUATION_METHODOLOGY.md`.

Beyond the evaluation crisis, the statistical design has genuine power problems in Blocks 2, 3, and 5 that would leave key claims (topology effects, MVQ curves, interaction effects) unsupported. The temperature-prompt confound threatens the interpretability of RQ1 (Disagreement Dividend). These are fixable within the existing architecture and approximate budget.

**My recommendation:** Fix items F1-F4, then S1-S4, then re-estimate costs. If total budget allows, run the revised design. If budget is tight, sacrifice Block 5 and redistribute. The paper will be stronger with four well-powered, cleanly evaluated blocks than with six blocks built on heuristic metrics and underpowered cells.

The team is close. The infrastructure is solid. The methodology thinking (per `EVALUATION_METHODOLOGY.md`) is excellent. They just need to connect the theory to the implementation.

---

*Dr. James Okonkwo*  
*Fellow, Royal Statistical Society*  
*"If your dependent variable doesn't measure what you claim, your p-values are just expensive random numbers."*
