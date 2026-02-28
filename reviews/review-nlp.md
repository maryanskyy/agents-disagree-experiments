# Committee Review: Metrics & Evaluation Assessment

**Reviewer:** Dr. Priya Sharma (NLP/LLM Evaluation)
**Date:** 2026-02-28
**Reviewed Materials:** EVALUATION_METHODOLOGY.md, EXPERIMENT_PLAN.md, llm_judge.py, metrics.py, disagreement.py, task configs, research questions

---

## Verdict: **Needs Revision Before Running**

## Confidence: **High**

The evaluation methodology document (EVALUATION_METHODOLOGY.md) is genuinely excellent — one of the most thorough threat analyses I have seen for multi-agent evaluation. But the implemented code diverges from it in fundamental ways. The methodology recommends pairwise comparison with Bradley-Terry; the code implements absolute scoring. The methodology requires multi-judge panels; the code uses a single judge. The methodology demands calibration anchors; the code has none. Running the experiment with the current implementation would produce results that the team's own methodology document argues are inadequate. This is fixable, but it must be fixed BEFORE spending $3-8K.

---

## 1. Metric Alignment Matrix

| Research Question | Required Measurement | Current Metric | Adequacy | Gap |
|---|---|---|---|---|
| **RQ1: Disagreement Dividend** — Q(d) inverted-U curve | Quality as function of disagreement level | LLM judge absolute score (0-1) + Jaccard-based disagreement | PARTIAL | Disagreement is surface-level only (Jaccard); quality via absolute scoring is subject to overrating that flattens the curve. Cannot reliably detect inverted-U shape. |
| **RQ2: Topology Effect** — quality-cost-latency tradeoffs | Comparative quality across topologies | LLM judge absolute score (0-1) | PARTIAL | Absolute scoring makes topologies look similar due to overrating compression. Pairwise comparison would force discrimination. Cost/latency metrics not in evaluation code. |
| **RQ3: MVQ Behavior** — threshold attainment vs quorum size | Quality threshold probability at each n | LLM judge absolute score + threshold check | PARTIAL | Threshold attainment requires calibrated absolute scores. Without calibration anchors, the threshold is meaningless — a biased judge with overrating tendency will show threshold attainment earlier than reality. |
| **RQ4: Quorum Paradox** — quality decrease when adding agents | Fine-grained quality discrimination at Q(n) vs Q(n+1) | LLM judge absolute score | INADEQUATE | This is the headline claim and it requires detecting SUBTLE quality decreases. Absolute scoring with a single judge and known overrating bias is the worst possible method. The methodology correctly flags 100% human validation for paradox cases, but no infrastructure exists. |

### Verdict on metric alignment:
- **RQ1-RQ3:** Metrics partially capture the intended phenomena but with known bias vulnerabilities that could produce misleading results. Fixable with code changes.
- **RQ4:** Current measurement approach is fundamentally inadequate for the claim being made. The Quorum Paradox requires the finest-grained quality discrimination at exactly the point where LLM-as-judge is weakest.

---

## 2. LLM-as-Judge Implementation: Critical Code-Methodology Disconnect

### 2.1 The Core Problem: Absolute Scoring vs. Pairwise Comparison

The evaluation methodology document explicitly states (Section 5.1):

> **Recommendation:** Use **pairwise comparison as the primary evaluation mode**, with Bradley-Terry aggregation to produce global rankings. [...] Specifically, adopt the **BT-sigma model** from Qian et al. (2026).

The code (llm_judge.py) implements:

```python
"Evaluate each candidate from 0.0 to 1.0.\n"
"Return strict JSON: {\"scores\":[...],\"rationale\":\"...\"}\n\n"
```

This is absolute scoring — precisely what the methodology warns against. The methodology document spends multiple pages explaining why pairwise comparison is "strictly preferred" for this use case, including:
- Eliminating absolute calibration issues
- Forcing discrimination between similar outputs (critical for Quorum Paradox detection)
- Well-founded statistical model (Bradley-Terry)

**This must change before running.** The entire aggregation pipeline, statistical analysis, and result interpretation depend on whether you use absolute or pairwise scoring. Retrofitting after data collection is not possible.

### 2.2 Position Randomization: Half-Implemented

The methodology recommends bidirectional evaluation: present each pair in BOTH orderings, check consistency, flag disagreements.

The code does ONE random shuffle per evaluation call:

```python
rng = random.Random(seed)
indices = list(range(len(outputs)))
rng.shuffle(indices)
```

This randomizes order (good) but does not evaluate in both orderings (incomplete). There is no consistency check, no tie detection, and no mechanism to flag uncertain items for human review.

### 2.3 Single Judge Model

The code takes one `judge_client`. The methodology requires three judges from at least two model families. There is no:
- Multi-judge orchestration
- Inter-rater reliability computation (Cohen's kappa)
- BT-sigma aggregation
- Judge calibration tracking

### 2.4 No Calibration Mechanism

The methodology specifies ~20 calibration anchor items with known quality levels interleaved in every batch. The code has zero calibration infrastructure. Without calibration:
- You cannot detect judge drift
- You cannot weight judges by reliability
- You cannot validate that the 0-1 scale means anything consistent

### 2.5 Heuristic Fallback Is Dangerously Misleading

The dry-run heuristic in metrics.py:

```python
def evaluate_analytical(reference: str, candidate: str) -> float:
    overlap = jaccard_similarity(reference, candidate)
    seq = sequence_similarity(reference, candidate)
    length_penalty = min(1.0, len(candidate.split()) / max(1, len(reference.split())))
    return max(0.0, min(1.0, 0.5 * overlap + 0.4 * seq + 0.1 * length_penalty))
```

This measures similarity between the **task prompt** and the **candidate output**. It rewards outputs that parrot the prompt back. This is not analytical quality — it is echo scoring. Any dry-run testing or development iteration using this heuristic will produce misleading signal about which pipeline configurations are "working."

The creative heuristic targets an average sentence length of 18 words and rewards lexical diversity. This penalizes both terse literary prose and complex academic sentences. It cannot distinguish a creative masterpiece from random diverse vocabulary.

**Recommendation:** Label these explicitly as `_placeholder_dev_only` and add an assertion or warning that fires if they are called outside dry-run mode.

---

## 3. Bias Risks Remaining

### 3.1 Mitigated (Partially)

| Bias | Mitigation Status | Residual Risk |
|---|---|---|
| **Position bias** | Randomization implemented, but single-pass only | MEDIUM — without bidirectional check, position effects survive in ~15-25% of cases per Wang et al. (2023) |
| **Self-preference** | Methodology says separate agent and judge pools | LOW if enforced — but not enforced in code. `judge_client` could be any model, including one used as a pipeline agent. |

### 3.2 Unmitigated

| Bias | Status | Impact on Results |
|---|---|---|
| **Verbosity bias** | NOT addressed in code | HIGH — multi-agent consensus outputs will be longer. Length normalization is mentioned in methodology but not implemented. No judge instruction to ignore length. No length-controlled analysis. This alone could produce a spurious "multi-agent wins" finding. |
| **Overlap/style bias (Fang et al. 2026)** | NOT addressed | HIGH — iteratively refined multi-agent outputs are "more LLM-like." LLM judges prefer LLM-like text. This is a confound that systematically inflates multi-agent scores. |
| **Overrating bias (Yu et al. 2026)** | NOT addressed | HIGH for RQ3/RQ4 — if the judge gives everything 0.7-0.9, quality differences between quorum sizes collapse. The paradox becomes undetectable. |
| **Rubric-Induced Preference Drift** | NOT addressed | MEDIUM — the rubrics in YAML are reasonable but untested. No sensitivity analysis planned in code. |
| **Blind evaluation gaps** | Partial — candidates are anonymized by position, but no stripping of multi-agent provenance signals | MEDIUM — consensus outputs may contain telltale artifacts (numbered perspectives, "synthesizing the above," hedged language from debate rounds). A judge could infer provenance. |

### 3.3 The Systematic Bias Direction

Here is what concerns me most: **three major unmitigated biases all push in the same direction.** Verbosity bias favors longer multi-agent outputs. Overlap/style bias favors more-LLM-like consensus text. Overrating bias compresses the scale, making it harder to detect when multi-agent is worse (Quorum Paradox). Together, these create a systematic pro-multi-agent bias that could produce a false Disagreement Dividend (RQ1) while hiding the Quorum Paradox (RQ4).

This is exactly the scenario a hostile reviewer would construct.

---

## 4. Benchmarking Validity

### 4.1 "Single Agent" as Control

The experiment plan uses single-agent output as the baseline. This is the correct control for the central question (does multi-agent improve over single?). However:

1. **Which single agent?** The plan uses three models (Opus, Gemini Pro, Gemini Flash) of very different capability. The single-agent baseline must be the BEST single agent, not the average. Otherwise you are comparing a 5-agent ensemble against a weak baseline, which inflates the apparent benefit of multi-agent.

2. **Temperature matching.** Multi-agent setups use temperature schedules for disagreement control. The single-agent baseline should use the SAME temperature as the mean temperature across multi-agent runs — or better, sweep temperature and report the best single-agent result at any temperature. A single agent at temperature 0.9 might match a 3-agent quorum at temperature 0.3.

3. **Cost-adjusted comparison.** A 5-agent pipeline makes 5x the API calls. The fair comparison is: does 5 agents with budget $X outperform 1 agent with 5 attempts and best-of-5 selection at the same budget $X? The experiment plan mentions cost-adjusted comparison for RQ2 but the evaluation code has no mechanism for this.

### 4.2 Missing Baselines

- **Best-of-N single agent:** Generate N independent outputs from the best model and select the best (by the same judge). This is the natural cost-matched baseline for multi-agent systems. Without this, the multi-agent benefit may be entirely attributable to having more samples, not to interaction/consensus.
- **Self-consistency / majority voting:** A single model generating multiple responses and self-selecting. Standard in reasoning literature. Necessary to isolate the benefit of agent INTERACTION from the benefit of SAMPLING.
- **Oracle upper bound:** For analytical tasks with verifiable answers, what score would a human expert achieve? Provides ceiling context.

The absence of best-of-N is the most concerning gap. If a hostile reviewer shows that best-of-5 with Claude Opus matches or exceeds the 5-agent quorum, the entire multi-agent narrative collapses.

---

## 5. Creative Task Evaluation

### 5.1 Can LLM-as-Judge Reliably Evaluate Creative Outputs?

Short answer: **not reliably enough for publication-grade claims without substantial human validation.**

The creative tasks in creative.yaml are excellent — genuinely challenging and diverse. But the evaluation rubrics require assessing "emotional resonance," "craft," "voice distinction," "stylistic inspiration vs. imitation," and "dramatic believability." These are precisely the dimensions where:

1. LLM judges exhibit the strongest overrating bias (Yu et al. 2026)
2. Human-LLM agreement is lowest (Zheng et al. 2023 report this)
3. Overlap/style bias is most dangerous (Fang et al. 2026) — the judge may penalize outputs that are genuinely creative but stylistically non-standard

### 5.2 Human Validation Requirements for Creative Tasks

The methodology recommends 25% human validation for creative tasks. I would argue this is the **minimum** and should be treated as a hard requirement, not aspirational. For any creative task result that supports the paper's claims (especially the Disagreement Dividend for creative tasks), human validation should be 100%.

Specifically:
- The claim that d*_creative > d*_analytical (from RQ1) is entirely about creative evaluation reliability
- If LLM judges systematically overrate all creative outputs, the inverted-U curve for creative tasks will be flattened, producing a false null or false location of d*

### 5.3 No Human Evaluation Infrastructure

Despite the methodology's clear requirements, there is no code, no tooling, no sampling logic, and no interface for human evaluation. This needs to exist before running — even a simple spreadsheet-based workflow with randomized presentation order.

---

## 6. Metric-to-RQ Gaps (Where Measurement Does Not Prove the Hypothesis)

### Gap 1: Disagreement Measurement Is Surface-Level (Affects RQ1 Fundamentally)

The research questions define disagreement as diversity of perspective and reasoning approach. The code measures it with Jaccard similarity on tokens. Two agents could give semantically identical analyses using different vocabulary and show high "disagreement" by Jaccard. Conversely, two agents could use identical vocabulary to express opposite conclusions and show low "disagreement."

**What this means:** The Q(d) curve in RQ1, where d = 1 - pairwise_similarity via Jaccard, may be measuring lexical diversity rather than genuine disagreement. The "Disagreement Dividend" could be a "Vocabulary Diversity Dividend" — a much weaker finding.

**Fix:** Add embedding-based semantic similarity (the methodology already recommends this). Use sentence-transformers to compute pairwise cosine similarity. Report both Jaccard and semantic disagreement and check whether the Q(d) relationship holds for both.

### Gap 2: Vote Entropy Requires Exact String Matching (Affects RQ3)

In disagreement.py:

```python
def response_entropy(outputs: list[str]) -> float:
    normalized = [normalize_text(text) for text in outputs]
    counts = Counter(normalized)
```

This counts exact normalized string matches. For open-ended generation (which is most of the task set), every output will be unique, and entropy will always be maximal. This metric is useless for generation tasks — it only works for multiple-choice or very short responses.

**Fix:** Cluster outputs by semantic similarity (using embeddings) before computing entropy, or drop this metric for generation tasks.

### Gap 3: No Mediation Analysis for Quorum Paradox (Affects RQ4)

The research questions specify two paradox mechanisms: (a) conformity pressure measured by similarity-to-centroid over debate rounds, and (b) correlation-induced diversity collapse. The evaluation code measures neither. There are no per-round metrics, no centroid computation, no correlation estimation.

Without mediation analysis, even if you detect the paradox, you cannot explain WHY it occurs — making the result descriptive rather than mechanistic. A hostile reviewer will note that without mechanism evidence, the "paradox" might be an artifact of evaluation noise.

### Gap 4: No Cost-Efficiency Metrics (Affects RQ2)

The experiment plan mentions cost-adjusted comparison, but evaluation code contains no cost tracking, no tokens-per-quality-point calculation, no Pareto frontier analysis. RQ2's hypothesis is specifically about quality-cost-latency tradeoffs — you need these in the evaluation pipeline, not as a post-hoc afterthought.

---

## 7. What Would a Hostile Reviewer Attack?

Ranked by severity:

### Attack 1: "Your implementation contradicts your own methodology" (DEVASTATING)

A reviewer who reads both EVALUATION_METHODOLOGY.md and the code will immediately note that the methodology recommends pairwise BT comparison while the code implements absolute scoring. This undermines the paper's credibility fundamentally — it suggests either the methodology was written as window dressing, or the team ran experiments before the methodology was finalized. Either way, the evaluation results become suspect.

### Attack 2: "Best-of-N is the obvious baseline and you did not include it" (MAJOR)

Without best-of-N, the multi-agent benefit cannot be separated from the sampling benefit. This is a standard critique in the ensemble learning literature and any ML reviewer will raise it.

### Attack 3: "Verbosity and style bias systematically favor your multi-agent condition" (MAJOR)

Multi-agent consensus outputs are likely longer and more polished. Three unmitigated biases (verbosity, overlap/style, overrating) all favor the multi-agent condition. Without length-controlled analysis and human validation, a reviewer can argue the results are explained by evaluation bias rather than genuine quality improvement.

### Attack 4: "Jaccard similarity does not measure real disagreement" (MAJOR)

The central variable of the paper — disagreement — is measured with a surface-level metric. If the Q(d) curve does not hold under semantic similarity, the core finding is an artifact.

### Attack 5: "16 tasks, single judge, no calibration — insufficient for your factorial design" (SIGNIFICANT)

16 tasks across 36+ experimental conditions means fewer than 1 observation per cell for some interactions. A single judge with no calibration provides a single biased measurement per output. The statistical analysis (ANOVA for interactions) requires much more data per cell than this provides.

### Attack 6: "Your Quorum Paradox claim has no human validation" (SIGNIFICANT)

The methodology requires 100% human validation for paradox cases. If the paper claims to demonstrate the paradox without human validation, it is vulnerable to dismissal as evaluation noise.

### Attack 7: "Creative task evaluation with LLM-only judges is methodologically unsound" (MODERATE)

Reviewers at *CL venues will flag this. Creative evaluation is the weakest link in LLM-as-judge methodology, and it is half the task set.

---

## 8. Critical Evaluation Gaps (Must Fix Before Running)

### MUST FIX (blocking — do not spend money until resolved)

**1. Implement pairwise comparison with Bradley-Terry aggregation.**
- Replace `score_candidates` absolute scoring with pairwise comparison
- Each pair evaluated in both orderings
- Aggregate using Bradley-Terry (or BT-sigma if implementing multi-judge)
- This is the single most important change

**2. Implement multi-judge panel (minimum 3 models, 2 families).**
- Orchestrate scoring across GPT-4o, Claude 3.5, Gemini Pro (or equivalent)
- Compute inter-judge Cohen's kappa per dimension
- Reject any dimension where kappa < 0.40

**3. Add embedding-based semantic similarity for disagreement measurement.**
- Supplement Jaccard with sentence-transformer cosine similarity
- This is required for the core RQ1 variable to be credible

**4. Add best-of-N baseline.**
- For each multi-agent condition with N agents, generate N independent outputs from the best single model
- Select best by the same evaluation method
- This is the cost-matched control

**5. Build human evaluation pipeline.**
- Even minimal: randomized pairwise presentation, 2 raters per item
- Required: 15% analytical, 25% creative, 100% paradox cases
- Without this infrastructure, the experiment produces unpublishable results

### SHOULD FIX (high priority, significant risk if omitted)

**6. Add verbosity controls.**
- Instruct judge to ignore length differences explicitly in prompt
- Compute output length per condition and run length-stratified analysis
- Consider truncation normalization

**7. Strip multi-agent provenance signals before evaluation.**
- Remove synthesis markers, numbered perspectives, debate artifacts
- Normalize formatting across single-agent and multi-agent outputs

**8. Add calibration anchor system.**
- 20 items with known quality levels
- Interleave in evaluation batches
- Track per-judge calibration accuracy

**9. Fix response_entropy for generation tasks.**
- Current exact-match clustering is useless for open-ended generation
- Implement embedding-based clustering before entropy computation

**10. Rename/guard heuristic metrics.**
- The evaluate_analytical function measures prompt echo, not analytical quality
- Add runtime guards preventing use outside dry-run mode
- Rename to _dev_heuristic_analytical to prevent confusion

### NICE TO HAVE (would strengthen, not blocking)

**11. Rubric sensitivity analysis.**
- Test 2-3 rubric phrasings on calibration items
- Report stability of rankings

**12. Per-round metrics for mediation analysis.**
- Compute similarity-to-centroid per debate round
- Track unique idea count per round (for paradox mechanism evidence)

**13. Cost tracking in evaluation pipeline.**
- Token counts per run
- Quality-per-dollar computation
- Pareto frontier visualization

---

## 9. Specific Code Changes Required

### llm_judge.py — Major Rewrite

```python
# CURRENT (absolute scoring):
"Evaluate each candidate from 0.0 to 1.0.\n"
"Return strict JSON: {\"scores\":[...],\"rationale\":\"...\"}\n\n"

# REQUIRED (pairwise comparison):
# New method: compare_pair(output_a, output_b) -> PairResult
# - Present pair in order A,B and B,A
# - Ask: "Which response is better for [dimension]? A, B, or tie."
# - Return: winner, confidence, consistency (both orderings agree?)
# - Aggregate across pairs using Bradley-Terry model
```

Required new class structure:
- `PairwiseJudge` with `compare_pair()` method
- `JudgePanel` orchestrating 3+ judge models
- `BradleyTerryAggregator` for producing global rankings
- `CalibrationTracker` for monitoring anchor accuracy
- Keep absolute scoring as supplementary (for dimension-level detail)

### disagreement.py — Add Semantic Metrics

```python
# ADD:
def semantic_pairwise_similarity(outputs, model_name="all-MiniLM-L6-v2"):
    """Embedding-based pairwise cosine similarity."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(outputs)
    # compute pairwise cosine similarities
    ...

def semantic_disagreement(outputs):
    """1 - mean pairwise semantic similarity."""
    return 1.0 - semantic_pairwise_similarity(outputs)

def clustered_entropy(outputs, similarity_threshold=0.85):
    """Cluster outputs by semantic similarity, then compute entropy over clusters."""
    ...
```

### metrics.py — Guard Heuristics

```python
def evaluate_analytical(reference, candidate):
    """DEV-ONLY heuristic. Measures prompt-output overlap, NOT analytical quality.
    
    WARNING: Do not use for production evaluation. This is a dry-run placeholder.
    """
    import warnings
    warnings.warn("evaluate_analytical is a dev heuristic, not a quality metric", stacklevel=2)
    ...
```

---

## 10. Summary Assessment

### What is done well:
- The EVALUATION_METHODOLOGY.md is outstanding — thorough, well-referenced, honest about limitations
- Task instances are carefully designed with rubrics
- The experiment plan has clear falsification criteria
- Position randomization (partial) is implemented
- The threat model is comprehensive

### What is critically missing:
- The code does not implement the methodology's own recommendations
- No pairwise comparison, no multi-judge, no calibration, no human eval pipeline
- Disagreement measured at surface level only
- Missing best-of-N baseline
- Systematic pro-multi-agent bias from three unmitigated sources

### Bottom line:
The team has done excellent theoretical preparation. The methodology document could stand on its own as a mini-survey. But the implementation gap is large enough that running the experiment now would produce results that the team's own methodology document would reject. The $3-8K investment is not at risk of being wasted — it is at risk of producing results that are unpublishable or, worse, misleading.

**Invest 1-2 weeks of engineering to align the code with the methodology, then run.** The methodology is sound. The code is not ready.

---

*Dr. Priya Sharma*
*Senior Research Scientist, LLM Evaluation*
*Area Chair, ACL/EMNLP*
