# Non-LLM Quality Assessment Frameworks for Multi-Agent Output Evaluation

**Author:** Elena Chen (PhD-1, Theoretical Track)
**Date:** 2026-02-28
**Task:** T-029a - Independent validation layer for multi-agent LLM pipeline outputs
**Context:** Analytical essays + creative writing from multi-agent pipelines; current LLM-as-judge (Bradley-Terry) produces 85-93% ties. We need non-LLM methods as a complementary sanity check.

---

## Table of Contents

1. [Motivation and Design Principles](#1-motivation-and-design-principles)
2. [Category 1: Information-Theoretic Metrics](#2-information-theoretic-metrics)
3. [Category 2: Textual Statistics](#3-textual-statistics)
4. [Category 3: Embedding-Based Metrics](#4-embedding-based-metrics)
5. [Category 4: Argument / Discourse Structure](#5-argument--discourse-structure)
6. [Category 5: Reference-Free Quality Estimation](#6-reference-free-quality-estimation)
7. [Category 6: Statistical Concordance Methods](#7-statistical-concordance-methods)
8. [Comparative Evaluation Matrix](#8-comparative-evaluation-matrix)
9. [Top 3 Recommendations](#9-top-3-recommendations)
10. [Implementation Plan](#10-implementation-plan)
11. [References](#11-references)

---

## 1. Motivation and Design Principles

Our experimental design relies on pairwise Bradley-Terry scoring via LLM judges. The pilot exposed a critical flaw: judges default to "tie" verdicts (85-93% of comparisons), producing degenerate preference matrices with near-zero discriminative power. This makes the Bradley-Terry parameter estimates unreliable -- the likelihood surface is nearly flat when most comparisons are ties.

**We do NOT need a replacement for LLM judges.** We need an *independent, orthogonal* validation layer that:

1. **Measures different properties** than holistic LLM judgment (which conflates fluency, relevance, coherence, and style into a single preference).
2. **Produces continuous scores** (not binary preferences) to enable finer-grained ranking.
3. **Is deterministic and reproducible** -- no stochastic LLM sampling variance.
4. **Is truly non-LLM** -- no generative model in the evaluation loop. Encoder-only models (BERT) are borderline; we will classify them explicitly.

**Key constraint for our use case:** We evaluate *open-ended generative text* (analytical essays and creative writing), NOT translation, summarization, or factual QA. This rules out reference-based metrics (BLEU, ROUGE, METEOR) entirely -- there is no gold reference. It also means metrics validated only on constrained-output tasks may not transfer.

---

## 2. Information-Theoretic Metrics

### 2.1 Compression Ratio as Quality Proxy

**Method:** Compress the output text using a general-purpose compressor (gzip, zlib, bz2). The compression ratio rho = |C(x)| / |x| measures information density: text with more unique information compresses less (higher rho), while repetitive or formulaic text compresses more (lower rho).

**Formal definition:**

    rho(x) = |C(x)| / |x|

where C(x) is the compressed representation and |.| denotes byte length.

**Evaluation:**

| Criterion | Assessment |
|---|---|
| Works for generative/subjective tasks? | **Partially.** Detects repetition and boilerplate, but cannot distinguish sophisticated from simplistic text if both are non-repetitive. High information density != high quality (random characters have maximum compression ratio). |
| Differentiates good from mediocre? | **Weak.** Can flag degenerate outputs (excessive repetition, copy-paste) but fails to distinguish "competent" from "excellent." Within the normal quality range, compression ratio is nearly constant for LLM outputs. |
| Truly non-LLM? | **Yes.** Pure algorithmic compression, no neural components. |
| Computational cost? | **Negligible.** Milliseconds per document. |
| Validated in research? | **Yes, but for different purposes.** Jiang et al. (2023) used gzip compression distance for text classification. Bennett et al. (2003) formalized Normalized Compression Distance (NCD). However, compression ratio as a *quality* metric for generation is not well-validated. |

**Verdict:** Warning -- **Useful only as a degeneracy detector** (flag texts with rho < threshold). Not discriminative within the normal quality range. Low priority for our use case.

**Variant worth exploring:** Normalized Compression Distance (NCD) *between* agent outputs could complement our existing cosine-similarity disagreement measure:

    NCD(x, y) = (|C(xy)| - min(|C(x)|, |C(y)|)) / max(|C(x)|, |C(y)|)

This is LLM-free and captures structural similarity beyond embedding space.

### 2.2 Token Distribution Entropy

**Method:** Compute the Shannon entropy of the unigram (or bigram) token distribution in the output:

    H(X) = -SUM_w p(w) * log2(p(w))

where p(w) is the relative frequency of token w in the output.

**Evaluation:**

| Criterion | Assessment |
|---|---|
| Works for generative/subjective tasks? | **Partially.** Higher entropy correlates with vocabulary richness, but does not capture semantic quality. |
| Differentiates good from mediocre? | **Weak.** LLM outputs all tend to have high entropy (diverse vocabulary is cheap for large models). More useful for detecting *degenerate* outputs (low entropy = stuck in a loop). |
| Truly non-LLM? | **Yes.** Pure statistical computation. |
| Computational cost? | **Negligible.** |
| Validated in research? | **Indirectly.** Entropy is standard in information theory. Hashimoto et al. (2019) used distribution-level statistics for evaluation but focused on diversity rather than quality. |

**Verdict:** Warning -- **Degeneracy detector only.** Same limitation as compression ratio. Not discriminative in the normal quality range.

### 2.3 Perplexity from a Small Local Model

**Method:** Compute the perplexity of the output using a small pre-trained language model (e.g., GPT-2 small, 124M parameters):

    PPL(x) = exp(-(1/N) * SUM_i log p_theta(x_i | x_{<i}))

Lower perplexity = the text is more "expected" by the model = higher fluency.

**Evaluation:**

| Criterion | Assessment |
|---|---|
| Works for generative/subjective tasks? | **For fluency only.** Perplexity measures how likely the text is under a language model's distribution. This correlates with fluency/naturalness but NOT with relevance, depth, or creativity. |
| Differentiates good from mediocre? | **Weak for our use case.** All LLM-generated text will have similar perplexity under GPT-2 because the outputs are already fluent. Perplexity distinguishes human-from-machine or fluent-from-disfluent, but not good-from-great analytical writing. |
| Truly non-LLM? | **BORDERLINE.** GPT-2 is technically a language model. However, it is used purely as a probability estimator (no generation, no judgment). I classify this as acceptable if we frame it as "fluency scoring via distributional statistics" rather than "LLM judgment." |
| Computational cost? | **Low-moderate.** GPT-2-small runs on CPU in seconds per document. Batch processing: minutes. |
| Validated in research? | **Yes.** Perplexity is the standard intrinsic metric for language models (Jelinek et al., 1977; Meister et al., 2021). Its correlation with human fluency judgments is documented (Kann et al., 2018) though it is an incomplete quality measure. |

**Verdict:** Warning -- **Include as a fluency floor check** -- flag outputs with anomalously high perplexity. But do NOT expect it to rank outputs within the normal quality range. Treat as a sanity filter, not a discriminator.

### 2.4 Summary: Information-Theoretic Metrics

These metrics share a fundamental limitation: **they measure distributional properties of the surface form, not semantic content.** For our use case (distinguishing quality levels of coherent, fluent LLM outputs), they operate in a regime where most outputs are "good enough" to produce similar scores. Their value is as **degeneracy detectors** and **sanity filters**, not as primary quality discriminators.

---

## 3. Textual Statistics

### 3.1 Readability Scores

**Methods:**
- **Flesch-Kincaid Grade Level:** 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
- **Coleman-Liau Index:** 0.0588*L - 0.296*S - 15.8 (L = avg letters per 100 words, S = avg sentences per 100 words)
- **Automated Readability Index (ARI):** 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43

**Evaluation:**

| Criterion | Assessment |
|---|---|
| Works for generative/subjective tasks? | **Yes, with caveats.** Readability is a meaningful property of analytical essays. For creative writing, "appropriate difficulty" may vary with genre. |
| Differentiates good from mediocre? | **Partially.** Can detect outputs that are too simplistic (grade level 5 for a graduate-level essay) or too convoluted (grade level 20+). But readability != quality. The relationship is **non-monotonic**: quality peaks at an appropriate level for the task. |
| Truly non-LLM? | **Yes.** Pure formula-based computation on surface features. |
| Computational cost? | **Negligible.** |
| Validated in research? | **Extensively**, but for measuring readability, not quality. Flesch (1948), Kincaid et al. (1975). NOT validated as a quality metric for generative AI output. |

**Verdict:** INCLUDE -- **Include as a task-appropriateness check.** Define expected readability ranges for analytical essays (grade 12-16) and creative writing (grade 8-14). Flag outliers. Cheap and interpretable.

### 3.2 Lexical Diversity (MTLD, vocd-D, TTR)

**Methods:**
- **Type-Token Ratio (TTR):** |unique words| / |total words| -- simple but length-dependent.
- **MTLD (Measure of Textual Lexical Diversity):** Average length of sequential word runs that maintain a TTR above a threshold (default 0.720). Robust to text length (McCarthy, 2005; McCarthy & Jarvis, 2010).
- **vocd-D:** Fits a curve to the TTR-vs-sample-size function across random samples. The parameter D indexes lexical diversity independent of text length (Malvern et al., 2004).

**Evaluation:**

| Criterion | Assessment |
|---|---|
| Works for generative/subjective tasks? | **Yes.** Lexical diversity is a recognized dimension of writing quality in composition studies (Crossley et al., 2011; McNamara et al., 2014). Higher MTLD correlates with higher human quality ratings for essays. |
| Differentiates good from mediocre? | **Moderate.** Published studies show significant correlations between MTLD and essay quality ratings (r ~ 0.3-0.5; Crossley et al., 2011). Not a strong discriminator alone, but meaningfully correlated. **This is one of the best-validated non-LLM metrics for essay quality.** |
| Truly non-LLM? | **Yes.** Pure statistical computation on word tokens. No neural components. |
| Computational cost? | **Low.** MTLD and vocd-D require tokenization and simple counting. Seconds per document. Python packages: `lexicalrichness`, `lexical_diversity`, or custom implementation. |
| Validated in research? | **Yes, specifically for writing quality.** McCarthy & Jarvis (2010) validated MTLD. Crossley et al. (2011, 2014) showed correlations with human essay ratings. McNamara et al. (2014) included it in the Coh-Metrix suite. Jarvis (2013) compared 12 lexical diversity measures. |

**Verdict:** **HIGH PRIORITY.** MTLD is the best single non-LLM metric for essay quality in the literature. It is validated, interpretable, length-independent, cheap, and truly non-neural. **Recommend as one of our top 3.**

### 3.3 Sentence Complexity (Dependency Tree Depth, Clause Count)

**Methods:**
- **Mean dependency tree depth:** Parse each sentence with spaCy or Stanza; compute the maximum depth of the dependency tree; average across sentences.
- **Mean clause count per sentence:** Count finite verb phrases or clausal dependencies (advcl, ccomp, xcomp, relcl in Universal Dependencies).
- **Mean sentence length (words):** Simple proxy for complexity.

**Evaluation:**

| Criterion | Assessment |
|---|---|
| Works for generative/subjective tasks? | **Yes.** Syntactic complexity is a standard dimension in writing assessment (Lu, 2010; Kyle & Crossley, 2018). |
| Differentiates good from mediocre? | **Moderate.** Correlates with essay quality in L2 writing studies (r ~ 0.2-0.4). For LLM-generated text, the question is whether multi-agent configurations produce systematically different syntactic profiles. |
| Truly non-LLM? | **Mostly yes.** SpaCy's dependency parser uses a small neural model (~15M params), but it is a task-specific parsing model, NOT a generative LLM. Analogous to using a POS tagger -- a tool, not a judge. |
| Computational cost? | **Low-moderate.** SpaCy parsing: milliseconds per sentence. Batch all outputs: minutes. |
| Validated in research? | **Yes.** Lu (2010) developed L2 Syntactic Complexity Analyzer. Kyle & Crossley (2018) validated TAASSC. Included in Coh-Metrix and TAALES. |

**Verdict:** INCLUDE -- **Include.** Cheap, validated, measures a different dimension than lexical diversity. Combine with MTLD for a richer textual statistics profile.

### 3.4 Text Length and Structural Completeness

**Method:** Word count, paragraph count, presence of expected structural elements.

**Evaluation:**

| Criterion | Assessment |
|---|---|
| Works for generative/subjective tasks? | **Yes.** Length is a known confounder in writing assessment (Chodorow & Burstein, 2004). |
| Differentiates good from mediocre? | **Weak on its own.** Necessary but not sufficient. |
| Truly non-LLM? | **Yes.** Counting. |
| Computational cost? | **Negligible.** |
| Validated? | **Yes, as a confounder.** Powers (2005) documented length-quality correlation. |

**Verdict:** INCLUDE -- **Include as a covariate**, not as a primary metric.

### 3.5 Summary: Textual Statistics

This category contains our **strongest candidates.** MTLD (lexical diversity) and syntactic complexity have the best empirical validation for essay quality among all non-LLM methods. They are cheap, interpretable, and measure properties orthogonal to LLM holistic judgment. Key limitation: effect sizes are moderate (r ~ 0.3-0.5), useful as one signal among several.

---

## 4. Embedding-Based Metrics

### 4.1 BERTScore -- Prompt-to-Output Relevance

**Method:** Use BERTScore (Zhang et al., 2020, ICLR 2020) to compute similarity between the task prompt and the generated output via token-level cosine similarities using contextual embeddings, aggregated via greedy matching.

**Evaluation:**

| Criterion | Assessment |
|---|---|
| Works for generative/subjective tasks? | **Partially.** Measures topical relevance but not depth, originality, or argumentation quality. |
| Differentiates good from mediocre? | **Weak.** If all agents respond to the same prompt, BERTScores will be similar. Detects off-topic responses but not quality gradations among on-topic ones. |
| Truly non-LLM? | **BORDERLINE.** BERT/DeBERTa are encoder-only transformers (110M-350M params). Not generative, no judgments -- they provide embeddings. "Encoder-based, non-generative" -- acceptable but should be flagged. |
| Computational cost? | **Moderate.** Requires GPU for efficiency. |
| Validated? | **Yes, extensively** for reference-based evaluation (Zhang et al., 2020). Prompt-to-output usage is less standard. |

**Verdict:** Warning -- **Lower priority.** We already have cosine similarity measurements. BERTScore adds token-level alignment but unlikely to discriminate quality in our setting. Include only for off-topic drift detection.

### 4.2 Cosine Similarity Between Agent Outputs (Already Implemented)

We already compute pairwise cosine similarity between agent outputs using sentence-transformers. This measures **inter-agent agreement/disagreement** in embedding space.

**Note:** The existing disagreement measure can be reinterpreted through the concordance lens (Section 7). High agreement + high quality scores = convergent quality. High agreement + low quality scores = convergent mediocrity. The interaction is informative.

### 4.3 Output Diversity Within a Multi-Agent Group

**Method:** For n agents responding to the same prompt, compute:
- **Mean pairwise cosine distance**
- **Centroid distance:** Average distance of each output embedding from the group centroid.
- **Semantic spread:** Trace of the covariance matrix of output embeddings.

**Verdict:** KEEP -- **Already implemented, keep as context metric.** Essential for interpreting quality scores across conditions. Not a quality metric per se.

### 4.4 Summary: Embedding-Based Metrics

Weakest candidates for *quality assessment* because they measure semantic similarity/distance rather than quality. Main value is **contextual**. The encoder-based nature makes them "borderline non-LLM."

---

## 5. Argument / Discourse Structure

### 5.1 Logical Connective Density

**Method:** Count discourse markers and logical connectives per sentence or per 100 words. Categories:
- **Causal:** therefore, because, thus, hence, consequently, as a result
- **Contrastive:** however, but, although, nevertheless, on the other hand, despite
- **Additive:** furthermore, moreover, additionally, also, in addition
- **Temporal/sequential:** first, then, subsequently, finally, meanwhile

**Evaluation:**

| Criterion | Assessment |
|---|---|
| Works for generative/subjective tasks? | **Yes, especially for analytical essays.** Discourse marker usage is well-studied for argumentative writing quality (Crossley et al., 2016; McNamara et al., 2014). Less relevant for creative writing. |
| Differentiates good from mediocre? | **Moderate for analytical essays.** Connective density correlates with essay coherence ratings (r ~ 0.2-0.35; Crossley et al., 2016). Crucially, the *type* of connective matters: causal and contrastive connectives indicate more sophisticated argumentation than additive ones. |
| Truly non-LLM? | **Yes.** Dictionary lookup + counting. No neural components. |
| Computational cost? | **Negligible.** Regex or word-list matching. |
| Validated? | **Yes.** Graesser et al. (2004) -- Coh-Metrix. Crossley et al. (2016) showed specific connective types predict essay quality. Halliday & Hasan (1976) -- foundational work on cohesion. |

**Verdict:** **HIGH PRIORITY for analytical essays.** When combined with connective *type* classification, this is a strong, validated, genuinely non-LLM metric for argumentative writing. Cheap and interpretable.

### 5.2 Claim Density

**Method:** Count distinct assertive statements (claims) per paragraph. Operationalize using sentence-level heuristics.

**Verdict:** Warning -- **Theoretically appealing but hard to operationalize reliably.** Heuristic approaches are noisy; robust approaches require neural models. Include as exploratory metric, not primary.

### 5.3 Evidence-to-Claim Ratio

Same limitations as claim density -- hard to operationalize without semantic understanding.

**Verdict:** Warning -- **Exploratory only.** Too noisy for reliable measurement.

### 5.4 Topic Coherence

**Method:** Measure whether text stays on-topic across paragraphs:
- **LDA-based coherence:** Fit LDA; measure topic distribution concentration.
- **Sliding-window coherence:** Cosine similarity between embeddings of adjacent paragraphs.
- **Coh-Metrix coherence:** Content word overlap between adjacent sentences.

**Evaluation:**

| Criterion | Assessment |
|---|---|
| Works for generative/subjective tasks? | **Yes.** Topic coherence is fundamental for both analytical and creative writing. |
| Differentiates good from mediocre? | **Moderate.** Coherence metrics from Coh-Metrix correlate with essay quality (r ~ 0.2-0.3; Crossley & McNamara, 2012). |
| Truly non-LLM? | **LDA-based: Yes.** Sliding-window with sentence-transformers: borderline. Coh-Metrix word-overlap: yes. |
| Validated? | **Yes.** Graesser et al. (2004, 2011). Crossley & McNamara (2012). Mimno et al. (2011). |

**Verdict:** INCLUDE -- **Include sliding-window coherence** as a secondary metric.

### 5.5 Summary: Argument/Discourse Structure

**Logical connective density** (with type classification) is the standout -- validated, cheap, interpretable, and truly non-LLM. Topic coherence is a useful secondary metric. Claim density and evidence ratios are theoretically appealing but currently hard to automate reliably.

---

## 6. Reference-Free Quality Estimation

### 6.1 BARTScore

**Paper:** Yuan et al. (2021), NeurIPS 2021. Uses BART (406M params) to score text by computing log-probability of generating the output.

**Is it an LLM in disguise?** **YES.** BART is a full autoregressive generative model. **Fails our "truly non-LLM" criterion.**

### 6.2 UniEval

**Paper:** Zhong et al. (2022), EMNLP 2022. Re-frames evaluation as Boolean QA using T5 (770M params).

**Is it an LLM in disguise?** **YES.** T5 is a generative encoder-decoder model. **Fails our criterion.**

### 6.3 G-Eval

**Paper:** Liu et al. (2023). Uses GPT-4 with CoT for evaluation.

**Is it an LLM?** **Obviously yes.** Literally LLM-as-judge.

### 6.4 Genuinely Non-LLM Reference-Free Approaches

The literature reveals that **truly non-LLM reference-free quality estimation for open-ended generation barely exists.** The field has moved almost entirely to neural metrics. The non-neural options are:

1. **Textual statistics suites** (Coh-Metrix, TAALES, TAASSC) -- see Sections 3.2-3.3.
2. **Discourse/argument structure heuristics** -- see Section 5.
3. **Information-theoretic measures** -- see Section 2.
4. **Statistical concordance among multiple outputs** -- see Section 7.

This is precisely why our task is important: we need to *construct* a non-LLM evaluation framework rather than adopt an off-the-shelf one.

---

## 7. Statistical Concordance Methods

### 7.1 Inter-Output Agreement as a Quality Signal

**Method:** Given n agents producing outputs for the same prompt:

    Agreement(X) = (2 / n(n-1)) * SUM_{i<j} sim(x_i, x_j)

**Hypothesis:** Higher inter-agent agreement signals higher quality (agents "converge" on the best response).

**Evaluation:**

| Criterion | Assessment |
|---|---|
| Works for generative/subjective tasks? | **Depends.** For analytical tasks with convergent answers, agreement is reasonable. For creative writing, agreement might indicate *lack of creativity*. |
| Differentiates good from mediocre? | **Conditional.** Informative only if agents are independently "noisy experts" (Condorcet assumption). If agents share systematic biases, agreement may reflect shared biases. |
| Truly non-LLM? | **Yes** (for the concordance computation itself). |
| Computational cost? | **Low.** |

### 7.2 Condorcet Jury Theorem Extensions

**Classical Condorcet (1785):** If each of n independent voters has probability p > 0.5 of choosing correctly, the majority vote converges to certainty as n -> infinity.

**Application to our setting -- key issues:**

1. **Independence assumption fails.** LLM agents are NOT independent voters. They share training data, architectural biases, and often base models. Ladha (1992) showed correlated voters can make majority voting *worse* than a single voter.
2. **Competence assumption is untestable.** We don't know if p > 0.5 for each agent for subjective tasks.
3. **Continuous vs. binary.** Classical Condorcet is for binary decisions.

**Relevant extensions:**
- **Hong & Page (2004):** "Diversity trumps ability" -- supports using different architectures/prompts.
- **List & Goodin (2001):** Supermajority rules are optimal when competence is uncertain.
- **Dietrich & List (2004):** Effective independence matters, not raw independence.

### 7.3 Wisdom-of-Crowds Scoring

**Method:** Use the *distribution* of outputs as quality signal:

1. **Centroid quality:** Centroid of agent output embeddings = "consensus answer."
2. **Outlier detection:** Outputs far from centroid may be creative or erroneous.
3. **Dispersion as difficulty proxy:** High dispersion signals task difficulty or ambiguity.

**Formal framework:**

- Centroid: e_bar = (1/n) * SUM_i e_i
- Centrality score: c_i = cos(e_i, e_bar)
- Dispersion: sigma^2 = (1/n) * SUM_i ||e_i - e_bar||^2

### 7.4 A Novel Composite: Agreement-Weighted Quality Scoring

**Proposal:** Combine concordance with textual quality metrics:

    Q_composite(x_i) = alpha * Q_textual(x_i) + (1-alpha) * c_i

where Q_textual is normalized textual statistics and c_i is centrality score. Weight alpha is task-dependent.

### 7.5 Summary

Condorcet provides **theoretical justification** for inter-agent agreement as quality signal, but independence assumption is severely violated. Agreement is most useful as a **context variable**, not standalone.

---

## 8. Comparative Evaluation Matrix

| Method | Generative Tasks | Discriminative Power | Truly Non-LLM | Cost | Validated | Overall |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Compression ratio | Partial | Low | Yes | Negligible | Not for quality | D |
| Token entropy | Partial | Low | Yes | Negligible | Indirect | D |
| Perplexity (GPT-2) | Fluency only | Low | Borderline | Low | For fluency | C |
| **Readability scores** | Yes | Non-monotonic | Yes | Negligible | For readability | B |
| **MTLD (lex. diversity)** | Yes | **Moderate (r~0.3-0.5)** | Yes | Low | **For essays** | **A** |
| **Syntactic complexity** | Yes | **Moderate (r~0.2-0.4)** | Yes* | Low | **For essays** | **A-** |
| Text length | Yes | Confounder | Yes | Negligible | As covariate | B- |
| BERTScore (prompt-sim) | Partial | Low (all on-topic) | Encoder-based | Moderate | For ref-based | C |
| Output diversity | Yes | Context only | Encoder-based | Low | Partial | B- |
| **Connective density** | Yes (essays) | **Moderate (r~0.2-0.35)** | Yes | Negligible | **Coh-Metrix** | **A-** |
| Claim density | Yes (essays) | Hard to operationalize | Yes (heuristic) | Low | Partial | C+ |
| Topic coherence | Yes | Moderate | Method-dependent | Low | Coh-Metrix | B |
| BARTScore | Yes | Good | **Is an LLM** | GPU needed | NeurIPS | **REJECTED** |
| UniEval | Yes | Good | **Is an LLM** | GPU needed | EMNLP | **REJECTED** |
| Agreement (Condorcet) | Convergent tasks | Conditional | Yes (method) | Low | Theory only | B |
| Centroid quality | Convergent tasks | Unvalidated | Encoder-based | Low | Novel | C+ |

*spaCy parser is a small neural model, but functionally a tool, not a judge.*

---

## 9. Top 3 Recommendations

### RECOMMENDATION 1: MTLD Lexical Diversity + vocd-D

**Why:** The single best-validated non-LLM metric for essay quality in the computational linguistics literature. MTLD was designed to be robust to text length and validated across multiple essay-scoring studies.

**Evidence:**
- McCarthy & Jarvis (2010): MTLD shows no significant correlation with text length while maintaining sensitivity to diversity differences.
- Crossley, Salsbury & McNamara (2011): MTLD correlates with writing proficiency ratings (r ~ 0.40-0.50).
- McNamara et al. (2014): Included in Coh-Metrix, the most widely-used computational text analysis tool in writing research.

**Implementation:**
- Python: `lexicalrichness` package or custom implementation.
- Input: tokenized text (standard spaCy tokenization suffices).
- Output: continuous score (higher = more lexically diverse).
- Time: <1 second per document.

**Complementarity with LLM judges:** LLM judges evaluate holistic quality. MTLD measures a specific, well-defined linguistic property (vocabulary richness) that is necessary but not sufficient for quality. Orthogonal dimensions.

**For our experiment:** Different multi-agent configurations may produce systematically different lexical diversity profiles. Debate-encouraging configurations may produce richer vocabulary than majority-vote (agents exposed to counterarguments adopt broader terminology). This is a *measurable, non-LLM-dependent* signal of the "disagreement dividend."

---

### RECOMMENDATION 2: Composite Discourse Metrics (Connective Density + Syntactic Complexity)

**Why:** Combines two well-validated dimensions of writing quality absent from LLM holistic judgment. Captures *argumentative sophistication* -- the degree to which text constructs complex, well-connected reasoning chains.

**Components:**
1. **Typed connective density:** Count logical connectives per sentence, classified by type (causal, contrastive, additive, temporal). Ratio of sophisticated connectives (causal + contrastive) to simple ones (additive). Higher ratio = more sophisticated argumentation.
2. **Mean dependency tree depth:** Average maximum depth of syntactic dependency trees. Deeper trees = more subordination = more complex structures.
3. **Clausal density:** Mean number of clausal complements (ccomp, xcomp, advcl, relcl) per sentence.

**Evidence:**
- Graesser et al. (2004, 2011): Coh-Metrix validates connectives and syntactic complexity as coherence/quality dimensions.
- Crossley et al. (2016): Specific connective types (causal, adversative) predict essay quality better than total count.
- Lu (2010): 14 syntactic complexity measures validated for writing quality.
- Kyle & Crossley (2018): TAASSC -- syntactic sophistication correlates with writing quality.

**Implementation:**
- Connectives: dictionary/regex lookup. ~100 English connectives, classified by type. Trivial.
- Dependency parsing: spaCy `en_core_web_sm` or `en_core_web_md`. Standard pipeline.
- Output: 4 continuous scores (causal/contrastive connective density, mean tree depth, clausal density).
- Time: seconds per document (dominated by dependency parsing).

**Complementarity:** LLM judges may reward "sounds good" without analyzing discourse structure. A text can sound fluent and relevant but lack argumentative depth. These metrics catch that gap.

**For our experiment:** Multi-agent debate configurations should produce higher causal/contrastive connective density (agents responding to counterarguments use "however," "because," "therefore" more). This is a *testable prediction*.

---

### RECOMMENDATION 3: Agreement-Conditioned Quality Profiling (Novel Composite)

**Why:** Not a single metric but a *framework* for interpreting metrics in the context of multi-agent agreement. Exploits our unique experimental structure (multiple agents, multiple configurations).

**Method:**

For each experimental condition (topology x consensus mechanism), compute:

1. **Inter-agent agreement** A_k: mean pairwise cosine similarity (already computed).
2. **Textual quality profile** q_k = [MTLD_k, LCD_k, SynComp_k, ReadGrade_k, Length_k]: average per-output textual statistics.
3. **Quality-Agreement quadrant:** Plot conditions in 2D space:

|  | High Quality Profile | Low Quality Profile |
|---|---|---|
| **High Agreement** | CONVERGENT QUALITY -- agents agree and produce sophisticated text. | CONVERGENT MEDIOCRITY -- agents agree on bland response. Correlated biases. |
| **Low Agreement** | DIVERSE QUALITY -- agents disagree but individually produce sophisticated text. The "disagreement dividend." | CONFUSED NOISE -- agents disagree and produce low-quality text. Configuration dysfunctional. |

4. **Statistical test:** For condition pairs, test quality profile differences using permutation-based Hotelling's T^2 (non-parametric; no normality assumption).

**Evidence:**
- Each component individually validated (Sections 3-5).
- Quadrant framework draws on Surowiecki (2004), Hong & Page (2004), Condorcet tradition.
- Permutation-based multivariate testing: Good (2005); Anderson (2001).

**Implementation:**
- Compute per-output quality vectors from Recommendations 1 and 2.
- Compute agreement from existing cosine similarities.
- Multivariate permutation test: `scipy.stats` or custom (~50 lines Python).
- Visualization: scatter plot with error ellipses.
- Time: minutes for entire corpus.

**Complementarity:** Directly addresses LLM judge failure mode (excessive ties). Even if LLM judges say "tie," the quality-agreement quadrant may reveal systematic differences in *how* outputs differ -- not just "which is better" (LLM question) but "in what measurable ways do they differ" (non-LLM question).

---

## 10. Implementation Plan

### Phase 1: Quick Wins (1-2 hours)

1. **MTLD and vocd-D** for all existing outputs.
   - Package: `lexicalrichness` (pip install)
   - Script: tokenize -> compute -> store as JSON alongside outputs.

2. **Connective density** (typed).
   - Build connective dictionary (100 entries, 4 types).
   - Count per sentence, normalize by sentence count.

3. **Readability scores** (Flesch-Kincaid, Coleman-Liau, ARI).
   - Package: `textstat` (pip install)
   - Three scores per output.

4. **Text length** (words, sentences, paragraphs).

### Phase 2: Dependency Parsing (2-3 hours)

5. **Syntactic complexity** via spaCy.
   - Install spaCy + `en_core_web_md`.
   - Compute: mean dependency tree depth, clausal density, mean sentence length.

6. **Topic coherence** (sliding-window cosine similarity between adjacent paragraphs).
   - Use existing sentence-transformer embeddings.

### Phase 3: Integration and Analysis (2-3 hours)

7. **Quality-agreement quadrant analysis.**
   - Combine metrics into per-output quality vectors.
   - Compute condition-level statistics.
   - Permutation-based multivariate tests.
   - Visualize quadrants.

8. **Correlation analysis with LLM judge scores.**
   - Spearman correlations between non-LLM metrics and LLM judge preferences.
   - Test whether non-LLM metrics can "break ties" that LLM judges could not.

### Output Format

Each output document gets a quality profile:

```json
{
  "output_id": "...",
  "condition": "...",
  "metrics": {
    "mtld": 85.3,
    "vocd_d": 72.1,
    "flesch_kincaid_grade": 13.2,
    "coleman_liau": 12.8,
    "ari": 14.1,
    "connective_density_total": 1.42,
    "connective_density_causal": 0.38,
    "connective_density_contrastive": 0.52,
    "connective_density_additive": 0.35,
    "connective_density_temporal": 0.17,
    "mean_dep_tree_depth": 5.7,
    "clausal_density": 1.83,
    "mean_sentence_length": 22.4,
    "word_count": 547,
    "paragraph_count": 5,
    "topic_coherence": 0.78
  }
}
```

---

## 11. References

- Anderson, M. J. (2001). A new method for non-parametric multivariate analysis of variance. *Austral Ecology*, 26(1), 32-46.
- Bennett, C. H., Gacs, P., Li, M., Vitanyi, P. M. B., & Zurek, W. H. (2003). Information distance. *IEEE Trans. Information Theory*, 44(4), 1407-1423.
- Chodorow, M., & Burstein, J. (2004). Beyond essay length: Evaluating e-rater's performance on TOEFL essays. *ETS Research Report Series*.
- Coleman, M., & Liau, T. L. (1975). A computer readability formula designed for machine scoring. *J. Applied Psychology*, 60(2), 283-284.
- Condorcet, M. de (1785). *Essai sur l'application de l'analyse a la probabilite des decisions rendues a la pluralite des voix.*
- Crossley, S. A., & McNamara, D. S. (2012). Predicting second language writing proficiency. *J. Research in Reading*, 35(2), 115-135.
- Crossley, S. A., Salsbury, T., & McNamara, D. S. (2011). Predicting the proficiency level of language learners using lexical indices. *Language Testing*, 29(2), 243-263.
- Crossley, S. A., Kyle, K., & McNamara, D. S. (2016). The tool for the automatic analysis of text cohesion (TAACO). *Behavior Research Methods*, 48(4), 1227-1237.
- Dietrich, F., & List, C. (2004). A model of jury decisions where all jurors have the same evidence. *Synthese*, 142(2), 175-202.
- Flesch, R. (1948). A new readability yardstick. *J. Applied Psychology*, 32(3), 221-233.
- Good, P. I. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
- Graesser, A. C., McNamara, D. S., Louwerse, M. M., & Cai, Z. (2004). Coh-Metrix: Analysis of text on cohesion and language. *Behavior Research Methods*, 36(2), 193-202.
- Graesser, A. C., McNamara, D. S., & Kulikowich, J. M. (2011). Coh-Metrix: Providing multilevel analyses of text characteristics. *Educational Researcher*, 40(5), 223-234.
- Halliday, M. A. K., & Hasan, R. (1976). *Cohesion in English*. Longman.
- Hashimoto, T. B., Zhang, H., & Liang, P. (2019). Unifying human and statistical evaluation with an information-theoretic framework. *NAACL-HLT 2019*.
- Hong, L., & Page, S. E. (2004). Groups of diverse problem solvers can outperform groups of high-ability problem solvers. *PNAS*, 101(46), 16385-16389.
- Jarvis, S. (2013). Defining and measuring lexical diversity. In *Vocabulary Knowledge* (pp. 13-44). John Benjamins.
- Jelinek, F., Mercer, R. L., Bahl, L. R., & Baker, J. K. (1977). Perplexity -- a measure of the difficulty of speech recognition tasks. *JASA*, 62(S1).
- Jiang, Z., Yang, M., Tsirlin, M., Tang, R., Dai, Y., & Lin, J. (2023). Low-resource text classification: A parameter-free classification method with compressors. *Findings of ACL 2023*.
- Kann, K., Stoyanov, V., Duh, K., & Hajic, J. (2018). Sentence-level fluency evaluation. *CoNLL 2018*.
- Kincaid, J. P., Fishburne, R. P., Rogers, R. L., & Chissom, B. S. (1975). Derivation of new readability formulas. Research Branch Report 8-75.
- Kyle, K., & Crossley, S. A. (2018). Measuring syntactic complexity in L2 writing. *Modern Language Journal*, 102(2), 333-349.
- Ladha, K. K. (1992). The Condorcet jury theorem, free speech, and correlated votes. *AJPS*, 36(3), 617-634.
- Li, J., Galley, M., Brockett, C., Gao, J., & Dolan, B. (2016). A diversity-promoting objective function for neural conversation models. *NAACL-HLT 2016*.
- List, C., & Goodin, R. E. (2001). Epistemic democracy: Generalizing the Condorcet jury theorem. *J. Political Philosophy*, 9(3), 277-306.
- Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG evaluation using GPT-4. *EMNLP 2023*.
- Lu, X. (2010). Automatic analysis of syntactic complexity in second language writing. *IJCL*, 15(4), 474-496.
- Malvern, D., Richards, B., Chipere, N., & Duran, P. (2004). *Lexical Diversity and Language Development*. Palgrave Macmillan.
- Mannes, A. E., Soll, J. B., & Larrick, R. P. (2014). The wisdom of select crowds. *JPSP*, 107(2), 276-299.
- McCarthy, P. M. (2005). *An assessment of the range and usefulness of lexical diversity measures and the potential of MTLD*. PhD dissertation, University of Memphis.
- McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study. *Behavior Research Methods*, 42(2), 381-392.
- McNamara, D. S., Graesser, A. C., McCarthy, P. M., & Cai, Z. (2014). *Automated Evaluation of Text and Discourse with Coh-Metrix*. Cambridge University Press.
- Meister, C., Cotterell, R., & Vieira, T. (2021). Language model evaluation beyond perplexity. *ACL 2021*.
- Mimno, D., Wallach, H. M., Talley, E., Leenders, M., & McCallum, A. (2011). Optimizing semantic coherence in topic models. *EMNLP 2011*.
- Powers, D. E. (2005). Wordiness: A selective review. ETS Research Memorandum RM-04-08.
- Stab, C., & Gurevych, I. (2017). Parsing argumentation structures in persuasive essays. *Computational Linguistics*, 43(3), 619-659.
- Surowiecki, J. (2004). *The Wisdom of Crowds*. Doubleday.
- Toulmin, S. E. (1958). *The Uses of Argument*. Cambridge University Press.
- Yuan, W., Neubig, G., & Liu, P. (2021). BARTScore: Evaluating generated text as text generation. *NeurIPS 2021*.
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. *ICLR 2020*.
- Zhong, M., Liu, Y., Xu, D., Zhu, C., & Zeng, M. (2022). Towards a unified multi-dimensional evaluator for text generation. *EMNLP 2022*.

---

## Appendix A: What We Explicitly Rejected and Why

| Method | Reason for Rejection |
|---|---|
| BLEU, ROUGE, METEOR | Reference-based; no gold reference for open-ended generation. |
| BARTScore | Generative LLM (BART, 406M) in evaluation loop. Violates independence. |
| UniEval | T5-based LLM making Boolean judgments. LLM in disguise. |
| G-Eval | Literally GPT-4. Not pretending to be non-LLM. |
| COMET/COMETKiwi | Trained for MT, not open-ended generation. Large encoder (XLM-R). |
| BERTScore (as primary) | Encoder-based (borderline) AND low discriminative power when all outputs address same prompt. |

## Appendix B: Sensitivity Analysis Considerations

Before deploying, validate sensitivity to *between-condition* differences:

1. **Within-condition variance:** Compute SD of each metric across agents per condition. If within > between variance, metric cannot discriminate conditions.
2. **Effect size estimation:** Cohen's d or eta^2 for each metric across conditions. Only medium+ effect sizes (d > 0.5) are useful.
3. **Multiple testing correction:** With ~15 metrics and multiple comparisons, apply Benjamini-Hochberg FDR correction.

---

*Last updated: 2026-02-28 | Elena Chen | T-029a*
