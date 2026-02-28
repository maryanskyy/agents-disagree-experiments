# Strategic Review: Publication Viability Assessment

**Reviewer:** Dr. Yuki Tanaka (PC member perspective - NeurIPS/ICML calibration)
**Date:** 2026-02-28
**Subject:** Will the $1,147 experiment produce a publishable paper at a top venue?

---

## Publication Verdict: Workshop First -> Full Paper Pipeline

Not "workshop-only" - but going directly to a NeurIPS/ICML main conference with this scale and these constraints would be a gamble with worse-than-even odds. The strongest path is: **NeurIPS 2026 Workshop** (e.g., Foundation Models, MASEC, or Agentic AI) -> incorporate feedback -> **AAMAS 2027 or ICML 2027 main conference** with expanded experiments.

Rationale below.

---

## 1. Novelty Check

### What exists already (as of early 2026)

The multi-agent LLM space has seen an explosion of work. Key threads relevant to this paper:

**Multi-agent debate and consensus:**
- **Du et al. (2023)**, "Improving Factuality and Reasoning in Language Models through Multiagent Debate" - established that multi-agent debate improves reasoning quality. High-citation foundation paper. Does NOT study the inverted-U or when debate *hurts*.
- **Liang et al. (2023)**, "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate" - studies diversity in debate but frames it as uniformly beneficial. No systematic disagreement-quality curve.
- **Chan et al. (2024)**, "ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate" - debate for evaluation, not generation quality.
- **Yao et al. (2025)**, "Roundtable Policy" - confidence-weighted consensus. Relevant but doesn't characterize the disagreement-quality relationship.
- **Cherian et al. (2025)**, "WISE" - weighted iterative debate. Focus on robust consensus, not on when consensus fails.

**Scaling and agent count effects:**
- **Li et al. (2024)**, "More Agents Is All You Need" - shows that scaling agents improves performance through majority voting. **Directly relevant.** Their claim is essentially the *opposite* of the Quorum Paradox. However, they study classification/reasoning with verifiable answers, not open-ended generation. The distinction matters - but a reviewer WILL cite this and ask why your results differ.
- **Wang et al. (2024)**, "ReConcile" - multi-agent reconciliation with confidence-weighted voting. Studies agent count scaling but doesn't systematically characterize non-monotonicity.

**Topology/orchestration:**
- **Wu et al. (2023)**, AutoGen - practical framework, no systematic topology comparison.
- **Hong et al. (2024)**, MetaGPT - role-based pipeline topology, no controlled comparison.
- **Qian et al. (2024)**, ChatDev - sequential pipeline for code generation. Shows topology matters but doesn't do controlled experiments.
- **AdaptOrch (various 2025 refs)** - adaptive orchestration. Cited in your research questions as something you extend.
- **Zhuge et al. (2024)**, "GPTSwarm: Language Agents as Optimizable Graphs" - graphs of agents with optimizable edges. Closest to your formal topology model. Does NOT do the systematic factorial comparison you propose.

**What's NOT been done (the gap):**
- **Nobody has published a systematic disagreement-quality curve (inverted-U).** The idea that moderate disagreement is optimal is widely intuited but never empirically characterized with controlled experiments.
- **Nobody has published a formal Quorum Paradox for generative tasks.** Li et al.'s "More Agents Is All You Need" is the closest, but it's about verifiable tasks where majority voting provably helps. The generative-task regime is different.
- **Nobody has published a controlled factorial comparison of topology x consensus x task type.** GPTSwarm and AdaptOrch study topology but not in this systematic, pre-registered way.
- **Nobody has published MVQ bounds.** The concept of a minimum viable quorum with formal guarantees is novel.

### Novelty verdict: **Novel enough, but the bar is rising fast.**

A reviewer would NOT say "this has been done." They WOULD say "parts of this are incremental extensions of known ideas." The Disagreement Dividend concept draws on diversity-quality tradeoffs studied in ensemble learning for decades. The Quorum Paradox has analogs in Condorcet jury theorem extensions. The topology comparison is a natural experiment that many groups could run.

**The novelty is in the COMBINATION:** formal framework + systematic empirical characterization + practical decision tools. No single component is a breakthrough; the integrated contribution is what makes it citable.

---

## 2. Story Strength

### If everything works perfectly:

Imagine every hypothesis confirms:
- Inverted-U disagreement curve with clear d* (creative: 0.4-0.6, analytical: 0.2-0.3)
- MVQ bound validated within 20% of empirical values
- Quorum Paradox at n=3 for creative tasks under debate, mediated by conformity pressure
- Significant topology x consensus interaction, with a clean decision matrix

**This is a Weak Accept to Accept at NeurIPS.** Not Strong Accept. Here's why:

**What makes it Weak Accept:**
- The scale (3 models, 2 task types, 16 tasks) is modest
- The formal results are "structural results with simulation-validated conditions" - honest epistemology, but reviewers comparing against papers with proper theorems will penalize
- The practical impact is incremental: the decision matrix is useful but doesn't change how people build systems fundamentally

**What would make it Strong Accept:**
1. **A surprising, counterintuitive finding that changes how people think.** The Quorum Paradox is the best candidate, but only if the effect is large and robust. If adding a 3rd agent to a creative task drops quality by 15%+, that's a finding people will remember.
2. **A formal result with real predictive power.** If the MVQ bound actually predicts n* within 1 agent for new tasks without calibration data, that's genuinely useful theory.
3. **Broader scale.** 5+ models from 3+ providers, 4+ task types, 50+ tasks would make the empirical contribution unassailable.
4. **A live deployment study.** Showing the decision matrix improves a real system's quality-cost tradeoff would be transformative.
5. **A mechanistic finding about WHY.** If the mediation analysis cleanly shows the causal chain (more agents -> conformity pressure -> diversity collapse -> quality drop), that's publishable cognitive science, not just engineering.

### At AAMAS specifically:
This would be a stronger submission at AAMAS (the premier multi-agent systems venue) than at NeurIPS. AAMAS values:
- Formal models of coordination
- Game-theoretic analysis of agent interaction
- Systematic empirical evaluation of multi-agent protocols

At AAMAS, this is an **Accept** if everything works. The formal-empirical integration and the Quorum Paradox would be seen as genuine contributions to the MAS literature.

---

## 3. Failure Scenarios

### Scenario A: Disagreement Dividend is flat
**What if:** Quality is monotonically decreasing with disagreement, or flat.

**Publishability:** YES, still publishable. A well-powered null result that *refutes* the intuition that "some disagreement is good" is itself interesting. It would contradict the diversity-quality narrative that pervades the multi-agent LLM literature. Frame as: "contrary to conventional wisdom, controlled disagreement does not improve multi-agent output quality for generative tasks."

**Minimum viable:** You'd need the result to be convincingly null (tight confidence intervals around zero effect), not just underpowered. With 2 reps and 16 tasks, a flat curve might be "inconclusive" rather than "convincingly null." This is the risk.

### Scenario B: Quorum Paradox doesn't appear
**What if:** Quality is monotonically non-decreasing with agent count across all conditions.

**Publishability:** HARDER. The Quorum Paradox is the paper's signature claim and most citable contribution. Without it, you have:
- A topology comparison (useful but not novel enough alone)
- MVQ curves (useful but incremental)
- Disagreement-quality curves (interesting but not groundbreaking if the shape is boring)

**Minimum viable in this scenario:** You'd need to reframe. "When Does Adding Agents Help? A Systematic Study of Multi-Agent Scaling for Generative Tasks" - emphasizing the positive scaling conditions and the topology/consensus moderators. This is publishable at a workshop but borderline for a main conference without the paradox.

### Scenario C: Topology effects are small
**What if:** All topologies perform similarly after cost normalization.

**Publishability:** YES - "topology doesn't matter (much)" is itself a finding. It would contradict the implicit assumption in AutoGen, CrewAI, etc. that topology choice is important. But this needs to be convincingly demonstrated, not just underpowered.

### Scenario D: Everything works but effect sizes are small
**What if:** All hypotheses confirm but Cohen's d < 0.2 everywhere.

**Publishability:** PROBLEMATIC. Small effects with this sample size will have wide confidence intervals. Reviewers will question practical significance. This is the most likely failure mode: directionally correct but unconvincing.

### Minimum viable set of positive findings for a publishable paper:
1. **At minimum ONE of:** clear Disagreement Dividend curve OR clear Quorum Paradox - at least one surprising, citable finding
2. **Plus:** clean topology x consensus interaction showing that "best topology depends on task type" (the decision matrix, even if imprecise)
3. **Plus:** the evaluation methodology paper-within-a-paper (the bias mitigation protocol is genuinely well-done and citable independently)

If you get all three, you have a workshop paper minimum. For main conference, you need finding #1 to be robust and surprising.

---

## 4. Scale Credibility

### Current scale: 3,456 runs, 3 models, 2 task types, 16 tasks

**Will reviewers say "too small"?**

**For NeurIPS/ICML: Yes, probably.** Recent multi-agent LLM papers at top venues:
- Du et al. (2023) tested on 6 benchmarks (GSM8K, MMLU, Chess, etc.) - but these are classification/reasoning with automatic metrics, not expensive generative evaluation
- Li et al. (2024) "More Agents" tested across multiple benchmarks with hundreds of problems
- Papers using LLM-as-judge typically evaluate 500-1000+ outputs

Your 16 tasks (8 analytical + 8 creative) is the weak point. Not the run count - 3,456 runs is respectable. The *task diversity* is what will be attacked. "Did you just find this works for your 8 creative writing prompts?"

**For AAMAS: More tolerant.** AAMAS accepts formal papers with modest experimental validation. The 3,456-run scale with the pre-registered factorial design would be seen as thorough.

**What's the minimum credible scale?**
- **3 model families** (you have 2 - Anthropic, Google. Need OpenAI or open-weight)
- **3+ task types** (you have 2 - adding summarization would help significantly)
- **25+ tasks per type** (you have 8 - this is the biggest weakness)
- **2+ repetitions per cell** (you have this for most blocks)

**My honest assessment:** The run count is fine. The model and task diversity are the limiting factors. If you can add 8 summarization tasks and one OpenAI model (even GPT-4o-mini as a "weak" agent), credibility improves dramatically.

---

## 5. Competitive Risk Assessment

### Who's working on this?

**High-activity groups:**
- **Microsoft Research (AutoGen team)** - Actively publishing on multi-agent orchestration. They have the engineering infrastructure to run large-scale experiments. BUT they tend to publish frameworks and demos, not controlled scientific studies.
- **Tsinghua NLP group (ChatDev, MetaGPT adjacent)** - Very productive on multi-agent systems. They could publish a systematic topology comparison.
- **Princeton/Stanford NLP** - Active in reasoning and debate. Could characterize agent-count scaling.
- **DeepMind/Google** - Could study this internally with massive compute. Unlikely to publish the dollar-1K-experiment version; more likely to publish a 100K-run version that makes yours obsolete.

### Scoop risk: **MODERATE**

The specific combination you're studying (disagreement dividend + quorum paradox + topology interaction + formal framework) is distinctive enough that an exact scoop is unlikely. But:

- **"More Agents Is All You Need" follow-ups** studying when *more agents hurts* are likely coming. The contrarian result is an obvious next paper.
- **Someone will publish a systematic topology comparison** soon. This is low-hanging fruit that any well-funded lab can pick.
- **The disagreement-quality curve** is conceptually simple enough that multiple groups could discover it independently.

### Speed importance: **MODERATE-HIGH**

You don't need to race, but you shouldn't dawdle. The window for being first on "disagreement quality curves for multi-agent LLM systems" is probably 6-12 months. If you wait for a full-scale experiment to get the numbers perfect, you risk being scooped on the conceptual contribution.

**This is the strongest argument for the workshop-first strategy:** Get a workshop paper out in 3 months to plant the flag. Then expand for the main conference.

---

## 6. Framing Recommendation

### Strongest framing: "When More Agents Hurt: The Quorum Paradox in Multi-Agent LLM Systems"

Lead with the counterintuitive finding. The Quorum Paradox is your most memorable, citable, tweetable contribution. It directly contradicts "More Agents Is All You Need" - and contrarian results get attention.

**Paper structure recommendation:**
1. **Hook:** "The dominant narrative says more agents = better. We show this is wrong for generative tasks."
2. **Framework:** Formal model (topology, consensus operators, quality decomposition) - keep it tight, 2 pages max
3. **The Paradox:** Block 4 results front and center, with mediation analysis showing WHY
4. **The Nuance:** Disagreement Dividend shows it's not that disagreement is bad - it's that UNCONTROLLED disagreement is bad. There's an optimum.
5. **Practical tools:** Decision matrix, MVQ guidelines
6. **Evaluation rigor:** The bias mitigation protocol as a methodological contribution

### What NOT to lead with:
- Don't lead with the formal framework. Reviewers see "novel formal framework for multi-agent LLM" and expect boring incremental formalism.
- Don't lead with the topology comparison. "We compared flat vs hierarchical vs quorum" sounds like a systems paper, not a science paper.
- Don't lead with MVQ. The bound is a nice contribution but not the story.

### Workshop vs. main conference:

| Path | Pros | Cons |
|------|------|------|
| **Direct to NeurIPS 2026 main** | Maximum prestige if accepted; faster timeline | ~30% acceptance odds with current scale; a desk reject or weak reject wastes 3 months |
| **NeurIPS 2026 Workshop** | ~60% acceptance odds; plants the flag; gets peer feedback; lower stakes | Delays main-conference version; workshop papers are less cited |
| **Direct to AAMAS 2027** | Best venue fit; ~50% acceptance odds; values formal+empirical | Later deadline (typically January); less ML-community visibility |
| **Workshop -> AAMAS 2027** | Highest overall probability of a strong main-conference paper | Slowest path; risk of being scooped between workshop and main |

**My recommendation:** Submit to a **NeurIPS 2026 Workshop** (deadline likely July-August 2026) with the current dollar-1,147 experiment. Use workshop feedback to identify which findings need strengthening. Then submit an **expanded version** to AAMAS 2027 or ICML 2027 with additional models, tasks, and human evaluation.

If the Quorum Paradox results are dramatic (>10% quality drop, confirmed by human evaluation), consider going directly to NeurIPS 2026 main conference. A dramatic, well-validated counterintuitive finding can overcome modest scale.

---

## Summary Scorecard

| Dimension | Assessment | Score |
|-----------|-----------|-------|
| **Novelty** | Combination is novel; individual components have precedents | 7/10 |
| **Story strength (best case)** | Weak Accept at NeurIPS; Accept at AAMAS | 6/10 |
| **Robustness to failure** | Quorum Paradox failure is dangerous; other failures are survivable | 5/10 |
| **Scale credibility** | Run count fine; task/model diversity is the weak link | 5/10 |
| **Competitive risk** | Moderate; 6-12 month window | 6/10 |
| **Evaluation methodology** | Exceptionally thorough; itself a contribution | 9/10 |
| **Cost efficiency** | dollar-1,147 for this breadth of claims is impressive | 8/10 |

### Overall: **Worth running, but manage expectations.**

This dollar-1,147 buys you a strong workshop paper with high probability and a main-conference paper with moderate probability. The expected value is positive. The evaluation methodology alone is worth documenting - it's more rigorous than most published multi-agent evaluation protocols.

**The biggest risk is not that the experiments fail - it's that the effects are real but too small to detect convincingly at this scale.** If I were advising, I'd earmark an additional dollar-500-800 for a follow-up expansion (more tasks, one more model family) contingent on promising initial results from Blocks 0-1.

---

## Strongest Contribution
The **Quorum Paradox** - showing that adding agents can HURT generative quality - combined with a mechanistic explanation (conformity pressure -> diversity collapse). If validated, this is the finding people will cite and remember. It directly challenges the "more agents is better" conventional wisdom.

## Weakest Link
**Task and model diversity.** 8 tasks per type x 3 models from 2 providers is the most attackable dimension. A reviewer will write: "These results may hold for 8 creative writing prompts evaluated by Claude and Gemini, but the authors provide no evidence of generalization." This is a factual criticism that is hard to rebut without more data.

## Minimum Viable Positive Findings
1. A statistically significant Quorum Paradox at one (task type, consensus) combination, confirmed by human evaluation
2. OR a clear inverted-U Disagreement Dividend with identifiable d*
3. PLUS topology x consensus interaction effects showing condition-dependence
4. PLUS the evaluation methodology contribution (bias mitigation protocol)

Any ONE of findings 1 or 2, plus findings 3 and 4, is sufficient for a workshop paper. Both findings 1 AND 2 are needed for a main conference.

## Competitive Risk Assessment
**Moderate.** The conceptual space (when do multi-agent systems fail?) is heating up. A 6-12 month window exists for being first on the systematic characterization. The specific formal framework (MVQ bounds, quality decomposition) provides some moat. Speed matters - don't over-optimize the experiment at the cost of delay.

## Framing Recommendation
Lead with the Quorum Paradox ("When More Agents Hurt"). Frame as an empirical science paper with formal grounding, not a systems/framework paper. Target NeurIPS 2026 Workshop first, then expand for AAMAS 2027 or ICML 2027 main conference. If Quorum Paradox effects are dramatic (>10% quality drop), go directly to NeurIPS main.

---

*Strategic assessment prepared independently. Calibrated against papers I've reviewed and seen accepted/rejected at NeurIPS 2023-2025 and AAMAS 2024-2025.*
