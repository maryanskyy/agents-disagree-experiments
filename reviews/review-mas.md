# Experiment Design Review — Multi-Agent Systems Perspective

**Reviewer:** Dr. Yuki Tanaka (Multi-Agent Systems, Game Theory, Distributed Coordination)
**Date:** 2026-02-28
**Subject:** Experiment framework for "When Agents Disagree" — pre-investment assessment

---

## Verdict: Invest with Changes

**Confidence:** Medium

---

## Top 3 Strengths

1. **Rigorous evaluation methodology with adversarial threat modeling.** The EVALUATION_METHODOLOGY document is among the most thorough I have seen for an LLM evaluation study. The systematic bias taxonomy (7 documented bias types), position randomization protocol, multi-judge panel design with BT-sigma aggregation, and the 100% human validation mandate for Quorum Paradox claims are exactly what a hostile reviewer would demand. The threat-model table mapping each bias to detection, mitigation, and validation is publication-ready.

2. **Well-structured block design with explicit falsification criteria.** The six-block structure maps cleanly onto the four research questions, with Block 0 providing calibration baselines that enable the formal framework validation in later blocks. The pre-registered falsification criteria (e.g., "H1 false if quality is monotonic decreasing with disagreement") are essential for credibility and rare in LLM systems papers. The block decomposition also allows selective execution — if Block 0 reveals unexpected calibration issues, the team can adapt before committing to the expensive factorial blocks.

3. **Strong formal-empirical integration.** The experiment plan explicitly connects each block to the paper's formal claims: Block 0 → mu/rho estimation for competence/correlation decomposition, Block 3 → MVQ theorem validation via threshold attainment, Block 4 → Quorum Paradox proposition testing. This bidirectional linkage between theory and experiment is the hallmark of a well-designed study and will strengthen the paper's contribution significantly beyond a pure empirical survey.

---

## Top 3 Risks

1. **Underpowered interaction effects and MVQ curves (RQ2 and RQ4 at risk).** Block 2 (topology comparison) and Block 5 (interaction probe) each run with only 1 repetition. For a 3x3x2 factorial (topology x consensus x task type), this yields exactly one observation per cell — making ANOVA-style interaction decomposition unreliable. Three-way interactions are effectively undetectable. Block 3 (MVQ curves) also uses only 1 repetition across 4 agent counts, meaning the quality curves Q(n) will have wide confidence intervals and the "knee point" identification for MVQ will be imprecise. These are not minor concerns: RQ2 asks for a *practitioner lookup table* and RQ4 claims *interaction effects explain more variance than either factor alone*. With 1-rep cells, neither claim survives scrutiny.

2. **Model diversity is insufficient for general claims.** Three models from only two providers (Anthropic, Google) is a narrow basis for claims about multi-agent coordination in general. The capability spread is essentially one-dimensional (strong → moderate → weak). Moreover, using Gemini Flash as both a pipeline agent *and* the default judge model (per the experiment plan: "Judge model defaults to Gemini Flash") creates a direct self-preference confound that the evaluation methodology itself identifies as a medium-severity threat. A reviewer will immediately flag this.

3. **Reduced task corpus limits generalizability.** The research questions specify "4 task types" (summarization, analysis, creative, multi-step), but the experiment only implements 2 (analytical, creative). Dropping summarization and multi-step reasoning means the Disagreement Dividend curve Q(d) is only characterized for half the claimed task space. With only 8 instances per family (16 total), domain-specific effects are masked, and any outlier task has disproportionate influence on aggregate results.

---

## Detailed Analysis

### 1. RQ-to-Experiment Mapping

| Research Question | Primary Block(s) | Key Metrics | Assessment |
|---|---|---|---|
| **RQ1: Disagreement Dividend** | Block 1 (disagreement sweep) | Q(d) curve, d* location | **Adequate with caveats.** Block 1 sweeps d in {1..5} across n in {2,3,5} with 2 reps = 480 runs. Sufficient for curve shape detection. But only tests flat topology + debate_then_vote — d* may be topology-dependent. |
| **RQ2: MVQ** | Block 3 (MVQ curves) | Threshold attainment P(Q>=theta) vs n | **Weak.** Only 1 rep x 1 topology (quorum) x 1 consensus (judge_based) = 64 runs. The RQ asks for n* as function of (theta, C, tau), but only one (C,tau) point is tested. The practitioner lookup table is unsupported by data. |
| **RQ3: Quorum Paradox** | Block 4 (asymmetric scaling) | QPI(n), monotonicity violation | **Adequate.** 4 reps x 16 tasks x 3 agent counts = 192 runs. Best-powered block. The paradox_strong_weak mix correctly operationalizes "1 strong + N-1 weak." |
| **RQ4: Topology-Consensus Interaction** | Block 2 + Block 5 | ANOVA decomposition, interaction F-statistics | **Underpowered.** 1-rep cells make interaction testing unreliable. Block 2 uses n in {2,5} and Block 5 uses n in {3,5}, so they don't share agent counts, complicating combined analysis. |

**Gaps:**
- RQ3's mediation analysis (conformity pressure, unique idea count, idea survival rate per the research questions) has no corresponding metrics or data collection in the experiment plan. These are mechanistic claims that require process-level data (intermediate debate round outputs, topic clustering), not just final quality scores.
- RQ1's prediction that d*_creative > d*_analytical requires enough precision in d* estimation across task types. With 2 reps and 8 tasks per family, confidence intervals on d* will likely overlap, making this comparison inconclusive.

### 2. Topology Representativeness

The three topologies (flat, hierarchical, quorum) represent distinct coordination patterns:
- **Flat:** fully connected, equal-weight participation (analog: peer committee)
- **Hierarchical:** tree-structured delegation (analog: organizational chain of command)
- **Quorum:** threshold-based collective decision (analog: distributed consensus protocol)

**What's missing:**
- **Pipeline/sequential:** Agents process serially, each refining the previous output. This is arguably the *most common* multi-agent LLM pattern in practice (chain-of-thought refinement, sequential critique-and-revise). Its absence is a significant gap — reviewers familiar with AutoGen, CrewAI, or LangGraph will expect it.
- **Ring topology:** Circular message passing where each agent sees only the previous agent's output. Interesting for studying information degradation and convergence dynamics.
- **The research questions mention "4 topologies"** — only 3 are implemented. The 4th was likely dropped during scale reduction, but this should be documented and justified.

The three chosen topologies are adequate for a *first* study but will draw reviewer comments about missing sequential/pipeline patterns.

### 3. Quorum Paradox Detectability

**Statistical power assessment for Block 4:**
- 64 paired observations per agent count transition (n=2 to 3, n=3 to 5)
- For Cohen's d = 0.4 (medium effect), paired t-test power is approximately 0.87 at alpha = 0.05
- For Cohen's d = 0.3 (small-medium), power is approximately 0.68 — borderline

**Structural concerns:**
- With only 2 model types (Opus, Flash), we cannot disentangle "adding *any* agent degrades quality" from "adding *this specific weak model* degrades quality." The paradox claim is about quorum size, but the experiment confounds quorum size with model composition. At n=2 it's {Opus, Flash}; at n=3 it's {Opus, Flash, Flash}; at n=5 it's {Opus, Flash, Flash, Flash, Flash}. The n=3 and n=5 conditions are increasingly Flash-dominated. A skeptical reviewer will argue the quality dip is simply "more weak model, worse output" rather than a genuine coordination paradox.
- **Remedy:** Include a homogeneous-strong control (e.g., 3x Opus) at n=3 to isolate the composition effect from the quorum-size effect. If 3x Opus doesn't dip but 1 Opus + 2 Flash does, the paradox is about asymmetry, not size per se. This distinction matters for the formal proposition.
- The quality thresholds in Block 3 ([0.60, 0.70, 0.80, 0.90]) are reasonable but need to be calibrated against Block 0 baselines.

### 4. Model Selection

**Current pool:** Claude Opus 4.6 (strong/expensive), Gemini 2.5 Pro (strong/moderate), Gemini 2.0 Flash (fast/cheap).

**Assessment:**
- Covers the capability-cost frontier adequately
- Sufficient for testing the strong-vs-weak asymmetry hypothesis
- Only 2 providers — provider-specific training artifacts could confound topology effects
- No open-weight model — limits reproducibility and generalizability claims
- Gemini Flash as both agent and default judge creates self-preference risk

**Should you add a 4th model?**
Yes, but strategically. I recommend **GPT-4o** (or o3-mini if cost is a concern):
- Adds a 3rd provider family, strengthening generalizability
- Enables n=4 with all distinct models, creating richer MVQ curves
- Eliminates the need to use a pipeline model as judge
- Cost increase is bounded: GPT-4o is comparable to Gemini 2.5 Pro in pricing

If a 4th model is infeasible, at minimum **replace Gemini Flash as the default judge** with a model not in the agent pool. Use GPT-4o-mini or Claude Haiku as the judge — both are cheap and from different families than at least one agent model.

### 5. Reduced Scale Risk Assessment

| What's Lost | Severity | Affected RQ |
|---|---|---|
| Interaction effects undetectable (1-rep cells) | **High** | RQ4 |
| MVQ curve precision (1-rep, no error bars) | **High** | RQ2 |
| d* comparison across task types (CI overlap) | **Medium** | RQ1 |
| Missing n=7 agent count | **Medium** | RQ2, RQ3 |
| Missing 2 task types (summarization, multi-step) | **Medium** | All |
| Mediation analysis data not collected | **Medium** | RQ3 |
| Three-way interactions invisible | **Low** | RQ4 |

**Most at-risk findings:**
1. The "decision matrix" (RQ4's most practical contribution per the research questions) requires reliable interaction effect estimates, which 1-rep cells cannot provide.
2. The practitioner lookup table for MVQ (RQ2) needs confidence intervals around n*, which 1-rep Block 3 cannot deliver.
3. The claim that d*_creative > d*_analytical (RQ1) will likely be inconclusive.

**What's preserved:**
- The Quorum Paradox detection (Block 4) is adequately powered with 4 reps.
- The Disagreement Dividend curve shape (Block 1) should be recoverable with 2 reps x 16 tasks.
- The calibration baselines (Block 0) are solid with 4 reps.

### 6. Investment Justification

**Cost-benefit analysis:**

| Factor | Assessment |
|---|---|
| $3-8K API cost | Reasonable for the scope. Many NeurIPS empirical papers spend more. |
| 24h runtime | Manageable with async concurrency. |
| 75 person-hours human evaluation | Significant but necessary per the methodology. Budget for this. |
| Publication potential | Conditional — the topic is timely and the formal framework is novel, but the reduced scale weakens several key claims. |

**Publishability at a top venue (AAMAS, NeurIPS, ICML):**
- With changes below: **plausible for AAMAS** (multi-agent focused, values formal-empirical integration), **borderline for NeurIPS/ICML** (need stronger empirical results or accept workshop track).
- Without changes: the underpowered interaction analysis and model diversity concerns would likely draw major revision requests, and the Quorum Paradox confound (model composition vs quorum size) is a potential reject-level issue at a top venue.

---

## Required Changes Before Running

1. **Separate judge and agent model pools.** Do not use Gemini Flash as both pipeline agent and default judge. Use GPT-4o-mini, Claude Haiku, or another model outside the agent pool. This is non-negotiable — the evaluation methodology document itself flags self-preference as a medium-severity threat, yet the experiment plan violates this by design.

2. **Add a homogeneous-strong control to Block 4.** Run n=3 x Opus-only alongside the paradox_strong_weak conditions. Without this, the Quorum Paradox claim is confounded with model composition effects. Cost: approximately 48 additional runs (3 x 16 tasks x 1 rep). Trivial relative to total budget.

3. **Increase Block 3 (MVQ) repetitions to at least 3.** Currently 1 rep x 64 runs yields no error bars on Q(n). With 3 reps (192 runs), you can report bootstrap CIs on threshold attainment curves. These curves are the empirical validation of your crown-jewel theorem — they cannot be noisy.

4. **Collect intermediate debate-round outputs for RQ3 mediation analysis.** The research questions promise conformity pressure measurement and idea survival rate analysis. The experiment infrastructure must log per-round agent responses, not just final consensus outputs. Without process data, the mechanistic claims in the Quorum Paradox proposition are unsupported.

5. **Document which 4th topology and which 2 additional consensus mechanisms were dropped from the research questions, and why.** The research questions specify "4 topologies x 5 consensus mechanisms" but the experiment implements 3 x 3. Reviewers will notice this discrepancy. If the cuts are principled (e.g., pipeline deferred to follow-up), state it explicitly in the experiment plan.

---

## Optional Improvements

1. **Add GPT-4o as a 4th agent model.** This adds provider diversity, enables all-distinct n=4 quorums, and strengthens generalizability. If cost is the constraint, replace Gemini 2.0 Flash with GPT-4o-mini in some conditions (comparable capability tier, different provider).

2. **Increase Block 2 or Block 5 repetitions to 2.** Even 2-rep cells in the factorial would substantially improve interaction effect estimates. If budget is tight, prioritize Block 5 (interaction probe) since that's specifically designed for ANOVA decomposition. Cost: approximately 288 additional runs.

3. **Add a pipeline/sequential topology.** This is the most practically relevant missing topology. Even testing it in Block 2 only (not the full factorial) would pre-empt the most obvious reviewer objection about topology coverage.

4. **Include 2-3 summarization tasks.** The research questions explicitly name summarization as a task type. Adding even a small summarization set would broaden the empirical coverage at minimal cost (reuse the same block structure, just additional task instances).

5. **Run a pilot batch (Block 0 + partial Block 1) first.** Before committing the full $3-8K, invest approximately $200-400 in Block 0 calibration and a slice of Block 1. This yields: (a) actual baseline quality estimates to calibrate thresholds, (b) empirical variance estimates for power analysis refinement, (c) validation that the infrastructure works end-to-end. This is standard practice and would reduce the risk of the larger investment.

6. **Pre-register the analysis plan on OSF or a similar platform.** The falsification criteria are already well-specified. Formal pre-registration would strengthen publication credibility, particularly for the Quorum Paradox claims which are the paper's most provocative finding.

---

## Summary Assessment

The experiment framework is thoughtfully designed with clear formal-empirical linkage and an unusually thorough evaluation methodology. The research questions are well-motivated and the block structure is logical. However, the reduced scale creates real risks for two of the four research questions (RQ2 and RQ4), the model pool is too narrow for general claims, and the judge-agent overlap is a preventable methodological error.

The five required changes above are feasible within the existing budget envelope (they add approximately 250 runs, roughly 15% more). They address the most likely reviewer objections and protect the paper's central claims — particularly the Quorum Paradox, which is the signature contribution but currently confounded.

With these changes, the experiment is worth the investment. The topic is timely, the formal framework is novel, and a clean empirical validation would make this a strong submission to AAMAS 2026 or a NeurIPS workshop. Without the changes, I would expect major revisions at best, with the judge self-preference issue and paradox confound as likely grounds for rejection.

---

*Review prepared independently. I have no prior relationship with the authors.*
