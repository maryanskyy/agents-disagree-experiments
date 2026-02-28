# RESEARCH PIVOT — Adopted 2026-02-28

## New Framing

**Old title**: "When Agents Disagree: Quorum-Based Consensus and Adaptive Orchestration Topology for Multi-Agent LLM Pipelines"

**New title**: "Topology Dominance: Why How You Wire Agents Matters More Than Which Model You Use"

**Subtitle/alternate**: "An Empirical Study of Orchestration Topology, Consensus Mechanisms, and Agent Scaling in Multi-Agent LLM Pipelines"

## Reordered Contributions (by strength of evidence)

1. **Topology Dominance** (LEAD) — Orchestration topology (hierarchical, quorum, flat, pipeline) has a larger effect on output quality than individual model capability. Hierarchical consensus win rate ~0.60 vs pipeline ~0.50. This challenges the "just use a bigger model" paradigm.

2. **Evaluation Methodology + BT Artifact** (NOVEL) — Discovery that Bradley-Terry scores are incomparable across different candidate set sizes. Consensus win rate as the corrected metric. Anyone doing multi-agent evaluation will need this.

3. **Practical Decision Framework** (UTILITY) — Given task type, budget, and quality target → recommended topology + agent count + consensus mechanism. Immediately actionable for practitioners.

4. **MVQ Bounds & Disagreement Analysis** (SUPPORTING) — Minimum viable quorum analysis and disagreement-quality relationship as secondary contributions.

## What We Dropped

- "Quorum Paradox" as headline claim — pilot showed it was ~97% BT artifact
- "Disagreement Dividend" inverted-U as primary finding — no signal in pilot (R²=0.001)
- Named phenomena that didn't survive empirical testing

## What This Means for the Paper Structure

- Introduction: lead with "topology matters more than model" — counterintuitive, practitioners care
- Related Work: position against "More Agents Is All You Need" (Li et al.) and ensemble scaling literature
- Methodology: highlight 3-provider design and evaluation correction as contributions
- Results: topology comparison first, then scaling, then disagreement
- Discussion: practical decision framework as the takeaway

## Impact on Experiment Design

No changes to the experiment itself — all blocks still run as planned. The data answers the same questions. We're just reordering which findings we lead with based on what the pilot actually showed.

## Adoption

This pivot was agreed upon 2026-02-28 after pilot review by 5 agents (Elena, Marcus, Tanaka, Sharma, Okonkwo). All review reports in `reviews/pilot-*.md`.
