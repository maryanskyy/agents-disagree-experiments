# The Selection Bottleneck

Experiment code and data for:

**The Selection Bottleneck: When and Why Diverse LLM Teams Outperform Homogeneous Ones**

*Artem Maryanskyy*

---

## Overview

This repository contains a resumable experiment framework for studying how **team composition** (diverse vs. homogeneous) and **aggregation mechanism** (judge-based selection vs. majority vote vs. synthesis) interact to determine multi-agent LLM pipeline quality.

Key findings:
- A diverse team with judge-based selection achieves **BT-WR = 0.810** vs. 0.512 for a homogeneous Opus team (Hedges' *g* = 2.71)
- Judge-based selection dramatically outperforms MoA-style synthesis (*g* = 3.86)
- Including a weaker, cheaper model paradoxically improves performance (exploratory, *g* = 0.87)
- Independent evaluation with a separate judge panel confirms all effects (Spearman ρ = 0.70)

---

## Repository Structure

```
├── config/
│   ├── models.yaml              # Model pool configuration
│   └── runspecs_v4_scale.json   # 210 experiment run specifications
├── paper/
│   ├── main.tex                 # Full paper (LaTeX)
│   ├── references.bib           # Bibliography
│   └── figures/                 # Generated figures
├── results/
│   └── v4/                      # V4 experiment results (210 runs)
│       ├── scale_diverse_strong_judge_based/
│       ├── scale_homo_opus_judge_based/
│       ├── scale_diverse_mixed_judge_based/
│       ├── scale_diverse_strong_simple_vote/
│       ├── scale_diverse_strong_synthesis/
│       └── decoupled_eval/      # Independent judge re-evaluation
├── scripts/
│   ├── run_experiment_v4_scale.py   # Main experiment runner
│   ├── run_decoupled_eval.py        # Independent evaluation pass
│   ├── generate_figures_v4.py       # Figure generation
│   └── analyze_decoupled.py         # Decoupled eval analysis
├── src/
│   ├── consensus/               # Aggregation mechanisms
│   ├── evaluation/              # Judge panel & metrics
│   ├── models/                  # API client adapters
│   └── utils/                   # Checkpointing, rate limiting
└── tasks/                       # 42 task definitions (7 categories × 6)
```

---

## Prerequisites

- Python 3.10+
- API keys for: Anthropic, OpenAI, Google (and optionally ZhiPu for GLM-5)

---

## Quick Start

```bash
git clone https://github.com/maryanskyy/agents-disagree-experiments.git
cd agents-disagree-experiments
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Run experiments
```bash
python scripts/run_experiment_v4_scale.py --resume
```

### Run decoupled evaluation
```bash
python scripts/run_decoupled_eval.py
```

### Generate figures
```bash
python scripts/generate_figures_v4.py
```

---

## Model Configuration

### Agent Pool (generate candidates)
| Model | Provider | Tier |
|-------|----------|------|
| claude-opus-4-6 | Anthropic | Strong |
| gpt-5.4 | OpenAI | Strong |
| gemini-2.5-pro | Google | Strong |
| claude-haiku-4-5 | Anthropic | Weak |
| gemini-2.5-flash | Google | Weak |

### Selector Judge Pool (evaluate & select)
| Model | Provider |
|-------|----------|
| claude-sonnet-4-6 | Anthropic |
| gpt-5-mini | OpenAI |
| deepseek-v3p2 | DeepSeek |

### Eval Judge Pool (independent verification)
| Model | Provider |
|-------|----------|
| gpt-4o-mini | OpenAI |
| gemini-2.0-flash-001 | Google |
| glm-5 | ZhiPu |

**Zero model overlap** between agent pool, selector judges, and eval judges.

---

## Proxy Support

The framework supports any OpenAI-compatible endpoint (LiteLLM, vLLM, etc.) via environment variables:

```bash
API_BASE_URL=http://your-endpoint/v1
API_ORG_ID=your-org-id        # optional
API_TOKEN_CMD=your-token-cmd   # optional: command that prints a bearer token
```

---

## Citation

```bibtex
@article{maryanskyy2026selection,
  title={The Selection Bottleneck: When and Why Diverse {LLM} Teams Outperform Homogeneous Ones},
  author={Maryanskyy, Artem},
  year={2026}
}
```

---

## License

MIT
