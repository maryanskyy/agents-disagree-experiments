"""Model catalog loading and normalization.

Supports both legacy config schema and v3 pool-based schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelCatalog:
    """Normalized model metadata used by runner and scripts."""

    models: dict[str, dict[str, Any]]
    agent_pool: list[str]
    judge_pool: list[str]
    judge_panel_size: int


def _normalize_model_entry(alias: str, entry: dict[str, Any]) -> dict[str, Any]:
    provider = str(entry["provider"])
    api_model = str(entry.get("api_model") or entry.get("model_id") or alias)

    if "pricing_per_1m_tokens" in entry:
        pricing = entry["pricing_per_1m_tokens"]
        input_cost = float(pricing["input"])
        output_cost = float(pricing["output"])
    else:
        input_cost = float(entry.get("input_cost_per_1m", 0.0))
        output_cost = float(entry.get("output_cost_per_1m", 0.0))

    rpm = int(entry.get("rpm", entry.get("rpm_limit", 60)))
    tpm = int(entry.get("tpm", entry.get("tpm_limit", 0)))

    return {
        "provider": provider,
        "api_model": api_model,
        "family": str(entry.get("family", alias)),
        "rpm": rpm,
        "tpm": tpm,
        "pricing_per_1m_tokens": {
            "input": input_cost,
            "output": output_cost,
        },
        "max_output_tokens": int(entry.get("max_output_tokens", 2048)),
        "quality_tier": str(entry.get("quality_tier", entry.get("tier", "unknown"))),
    }


def _normalize_legacy(raw: dict[str, Any]) -> ModelCatalog:
    models = {alias: _normalize_model_entry(alias, cfg) for alias, cfg in raw["models"].items()}

    raw_agent_pool = raw.get("agent_pool", {})
    if isinstance(raw_agent_pool, dict) and "models" in raw_agent_pool:
        agent_pool = [str(name) for name in raw_agent_pool.get("models", [])]
    else:
        agent_pool = [str(name) for name in raw_agent_pool] if isinstance(raw_agent_pool, list) else []

    judge_cfg = raw.get("judge_pool", {})
    if isinstance(judge_cfg, dict) and all(isinstance(v, dict) for v in judge_cfg.values()):
        judge_pool = [str(name) for name in judge_cfg.keys()]
        panel_size = int(raw.get("judge_panel_size", min(3, len(judge_pool))))
    else:
        primary = [str(name) for name in judge_cfg.get("primary_models", [])]
        reserve = [str(name) for name in judge_cfg.get("reserve_models", [])]
        judge_pool: list[str] = []
        for model in [*primary, *reserve]:
            if model not in judge_pool:
                judge_pool.append(model)
        panel_size = int(judge_cfg.get("panel_size", min(3, len(judge_pool))))

    if not agent_pool:
        # Legacy fallback: all non-judge models are candidate agents.
        judge_set = set(judge_pool)
        agent_pool = [name for name in models if name not in judge_set]

    return ModelCatalog(models=models, agent_pool=agent_pool, judge_pool=judge_pool, judge_panel_size=panel_size)


def _normalize_pool_schema(raw: dict[str, Any]) -> ModelCatalog:
    agent_cfg = raw.get("agent_pool", {})
    judge_cfg = raw.get("judge_pool", {})

    if not isinstance(agent_cfg, dict) or not isinstance(judge_cfg, dict):
        raise ValueError("Pool schema requires mapping-style agent_pool and judge_pool entries")

    models: dict[str, dict[str, Any]] = {}
    for alias, entry in agent_cfg.items():
        models[str(alias)] = _normalize_model_entry(str(alias), entry)
    for alias, entry in judge_cfg.items():
        models[str(alias)] = _normalize_model_entry(str(alias), entry)

    agent_pool = [str(name) for name in agent_cfg.keys()]
    judge_pool = [str(name) for name in judge_cfg.keys()]
    panel_size = int(raw.get("judge_panel_size", min(3, len(judge_pool))))

    overlap = set(agent_pool) & set(judge_pool)
    if overlap:
        raise ValueError(f"Agent/judge pools must not overlap; found: {sorted(overlap)}")

    return ModelCatalog(models=models, agent_pool=agent_pool, judge_pool=judge_pool, judge_panel_size=panel_size)


def load_model_catalog(*, config_path: Path | None = None, raw_config: dict[str, Any] | None = None) -> ModelCatalog:
    """Load and normalize model config into a stable internal schema."""
    if raw_config is None:
        if config_path is None:
            raise ValueError("Either config_path or raw_config must be provided")
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(raw_config, dict):
        raise ValueError("Model config must be a mapping")

    if "models" in raw_config:
        return _normalize_legacy(raw_config)

    if "agent_pool" in raw_config and "judge_pool" in raw_config:
        return _normalize_pool_schema(raw_config)

    raise ValueError("Unsupported model config schema")

