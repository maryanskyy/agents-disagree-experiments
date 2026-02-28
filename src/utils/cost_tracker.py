"""Real-time token and cost accounting for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass(slots=True)
class PriceConfig:
    """Per-model token pricing in USD per 1M tokens."""

    input_per_million: float
    output_per_million: float


class CostTracker:
    """Tracks cumulative cost by model and block; logs every API call."""

    def __init__(self, pricing: dict[str, PriceConfig], results_dir: Path) -> None:
        self.pricing = pricing
        self.results_dir = results_dir
        self._lock = Lock()

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.calls = 0

        self.by_model: dict[str, dict[str, float]] = {}
        self.by_block: dict[str, dict[str, float]] = {}

        self.log_path = results_dir / "cost_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _compute_call_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        price = self.pricing.get(model_name)
        if price is None:
            return 0.0
        in_cost = (input_tokens / 1_000_000) * price.input_per_million
        out_cost = (output_tokens / 1_000_000) * price.output_per_million
        return in_cost + out_cost

    def record_call(
        self,
        *,
        block_id: str,
        run_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        tag: str,
    ) -> float:
        """Record one model call and append JSONL cost log entry."""
        with self._lock:
            call_cost = self._compute_call_cost(model_name, input_tokens, output_tokens)
            self.calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += call_cost

            self.by_model.setdefault(model_name, {"cost_usd": 0.0, "input_tokens": 0.0, "output_tokens": 0.0})
            self.by_model[model_name]["cost_usd"] += call_cost
            self.by_model[model_name]["input_tokens"] += input_tokens
            self.by_model[model_name]["output_tokens"] += output_tokens

            self.by_block.setdefault(block_id, {"cost_usd": 0.0, "calls": 0.0})
            self.by_block[block_id]["cost_usd"] += call_cost
            self.by_block[block_id]["calls"] += 1

            entry = {
                "run_id": run_id,
                "block_id": block_id,
                "model_name": model_name,
                "tag": tag,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "cost_usd": round(call_cost, 8),
                "total_cost_usd": round(self.total_cost_usd, 8),
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

            return call_cost

    def snapshot(self) -> dict[str, Any]:
        """Return cumulative cost snapshot."""
        with self._lock:
            return {
                "calls": self.calls,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "by_model": self.by_model,
                "by_block": self.by_block,
            }