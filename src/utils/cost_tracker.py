"""Token accounting for experiment runs.

Tracks cumulative token usage by model and block; logs every API call.
Cost computation has been removed — use public API pricing externally if needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class CostTracker:
    """Tracks cumulative token usage by model and block; logs every API call."""

    def __init__(self, results_dir: Path, **_kwargs: Any) -> None:
        self.results_dir = results_dir
        self._lock = Lock()

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.calls = 0

        self.by_model: dict[str, dict[str, float]] = {}
        self.by_block: dict[str, dict[str, float]] = {}

        self.log_path = results_dir / "token_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

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
        """Record one model call and append JSONL token log entry."""
        with self._lock:
            self.calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            self.by_model.setdefault(model_name, {"input_tokens": 0, "output_tokens": 0})
            self.by_model[model_name]["input_tokens"] += input_tokens
            self.by_model[model_name]["output_tokens"] += output_tokens

            self.by_block.setdefault(block_id, {"calls": 0})
            self.by_block[block_id]["calls"] += 1

            entry = {
                "run_id": run_id,
                "block_id": block_id,
                "model_name": model_name,
                "tag": tag,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

            return 0.0  # no cost tracking

    def snapshot(self) -> dict[str, Any]:
        """Return cumulative token usage snapshot."""
        with self._lock:
            return {
                "calls": self.calls,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "by_model": self.by_model,
                "by_block": self.by_block,
            }
