"""Asynchronous, resumable experiment runner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import random
import time
from typing import Any

import yaml

from .consensus import (
    AgentOutput,
    DebateThenVoteConsensus,
    JudgeBasedConsensus,
    SimpleVoteConsensus,
)
from .evaluation.disagreement import disagreement_summary
from .evaluation.llm_judge import LLMJudge
from .evaluation.metrics import evaluate_analytical, evaluate_creative
from .manifest import ExperimentManifest, RunSpec
from .models import AnthropicModelClient, BaseModelClient, GoogleModelClient
from .tasks.loader import TaskInstance, load_all_tasks
from .topologies import FlatTopology, HierarchicalTopology, QuorumTopology
from .utils.checkpoint import CheckpointManager, ProgressSnapshot, utc_now_iso
from .utils.cost_tracker import CostTracker, PriceConfig
from .utils.rate_limiter import AsyncRateLimiter, retry_with_backoff


@dataclass(slots=True)
class RunnerConfig:
    """Runtime configuration for ExperimentRunner."""

    models_config_path: Path
    task_dir: Path
    results_dir: Path
    max_concurrent: int = 10
    max_retries: int = 5
    progress_every: int = 10
    dry_run: bool = False
    resume: bool = True
    seed: int = 42


class ExperimentRunner:
    """Coordinates concurrent execution with checkpointing and cost tracking."""

    def __init__(self, config: RunnerConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

        models_config = yaml.safe_load(config.models_config_path.read_text(encoding="utf-8"))
        self.models_config = models_config

        self.task_catalog = load_all_tasks(config.task_dir)
        self.checkpoint = CheckpointManager(config.results_dir)
        self.rate_limiter = AsyncRateLimiter()
        self._random = random.Random(config.seed)
        self._state_lock = asyncio.Lock()

        pricing: dict[str, PriceConfig] = {}
        for model_name, entry in models_config["models"].items():
            pricing[model_name] = PriceConfig(
                input_per_million=float(entry["pricing_per_1m_tokens"]["input"]),
                output_per_million=float(entry["pricing_per_1m_tokens"]["output"]),
            )
        self.cost_tracker = CostTracker(pricing=pricing, results_dir=config.results_dir)

        self._client_cache: dict[str, BaseModelClient] = {}
        self._judge: LLMJudge | None = None

        self.topologies = {
            "flat": FlatTopology(),
            "hierarchical": HierarchicalTopology(),
            "quorum": QuorumTopology(),
        }

    async def run_manifest(self, manifest: ExperimentManifest) -> dict[str, Any]:
        """Execute all pending runs and return summary."""
        started = time.perf_counter()
        self.checkpoint.save_manifest_snapshot(manifest.to_dict())

        completed_ids = self.checkpoint.load_completed_ids() if self.config.resume else set()
        pending = [run for run in manifest.runs if run.id not in completed_ids]

        self.logger.info("Manifest runs total=%d pending=%d completed=%d", len(manifest.runs), len(pending), len(completed_ids))

        if not pending:
            return {
                "status": "nothing_to_do",
                "total_runs": len(manifest.runs),
                "completed_runs": len(completed_ids),
                "cost": self.cost_tracker.snapshot(),
            }

        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        state = {"completed": len(completed_ids), "failed": 0, "started": len(completed_ids)}

        async def _run_with_semaphore(run_spec: RunSpec) -> None:
            async with semaphore:
                await self._execute_one(run_spec, state=state, total=len(manifest.runs), start_time=started)

        await asyncio.gather(*[_run_with_semaphore(run) for run in pending])

        elapsed = time.perf_counter() - started
        summary = {
            "status": "done",
            "total_runs": len(manifest.runs),
            "completed_runs": state["completed"],
            "failed_runs": state["failed"],
            "elapsed_seconds": elapsed,
            "cost": self.cost_tracker.snapshot(),
        }
        self.logger.info("Finished all runs in %.1fs (failed=%d)", elapsed, state["failed"])
        return summary

    async def _execute_one(self, run_spec: RunSpec, *, state: dict[str, int], total: int, start_time: float) -> None:
        """Execute one run and persist result atomically."""

        async def _core() -> dict[str, Any]:
            return await self._execute_run(run_spec)

        try:
            result = await retry_with_backoff(
                _core,
                max_retries=self.config.max_retries,
                base_delay=1.0,
                retryable_exceptions=(Exception,),
            )
            self.checkpoint.save_result(run_spec.block_id, run_spec.id, result)
            success = True
        except Exception as exc:  # pragma: no cover - safety path
            result = {
                "run_id": run_spec.id,
                "block_id": run_spec.block_id,
                "status": "failed",
                "error": repr(exc),
                "timestamp_utc": utc_now_iso(),
            }
            self.checkpoint.save_result(run_spec.block_id, run_spec.id, result)
            self.logger.exception("Run failed: %s", run_spec.id)
            success = False

        async with self._state_lock:
            state["completed"] += 1
            if not success:
                state["failed"] += 1

            done = state["completed"]
            if done % self.config.progress_every == 0 or done == total:
                elapsed = max(0.001, time.perf_counter() - start_time)
                rate = done / elapsed
                remaining = max(0, total - done)
                eta = remaining / max(rate, 1e-6)
                cost_snapshot = self.cost_tracker.snapshot()
                progress = ProgressSnapshot(
                    timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
                    completed_runs=done,
                    pending_runs=remaining,
                    failed_runs=state["failed"],
                    eta_seconds=eta,
                    estimated_total_cost_usd=float(cost_snapshot["total_cost_usd"]),
                    estimated_cost_by_model={
                        model: float(values["cost_usd"])
                        for model, values in cost_snapshot["by_model"].items()
                    },
                )
                self.checkpoint.update_progress(progress)
                self.logger.info(
                    "Progress: %d/%d complete | failed=%d | eta=%.1fs | cost=$%.4f",
                    done,
                    total,
                    state["failed"],
                    eta,
                    cost_snapshot["total_cost_usd"],
                )

    async def _execute_run(self, run_spec: RunSpec) -> dict[str, Any]:
        task = self._get_task(run_spec.task_type, run_spec.task_id)
        system_prompts = self._build_system_prompts(run_spec)
        topology = self.topologies[run_spec.topology]
        consensus = await self._build_consensus(run_spec)

        async def invoke_agent(
            agent_idx: int,
            model_alias: str,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            context: dict[str, Any],
        ) -> AgentOutput:
            model_cfg = self.models_config["models"][model_alias]
            provider = str(model_cfg["provider"])
            rpm = int(model_cfg.get("rpm", 60))
            max_tokens = int(model_cfg.get("max_output_tokens", 1024))

            await self.rate_limiter.acquire(key=f"{provider}:{model_alias}", rpm=rpm)
            client = self._get_client(model_alias)
            response = await client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                metadata={"run_id": run_spec.id, "agent_idx": agent_idx, **context},
            )
            self.cost_tracker.record_call(
                block_id=run_spec.block_id,
                run_id=run_spec.id,
                model_name=model_alias,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                tag=str(context.get("phase", "response")),
            )

            return AgentOutput(
                agent_id=f"agent_{agent_idx}",
                model_name=model_alias,
                text=response.text,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                metadata={"phase": context.get("phase", "main")},
            )

        topology_result = await topology.execute(
            task_type=task.task_type,
            task_prompt=task.prompt,
            rubric=task.rubric,
            model_assignment=run_spec.model_assignment,
            system_prompts=system_prompts,
            temperature=run_spec.temperature,
            invoke_agent=invoke_agent,
            consensus=consensus,
            context={"seed": self.config.seed + run_spec.repetition},
        )

        output_texts = [o.text for o in topology_result.outputs]
        disagreement = disagreement_summary(output_texts)
        final_text = topology_result.consensus.selected_text

        quality_score = (
            evaluate_analytical(task.prompt, final_text)
            if run_spec.task_type == "analytical"
            else evaluate_creative(final_text)
        )

        threshold_met = (
            quality_score >= run_spec.quality_threshold
            if run_spec.quality_threshold is not None
            else None
        )

        return {
            "run_id": run_spec.id,
            "block_id": run_spec.block_id,
            "status": "ok",
            "timestamp_utc": utc_now_iso(),
            "config": {
                "task_type": run_spec.task_type,
                "task_id": run_spec.task_id,
                "topology": run_spec.topology,
                "consensus": run_spec.consensus,
                "agent_count": run_spec.agent_count,
                "model_assignment": run_spec.model_assignment,
                "disagreement_level": run_spec.disagreement_level,
                "temperature": run_spec.temperature,
                "prompt_strategy": run_spec.prompt_strategy,
                "repetition": run_spec.repetition,
                "quality_threshold": run_spec.quality_threshold,
            },
            "task": {
                "title": task.title,
                "rubric": task.rubric,
            },
            "outputs": [
                {
                    "agent_id": out.agent_id,
                    "model_name": out.model_name,
                    "text": out.text,
                    "input_tokens": out.input_tokens,
                    "output_tokens": out.output_tokens,
                    "latency_ms": out.latency_ms,
                }
                for out in topology_result.outputs
            ],
            "consensus": {
                "method": topology_result.consensus.method,
                "selected_agent_id": topology_result.consensus.selected_agent_id,
                "selected_text": topology_result.consensus.selected_text,
                "confidence": topology_result.consensus.confidence,
                "scores": topology_result.consensus.scores,
                "metadata": topology_result.consensus.metadata,
            },
            "topology": {
                "name": run_spec.topology,
                "rounds": topology_result.rounds,
                "metadata": topology_result.metadata,
            },
            "evaluation": {
                "quality_score": quality_score,
                "threshold_met": threshold_met,
                "disagreement": disagreement,
            },
        }

    def _get_task(self, task_type: str, task_id: str) -> TaskInstance:
        try:
            return self.task_catalog[task_type][task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown task {task_type}:{task_id}") from exc

    def _build_system_prompts(self, run_spec: RunSpec) -> list[str]:
        base = "You are an expert agent. Be explicit, structured, and self-critical."
        prompts: list[str] = []
        for idx in range(run_spec.agent_count):
            if run_spec.prompt_strategy == "identical":
                prompts.append(base)
            elif run_spec.prompt_strategy == "slight_variation":
                prompts.append(base + f" Focus area {idx + 1}: emphasize precision and concise argumentation.")
            elif run_spec.prompt_strategy == "perspective_diversity":
                perspectives = [
                    "adopt a skeptical perspective",
                    "adopt a systems-level perspective",
                    "adopt a pragmatic implementation perspective",
                    "adopt a risk-management perspective",
                    "adopt a contrarian perspective",
                ]
                prompts.append(base + ". " + perspectives[idx % len(perspectives)] + ".")
            elif run_spec.prompt_strategy == "perspective_plus_model_mix":
                prompts.append(base + f" Take perspective #{idx + 1} and challenge hidden assumptions aggressively.")
            else:
                prompts.append(base + " Use adversarial reasoning to expose weak claims before finalizing.")
        return prompts

    async def _build_consensus(self, run_spec: RunSpec):
        if run_spec.consensus == "simple_vote":
            return SimpleVoteConsensus()
        if run_spec.consensus == "debate_then_vote":
            return DebateThenVoteConsensus()
        if run_spec.consensus == "judge_based":
            if self._judge is None:
                judge_model = self.models_config["judge"]["model"]
                judge_client = self._get_client(judge_model)
                self._judge = LLMJudge(judge_client=judge_client, dry_run=self.config.dry_run)
            return JudgeBasedConsensus(judge=self._judge)
        raise ValueError(f"Unsupported consensus type: {run_spec.consensus}")

    def _get_client(self, model_alias: str) -> BaseModelClient:
        if model_alias in self._client_cache:
            return self._client_cache[model_alias]

        model_cfg = self.models_config["models"][model_alias]
        provider = model_cfg["provider"]
        api_model = model_cfg["api_model"]

        if provider == "anthropic":
            client = AnthropicModelClient(model_alias=model_alias, api_model=api_model, dry_run=self.config.dry_run)
        elif provider == "google":
            client = GoogleModelClient(model_alias=model_alias, api_model=api_model, dry_run=self.config.dry_run)
        else:
            raise ValueError(f"Unknown provider '{provider}' for model '{model_alias}'")

        self._client_cache[model_alias] = client
        return client

    async def close(self) -> None:
        """Close all clients."""
        await asyncio.gather(*[client.close() for client in self._client_cache.values()])