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
from .evaluation.human_eval import HumanEvalManager
from .evaluation.llm_judge import JudgePanel, PairwiseJudge
from .manifest import ExperimentManifest, RunSpec
from .models import AnthropicModelClient, BaseModelClient, GoogleModelClient
from .tasks.loader import TaskInstance, load_all_tasks
from .topologies import (
    BestOfNTopology,
    FlatTopology,
    HierarchicalTopology,
    PipelineTopology,
    QuorumTopology,
)
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
    max_cost_usd: float = 1500.0


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
        self.human_eval = HumanEvalManager(config.results_dir / "human_eval", random_sample_rate=0.15, seed=config.seed)

        self.topologies = {
            "flat": FlatTopology(),
            "hierarchical": HierarchicalTopology(),
            "quorum": QuorumTopology(),
            "pipeline": PipelineTopology(),
            "best_of_n": BestOfNTopology(),
        }

        self._pause_requested = False
        self._pause_reason: str | None = None
        self._runtime_state: dict[str, int] | None = None
        self._runtime_total = 0
        self._runtime_started = 0.0

    async def run_manifest(self, manifest: ExperimentManifest) -> dict[str, Any]:
        """Execute all pending runs and return summary."""
        started = time.perf_counter()
        self.checkpoint.save_manifest_snapshot(manifest.to_dict())

        completed_ids = self.checkpoint.load_completed_ids() if self.config.resume else set()
        pending = [run for run in manifest.runs if run.id not in completed_ids]

        self.logger.info(
            "Manifest runs total=%d pending=%d completed=%d",
            len(manifest.runs),
            len(pending),
            len(completed_ids),
        )

        if not pending:
            return {
                "status": "nothing_to_do",
                "total_runs": len(manifest.runs),
                "completed_runs": len(completed_ids),
                "cost": self.cost_tracker.snapshot(),
            }

        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        state = {"completed": len(completed_ids), "failed": 0, "started": len(completed_ids)}

        self._pause_requested = False
        self._pause_reason = None
        self._runtime_state = state
        self._runtime_total = len(manifest.runs)
        self._runtime_started = started

        async def _run_with_semaphore(run_spec: RunSpec) -> None:
            async with semaphore:
                if self._pause_requested:
                    return
                await self._execute_one(run_spec, state=state, total=len(manifest.runs), start_time=started)

        await asyncio.gather(*[_run_with_semaphore(run) for run in pending])

        elapsed = time.perf_counter() - started
        remaining = max(0, len(manifest.runs) - state["completed"])
        summary = {
            "status": "paused_max_cost" if self._pause_requested else "done",
            "total_runs": len(manifest.runs),
            "completed_runs": state["completed"],
            "failed_runs": state["failed"],
            "pending_runs": remaining,
            "elapsed_seconds": elapsed,
            "cost": self.cost_tracker.snapshot(),
        }
        if self._pause_reason:
            summary["pause_reason"] = self._pause_reason

        self.logger.info(
            "Finished run loop in %.1fs (failed=%d, pending=%d, paused=%s)",
            elapsed,
            state["failed"],
            remaining,
            self._pause_requested,
        )
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
                progress = self._build_progress_snapshot(done=done, total=total, failed=state["failed"], start_time=start_time)
                self.checkpoint.update_progress(progress)
                self.logger.info(
                    "Progress: %d/%d complete | failed=%d | eta=%.1fs | cost=$%.4f",
                    done,
                    total,
                    state["failed"],
                    progress.eta_seconds or 0.0,
                    progress.estimated_total_cost_usd,
                )

    def _build_progress_snapshot(self, *, done: int, total: int, failed: int, start_time: float) -> ProgressSnapshot:
        elapsed = max(0.001, time.perf_counter() - start_time)
        rate = done / elapsed
        remaining = max(0, total - done)
        eta = remaining / max(rate, 1e-6)
        cost_snapshot = self.cost_tracker.snapshot()
        return ProgressSnapshot(
            timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
            completed_runs=done,
            pending_runs=remaining,
            failed_runs=failed,
            eta_seconds=eta,
            estimated_total_cost_usd=float(cost_snapshot["total_cost_usd"]),
            estimated_cost_by_model={
                model: float(values["cost_usd"])
                for model, values in cost_snapshot["by_model"].items()
            },
            status="paused_max_cost" if self._pause_requested else "running",
            warning=self._pause_reason,
            max_cost_usd=float(self.config.max_cost_usd),
        )

    def _check_cost_guardrail(self) -> None:
        if self._pause_requested:
            return
        total_cost = float(self.cost_tracker.snapshot().get("total_cost_usd", 0.0))
        if total_cost >= float(self.config.max_cost_usd):
            self._pause_requested = True
            self._pause_reason = (
                f"Max cost guardrail triggered at ${total_cost:.2f} "
                f"(limit=${self.config.max_cost_usd:.2f}). Execution paused."
            )
            self.logger.warning(self._pause_reason)

            if self._runtime_state is not None:
                progress = self._build_progress_snapshot(
                    done=int(self._runtime_state.get("completed", 0)),
                    total=self._runtime_total,
                    failed=int(self._runtime_state.get("failed", 0)),
                    start_time=self._runtime_started,
                )
                self.checkpoint.update_progress(progress)

    async def _execute_run(self, run_spec: RunSpec) -> dict[str, Any]:
        task = self._get_task(run_spec.task_type, run_spec.task_id)
        system_prompts = self._build_system_prompts(run_spec)
        topology = self.topologies[run_spec.topology]

        judge_panel = self._build_judge_panel(run_spec)
        consensus = await self._build_consensus(run_spec, judge_panel=judge_panel)

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
            self._check_cost_guardrail()

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
        output_agent_ids = [o.agent_id for o in topology_result.outputs]
        disagreement = disagreement_summary(output_texts)
        final_text = topology_result.consensus.selected_text

        panel_payload, quality_score, selected_per_judge_scores = await self._evaluate_final_quality(
            run_spec=run_spec,
            task=task,
            output_texts=output_texts,
            output_agent_ids=output_agent_ids,
            final_text=final_text,
            selected_agent_id=topology_result.consensus.selected_agent_id,
            consensus_metadata=topology_result.consensus.metadata,
            judge_panel=judge_panel,
        )

        threshold_met = (
            quality_score >= run_spec.quality_threshold
            if run_spec.quality_threshold is not None
            else None
        )
        thresholds = {
            str(threshold): quality_score >= threshold
            for threshold in run_spec.posthoc_quality_thresholds
        }

        debate_rounds = self._extract_debate_rounds(
            run_spec=run_spec,
            topology_result=topology_result,
        )

        result_payload = {
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
                "posthoc_quality_thresholds": run_spec.posthoc_quality_thresholds,
            },
            "task": {
                "title": task.title,
                "prompt": task.prompt,
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
            "debate_rounds": debate_rounds,
            "evaluation": {
                "quality_score": quality_score,
                "threshold_met": threshold_met,
                "thresholds": thresholds,
                "selected_per_judge_scores": selected_per_judge_scores,
                "judge_panel": panel_payload,
                "disagreement": disagreement,
            },
        }

        human_decision = self.human_eval.decide(result_payload)
        if human_decision.should_flag:
            sheet_path = self.human_eval.create_sheet(result_payload, human_decision.reasons)
            result_payload["evaluation"]["human_review"] = {
                "flagged": True,
                "reasons": human_decision.reasons,
                "sheet_path": str(sheet_path),
            }
        else:
            result_payload["evaluation"]["human_review"] = {
                "flagged": False,
                "reasons": [],
                "sheet_path": None,
            }

        return result_payload

    async def _evaluate_final_quality(
        self,
        *,
        run_spec: RunSpec,
        task: TaskInstance,
        output_texts: list[str],
        output_agent_ids: list[str],
        final_text: str,
        selected_agent_id: str | None,
        consensus_metadata: dict[str, Any],
        judge_panel: JudgePanel,
    ) -> tuple[dict[str, Any], float, dict[str, float]]:
        # Reuse panel evaluation from judge-based consensus when possible.
        panel_payload = consensus_metadata.get("judge_panel") if run_spec.consensus == "judge_based" else None

        if panel_payload is not None:
            score_map = panel_payload.get("bt_scores", {})
            selected_id = selected_agent_id
            if selected_id not in score_map:
                # Fallback by text match if selected ID came from hierarchical cluster IDs.
                for agent_id, text in zip(output_agent_ids, output_texts):
                    if text == final_text and agent_id in score_map:
                        selected_id = agent_id
                        break
            if selected_id in score_map:
                quality = float(score_map[selected_id])
                per_judge = {
                    judge: float(scores.get(selected_id, 0.0))
                    for judge, scores in panel_payload.get("per_judge_bt_scores", {}).items()
                }
                return panel_payload, quality, per_judge

        # Otherwise score final output against candidate pool explicitly.
        candidate_texts = [final_text, *output_texts]
        candidate_ids = ["final_consensus", *output_agent_ids]
        panel_eval = await judge_panel.evaluate_candidates(
            task_type=task.task_type,
            task_prompt=task.prompt,
            rubric=task.rubric,
            outputs=candidate_texts,
            seed=self.config.seed + run_spec.repetition + 999,
        )
        payload = panel_eval.to_dict(candidate_ids=candidate_ids)
        quality = float(payload["bt_scores"].get("final_consensus", 0.0))
        per_judge = {
            judge: float(scores.get("final_consensus", 0.0))
            for judge, scores in payload.get("per_judge_bt_scores", {}).items()
        }
        return payload, quality, per_judge

    def _extract_debate_rounds(self, *, run_spec: RunSpec, topology_result) -> list[dict[str, Any]]:
        metadata_rounds = topology_result.metadata.get("debate_rounds") if topology_result.metadata else None
        if metadata_rounds:
            return metadata_rounds

        if run_spec.consensus == "debate_then_vote":
            return [
                {
                    "round": 1,
                    "phase": "initial_outputs",
                    "agent_outputs": [
                        {"agent_id": out.agent_id, "model_name": out.model_name, "text": out.text}
                        for out in topology_result.outputs
                    ],
                },
                {
                    "round": 2,
                    "phase": "final_consensus",
                    "agent_outputs": [
                        {
                            "agent_id": topology_result.consensus.selected_agent_id or "consensus",
                            "model_name": "consensus",
                            "text": topology_result.consensus.selected_text,
                        }
                    ],
                },
            ]

        return []

    def _get_task(self, task_type: str, task_id: str) -> TaskInstance:
        try:
            return self.task_catalog[task_type][task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown task {task_type}:{task_id}") from exc

    def _build_system_prompts(self, run_spec: RunSpec) -> list[str]:
        base = "You are an expert agent. Be explicit, structured, and self-critical."
        perspectives = [
            "adopt a skeptical perspective",
            "adopt a systems-level perspective",
            "adopt a pragmatic implementation perspective",
            "adopt a risk-management perspective",
            "adopt a contrarian perspective",
        ]
        rng = random.Random(f"{self.config.seed}:{run_spec.id}")
        shuffled_perspectives = list(perspectives)
        rng.shuffle(shuffled_perspectives)

        prompts: list[str] = []
        for idx in range(run_spec.agent_count):
            if run_spec.prompt_strategy == "identical":
                prompts.append(base)
            elif run_spec.prompt_strategy == "slight_variation":
                prompts.append(base + f" Focus area {idx + 1}: emphasize precision and concise argumentation.")
            elif run_spec.prompt_strategy == "perspective_diversity":
                prompts.append(base + ". " + shuffled_perspectives[idx % len(shuffled_perspectives)] + ".")
            elif run_spec.prompt_strategy == "perspective_plus_model_mix":
                prompts.append(
                    base
                    + ". "
                    + shuffled_perspectives[idx % len(shuffled_perspectives)]
                    + ". Challenge hidden assumptions and present an alternative framing before concluding."
                )
            else:
                prompts.append(base + " Use adversarial reasoning to expose weak claims before finalizing.")
        return prompts

    async def _build_consensus(self, run_spec: RunSpec, *, judge_panel: JudgePanel):
        if run_spec.consensus == "simple_vote":
            return SimpleVoteConsensus()
        if run_spec.consensus == "debate_then_vote":
            return DebateThenVoteConsensus()
        if run_spec.consensus == "judge_based":
            return JudgeBasedConsensus(judge_panel=judge_panel)
        raise ValueError(f"Unsupported consensus type: {run_spec.consensus}")

    def _build_judge_panel(self, run_spec: RunSpec) -> JudgePanel:
        selected_models = self._select_judge_models(run_spec.model_assignment)
        judges: list[PairwiseJudge] = []

        for judge_model in selected_models:
            judge_client = self._get_client(judge_model)

            def _recorder(model_name: str, response, tag: str, *, _block=run_spec.block_id, _run=run_spec.id):
                self.cost_tracker.record_call(
                    block_id=_block,
                    run_id=_run,
                    model_name=model_name,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=response.latency_ms,
                    tag=tag,
                )
                self._check_cost_guardrail()

            judges.append(
                PairwiseJudge(
                    judge_client=judge_client,
                    dry_run=self.config.dry_run,
                    call_recorder=_recorder,
                )
            )

        return JudgePanel(judges=judges)

    def _select_judge_models(self, agent_models: list[str]) -> list[str]:
        judge_cfg = self.models_config.get("judge_pool", {})
        panel_size = int(judge_cfg.get("panel_size", 3))
        primary = [str(m) for m in judge_cfg.get("primary_models", [])]
        reserve = [str(m) for m in judge_cfg.get("reserve_models", [])]

        pool = []
        for model in [*primary, *reserve]:
            if model not in pool:
                pool.append(model)

        if not pool:
            raise ValueError("judge_pool is empty in models config")

        agent_set = set(agent_models)
        if bool(judge_cfg.get("exclude_agent_models", True)):
            pool = [m for m in pool if m not in agent_set]

        if bool(judge_cfg.get("prefer_different_families", True)):
            run_families = {
                str(self.models_config["models"][model].get("family", model))
                for model in agent_models
                if model in self.models_config["models"]
            }
            preferred = [
                m
                for m in pool
                if str(self.models_config["models"].get(m, {}).get("family", m)) not in run_families
            ]
            fallback = [m for m in pool if m not in preferred]
            ordered = preferred + fallback
        else:
            ordered = pool

        if len(ordered) < panel_size:
            raise ValueError(
                f"Not enough judges after exclusions: need {panel_size}, have {len(ordered)} (run models={agent_models})"
            )
        return ordered[:panel_size]

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
