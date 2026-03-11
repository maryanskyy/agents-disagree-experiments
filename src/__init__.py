"""Agents Disagree experiment framework package."""

from .manifest import ExperimentManifest, RunSpec
from .runner import ExperimentRunner, RunnerConfig

__all__ = ["ExperimentManifest", "RunSpec", "ExperimentRunner", "RunnerConfig"]