"""Consensus implementations."""

from .base import AgentOutput, BaseConsensus, ConsensusResult
from .debate import DebateThenVoteConsensus
from .judge import JudgeBasedConsensus
from .vote import SimpleVoteConsensus

__all__ = [
    "AgentOutput",
    "BaseConsensus",
    "ConsensusResult",
    "SimpleVoteConsensus",
    "DebateThenVoteConsensus",
    "JudgeBasedConsensus",
]