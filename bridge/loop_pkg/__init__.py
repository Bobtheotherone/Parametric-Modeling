"""Loop package: Orchestration loop components.

This package contains the refactored orchestration loop components
for improved readability and maintainability within tooling constraints.

The main entrypoint remains bridge/loop.py which imports from this package.
"""

from bridge.loop_pkg.config import AGENTS, ParallelSettings, RunConfig, RunState
from bridge.loop_pkg.policy import AgentPolicy, AgentPolicyViolation, get_agent_policy, set_agent_policy
from bridge.loop_pkg.turn_normalizer import (
    NormalizationResult,
    TurnNormalizer,
    normalize_agent_output,
    validate_turn_lenient,
)

__all__ = [
    "AGENTS",
    "AgentPolicy",
    "AgentPolicyViolation",
    "ParallelSettings",
    "RunConfig",
    "RunState",
    "get_agent_policy",
    "set_agent_policy",
    # Turn normalization
    "NormalizationResult",
    "TurnNormalizer",
    "normalize_agent_output",
    "validate_turn_lenient",
]
