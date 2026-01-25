"""Agent policy enforcement for the orchestration loop."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

from bridge.loop_pkg.config import AGENTS


class AgentPolicyViolation(Exception):
    """Raised when code attempts to use an agent that violates the policy."""

    pass


@dataclasses.dataclass
class AgentPolicy:
    """Centralized agent selection policy.

    When forced_agent is set (via --only-codex or --only-claude), ALL agent
    selections must go through this policy and will be overridden to use
    only the forced agent.
    """

    forced_agent: str | None = None  # Set by --only-* flags
    allowed_agents: tuple[str, ...] = AGENTS
    runs_dir: Path | None = None  # For writing violation artifacts

    def enforce(self, requested_agent: str, context: str = "") -> str:
        """Enforce the agent policy, returning the agent to use.

        Args:
            requested_agent: The agent that was requested
            context: Description of where this request originated (for error messages)

        Returns:
            The agent to actually use (forced_agent if set, otherwise requested)

        Raises:
            AgentPolicyViolation: If forced mode is active and code tries to use wrong agent
        """
        if self.forced_agent:
            if requested_agent != self.forced_agent and requested_agent in AGENTS:
                # Log the override
                print(f"[AgentPolicy] OVERRIDE: {requested_agent} -> {self.forced_agent} ({context})")
            return self.forced_agent

        # No forced agent - verify requested is allowed
        if requested_agent not in self.allowed_agents:
            if self.allowed_agents:
                return self.allowed_agents[0]
            return AGENTS[0]

        return requested_agent

    def enforce_strict(self, requested_agent: str, context: str = "") -> str:
        """Strict enforcement - raises exception if wrong agent is requested.

        Use this for code paths that should NEVER attempt to use the wrong agent
        (e.g., fallback logic that might try to switch agents).
        """
        if self.forced_agent and requested_agent != self.forced_agent:
            msg = (
                f"AGENT POLICY VIOLATION: Attempted to use '{requested_agent}' "
                f"when --only-{self.forced_agent} is active. Context: {context}"
            )
            self._write_violation_artifact(msg, requested_agent, context)
            raise AgentPolicyViolation(msg)
        return self.enforce(requested_agent, context)

    def _write_violation_artifact(self, msg: str, requested: str, context: str) -> None:
        """Write an artifact explaining the policy violation."""
        if not self.runs_dir:
            return
        artifact_path = self.runs_dir / "agent_policy_violation.txt"
        content = f"""AGENT POLICY VIOLATION
======================

Timestamp: {dt.datetime.utcnow().isoformat()}Z
Forced Agent: {self.forced_agent}
Requested Agent: {requested}
Context: {context}

Message:
{msg}

This file was created because code attempted to invoke an agent that
violates the --only-{self.forced_agent} flag. This indicates a bug in
the orchestrator's agent selection logic.
"""
        try:
            artifact_path.write_text(content, encoding="utf-8")
            print(f"[AgentPolicy] Violation artifact written to: {artifact_path}")
        except Exception as e:
            print(f"[AgentPolicy] Failed to write violation artifact: {e}")

    def is_forced_mode(self) -> bool:
        """Return True if a forced agent mode is active."""
        return self.forced_agent is not None

    def get_prompt_header(self) -> str:
        """Get a header to inject into prompts when in forced mode.

        This tells the agent it's the only one and must implement, not just review.
        """
        if not self.forced_agent:
            return ""

        return f"""## AGENT POLICY OVERRIDE

**IMPORTANT**: You are running in `--only-{self.forced_agent}` mode.

- You are the ONLY agent allowed in this session.
- You MUST implement all changes yourself. Do NOT suggest handing off to another agent.
- You MUST verify your own changes. Do NOT assume another agent will review.
- Set `next_agent` to `"{self.forced_agent}"` in your response (it will be enforced anyway).
- Focus on both implementation AND verification - you are responsible for the full cycle.

"""


# Global policy instance (set during main() based on CLI flags)
_agent_policy: AgentPolicy | None = None


def get_agent_policy() -> AgentPolicy:
    """Get the global agent policy. Returns a default policy if not set."""
    global _agent_policy
    if _agent_policy is None:
        _agent_policy = AgentPolicy()
    return _agent_policy


def set_agent_policy(policy: AgentPolicy) -> None:
    """Set the global agent policy."""
    global _agent_policy
    _agent_policy = policy
