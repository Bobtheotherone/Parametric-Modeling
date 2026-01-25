# SPDX-License-Identifier: MIT
"""Unit tests for bridge/loop_pkg/config.py and bridge/loop_pkg/policy.py.

Tests the orchestrator configuration and policy enforcement modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.loop_pkg.config import (
    AGENTS,
    AgentCapabilities,
    ParallelSettings,
    RunConfig,
    RunState,
)
from bridge.loop_pkg.policy import (
    AgentPolicy,
    AgentPolicyViolation,
    get_agent_policy,
    set_agent_policy,
)

# -----------------------------------------------------------------------------
# ParallelSettings tests
# -----------------------------------------------------------------------------


class TestParallelSettings:
    """Tests for ParallelSettings dataclass."""

    def test_default_values(self) -> None:
        """ParallelSettings has expected defaults."""
        settings = ParallelSettings()
        assert settings.max_workers_default == 8
        assert settings.cpu_intensive_threshold_pct == 40.0
        assert settings.mem_intensive_threshold_pct == 40.0
        assert settings.sample_interval_s == 1.0
        assert settings.consecutive_samples == 3
        assert settings.kill_grace_s == 8.0
        assert settings.terminal_max_bytes_per_worker == 40000
        assert settings.terminal_max_line_length == 600
        assert settings.disable_gpu_by_default is True

    def test_frozen(self) -> None:
        """ParallelSettings is frozen (immutable)."""
        settings = ParallelSettings()
        with pytest.raises(AttributeError):
            settings.max_workers_default = 16  # type: ignore

    def test_custom_values(self) -> None:
        """ParallelSettings accepts custom values."""
        settings = ParallelSettings(
            max_workers_default=4,
            cpu_intensive_threshold_pct=50.0,
            mem_intensive_threshold_pct=60.0,
        )
        assert settings.max_workers_default == 4
        assert settings.cpu_intensive_threshold_pct == 50.0
        assert settings.mem_intensive_threshold_pct == 60.0


# -----------------------------------------------------------------------------
# AgentCapabilities tests
# -----------------------------------------------------------------------------


class TestAgentCapabilities:
    """Tests for AgentCapabilities dataclass."""

    def test_default_values(self) -> None:
        """AgentCapabilities has expected defaults (all True)."""
        caps = AgentCapabilities()
        assert caps.supports_tools is True
        assert caps.supports_fs_read is True
        assert caps.supports_fs_write is True
        assert caps.supports_bash is True
        assert caps.supports_write_access is True

    def test_frozen(self) -> None:
        """AgentCapabilities is frozen (immutable)."""
        caps = AgentCapabilities()
        with pytest.raises(AttributeError):
            caps.supports_tools = False  # type: ignore

    def test_custom_values(self) -> None:
        """AgentCapabilities accepts custom values."""
        caps = AgentCapabilities(
            supports_tools=False,
            supports_bash=False,
        )
        assert caps.supports_tools is False
        assert caps.supports_bash is False
        assert caps.supports_fs_read is True  # Default preserved


# -----------------------------------------------------------------------------
# RunConfig tests
# -----------------------------------------------------------------------------


class TestRunConfig:
    """Tests for RunConfig dataclass."""

    def test_basic_config(self) -> None:
        """RunConfig stores configuration correctly."""
        config = RunConfig(
            max_calls_per_agent=10,
            quota_retry_attempts=3,
            max_total_calls=20,
            max_json_correction_attempts=2,
            fallback_order=["codex", "claude"],
            enable_agents=["codex", "claude"],
            smoke_route=("codex", "claude"),
            agent_scripts={"codex": "agents/codex.sh", "claude": "agents/claude.sh"},
            agent_models={"codex": "gpt-4o", "claude": "claude-3"},
            quota_error_patterns={"codex": ["rate limit"]},
            supports_write_access={"codex": True, "claude": True},
            agent_capabilities={"codex": AgentCapabilities(), "claude": AgentCapabilities()},
            parallel=ParallelSettings(),
        )
        assert config.max_calls_per_agent == 10
        assert config.quota_retry_attempts == 3
        assert config.smoke_route == ("codex", "claude")
        assert "codex" in config.agent_scripts
        assert "claude" in config.agent_models

    def test_frozen(self) -> None:
        """RunConfig is frozen (immutable)."""
        config = RunConfig(
            max_calls_per_agent=10,
            quota_retry_attempts=3,
            max_total_calls=20,
            max_json_correction_attempts=2,
            fallback_order=["codex", "claude"],
            enable_agents=["codex", "claude"],
            smoke_route=("codex", "claude"),
            agent_scripts={},
            agent_models={},
            quota_error_patterns={},
            supports_write_access={},
            agent_capabilities={},
            parallel=ParallelSettings(),
        )
        with pytest.raises(AttributeError):
            config.max_calls_per_agent = 20  # type: ignore


# -----------------------------------------------------------------------------
# RunState tests
# -----------------------------------------------------------------------------


class TestRunState:
    """Tests for RunState dataclass."""

    def test_default_state(self, tmp_path: Path) -> None:
        """RunState initializes with correct defaults."""
        state = RunState(
            run_id="test-run",
            project_root=tmp_path,
            runs_dir=tmp_path / "runs",
            schema_path=tmp_path / "schema.json",
            system_prompt_path=tmp_path / "prompt.txt",
            design_doc_path=tmp_path / "DESIGN_DOCUMENT.md",
        )
        assert state.run_id == "test-run"
        assert state.total_calls == 0
        assert state.call_counts == {"codex": 0, "claude": 0}
        assert state.quota_failures == {"codex": 0, "claude": 0}
        assert state.disabled_by_quota == {"codex": False, "claude": False}
        assert state.history == []
        assert state.grant_write_access is False

    def test_mutable(self, tmp_path: Path) -> None:
        """RunState is mutable (not frozen)."""
        state = RunState(
            run_id="test-run",
            project_root=tmp_path,
            runs_dir=tmp_path / "runs",
            schema_path=tmp_path / "schema.json",
            system_prompt_path=tmp_path / "prompt.txt",
            design_doc_path=tmp_path / "DESIGN_DOCUMENT.md",
        )
        # These should work without raising
        state.total_calls = 5
        state.call_counts["codex"] = 3
        assert state.total_calls == 5
        assert state.call_counts["codex"] == 3


# -----------------------------------------------------------------------------
# AGENTS constant tests
# -----------------------------------------------------------------------------


class TestAgentsConstant:
    """Tests for AGENTS constant."""

    def test_contains_codex_and_claude(self) -> None:
        """AGENTS contains the expected agent names."""
        assert "codex" in AGENTS
        assert "claude" in AGENTS
        assert len(AGENTS) == 2


# -----------------------------------------------------------------------------
# AgentPolicy tests
# -----------------------------------------------------------------------------


class TestAgentPolicy:
    """Tests for AgentPolicy class."""

    def test_default_policy_no_forced_agent(self) -> None:
        """Default policy does not force any agent."""
        policy = AgentPolicy()
        assert policy.forced_agent is None
        assert policy.is_forced_mode() is False

    def test_enforce_with_no_forced_agent(self) -> None:
        """enforce() returns requested agent when no forced agent."""
        policy = AgentPolicy()
        assert policy.enforce("codex") == "codex"
        assert policy.enforce("claude") == "claude"

    def test_enforce_with_forced_agent(self) -> None:
        """enforce() returns forced agent regardless of request."""
        policy = AgentPolicy(forced_agent="claude")
        assert policy.enforce("codex") == "claude"
        assert policy.enforce("claude") == "claude"
        assert policy.is_forced_mode() is True

    def test_enforce_strict_raises_on_violation(self) -> None:
        """enforce_strict() raises AgentPolicyViolation when wrong agent requested."""
        policy = AgentPolicy(forced_agent="codex")
        with pytest.raises(AgentPolicyViolation) as exc_info:
            policy.enforce_strict("claude", context="test context")
        assert "claude" in str(exc_info.value)
        assert "codex" in str(exc_info.value)

    def test_enforce_strict_allows_correct_agent(self) -> None:
        """enforce_strict() allows the forced agent."""
        policy = AgentPolicy(forced_agent="codex")
        result = policy.enforce_strict("codex", context="test context")
        assert result == "codex"

    def test_enforce_with_unknown_agent(self) -> None:
        """enforce() handles unknown agents gracefully."""
        policy = AgentPolicy()
        result = policy.enforce("unknown_agent")
        assert result in AGENTS  # Should return first allowed

    def test_get_prompt_header_no_forced(self) -> None:
        """get_prompt_header() returns empty string when not forced."""
        policy = AgentPolicy()
        assert policy.get_prompt_header() == ""

    def test_get_prompt_header_forced(self) -> None:
        """get_prompt_header() returns override message when forced."""
        policy = AgentPolicy(forced_agent="claude")
        header = policy.get_prompt_header()
        assert "AGENT POLICY OVERRIDE" in header
        assert "claude" in header
        assert "--only-claude" in header

    def test_write_violation_artifact(self, tmp_path: Path) -> None:
        """Policy writes violation artifact when runs_dir is set."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        policy = AgentPolicy(forced_agent="codex", runs_dir=runs_dir)

        try:
            policy.enforce_strict("claude", context="test violation")
        except AgentPolicyViolation:
            pass

        artifact_path = runs_dir / "agent_policy_violation.txt"
        assert artifact_path.exists()
        content = artifact_path.read_text()
        assert "AGENT POLICY VIOLATION" in content
        assert "codex" in content
        assert "claude" in content


# -----------------------------------------------------------------------------
# Global policy tests
# -----------------------------------------------------------------------------


class TestGlobalPolicy:
    """Tests for global policy functions."""

    def test_get_agent_policy_returns_default(self) -> None:
        """get_agent_policy() returns a default policy initially."""
        # Reset global state first
        set_agent_policy(AgentPolicy())
        policy = get_agent_policy()
        assert policy is not None
        assert policy.forced_agent is None

    def test_set_and_get_policy(self) -> None:
        """set_agent_policy() and get_agent_policy() work together."""
        custom_policy = AgentPolicy(forced_agent="claude")
        set_agent_policy(custom_policy)
        retrieved = get_agent_policy()
        assert retrieved.forced_agent == "claude"
        # Restore default
        set_agent_policy(AgentPolicy())
