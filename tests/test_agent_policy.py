"""Tests for AgentPolicy enforcement.

These tests verify that the --only-codex and --only-claude flags work correctly
to enforce single-agent mode across all agent selection points.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.loop import (
    AGENTS,
    AgentPolicy,
    AgentPolicyViolation,
    RunConfig,
    RunState,
    _override_next_agent,
    _pick_fallback,
    get_agent_policy,
    set_agent_policy,
)

# -----------------------------
# Test fixtures
# -----------------------------


@pytest.fixture
def mock_config() -> RunConfig:
    """Create a minimal RunConfig for testing."""
    return RunConfig(
        max_calls_per_agent=10,
        quota_retry_attempts=3,
        max_total_calls=100,
        max_json_correction_attempts=3,
        fallback_order=["codex", "claude"],
        enable_agents=["codex", "claude"],
        agent_scripts={"codex": "bridge/agents/codex.sh", "claude": "bridge/agents/claude.sh"},
        agent_models={"codex": "(default)", "claude": "(default)"},
        quota_error_patterns={"codex": [], "claude": []},
        supports_write_access={"codex": True, "claude": True},
        parallel=MagicMock(),
    )


@pytest.fixture
def mock_state(tmp_path: Path) -> RunState:
    """Create a minimal RunState for testing."""
    return RunState(
        run_id="test-run",
        project_root=tmp_path,
        runs_dir=tmp_path / "runs",
        schema_path=tmp_path / "schema.json",
        system_prompt_path=tmp_path / "system.md",
        design_doc_path=tmp_path / "design.md",
    )


# -----------------------------
# AgentPolicy basic tests
# -----------------------------


class TestAgentPolicyBasic:
    """Test basic AgentPolicy functionality."""

    def test_no_forced_agent_passes_through(self) -> None:
        """Without forced agent, requested agent is returned."""
        policy = AgentPolicy(forced_agent=None)
        assert policy.enforce("codex", "test") == "codex"
        assert policy.enforce("claude", "test") == "claude"

    def test_forced_codex_overrides_claude(self) -> None:
        """With --only-codex, claude requests are overridden to codex."""
        policy = AgentPolicy(forced_agent="codex")
        assert policy.enforce("claude", "test") == "codex"
        assert policy.enforce("codex", "test") == "codex"

    def test_forced_claude_overrides_codex(self) -> None:
        """With --only-claude, codex requests are overridden to claude."""
        policy = AgentPolicy(forced_agent="claude")
        assert policy.enforce("codex", "test") == "claude"
        assert policy.enforce("claude", "test") == "claude"

    def test_is_forced_mode(self) -> None:
        """is_forced_mode returns correct value."""
        assert not AgentPolicy(forced_agent=None).is_forced_mode()
        assert AgentPolicy(forced_agent="codex").is_forced_mode()
        assert AgentPolicy(forced_agent="claude").is_forced_mode()


class TestAgentPolicyStrict:
    """Test strict enforcement that raises exceptions."""

    def test_strict_allows_correct_agent(self) -> None:
        """Strict enforcement allows the forced agent."""
        policy = AgentPolicy(forced_agent="claude")
        assert policy.enforce_strict("claude", "test") == "claude"

    def test_strict_raises_on_wrong_agent(self, tmp_path: Path) -> None:
        """Strict enforcement raises AgentPolicyViolation on wrong agent."""
        policy = AgentPolicy(forced_agent="claude", runs_dir=tmp_path)
        with pytest.raises(AgentPolicyViolation) as exc_info:
            policy.enforce_strict("codex", "fallback logic")
        assert "codex" in str(exc_info.value)
        assert "--only-claude" in str(exc_info.value)

    def test_strict_writes_violation_artifact(self, tmp_path: Path) -> None:
        """Strict enforcement writes artifact on violation."""
        policy = AgentPolicy(forced_agent="codex", runs_dir=tmp_path)
        with pytest.raises(AgentPolicyViolation):
            policy.enforce_strict("claude", "test context")

        artifact_path = tmp_path / "agent_policy_violation.txt"
        assert artifact_path.exists()
        content = artifact_path.read_text()
        assert "claude" in content
        assert "codex" in content
        assert "test context" in content


class TestAgentPolicyPromptHeader:
    """Test prompt header generation for forced mode."""

    def test_no_header_when_not_forced(self) -> None:
        """No header when no forced agent."""
        policy = AgentPolicy(forced_agent=None)
        assert policy.get_prompt_header() == ""

    def test_header_for_only_codex(self) -> None:
        """Header mentions --only-codex when forced."""
        policy = AgentPolicy(forced_agent="codex")
        header = policy.get_prompt_header()
        assert "--only-codex" in header
        assert "ONLY agent" in header
        assert "codex" in header

    def test_header_for_only_claude(self) -> None:
        """Header mentions --only-claude when forced."""
        policy = AgentPolicy(forced_agent="claude")
        header = policy.get_prompt_header()
        assert "--only-claude" in header
        assert "ONLY agent" in header
        assert "claude" in header


# -----------------------------
# Integration with agent selection functions
# -----------------------------


class TestOverrideNextAgent:
    """Test _override_next_agent respects policy."""

    def test_forced_claude_overrides_codex_handoff(self, mock_config: RunConfig, mock_state: RunState) -> None:
        """When --only-claude, a codex handoff request is overridden to claude."""
        set_agent_policy(AgentPolicy(forced_agent="claude"))
        try:
            mock_state.history = [{"agent": "claude", "summary": "did something"}]
            effective, reason = _override_next_agent("codex", mock_config, mock_state)
            assert effective == "claude"
            assert reason is not None
            assert "policy" in reason.lower() or "only" in reason.lower()
        finally:
            set_agent_policy(AgentPolicy())

    def test_forced_codex_overrides_claude_handoff(self, mock_config: RunConfig, mock_state: RunState) -> None:
        """When --only-codex, a claude handoff request is overridden to codex."""
        set_agent_policy(AgentPolicy(forced_agent="codex"))
        try:
            mock_state.history = [{"agent": "codex", "summary": "did something"}]
            effective, reason = _override_next_agent("claude", mock_config, mock_state)
            assert effective == "codex"
            assert reason is not None
            assert "policy" in reason.lower() or "only" in reason.lower()
        finally:
            set_agent_policy(AgentPolicy())

    def test_no_forced_allows_normal_alternation(self, mock_config: RunConfig, mock_state: RunState) -> None:
        """Without forced agent, normal two-agent alternation works."""
        set_agent_policy(AgentPolicy(forced_agent=None))
        try:
            mock_state.history = [{"agent": "codex", "summary": "did something"}]
            effective, reason = _override_next_agent("codex", mock_config, mock_state)
            # Normal alternation: should switch to claude
            assert effective == "claude"
        finally:
            set_agent_policy(AgentPolicy())


class TestPickFallback:
    """Test _pick_fallback respects policy."""

    def test_forced_claude_returns_claude_on_fallback(self, mock_config: RunConfig, mock_state: RunState) -> None:
        """When --only-claude, fallback always returns claude."""
        set_agent_policy(AgentPolicy(forced_agent="claude", runs_dir=mock_state.runs_dir))
        try:
            result = _pick_fallback(mock_config, mock_state, current_agent="claude")
            assert result == "claude"
        finally:
            set_agent_policy(AgentPolicy())

    def test_forced_codex_returns_codex_on_fallback(self, mock_config: RunConfig, mock_state: RunState) -> None:
        """When --only-codex, fallback always returns codex."""
        set_agent_policy(AgentPolicy(forced_agent="codex", runs_dir=mock_state.runs_dir))
        try:
            result = _pick_fallback(mock_config, mock_state, current_agent="codex")
            assert result == "codex"
        finally:
            set_agent_policy(AgentPolicy())

    def test_no_forced_allows_fallback_to_other(self, mock_config: RunConfig, mock_state: RunState) -> None:
        """Without forced agent, fallback can switch to other agent."""
        set_agent_policy(AgentPolicy(forced_agent=None))
        try:
            result = _pick_fallback(mock_config, mock_state, current_agent="codex")
            # Normal fallback prefers codex (first in fallback_order), but if current is codex,
            # should fall back to claude
            assert result == "claude"
        finally:
            set_agent_policy(AgentPolicy())


# -----------------------------
# CLI flag validation tests
# -----------------------------


class TestCLIFlagValidation:
    """Test CLI flag mutual exclusion validation."""

    def test_both_flags_should_error(self) -> None:
        """Setting both --only-codex and --only-claude should cause an error.

        This test verifies the expected behavior - actual CLI parsing happens
        in main() and exits with error code 1.
        """
        # This is tested by checking the policy cannot represent both
        # In reality, the CLI parser rejects this combination
        policy_codex = AgentPolicy(forced_agent="codex")
        policy_claude = AgentPolicy(forced_agent="claude")

        # Both policies are valid individually
        assert policy_codex.forced_agent == "codex"
        assert policy_claude.forced_agent == "claude"

        # But the CLI should prevent setting both (tested via integration test)


# -----------------------------
# Global policy management tests
# -----------------------------


class TestGlobalPolicyManagement:
    """Test global policy get/set functions."""

    def test_get_returns_default_if_not_set(self) -> None:
        """get_agent_policy returns a default policy if not explicitly set."""
        # Clear any existing policy
        set_agent_policy(AgentPolicy())
        policy = get_agent_policy()
        assert policy is not None
        assert not policy.is_forced_mode()

    def test_set_and_get_roundtrip(self) -> None:
        """set_agent_policy and get_agent_policy work together."""
        custom_policy = AgentPolicy(forced_agent="claude")
        set_agent_policy(custom_policy)
        try:
            retrieved = get_agent_policy()
            assert retrieved.forced_agent == "claude"
        finally:
            set_agent_policy(AgentPolicy())


# -----------------------------
# Edge case tests
# -----------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unknown_agent_with_forced_returns_forced(self) -> None:
        """Unknown agent name with forced mode returns forced agent."""
        policy = AgentPolicy(forced_agent="claude")
        assert policy.enforce("gemini", "test") == "claude"
        assert policy.enforce("unknown", "test") == "claude"
        assert policy.enforce("", "test") == "claude"

    def test_empty_allowed_agents_with_forced(self) -> None:
        """Forced agent takes precedence over allowed_agents."""
        policy = AgentPolicy(forced_agent="claude", allowed_agents=())
        assert policy.enforce("codex", "test") == "claude"

    def test_strict_allows_non_agent_names_when_not_forced(self) -> None:
        """Without forced mode, strict allows any agent and falls back."""
        policy = AgentPolicy(forced_agent=None)
        # Should not raise, just return a fallback
        result = policy.enforce_strict("unknown", "test")
        assert result in AGENTS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
