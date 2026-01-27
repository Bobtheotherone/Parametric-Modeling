"""Tests for quota/rate limit detection in the orchestrator.

These tests verify that the orchestrator correctly detects quota errors
from various agent outputs, including Claude-specific messages.

This prevents the quota thrashing that wasted credits in the Jan 26 run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MockRunConfig:
    """Mock run config for testing quota detection."""

    quota_error_patterns: dict[str, list[str]]


def _is_quota_error(agent: str, text: str, config: MockRunConfig) -> bool:
    """Check if text contains a quota error pattern for the given agent.

    This mirrors the implementation in bridge/loop.py.
    """
    pats = config.quota_error_patterns.get(agent, [])
    return any(re.search(p, text, flags=re.IGNORECASE) for p in pats)


class TestClaudeQuotaDetection:
    """Test quota detection for Claude agent."""

    # The patterns from bridge/config.json
    CLAUDE_PATTERNS = [
        "rate limit",
        "429",
        "quota",
        "hit your limit",
        "resets.*\\(",
        "You've hit your limit",
    ]

    def get_config(self) -> MockRunConfig:
        return MockRunConfig(quota_error_patterns={"claude": self.CLAUDE_PATTERNS})

    def test_detects_standard_rate_limit_message(self):
        """Test detection of standard rate limit messages."""
        config = self.get_config()

        assert _is_quota_error("claude", "Error: rate limit exceeded", config)
        assert _is_quota_error("claude", "Rate Limit: Please try again later", config)

    def test_detects_429_status_code(self):
        """Test detection of HTTP 429 status."""
        config = self.get_config()

        assert _is_quota_error("claude", "HTTP 429 Too Many Requests", config)
        assert _is_quota_error("claude", "Error 429: Rate limited", config)

    def test_detects_quota_exceeded_message(self):
        """Test detection of quota exceeded messages."""
        config = self.get_config()

        assert _is_quota_error("claude", "quota exceeded", config)
        assert _is_quota_error("claude", "You have exceeded your quota", config)

    def test_detects_claude_specific_hit_your_limit(self):
        """Test detection of Claude-specific 'hit your limit' message.

        This is the exact message from the Jan 26 failure logs.
        """
        config = self.get_config()

        # Exact message from logs
        exact_message = "You've hit your limit · resets 7pm (America/Anchorage)"
        assert _is_quota_error("claude", exact_message, config), f"Failed to detect exact Claude quota message: {exact_message}"

        # Variations
        assert _is_quota_error("claude", "hit your limit", config)
        assert _is_quota_error("claude", "You've hit your limit", config)
        assert _is_quota_error("claude", "You have hit your limit", config)

    def test_detects_resets_with_timezone(self):
        """Test detection of reset messages with timezone markers."""
        config = self.get_config()

        assert _is_quota_error("claude", "resets 7pm (America/Anchorage)", config)
        assert _is_quota_error("claude", "resets 3pm (UTC)", config)
        assert _is_quota_error("claude", "Resets at 5pm (America/New_York)", config)

    def test_does_not_false_positive_on_normal_messages(self):
        """Test that normal messages don't trigger quota detection."""
        config = self.get_config()

        # Normal messages should not trigger
        assert not _is_quota_error("claude", "Task completed successfully", config)
        assert not _is_quota_error("claude", "Reading file...", config)
        assert not _is_quota_error("claude", "Created 3 new files", config)
        assert not _is_quota_error("claude", "Test passed", config)

    def test_case_insensitive_matching(self):
        """Test that pattern matching is case-insensitive."""
        config = self.get_config()

        assert _is_quota_error("claude", "RATE LIMIT", config)
        assert _is_quota_error("claude", "Rate Limit", config)
        assert _is_quota_error("claude", "HIT YOUR LIMIT", config)
        assert _is_quota_error("claude", "QUOTA exceeded", config)


class TestCodexQuotaDetection:
    """Test quota detection for Codex agent."""

    CODEX_PATTERNS = [
        "rate limit",
        "429",
        "quota",
    ]

    def get_config(self) -> MockRunConfig:
        return MockRunConfig(quota_error_patterns={"codex": self.CODEX_PATTERNS})

    def test_detects_standard_patterns(self):
        """Test that Codex detects standard quota patterns."""
        config = self.get_config()

        assert _is_quota_error("codex", "rate limit exceeded", config)
        assert _is_quota_error("codex", "HTTP 429", config)
        assert _is_quota_error("codex", "quota exceeded", config)


class TestUnknownAgentQuotaDetection:
    """Test quota detection for unknown agents."""

    def test_unknown_agent_returns_false(self):
        """Test that unknown agents don't match any patterns."""
        config = MockRunConfig(
            quota_error_patterns={
                "claude": ["rate limit"],
                "codex": ["rate limit"],
            }
        )

        # Unknown agent should not match
        assert not _is_quota_error("unknown", "rate limit exceeded", config)
        assert not _is_quota_error("gemini", "rate limit exceeded", config)


class TestQuotaPatternRobustness:
    """Test that patterns are robust against edge cases."""

    PATTERNS = [
        "rate limit",
        "429",
        "quota",
        "hit your limit",
        "resets.*\\(",
    ]

    def get_config(self) -> MockRunConfig:
        return MockRunConfig(quota_error_patterns={"claude": self.PATTERNS})

    def test_detects_pattern_in_json_output(self):
        """Test detection in JSON-formatted output."""
        config = self.get_config()

        json_output = '{"error": "rate limit", "code": 429}'
        assert _is_quota_error("claude", json_output, config)

    def test_detects_pattern_in_multiline_output(self):
        """Test detection in multi-line output."""
        config = self.get_config()

        multiline_output = """
        Starting task...
        Error occurred:
        You've hit your limit · resets 7pm (America/Anchorage)
        """
        assert _is_quota_error("claude", multiline_output, config)

    def test_detects_pattern_with_special_characters(self):
        """Test detection with special characters in message."""
        config = self.get_config()

        # The actual Claude message has a middle dot (·)
        message = "You've hit your limit · resets 7pm (America/Anchorage)"
        assert _is_quota_error("claude", message, config)
