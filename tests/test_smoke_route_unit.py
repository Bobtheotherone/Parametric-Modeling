# SPDX-License-Identifier: MIT
"""Unit tests for bridge/smoke_route.py.

Tests the pure functions in the smoke_route module which handle
routing agent calls through a predefined route sequence.

Key functions tested:
- next_agent_for_route: Get agent at a route index
- smoke_route_override_reason: Generate override reason strings
- resolve_smoke_route: Combine routing with reason generation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.smoke_route import (
    next_agent_for_route,
    resolve_smoke_route,
    smoke_route_override_reason,
)

# -----------------------------------------------------------------------------
# next_agent_for_route tests
# -----------------------------------------------------------------------------


class TestNextAgentForRoute:
    """Tests for next_agent_for_route function."""

    def test_single_agent_route_index_0(self) -> None:
        """Single-agent route returns agent at index 0."""
        route = ["codex"]
        assert next_agent_for_route(route, 0) == "codex"

    def test_two_agent_route_indices(self) -> None:
        """Two-agent route returns correct agent at each index."""
        route = ["codex", "claude"]
        assert next_agent_for_route(route, 0) == "codex"
        assert next_agent_for_route(route, 1) == "claude"

    def test_three_agent_route(self) -> None:
        """Three-agent route returns correct agents."""
        route = ["codex", "claude", "codex"]
        assert next_agent_for_route(route, 0) == "codex"
        assert next_agent_for_route(route, 1) == "claude"
        assert next_agent_for_route(route, 2) == "codex"

    def test_empty_route_raises_valueerror(self) -> None:
        """Empty route raises ValueError."""
        with pytest.raises(ValueError, match="at least one agent"):
            next_agent_for_route([], 0)

    def test_negative_index_raises_indexerror(self) -> None:
        """Negative index raises IndexError."""
        route = ["codex", "claude"]
        with pytest.raises(IndexError, match="out of range"):
            next_agent_for_route(route, -1)

    def test_index_too_large_raises_indexerror(self) -> None:
        """Index >= route length raises IndexError."""
        route = ["codex", "claude"]
        with pytest.raises(IndexError, match="out of range"):
            next_agent_for_route(route, 2)

    def test_accepts_tuple(self) -> None:
        """Function accepts tuple as route."""
        route = ("codex", "claude")
        assert next_agent_for_route(route, 0) == "codex"
        assert next_agent_for_route(route, 1) == "claude"

    def test_repeated_agent_in_route(self) -> None:
        """Route can have same agent repeated."""
        route = ["codex", "codex", "codex"]
        assert next_agent_for_route(route, 0) == "codex"
        assert next_agent_for_route(route, 1) == "codex"
        assert next_agent_for_route(route, 2) == "codex"


# -----------------------------------------------------------------------------
# smoke_route_override_reason tests
# -----------------------------------------------------------------------------


class TestSmokeRouteOverrideReason:
    """Tests for smoke_route_override_reason function."""

    def test_no_override_returns_none(self) -> None:
        """When requested equals routed, returns None."""
        result = smoke_route_override_reason(
            requested="codex",
            routed="codex",
            index=0,
            route_len=2,
        )
        assert result is None

    def test_override_returns_reason_string(self) -> None:
        """When overridden, returns explanation string."""
        result = smoke_route_override_reason(
            requested="claude",
            routed="codex",
            index=0,
            route_len=2,
        )
        assert result is not None
        assert "smoke route override" in result
        assert "claude" in result
        assert "codex" in result

    def test_reason_includes_index_info(self) -> None:
        """Override reason includes 1-based index info."""
        result = smoke_route_override_reason(
            requested="claude",
            routed="codex",
            index=0,
            route_len=3,
        )
        assert result is not None
        assert "1/3" in result  # 1-based index

    def test_reason_format_second_index(self) -> None:
        """Override reason for second index."""
        result = smoke_route_override_reason(
            requested="codex",
            routed="claude",
            index=1,
            route_len=3,
        )
        assert result is not None
        assert "2/3" in result

    def test_reason_shows_direction(self) -> None:
        """Override reason shows 'requested -> routed' direction."""
        result = smoke_route_override_reason(
            requested="claude",
            routed="codex",
            index=0,
            route_len=2,
        )
        assert result is not None
        assert "claude -> codex" in result


# -----------------------------------------------------------------------------
# resolve_smoke_route tests
# -----------------------------------------------------------------------------


class TestResolveSmokeRoute:
    """Tests for resolve_smoke_route function."""

    def test_returns_routed_agent_and_reason_tuple(self) -> None:
        """Returns tuple of (agent, reason)."""
        route = ["codex", "claude"]
        result = resolve_smoke_route(requested="claude", route=route, index=0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_no_override_reason_is_none(self) -> None:
        """When route matches request, reason is None."""
        route = ["codex", "claude"]
        agent, reason = resolve_smoke_route(requested="codex", route=route, index=0)
        assert agent == "codex"
        assert reason is None

    def test_override_has_reason(self) -> None:
        """When route overrides request, reason is provided."""
        route = ["codex", "claude"]
        agent, reason = resolve_smoke_route(requested="claude", route=route, index=0)
        assert agent == "codex"
        assert reason is not None
        assert "override" in reason

    def test_second_index_routing(self) -> None:
        """Routing at second index works correctly."""
        route = ["codex", "claude"]
        agent, reason = resolve_smoke_route(requested="codex", route=route, index=1)
        assert agent == "claude"
        assert reason is not None
        assert "2/2" in reason

    def test_empty_route_raises(self) -> None:
        """Empty route raises ValueError."""
        with pytest.raises(ValueError, match="at least one agent"):
            resolve_smoke_route(requested="codex", route=[], index=0)

    def test_index_out_of_bounds_raises(self) -> None:
        """Index out of bounds raises IndexError."""
        route = ["codex"]
        with pytest.raises(IndexError, match="out of range"):
            resolve_smoke_route(requested="codex", route=route, index=1)


# -----------------------------------------------------------------------------
# Edge case and integration tests
# -----------------------------------------------------------------------------


class TestSmokeRouteEdgeCases:
    """Edge case and integration tests for smoke_route module."""

    def test_long_route_all_indices(self) -> None:
        """Long route returns correct agents at all indices."""
        route = ["codex", "claude", "codex", "claude", "codex"]
        expected = ["codex", "claude", "codex", "claude", "codex"]
        for i, expected_agent in enumerate(expected):
            agent, _ = resolve_smoke_route(requested="any", route=route, index=i)
            assert agent == expected_agent

    def test_single_element_route(self) -> None:
        """Single-element route works at index 0."""
        route = ["claude"]
        agent, reason = resolve_smoke_route(requested="codex", route=route, index=0)
        assert agent == "claude"
        assert reason is not None
        assert "1/1" in reason

    def test_route_with_custom_agent_names(self) -> None:
        """Route can contain arbitrary agent names."""
        route = ["agent_a", "agent_b", "agent_c"]
        assert next_agent_for_route(route, 0) == "agent_a"
        assert next_agent_for_route(route, 1) == "agent_b"
        assert next_agent_for_route(route, 2) == "agent_c"

    def test_generator_input_normalized(self) -> None:
        """Generator input is normalized to tuple."""

        def gen():
            yield "codex"
            yield "claude"

        # This should work because _normalize_route uses tuple()
        agent = next_agent_for_route(list(gen()), 0)
        assert agent == "codex"

    def test_reason_string_is_human_readable(self) -> None:
        """Override reason is clear and human-readable."""
        result = smoke_route_override_reason(
            requested="claude",
            routed="codex",
            index=2,
            route_len=5,
        )
        assert result is not None
        # Should mention override, have index info, and show the change
        assert "smoke route" in result.lower()
        assert "3/5" in result
        assert "->" in result
