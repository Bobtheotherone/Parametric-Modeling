# SPDX-License-Identifier: MIT
"""Additional unit tests for bridge/smoke_route.py _normalize_route internal function.

Supplements test_smoke_route_unit.py with coverage for the internal
_normalize_route helper and additional edge cases not covered:
- _normalize_route: Normalize various sequence types to tuple
- Additional edge cases for input validation
- Type coercion behavior
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import module to access internal function
from bridge import smoke_route

# Access internal function
_normalize_route = smoke_route._normalize_route

# Also import public functions for integration tests
from bridge.smoke_route import (
    next_agent_for_route,
    resolve_smoke_route,
    smoke_route_override_reason,
)

# -----------------------------------------------------------------------------
# _normalize_route Tests
# -----------------------------------------------------------------------------


class TestNormalizeRoute:
    """Tests for _normalize_route internal helper function."""

    def test_list_normalized_to_tuple(self) -> None:
        """List input is normalized to tuple."""
        result = _normalize_route(["codex", "claude"])
        assert isinstance(result, tuple)
        assert result == ("codex", "claude")

    def test_tuple_unchanged(self) -> None:
        """Tuple input remains a tuple."""
        result = _normalize_route(("codex", "claude"))
        assert isinstance(result, tuple)
        assert result == ("codex", "claude")

    def test_single_element_list(self) -> None:
        """Single-element list is normalized."""
        result = _normalize_route(["codex"])
        assert result == ("codex",)

    def test_empty_list_raises_valueerror(self) -> None:
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="at least one agent"):
            _normalize_route([])

    def test_empty_tuple_raises_valueerror(self) -> None:
        """Empty tuple raises ValueError."""
        with pytest.raises(ValueError, match="at least one agent"):
            _normalize_route(())

    def test_preserves_element_order(self) -> None:
        """Element order is preserved during normalization."""
        result = _normalize_route(["a", "b", "c", "d", "e"])
        assert result == ("a", "b", "c", "d", "e")

    def test_preserves_duplicate_elements(self) -> None:
        """Duplicate elements are preserved."""
        result = _normalize_route(["codex", "codex", "codex"])
        assert result == ("codex", "codex", "codex")

    def test_long_route_normalized(self) -> None:
        """Long route is handled correctly."""
        long_route = ["agent"] * 100
        result = _normalize_route(long_route)
        assert len(result) == 100
        assert all(a == "agent" for a in result)

    def test_unicode_agent_names(self) -> None:
        """Unicode agent names are preserved."""
        result = _normalize_route(["代理1", "代理2"])
        assert result == ("代理1", "代理2")

    def test_mixed_case_agent_names(self) -> None:
        """Mixed case agent names are preserved (no case normalization)."""
        result = _normalize_route(["Codex", "CLAUDE", "agent"])
        assert result == ("Codex", "CLAUDE", "agent")


# -----------------------------------------------------------------------------
# Additional Edge Cases for Public Functions
# -----------------------------------------------------------------------------


class TestNextAgentForRouteEdgeCases:
    """Additional edge case tests for next_agent_for_route."""

    def test_large_index_raises_clearly(self) -> None:
        """Large out-of-bounds index gives clear error message."""
        route = ["codex"]
        with pytest.raises(IndexError) as exc_info:
            next_agent_for_route(route, 1000)
        assert "1000" in str(exc_info.value)
        assert "1" in str(exc_info.value)  # route length

    def test_zero_index_always_valid_for_non_empty(self) -> None:
        """Index 0 is always valid for non-empty routes."""
        for length in range(1, 10):
            route = [f"agent_{i}" for i in range(length)]
            agent = next_agent_for_route(route, 0)
            assert agent == "agent_0"

    def test_last_index_access(self) -> None:
        """Last valid index is accessible."""
        route = ["a", "b", "c", "d", "e"]
        agent = next_agent_for_route(route, 4)
        assert agent == "e"

    def test_whitespace_agent_name_preserved(self) -> None:
        """Agent names with whitespace are preserved."""
        route = [" codex ", "  claude  "]
        assert next_agent_for_route(route, 0) == " codex "
        assert next_agent_for_route(route, 1) == "  claude  "


class TestSmokeRouteOverrideReasonEdgeCases:
    """Additional edge case tests for smoke_route_override_reason."""

    def test_same_agent_different_cases_not_equal(self) -> None:
        """Agents that differ only in case are considered different."""
        result = smoke_route_override_reason(
            requested="Codex",
            routed="codex",
            index=0,
            route_len=1,
        )
        # Different cases means override occurred
        assert result is not None
        assert "Codex -> codex" in result

    def test_zero_index_formats_as_1_based(self) -> None:
        """Zero-based index is formatted as 1-based for humans."""
        result = smoke_route_override_reason(
            requested="a",
            routed="b",
            index=0,
            route_len=5,
        )
        assert result is not None
        assert "1/5" in result

    def test_last_index_formats_correctly(self) -> None:
        """Last index formats correctly as 1-based."""
        result = smoke_route_override_reason(
            requested="a",
            routed="b",
            index=4,
            route_len=5,
        )
        assert result is not None
        assert "5/5" in result

    def test_single_route_index_format(self) -> None:
        """Single-element route index formats as 1/1."""
        result = smoke_route_override_reason(
            requested="a",
            routed="b",
            index=0,
            route_len=1,
        )
        assert result is not None
        assert "1/1" in result

    def test_reason_contains_all_info(self) -> None:
        """Override reason contains all relevant information."""
        result = smoke_route_override_reason(
            requested="claude",
            routed="codex",
            index=2,
            route_len=10,
        )
        assert result is not None
        assert "smoke route override" in result
        assert "3/10" in result
        assert "claude" in result
        assert "codex" in result


class TestResolveSmokeRouteEdgeCases:
    """Additional edge case tests for resolve_smoke_route."""

    def test_resolve_returns_consistent_types(self) -> None:
        """resolve_smoke_route always returns (str, str | None)."""
        route = ["codex", "claude"]

        # No override case
        agent1, reason1 = resolve_smoke_route(requested="codex", route=route, index=0)
        assert isinstance(agent1, str)
        assert reason1 is None

        # Override case
        agent2, reason2 = resolve_smoke_route(requested="claude", route=route, index=0)
        assert isinstance(agent2, str)
        assert isinstance(reason2, str)

    def test_resolve_at_boundary_indices(self) -> None:
        """resolve_smoke_route works at boundary indices."""
        route = ["a", "b", "c"]

        # First index
        agent1, _ = resolve_smoke_route(requested="x", route=route, index=0)
        assert agent1 == "a"

        # Last index
        agent2, _ = resolve_smoke_route(requested="x", route=route, index=2)
        assert agent2 == "c"

    def test_resolve_with_identical_route_elements(self) -> None:
        """resolve_smoke_route works when all route elements are identical."""
        route = ["codex", "codex", "codex"]
        for i in range(3):
            agent, reason = resolve_smoke_route(requested="codex", route=route, index=i)
            assert agent == "codex"
            assert reason is None  # No override since request matches

    def test_resolve_with_alternating_route(self) -> None:
        """resolve_smoke_route works with alternating route pattern."""
        route = ["codex", "claude", "codex", "claude"]
        expected_agents = ["codex", "claude", "codex", "claude"]

        for i, expected in enumerate(expected_agents):
            agent, _ = resolve_smoke_route(requested="any", route=route, index=i)
            assert agent == expected


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestSmokeRouteIntegration:
    """Integration tests for smoke_route module functions working together."""

    def test_full_route_traversal(self) -> None:
        """Traverse entire route verifying each step."""
        route = ["codex", "claude", "codex"]

        for i in range(len(route)):
            agent = next_agent_for_route(route, i)
            resolved_agent, reason = resolve_smoke_route(
                requested="other",
                route=route,
                index=i,
            )
            assert agent == resolved_agent
            assert reason is not None  # "other" never matches

    def test_requested_matches_all_in_route(self) -> None:
        """When requested matches current route agent, no override."""
        route = ["codex", "claude", "codex"]

        for i, expected in enumerate(route):
            agent, reason = resolve_smoke_route(
                requested=expected,
                route=route,
                index=i,
            )
            assert agent == expected
            assert reason is None  # No override needed

    def test_round_trip_consistency(self) -> None:
        """Ensure _normalize_route + next_agent_for_route is consistent."""
        routes_to_test = [
            ["codex"],
            ["codex", "claude"],
            ["a", "b", "c", "d", "e"],
            ("tuple", "route"),
        ]

        for route in routes_to_test:
            normalized = _normalize_route(route)
            for i in range(len(normalized)):
                agent = next_agent_for_route(route, i)
                assert agent == normalized[i]
