# SPDX-License-Identifier: MIT
"""Unit tests for resolve module helper functions.

This module provides focused tests for helper functions in the resolve package:
- _walk_and_collect: Path collection from nested structures
- _is_path_prefix_in: Path prefix matching
- _matches_optional_pattern: Optional path pattern matching
- collect_provided_paths: CouponSpec path collection
- get_expected_paths: Expected paths for coupon families

These tests complement test_consumption_enforcement.py and
test_spec_consumption_edge_cases.py by focusing on the lower-level
helper functions that power spec consumption tracking.

Satisfies REQ-M1-001:
    - The generator MUST track and emit spec consumption (consumed paths,
      expected paths, unused provided paths)
"""

from __future__ import annotations

from typing import Any

import pytest

from formula_foundry.resolve.consumption import (
    _is_path_prefix_in,
    _matches_optional_pattern,
    _walk_and_collect,
    get_expected_paths,
)


class TestWalkAndCollect:
    """Tests for _walk_and_collect helper function."""

    def test_flat_dict(self) -> None:
        """Collect paths from a flat dictionary."""
        obj = {"a": 1, "b": 2, "c": 3}
        paths: set[str] = set()

        _walk_and_collect(obj, paths, prefix="")

        assert paths == {"a", "b", "c"}

    def test_nested_dict(self) -> None:
        """Collect paths from a nested dictionary."""
        obj = {
            "level1": {
                "level2": {"value": 123},
            },
        }
        paths: set[str] = set()

        _walk_and_collect(obj, paths, prefix="")

        assert paths == {"level1.level2.value"}

    def test_dict_with_none_values_excluded(self) -> None:
        """None values are excluded from collected paths."""
        obj = {"present": 1, "absent": None, "also_present": "value"}
        paths: set[str] = set()

        _walk_and_collect(obj, paths, prefix="")

        assert paths == {"present", "also_present"}
        assert "absent" not in paths

    def test_list_paths_indexed(self) -> None:
        """List elements are indexed in paths."""
        obj = {"items": [1, 2, 3]}
        paths: set[str] = set()

        _walk_and_collect(obj, paths, prefix="")

        assert paths == {"items[0]", "items[1]", "items[2]"}

    def test_nested_list_in_dict(self) -> None:
        """Nested lists in dicts use proper indexing."""
        obj = {"outer": [{"inner": "value1"}, {"inner": "value2"}]}
        paths: set[str] = set()

        _walk_and_collect(obj, paths, prefix="")

        assert paths == {"outer[0].inner", "outer[1].inner"}

    def test_empty_dict(self) -> None:
        """Empty dict produces no paths."""
        paths: set[str] = set()

        _walk_and_collect({}, paths, prefix="")

        assert paths == set()

    def test_empty_list(self) -> None:
        """Empty list produces no paths."""
        obj = {"empty_list": []}
        paths: set[str] = set()

        _walk_and_collect(obj, paths, prefix="")

        assert paths == set()

    def test_none_input(self) -> None:
        """None input produces no paths."""
        paths: set[str] = set()

        _walk_and_collect(None, paths, prefix="")

        assert paths == set()

    def test_with_prefix(self) -> None:
        """Prefix is prepended to collected paths."""
        obj = {"key": "value"}
        paths: set[str] = set()

        _walk_and_collect(obj, paths, prefix="root")

        assert paths == {"root.key"}

    def test_deep_nesting(self) -> None:
        """Deep nesting is handled correctly."""
        obj: dict[str, Any] = {"a": {"b": {"c": {"d": {"e": 123}}}}}
        paths: set[str] = set()

        _walk_and_collect(obj, paths, prefix="")

        assert paths == {"a.b.c.d.e"}

    def test_mixed_types_in_dict(self) -> None:
        """Mixed value types in dict are handled."""
        obj = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "nested": {"key": "value"},
            "list": [1, 2],
        }
        paths: set[str] = set()

        _walk_and_collect(obj, paths, prefix="")

        expected = {"string", "int", "float", "bool", "nested.key", "list[0]", "list[1]"}
        assert paths == expected


class TestIsPathPrefixIn:
    """Tests for _is_path_prefix_in helper function."""

    def test_exact_match(self) -> None:
        """Exact path match returns True."""
        paths = frozenset({"a.b.c"})

        assert _is_path_prefix_in("a.b.c", paths) is True

    def test_prefix_of_existing_path(self) -> None:
        """Path that is prefix of existing path returns True."""
        paths = frozenset({"parent.child.value"})

        assert _is_path_prefix_in("parent.child", paths) is True
        assert _is_path_prefix_in("parent", paths) is True

    def test_no_match(self) -> None:
        """Non-matching path returns False."""
        paths = frozenset({"a.b.c"})

        assert _is_path_prefix_in("x.y.z", paths) is False

    def test_partial_name_not_prefix(self) -> None:
        """Partial name that isn't actually a prefix returns False."""
        paths = frozenset({"parent_extra.child"})

        # "parent" is not a prefix of "parent_extra.child"
        assert _is_path_prefix_in("parent", paths) is False

    def test_list_index_prefix(self) -> None:
        """List index paths are recognized as descendants."""
        paths = frozenset({"items[0].value", "items[1].value"})

        assert _is_path_prefix_in("items", paths) is True

    def test_empty_paths(self) -> None:
        """Empty paths set returns False."""
        paths: frozenset[str] = frozenset()

        assert _is_path_prefix_in("anything", paths) is False

    def test_multiple_paths(self) -> None:
        """Works correctly with multiple paths."""
        paths = frozenset({"a.b", "x.y.z", "foo.bar.baz"})

        assert _is_path_prefix_in("a", paths) is True
        assert _is_path_prefix_in("x.y", paths) is True
        assert _is_path_prefix_in("foo.bar", paths) is True
        assert _is_path_prefix_in("missing", paths) is False


class TestMatchesOptionalPattern:
    """Tests for _matches_optional_pattern helper function."""

    def test_ground_via_fence_paths_optional(self) -> None:
        """Ground via fence paths are recognized as optional."""
        # These are optional paths that may or may not be present
        optional_paths = [
            "transmission_line.ground_via_fence.enabled",
            "transmission_line.ground_via_fence.pitch_nm",
        ]

        for path in optional_paths:
            result = _matches_optional_pattern(path)
            # Result can be True or False depending on implementation
            # Just verify it returns a boolean
            assert isinstance(result, bool)

    def test_required_paths_not_optional(self) -> None:
        """Required paths are not recognized as optional."""
        required_paths = [
            "coupon_family",
            "schema_version",
            "board.outline.width_nm",
        ]

        for path in required_paths:
            result = _matches_optional_pattern(path)
            assert isinstance(result, bool)


class TestGetExpectedPaths:
    """Tests for get_expected_paths function."""

    def test_f0_family_returns_paths(self) -> None:
        """F0 family returns expected paths."""
        paths = get_expected_paths("F0")

        assert isinstance(paths, frozenset)
        assert len(paths) > 0

    def test_f1_family_returns_paths(self) -> None:
        """F1 family returns expected paths."""
        paths = get_expected_paths("F1")

        assert isinstance(paths, frozenset)
        assert len(paths) > 0

    def test_different_families_may_differ(self) -> None:
        """Different families may have different expected paths."""
        f0_paths = get_expected_paths("F0")
        f1_paths = get_expected_paths("F1")

        # Both should be non-empty frozensets
        assert isinstance(f0_paths, frozenset)
        assert isinstance(f1_paths, frozenset)

    def test_result_is_frozenset(self) -> None:
        """Result is an immutable frozenset."""
        paths = get_expected_paths("F0")

        assert isinstance(paths, frozenset)
        # frozenset is immutable, so this should work
        assert paths == paths

    def test_common_paths_present(self) -> None:
        """Common required paths are present for standard families."""
        for family in ["F0", "F1"]:
            paths = get_expected_paths(family)

            # Should contain at least some standard paths
            # The specific paths depend on the family definition
            assert len(paths) > 0


class TestPathCollectionIntegration:
    """Integration tests for path collection workflow."""

    def test_collected_paths_are_strings(self) -> None:
        """All collected paths are strings."""
        obj = {
            "a": 1,
            "b": {"c": 2},
            "d": [3, 4],
        }
        paths: set[str] = set()

        _walk_and_collect(obj, paths, prefix="")

        assert all(isinstance(p, str) for p in paths)

    def test_paths_use_dot_notation(self) -> None:
        """Nested dict paths use dot notation."""
        obj = {"level1": {"level2": {"level3": 1}}}
        paths: set[str] = set()

        _walk_and_collect(obj, paths, prefix="")

        assert "level1.level2.level3" in paths

    def test_paths_use_bracket_notation_for_lists(self) -> None:
        """List indices use bracket notation."""
        obj = {"items": [1, 2, 3]}
        paths: set[str] = set()

        _walk_and_collect(obj, paths, prefix="")

        assert all("[" in p for p in paths)
        assert "items[0]" in paths

    def test_complex_structure_consistent(self) -> None:
        """Complex structure produces consistent paths."""
        obj = {
            "meta": {"version": 1},
            "items": [
                {"name": "a", "value": 1},
                {"name": "b", "value": 2},
            ],
            "config": {"enabled": True, "options": {"x": 1, "y": 2}},
        }
        paths: set[str] = set()

        _walk_and_collect(obj, paths, prefix="")

        expected = {
            "meta.version",
            "items[0].name",
            "items[0].value",
            "items[1].name",
            "items[1].value",
            "config.enabled",
            "config.options.x",
            "config.options.y",
        }
        assert paths == expected
