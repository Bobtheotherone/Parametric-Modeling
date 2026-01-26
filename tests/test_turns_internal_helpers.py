# SPDX-License-Identifier: MIT
"""Unit tests for bridge/turns.py internal helper functions.

Supplements test_bridge_turns_unit.py with coverage for internal helpers:
- _to_str_list: Convert various inputs to string lists
- _join_summary: Join summary and error detail strings

These are internal functions but are critical for the correctness of
build_error_turn and should be tested directly for edge cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import internal functions (using module-level access)
from bridge import turns

# Access internal functions via module
_to_str_list = turns._to_str_list
_join_summary = turns._join_summary


# -----------------------------------------------------------------------------
# _to_str_list Tests
# -----------------------------------------------------------------------------


class TestToStrList:
    """Tests for _to_str_list internal helper function."""

    def test_none_returns_empty_list(self) -> None:
        """None input returns empty list."""
        assert _to_str_list(None) == []

    def test_empty_list_returns_empty_list(self) -> None:
        """Empty list returns empty list."""
        assert _to_str_list([]) == []

    def test_list_of_strings_preserved(self) -> None:
        """List of strings is preserved."""
        result = _to_str_list(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_string_input_returns_empty_list(self) -> None:
        """String input (not a sequence of strings) returns empty list."""
        # Strings are sequences but should be treated specially
        result = _to_str_list("abc")
        assert result == []

    def test_bytes_input_returns_empty_list(self) -> None:
        """Bytes input returns empty list."""
        result = _to_str_list(b"abc")
        assert result == []

    def test_int_input_returns_empty_list(self) -> None:
        """Integer input returns empty list."""
        result = _to_str_list(123)
        assert result == []

    def test_dict_input_returns_empty_list(self) -> None:
        """Dict input returns empty list."""
        result = _to_str_list({"key": "value"})
        assert result == []

    def test_mixed_list_filters_non_strings(self) -> None:
        """List with mixed types filters out non-strings."""
        result = _to_str_list(["valid", 123, None, "also_valid", True])
        assert result == ["valid", "also_valid"]

    def test_whitespace_strings_filtered(self) -> None:
        """Whitespace-only strings are filtered out."""
        result = _to_str_list(["valid", "  ", "", "also_valid", "\t\n"])
        assert result == ["valid", "also_valid"]

    def test_strings_are_stripped(self) -> None:
        """Strings are stripped of leading/trailing whitespace."""
        result = _to_str_list(["  hello  ", "\tworld\t"])
        assert result == ["hello", "world"]

    def test_tuple_of_strings_works(self) -> None:
        """Tuple of strings works like list."""
        result = _to_str_list(("a", "b", "c"))
        assert result == ["a", "b", "c"]

    def test_empty_string_filtered(self) -> None:
        """Empty string is filtered out."""
        result = _to_str_list(["", "valid", ""])
        assert result == ["valid"]

    def test_unicode_strings_preserved(self) -> None:
        """Unicode strings are preserved."""
        result = _to_str_list(["hello", "世界", "🚀"])
        assert result == ["hello", "世界", "🚀"]

    def test_nested_list_not_flattened(self) -> None:
        """Nested lists are filtered (not flattened)."""
        result = _to_str_list(["valid", ["nested"], "also_valid"])
        # Nested list is not a string, so it's filtered
        assert result == ["valid", "also_valid"]


# -----------------------------------------------------------------------------
# _join_summary Tests
# -----------------------------------------------------------------------------


class TestJoinSummary:
    """Tests for _join_summary internal helper function."""

    def test_summary_only(self) -> None:
        """Summary alone is returned as-is."""
        result = _join_summary("Main summary", None)
        assert result == "Main summary"

    def test_summary_with_error_detail(self) -> None:
        """Summary and error detail are joined with newline."""
        result = _join_summary("Main summary", "Error occurred")
        assert result == "Main summary\nError occurred"

    def test_empty_summary_with_error_detail(self) -> None:
        """Empty summary with error detail returns just error detail."""
        result = _join_summary("", "Error only")
        assert result == "Error only"

    def test_none_summary_with_error_detail(self) -> None:
        """None summary with error detail returns just error detail."""
        result = _join_summary(None, "Error only")
        assert result == "Error only"

    def test_whitespace_summary_with_error_detail(self) -> None:
        """Whitespace-only summary with error detail returns just error detail."""
        result = _join_summary("   ", "Error only")
        assert result == "Error only"

    def test_summary_with_empty_error_detail(self) -> None:
        """Summary with empty error detail returns just summary."""
        result = _join_summary("Main summary", "")
        assert result == "Main summary"

    def test_summary_with_none_error_detail(self) -> None:
        """Summary with None error detail returns just summary."""
        result = _join_summary("Main summary", None)
        assert result == "Main summary"

    def test_summary_with_whitespace_error_detail(self) -> None:
        """Summary with whitespace-only error detail returns just summary."""
        result = _join_summary("Main summary", "   ")
        assert result == "Main summary"

    def test_both_empty(self) -> None:
        """Both empty returns empty string."""
        result = _join_summary("", "")
        assert result == ""

    def test_both_none(self) -> None:
        """Both None returns empty string."""
        result = _join_summary(None, None)
        assert result == ""

    def test_summary_stripped(self) -> None:
        """Summary is stripped of whitespace."""
        result = _join_summary("  summary  ", None)
        assert result == "summary"

    def test_error_detail_stripped(self) -> None:
        """Error detail is stripped of whitespace."""
        result = _join_summary("summary", "  error  ")
        assert result == "summary\nerror"

    def test_multiline_summary(self) -> None:
        """Multiline summary is preserved."""
        result = _join_summary("Line 1\nLine 2", "Error")
        assert result == "Line 1\nLine 2\nError"

    def test_multiline_error_detail(self) -> None:
        """Multiline error detail is preserved."""
        result = _join_summary("Summary", "Error 1\nError 2")
        assert result == "Summary\nError 1\nError 2"

    def test_unicode_content(self) -> None:
        """Unicode content is preserved."""
        result = _join_summary("Summary 世界", "Error 🚀")
        assert result == "Summary 世界\nError 🚀"


# -----------------------------------------------------------------------------
# Integration tests with build_error_turn
# -----------------------------------------------------------------------------


class TestInternalHelpersIntegration:
    """Integration tests verifying internal helpers work correctly with build_error_turn."""

    def test_build_error_turn_uses_to_str_list_for_gates(self) -> None:
        """build_error_turn correctly filters gates_passed through _to_str_list."""
        from bridge.turns import build_error_turn

        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
            gates_passed=["valid", 123, None, "also_valid"],
        )
        # Non-strings should be filtered
        assert turn["gates_passed"] == ["valid", "also_valid"]

    def test_build_error_turn_uses_join_summary_for_error(self) -> None:
        """build_error_turn correctly joins summary and error_detail."""
        from bridge.turns import build_error_turn

        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Main issue",
            error_detail="Stack trace here",
        )
        assert "Main issue" in turn["summary"]
        assert "Stack trace here" in turn["summary"]

    def test_build_error_turn_handles_none_gates_passed(self) -> None:
        """build_error_turn handles None gates_passed."""
        from bridge.turns import build_error_turn

        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
            gates_passed=None,
        )
        assert turn["gates_passed"] == []

    def test_build_error_turn_strips_whitespace_in_gates(self) -> None:
        """build_error_turn strips whitespace from gates_passed entries."""
        from bridge.turns import build_error_turn

        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
            gates_passed=["  ruff  ", "\tmypy\t"],
        )
        assert turn["gates_passed"] == ["ruff", "mypy"]
