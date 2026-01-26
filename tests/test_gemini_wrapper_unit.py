# SPDX-License-Identifier: MIT
"""Unit tests for bridge/gemini.py - Gemini wrapper helper functions.

Tests cover:
- _extract_enum: Extracting enum values from JSON schema
- _select_from_allowed: Selecting agent from allowed list
- _extract_milestone: Extracting milestone ID from prompt text
- _extract_stats_refs: Extracting stats refs from prompt text
- _load_stats_ids: Loading stats IDs from STATS.md
- _env_truthy: Parsing truthy environment variable values
- _extract_json_payload: Extracting JSON payloads from text
- _candidate_json_texts: Generating candidate JSON texts
- _strip_code_fence: Stripping markdown code fences
- _try_parse_json: Parsing JSON with fallback handling
- _extract_balanced_json: Extracting balanced JSON from text
- _unwrap_payload: Unwrapping nested payloads
- _validate_turn_minimal: Minimal turn validation without jsonschema
- _coerce_enum: Coercing values to valid enum entries
- _truncate: Truncating text to a limit
- _format_process_error: Formatting process errors
- _format_parse_error: Formatting parse errors
- _format_validation_error: Formatting validation errors

Run with: pytest tests/test_gemini_wrapper_unit.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.gemini import (
    REQUIRED_KEYS,
    _candidate_json_texts,
    _coerce_enum,
    _env_truthy,
    _extract_balanced_json,
    _extract_enum,
    _extract_json_payload,
    _extract_milestone,
    _extract_stats_refs,
    _format_parse_error,
    _format_process_error,
    _format_validation_error,
    _load_stats_ids,
    _select_from_allowed,
    _strip_code_fence,
    _truncate,
    _try_parse_json,
    _unwrap_payload,
    _validate_turn_minimal,
)

# -----------------------------------------------------------------------------
# _extract_enum tests
# -----------------------------------------------------------------------------


class TestExtractEnum:
    """Tests for _extract_enum function."""

    def test_extracts_enum_from_schema(self) -> None:
        """Extract enum values from a valid schema."""
        schema = {
            "properties": {
                "agent": {"enum": ["codex", "claude"]},
            },
        }
        result = _extract_enum(schema, "agent")
        assert result == ["codex", "claude"]

    def test_returns_empty_for_missing_key(self) -> None:
        """Returns empty list when key not in schema."""
        schema = {
            "properties": {
                "agent": {"enum": ["codex", "claude"]},
            },
        }
        result = _extract_enum(schema, "phase")
        assert result == []

    def test_returns_empty_for_no_enum(self) -> None:
        """Returns empty list when no enum defined."""
        schema = {
            "properties": {
                "agent": {"type": "string"},
            },
        }
        result = _extract_enum(schema, "agent")
        assert result == []

    def test_returns_empty_for_none_schema(self) -> None:
        """Returns empty list for None schema."""
        result = _extract_enum(None, "agent")
        assert result == []

    def test_returns_empty_for_non_dict_schema(self) -> None:
        """Returns empty list for non-dict schema."""
        result = _extract_enum("not a dict", "agent")  # type: ignore
        assert result == []

    def test_filters_non_string_enum_values(self) -> None:
        """Filters out non-string enum values."""
        schema = {
            "properties": {
                "agent": {"enum": ["codex", 123, None, "claude"]},
            },
        }
        result = _extract_enum(schema, "agent")
        assert result == ["codex", "claude"]


# -----------------------------------------------------------------------------
# _select_from_allowed tests
# -----------------------------------------------------------------------------


class TestSelectFromAllowed:
    """Tests for _select_from_allowed function."""

    def test_selects_preferred_when_in_allowed(self) -> None:
        """Selects preferred when it's in allowed list."""
        result = _select_from_allowed("claude", ["codex", "claude"])
        assert result == "claude"

    def test_falls_back_to_gemini_when_preferred_not_allowed(self) -> None:
        """Falls back to 'gemini' if preferred not in list."""
        result = _select_from_allowed("invalid", ["codex", "gemini", "claude"])
        assert result == "gemini"

    def test_falls_back_to_first_when_gemini_not_available(self) -> None:
        """Falls back to first item when gemini not available."""
        result = _select_from_allowed("invalid", ["codex", "claude"])
        assert result == "codex"

    def test_returns_codex_for_empty_list(self) -> None:
        """Returns 'codex' for empty allowed list."""
        result = _select_from_allowed("anything", [])
        assert result == "codex"


# -----------------------------------------------------------------------------
# _extract_milestone tests
# -----------------------------------------------------------------------------


class TestExtractMilestone:
    """Tests for _extract_milestone function."""

    def test_extracts_milestone_with_bold_format(self) -> None:
        """Extracts milestone from **Milestone:** format."""
        text = "Some text **Milestone:** M2 more text"
        result = _extract_milestone(text)
        assert result == "M2"

    def test_extracts_milestone_with_bold_id_format(self) -> None:
        """Extracts milestone from **Milestone ID:** format."""
        text = "Some text **Milestone ID:** M3 more text"
        result = _extract_milestone(text)
        assert result == "M3"

    def test_extracts_milestone_from_line_start(self) -> None:
        """Extracts milestone from line-start format."""
        text = "Some text\nMilestone: M1\nmore text"
        result = _extract_milestone(text)
        assert result == "M1"

    def test_returns_none_for_no_milestone(self) -> None:
        """Returns None when no milestone found."""
        text = "This text has no milestone information"
        result = _extract_milestone(text)
        assert result is None

    def test_case_insensitive(self) -> None:
        """Milestone extraction is case-insensitive."""
        text = "**MILESTONE:** M5"
        result = _extract_milestone(text)
        assert result == "M5"


# -----------------------------------------------------------------------------
# _extract_stats_refs tests
# -----------------------------------------------------------------------------


class TestExtractStatsRefs:
    """Tests for _extract_stats_refs function."""

    def test_extracts_stats_refs_from_text(self) -> None:
        """Extracts stats refs matching pattern [A-Z]{2,}-\\d+."""
        text = "Referenced CX-1 and CL-1 in this document"
        result = _extract_stats_refs(text)
        assert "CX-1" in result
        assert "CL-1" in result

    def test_deduplicates_refs(self) -> None:
        """Deduplicates repeated refs."""
        text = "See CX-1 and also CX-1 again"
        result = _extract_stats_refs(text)
        assert result.count("CX-1") == 1

    def test_preserves_order(self) -> None:
        """Preserves order of first occurrence."""
        text = "First CL-1 then CX-1 then CL-1"
        result = _extract_stats_refs(text)
        assert result == ["CL-1", "CX-1"]

    def test_returns_empty_for_no_matches(self) -> None:
        """Returns empty list when no matches."""
        text = "No stats refs here"
        result = _extract_stats_refs(text)
        assert result == []


# -----------------------------------------------------------------------------
# _load_stats_ids tests
# -----------------------------------------------------------------------------


class TestLoadStatsIds:
    """Tests for _load_stats_ids function."""

    def test_loads_stats_ids_from_file(self, tmp_path: Path) -> None:
        """Loads stats IDs from STATS.md file."""
        stats_path = tmp_path / "STATS.md"
        stats_path.write_text("## Stats\n- CX-1: Codex\n- CL-1: Claude\n")
        result = _load_stats_ids(tmp_path)
        assert result is not None
        assert "CX-1" in result
        assert "CL-1" in result

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """Returns None when STATS.md doesn't exist."""
        result = _load_stats_ids(tmp_path)
        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path: Path) -> None:
        """Returns None for empty/no-matches file."""
        stats_path = tmp_path / "STATS.md"
        stats_path.write_text("No stats IDs here")
        result = _load_stats_ids(tmp_path)
        assert result is None


# -----------------------------------------------------------------------------
# _env_truthy tests
# -----------------------------------------------------------------------------


class TestEnvTruthy:
    """Tests for _env_truthy function."""

    def test_returns_true_for_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns True for '1'."""
        monkeypatch.setenv("TEST_VAR", "1")
        assert _env_truthy("TEST_VAR") is True

    def test_returns_true_for_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns True for 'true'."""
        monkeypatch.setenv("TEST_VAR", "true")
        assert _env_truthy("TEST_VAR") is True

    def test_returns_true_for_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns True for 'yes'."""
        monkeypatch.setenv("TEST_VAR", "yes")
        assert _env_truthy("TEST_VAR") is True

    def test_returns_true_for_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns True for 'on'."""
        monkeypatch.setenv("TEST_VAR", "on")
        assert _env_truthy("TEST_VAR") is True

    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Case insensitive matching."""
        monkeypatch.setenv("TEST_VAR", "TRUE")
        assert _env_truthy("TEST_VAR") is True

    def test_returns_false_for_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False for '0'."""
        monkeypatch.setenv("TEST_VAR", "0")
        assert _env_truthy("TEST_VAR") is False

    def test_returns_false_for_unset(self) -> None:
        """Returns False for unset variable."""
        assert _env_truthy("DEFINITELY_NOT_SET_VAR_12345") is False


# -----------------------------------------------------------------------------
# _try_parse_json tests
# -----------------------------------------------------------------------------


class TestTryParseJson:
    """Tests for _try_parse_json function."""

    def test_parses_valid_dict(self) -> None:
        """Parses valid JSON dict."""
        result = _try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_returns_none_for_invalid_json(self) -> None:
        """Returns None for invalid JSON."""
        result = _try_parse_json("not json")
        assert result is None

    def test_returns_none_for_json_array(self) -> None:
        """Returns None for JSON array (not dict)."""
        result = _try_parse_json("[1, 2, 3]")
        assert result is None

    def test_unwraps_double_encoded_json(self) -> None:
        """Unwraps double-encoded JSON string."""
        inner = '{"key": "value"}'
        outer = json.dumps(inner)  # Double encode
        result = _try_parse_json(outer)
        assert result == {"key": "value"}

    def test_returns_none_for_string_not_json(self) -> None:
        """Returns None for JSON string that's not JSON inside."""
        result = _try_parse_json('"just a string"')
        assert result is None


# -----------------------------------------------------------------------------
# _strip_code_fence tests
# -----------------------------------------------------------------------------


class TestStripCodeFence:
    """Tests for _strip_code_fence function."""

    def test_strips_json_code_fence(self) -> None:
        """Strips ```json code fence."""
        text = '```json\n{"key": "value"}\n```'
        result = _strip_code_fence(text)
        assert result == '{"key": "value"}'

    def test_strips_generic_code_fence(self) -> None:
        """Strips generic ``` code fence."""
        text = '```\n{"key": "value"}\n```'
        result = _strip_code_fence(text)
        assert result == '{"key": "value"}'

    def test_handles_no_ending_fence(self) -> None:
        """Handles text with no ending fence."""
        text = '```json\n{"key": "value"}'
        result = _strip_code_fence(text)
        assert '{"key": "value"}' in result


# -----------------------------------------------------------------------------
# _extract_balanced_json tests
# -----------------------------------------------------------------------------


class TestExtractBalancedJson:
    """Tests for _extract_balanced_json function."""

    def test_extracts_simple_json(self) -> None:
        """Extracts simple JSON object."""
        text = 'prefix {"key": "value"} suffix'
        result = _extract_balanced_json(text)
        assert result == '{"key": "value"}'

    def test_extracts_nested_json(self) -> None:
        """Extracts nested JSON object."""
        text = 'before {"outer": {"inner": 123}} after'
        result = _extract_balanced_json(text)
        assert result == '{"outer": {"inner": 123}}'

    def test_handles_strings_with_braces(self) -> None:
        """Handles strings containing braces."""
        text = '{"msg": "hello {world}"}'
        result = _extract_balanced_json(text)
        assert result == '{"msg": "hello {world}"}'

    def test_handles_escaped_quotes(self) -> None:
        """Handles escaped quotes in strings."""
        text = r'{"msg": "say \"hi\""}'
        result = _extract_balanced_json(text)
        assert result == r'{"msg": "say \"hi\""}'

    def test_returns_none_for_no_json(self) -> None:
        """Returns None when no JSON found."""
        text = "no json here"
        result = _extract_balanced_json(text)
        assert result is None

    def test_returns_none_for_unbalanced(self) -> None:
        """Returns None for unbalanced braces."""
        text = '{"key": "value"'
        result = _extract_balanced_json(text)
        assert result is None


# -----------------------------------------------------------------------------
# _unwrap_payload tests
# -----------------------------------------------------------------------------


class TestUnwrapPayload:
    """Tests for _unwrap_payload function."""

    def test_unwraps_turn_wrapper(self) -> None:
        """Unwraps {'turn': {...}} wrapper."""
        obj = {"turn": {"summary": "test"}}
        result = _unwrap_payload(obj)
        assert result == {"summary": "test"}

    def test_unwraps_result_dict(self) -> None:
        """Unwraps {'result': {...}} wrapper."""
        obj = {"result": {"summary": "test"}}
        result = _unwrap_payload(obj)
        assert result == {"summary": "test"}

    def test_unwraps_result_string(self) -> None:
        """Unwraps {'result': 'json string'}."""
        obj = {"result": '{"summary": "test"}'}
        result = _unwrap_payload(obj)
        assert result == {"summary": "test"}

    def test_unwraps_response_wrapper(self) -> None:
        """Unwraps {'response': {...}} wrapper."""
        obj = {"response": {"summary": "test"}}
        result = _unwrap_payload(obj)
        assert result == {"summary": "test"}

    def test_returns_unchanged_for_no_wrapper(self) -> None:
        """Returns unchanged for direct payload."""
        obj = {"summary": "test"}
        result = _unwrap_payload(obj)
        assert result == {"summary": "test"}


# -----------------------------------------------------------------------------
# _candidate_json_texts tests
# -----------------------------------------------------------------------------


class TestCandidateJsonTexts:
    """Tests for _candidate_json_texts function."""

    def test_yields_stripped_text(self) -> None:
        """Yields stripped text as first candidate."""
        text = '  {"key": "value"}  '
        candidates = list(_candidate_json_texts(text))
        assert '{"key": "value"}' in candidates

    def test_yields_code_fence_stripped(self) -> None:
        """Yields code-fence-stripped text."""
        text = '```json\n{"key": "value"}\n```'
        candidates = list(_candidate_json_texts(text))
        assert any('{"key": "value"}' in c for c in candidates)

    def test_yields_balanced_json(self) -> None:
        """Yields balanced JSON extraction."""
        text = 'prefix {"key": "value"} suffix'
        candidates = list(_candidate_json_texts(text))
        assert '{"key": "value"}' in candidates


# -----------------------------------------------------------------------------
# _extract_json_payload tests
# -----------------------------------------------------------------------------


class TestExtractJsonPayload:
    """Tests for _extract_json_payload function."""

    def test_extracts_clean_json(self) -> None:
        """Extracts clean JSON payload."""
        text = '{"summary": "test"}'
        result = _extract_json_payload(text)
        assert result == {"summary": "test"}

    def test_extracts_from_code_fence(self) -> None:
        """Extracts JSON from code fence."""
        text = '```json\n{"summary": "test"}\n```'
        result = _extract_json_payload(text)
        assert result == {"summary": "test"}

    def test_extracts_from_prose(self) -> None:
        """Extracts JSON from surrounding prose."""
        text = 'Here is the output:\n{"summary": "test"}\nEnd.'
        result = _extract_json_payload(text)
        assert result == {"summary": "test"}

    def test_returns_none_for_empty(self) -> None:
        """Returns None for empty text."""
        result = _extract_json_payload("")
        assert result is None

    def test_returns_none_for_no_json(self) -> None:
        """Returns None for text with no JSON."""
        result = _extract_json_payload("no json here")
        assert result is None


# -----------------------------------------------------------------------------
# _validate_turn_minimal tests
# -----------------------------------------------------------------------------


class TestValidateTurnMinimal:
    """Tests for _validate_turn_minimal function."""

    def _make_valid_turn(self) -> dict[str, Any]:
        """Create a valid turn for testing."""
        return {
            "agent": "codex",
            "milestone_id": "M0",
            "phase": "plan",
            "work_completed": False,
            "project_complete": False,
            "summary": "Test summary",
            "gates_passed": [],
            "requirement_progress": {
                "covered_req_ids": [],
                "tests_added_or_modified": [],
                "commands_run": [],
            },
            "next_agent": "claude",
            "next_prompt": "",
            "delegate_rationale": "",
            "stats_refs": ["CX-1"],
            "needs_write_access": False,
            "artifacts": [],
        }

    def test_valid_turn_passes(self) -> None:
        """Valid turn passes validation."""
        turn = self._make_valid_turn()
        valid, err = _validate_turn_minimal(turn, ["codex", "claude"], ["plan", "implement"])
        assert valid
        assert err == ""

    def test_missing_key_fails(self) -> None:
        """Missing required key fails validation."""
        turn = self._make_valid_turn()
        del turn["summary"]
        valid, err = _validate_turn_minimal(turn, ["codex", "claude"], ["plan"])
        assert not valid
        assert "missing key" in err

    def test_invalid_agent_fails(self) -> None:
        """Invalid agent fails validation."""
        turn = self._make_valid_turn()
        turn["agent"] = "invalid"
        valid, err = _validate_turn_minimal(turn, ["codex", "claude"], ["plan"])
        assert not valid
        assert "invalid agent" in err

    def test_invalid_phase_fails(self) -> None:
        """Invalid phase fails validation."""
        turn = self._make_valid_turn()
        turn["phase"] = "invalid"
        valid, err = _validate_turn_minimal(turn, ["codex", "claude"], ["plan", "implement"])
        assert not valid
        assert "invalid phase" in err

    def test_non_boolean_work_completed_fails(self) -> None:
        """Non-boolean work_completed fails validation."""
        turn = self._make_valid_turn()
        turn["work_completed"] = "true"  # String not bool
        valid, err = _validate_turn_minimal(turn, ["codex", "claude"], ["plan"])
        assert not valid
        assert "boolean" in err

    def test_empty_stats_refs_fails(self) -> None:
        """Empty stats_refs fails validation."""
        turn = self._make_valid_turn()
        turn["stats_refs"] = []
        valid, err = _validate_turn_minimal(turn, ["codex", "claude"], ["plan"])
        assert not valid
        assert "stats_refs" in err

    def test_invalid_artifacts_fails(self) -> None:
        """Invalid artifacts structure fails validation."""
        turn = self._make_valid_turn()
        turn["artifacts"] = [{"path": "file.py"}]  # Missing description
        valid, err = _validate_turn_minimal(turn, ["codex", "claude"], ["plan"])
        assert not valid
        assert "artifact" in err


# -----------------------------------------------------------------------------
# _coerce_enum tests
# -----------------------------------------------------------------------------


class TestCoerceEnum:
    """Tests for _coerce_enum function."""

    def test_returns_value_when_valid(self) -> None:
        """Returns value when it's in allowed list."""
        result = _coerce_enum("claude", ["codex", "claude"], "default")
        assert result == "claude"

    def test_returns_first_allowed_when_invalid(self) -> None:
        """Returns first allowed when value invalid."""
        result = _coerce_enum("invalid", ["codex", "claude"], "default")
        assert result == "codex"

    def test_returns_fallback_when_empty_allowed(self) -> None:
        """Returns fallback when allowed list empty."""
        result = _coerce_enum("anything", [], "default")
        assert result == "default"

    def test_returns_first_for_non_string_value(self) -> None:
        """Returns first allowed for non-string value."""
        result = _coerce_enum(123, ["codex", "claude"], "default")  # type: ignore
        assert result == "codex"


# -----------------------------------------------------------------------------
# _truncate tests
# -----------------------------------------------------------------------------


class TestTruncate:
    """Tests for _truncate function."""

    def test_short_text_unchanged(self) -> None:
        """Short text is returned unchanged."""
        result = _truncate("short", limit=100)
        assert result == "short"

    def test_long_text_truncated(self) -> None:
        """Long text is truncated."""
        result = _truncate("x" * 100, limit=50)
        assert len(result) < 100
        assert "truncated" in result

    def test_default_limit(self) -> None:
        """Default limit is 800."""
        result = _truncate("x" * 1000)
        assert len(result) < 1000


# -----------------------------------------------------------------------------
# _format_process_error tests
# -----------------------------------------------------------------------------


class TestFormatProcessError:
    """Tests for _format_process_error function."""

    def test_includes_exit_code(self) -> None:
        """Includes exit code in output."""
        import subprocess

        result = subprocess.CompletedProcess(
            args=["test"],
            returncode=1,
            stdout="out",
            stderr="err",
        )
        formatted = _format_process_error(result)
        assert "exit_code=1" in formatted

    def test_includes_stderr_when_present(self) -> None:
        """Includes stderr in output."""
        import subprocess

        result = subprocess.CompletedProcess(
            args=["test"],
            returncode=1,
            stdout="",
            stderr="error message",
        )
        formatted = _format_process_error(result)
        assert "stderr=error message" in formatted


# -----------------------------------------------------------------------------
# _format_parse_error tests
# -----------------------------------------------------------------------------


class TestFormatParseError:
    """Tests for _format_parse_error function."""

    def test_includes_raw_output(self) -> None:
        """Includes raw output in error."""
        formatted = _format_parse_error("raw text here")
        assert "raw_output=raw text here" in formatted

    def test_handles_empty_output(self) -> None:
        """Handles empty output."""
        formatted = _format_parse_error("")
        assert "raw_output_empty" in formatted


# -----------------------------------------------------------------------------
# _format_validation_error tests
# -----------------------------------------------------------------------------


class TestFormatValidationError:
    """Tests for _format_validation_error function."""

    def test_includes_error_and_output(self) -> None:
        """Includes both error and raw output."""
        formatted = _format_validation_error("validation failed", "raw text")
        assert "validation_error=validation failed" in formatted
        assert "raw_output=raw text" in formatted

    def test_handles_empty_output(self) -> None:
        """Handles empty raw output."""
        formatted = _format_validation_error("validation failed", "")
        assert "validation_error=validation failed" in formatted
        assert "raw_output" not in formatted


# -----------------------------------------------------------------------------
# REQUIRED_KEYS constant tests
# -----------------------------------------------------------------------------


class TestRequiredKeys:
    """Tests for REQUIRED_KEYS constant."""

    def test_required_keys_contains_expected(self) -> None:
        """REQUIRED_KEYS contains expected keys."""
        expected = [
            "agent",
            "milestone_id",
            "phase",
            "work_completed",
            "project_complete",
            "summary",
            "stats_refs",
            "artifacts",
        ]
        for key in expected:
            assert key in REQUIRED_KEYS, f"Missing required key: {key}"

    def test_required_keys_count(self) -> None:
        """REQUIRED_KEYS has expected count."""
        assert len(REQUIRED_KEYS) == 14
