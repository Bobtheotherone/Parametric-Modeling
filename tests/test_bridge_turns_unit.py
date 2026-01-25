# SPDX-License-Identifier: MIT
"""Unit tests for bridge/turns.py.

Tests the helper functions for building schema-valid agent turns.
Key functions tested:
- build_error_turn: Build schema-valid error turn with safe defaults
- error_turn: Alias for build_error_turn
- normalize_stats_refs: Normalize stats refs with fallback
- normalize_requirement_progress: Normalize requirement progress entries
- normalize_artifacts: Normalize artifact list to required schema shape
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.turns import (
    AGENTS,
    VALID_PHASES,
    build_error_turn,
    error_turn,
    normalize_artifacts,
    normalize_requirement_progress,
    normalize_stats_refs,
)

# -----------------------------------------------------------------------------
# normalize_stats_refs tests
# -----------------------------------------------------------------------------


class TestNormalizeStatsRefs:
    """Tests for normalize_stats_refs function."""

    def test_valid_refs_preserved(self) -> None:
        """Valid stats refs are preserved."""
        refs = normalize_stats_refs("codex", ["CX-1"], {"CX-1", "CL-1"})
        assert refs == ["CX-1"]

    def test_multiple_valid_refs(self) -> None:
        """Multiple valid refs are all preserved."""
        refs = normalize_stats_refs("codex", ["CX-1", "CL-1"], {"CX-1", "CL-1"})
        assert refs == ["CX-1", "CL-1"]

    def test_filters_invalid_refs(self) -> None:
        """Invalid refs are filtered out when stats_id_set is provided."""
        refs = normalize_stats_refs("codex", ["CX-1", "INVALID"], {"CX-1", "CL-1"})
        assert refs == ["CX-1"]
        assert "INVALID" not in refs

    def test_all_invalid_falls_back_to_agent_default(self) -> None:
        """When all refs are invalid, falls back to agent-appropriate default."""
        refs = normalize_stats_refs("codex", ["INVALID-1", "INVALID-2"], {"CX-1", "CL-1"})
        assert refs == ["CX-1"]  # Default for codex

    def test_empty_refs_falls_back_to_agent_default(self) -> None:
        """Empty refs falls back to agent-appropriate default."""
        refs = normalize_stats_refs("codex", [], {"CX-1", "CL-1"})
        assert refs == ["CX-1"]

    def test_none_refs_falls_back_to_agent_default(self) -> None:
        """None refs falls back to agent-appropriate default."""
        refs = normalize_stats_refs("codex", None, {"CX-1", "CL-1"})
        assert refs == ["CX-1"]

    def test_claude_agent_default(self) -> None:
        """Claude agent gets CL-1 as default."""
        refs = normalize_stats_refs("claude", None, {"CX-1", "CL-1"})
        assert refs == ["CL-1"]

    def test_no_stats_id_set_uses_fallback(self) -> None:
        """Without stats_id_set, refs pass through unfiltered."""
        refs = normalize_stats_refs("codex", ["CUSTOM-1"], None)
        assert refs == ["CUSTOM-1"]

    def test_no_stats_id_set_empty_refs_gets_default(self) -> None:
        """Without stats_id_set and empty refs, gets default."""
        refs = normalize_stats_refs("codex", [], None)
        assert refs == ["CX-1"]

    def test_unknown_agent_falls_back_to_cx1(self) -> None:
        """Unknown agent falls back to CX-1."""
        refs = normalize_stats_refs("unknown_agent", [], {"CX-1", "CL-1"})
        assert refs == ["CX-1"]

    def test_fallback_uses_first_sorted_when_default_not_in_set(self) -> None:
        """When default not in set, uses first sorted entry."""
        refs = normalize_stats_refs("codex", [], {"CL-1"})  # CX-1 not in set
        assert refs == ["CL-1"]


# -----------------------------------------------------------------------------
# normalize_requirement_progress tests
# -----------------------------------------------------------------------------


class TestNormalizeRequirementProgress:
    """Tests for normalize_requirement_progress function."""

    def test_valid_progress_preserved(self) -> None:
        """Valid requirement progress is preserved."""
        val = {
            "covered_req_ids": ["REQ-001"],
            "tests_added_or_modified": ["test_foo.py"],
            "commands_run": ["pytest"],
        }
        result = normalize_requirement_progress(val)
        assert result["covered_req_ids"] == ["REQ-001"]
        assert result["tests_added_or_modified"] == ["test_foo.py"]
        assert result["commands_run"] == ["pytest"]

    def test_missing_keys_default_to_empty_lists(self) -> None:
        """Missing keys default to empty lists."""
        result = normalize_requirement_progress({})
        assert result["covered_req_ids"] == []
        assert result["tests_added_or_modified"] == []
        assert result["commands_run"] == []

    def test_none_input_defaults_all_empty(self) -> None:
        """None input defaults all fields to empty lists."""
        result = normalize_requirement_progress(None)
        assert result["covered_req_ids"] == []
        assert result["tests_added_or_modified"] == []
        assert result["commands_run"] == []

    def test_partial_keys_filled(self) -> None:
        """Partial keys are filled, missing ones get empty lists."""
        val = {"covered_req_ids": ["REQ-001"]}
        result = normalize_requirement_progress(val)
        assert result["covered_req_ids"] == ["REQ-001"]
        assert result["tests_added_or_modified"] == []
        assert result["commands_run"] == []

    def test_non_string_values_filtered(self) -> None:
        """Non-string values in lists are filtered out."""
        val = {
            "covered_req_ids": ["REQ-001", 123, None, "REQ-002"],
            "tests_added_or_modified": [],
            "commands_run": [],
        }
        result = normalize_requirement_progress(val)
        # Only strings should remain
        assert "REQ-001" in result["covered_req_ids"]
        assert "REQ-002" in result["covered_req_ids"]

    def test_whitespace_only_strings_filtered(self) -> None:
        """Whitespace-only strings are filtered out."""
        val = {
            "covered_req_ids": ["REQ-001", "  ", ""],
            "tests_added_or_modified": [],
            "commands_run": [],
        }
        result = normalize_requirement_progress(val)
        assert result["covered_req_ids"] == ["REQ-001"]


# -----------------------------------------------------------------------------
# normalize_artifacts tests
# -----------------------------------------------------------------------------


class TestNormalizeArtifacts:
    """Tests for normalize_artifacts function."""

    def test_valid_artifacts_preserved(self) -> None:
        """Valid artifacts are preserved."""
        artifacts = [{"path": "src/foo.py", "description": "New module"}]
        result = normalize_artifacts(artifacts)
        assert len(result) == 1
        assert result[0]["path"] == "src/foo.py"
        assert result[0]["description"] == "New module"

    def test_multiple_artifacts(self) -> None:
        """Multiple valid artifacts are all preserved."""
        artifacts = [
            {"path": "src/foo.py", "description": "New module"},
            {"path": "tests/test_foo.py", "description": "Tests"},
        ]
        result = normalize_artifacts(artifacts)
        assert len(result) == 2

    def test_none_input_returns_empty_list(self) -> None:
        """None input returns empty list."""
        result = normalize_artifacts(None)
        assert result == []

    def test_empty_list_returns_empty_list(self) -> None:
        """Empty list returns empty list."""
        result = normalize_artifacts([])
        assert result == []

    def test_missing_path_filtered(self) -> None:
        """Artifacts without path are filtered out."""
        artifacts = [{"description": "No path"}]
        result = normalize_artifacts(artifacts)
        assert result == []

    def test_missing_description_filtered(self) -> None:
        """Artifacts without description are filtered out."""
        artifacts = [{"path": "src/foo.py"}]
        result = normalize_artifacts(artifacts)
        assert result == []

    def test_empty_path_filtered(self) -> None:
        """Artifacts with empty path are filtered out."""
        artifacts = [{"path": "", "description": "Empty path"}]
        result = normalize_artifacts(artifacts)
        assert result == []

    def test_whitespace_path_filtered(self) -> None:
        """Artifacts with whitespace-only path are filtered out."""
        artifacts = [{"path": "   ", "description": "Whitespace path"}]
        result = normalize_artifacts(artifacts)
        assert result == []

    def test_non_dict_items_filtered(self) -> None:
        """Non-dict items in the list are filtered out."""
        artifacts = [
            {"path": "src/foo.py", "description": "Valid"},
            "not a dict",
            123,
            None,
        ]
        result = normalize_artifacts(artifacts)
        assert len(result) == 1
        assert result[0]["path"] == "src/foo.py"

    def test_path_and_description_trimmed(self) -> None:
        """Path and description are trimmed of whitespace."""
        artifacts = [{"path": "  src/foo.py  ", "description": "  Desc  "}]
        result = normalize_artifacts(artifacts)
        assert result[0]["path"] == "src/foo.py"
        assert result[0]["description"] == "Desc"

    def test_non_string_path_filtered(self) -> None:
        """Non-string path values are filtered out."""
        artifacts = [{"path": 123, "description": "Desc"}]
        result = normalize_artifacts(artifacts)
        assert result == []

    def test_non_sequence_input_returns_empty(self) -> None:
        """Non-sequence input returns empty list."""
        result = normalize_artifacts("not a sequence")  # type: ignore
        assert result == []


# -----------------------------------------------------------------------------
# build_error_turn tests
# -----------------------------------------------------------------------------


class TestBuildErrorTurn:
    """Tests for build_error_turn function."""

    def test_minimal_turn(self) -> None:
        """Minimal turn with required fields only."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test summary",
        )
        assert turn["agent"] == "codex"
        assert turn["milestone_id"] == "M0"
        assert turn["summary"] == "Test summary"
        assert turn["work_completed"] is False
        assert turn["project_complete"] is False

    def test_includes_all_required_keys(self) -> None:
        """Turn includes all required schema keys."""
        turn = build_error_turn(
            agent="claude",
            milestone_id="M1",
            summary="Test",
        )
        required_keys = {
            "agent",
            "milestone_id",
            "phase",
            "work_completed",
            "project_complete",
            "summary",
            "gates_passed",
            "requirement_progress",
            "next_agent",
            "next_prompt",
            "delegate_rationale",
            "stats_refs",
            "needs_write_access",
            "artifacts",
        }
        assert set(turn.keys()) == required_keys

    def test_error_detail_appended_to_summary(self) -> None:
        """Error detail is appended to summary."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Main summary",
            error_detail="Additional detail",
        )
        assert "Main summary" in turn["summary"]
        assert "Additional detail" in turn["summary"]

    def test_stats_refs_normalized(self) -> None:
        """Stats refs are normalized with fallback."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
            stats_refs=None,
            stats_id_set={"CX-1", "CL-1"},
        )
        assert "CX-1" in turn["stats_refs"]

    def test_default_phase_is_plan(self) -> None:
        """Default phase is 'plan'."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
        )
        assert turn["phase"] == "plan"

    def test_custom_phase_accepted(self) -> None:
        """Custom valid phase is accepted."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
            phase="implement",
        )
        assert turn["phase"] == "implement"

    def test_invalid_phase_falls_back_to_default(self) -> None:
        """Invalid phase falls back to default."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
            phase="invalid_phase",
        )
        assert turn["phase"] == "plan"

    def test_next_agent_defaults_to_other(self) -> None:
        """Next agent defaults to the other agent."""
        turn_codex = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
        )
        assert turn_codex["next_agent"] == "claude"

        turn_claude = build_error_turn(
            agent="claude",
            milestone_id="M0",
            summary="Test",
        )
        assert turn_claude["next_agent"] == "codex"

    def test_explicit_next_agent(self) -> None:
        """Explicit next_agent is used when provided."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
            next_agent="codex",
        )
        assert turn["next_agent"] == "codex"

    def test_needs_write_access_defaults_true(self) -> None:
        """needs_write_access defaults to True."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
        )
        assert turn["needs_write_access"] is True

    def test_needs_write_access_can_be_false(self) -> None:
        """needs_write_access can be set to False."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
            needs_write_access=False,
        )
        assert turn["needs_write_access"] is False

    def test_work_completed_boolean(self) -> None:
        """work_completed is properly converted to boolean."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
            work_completed=True,
        )
        assert turn["work_completed"] is True

    def test_project_complete_boolean(self) -> None:
        """project_complete is properly converted to boolean."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
            project_complete=True,
        )
        assert turn["project_complete"] is True

    def test_gates_passed_normalized(self) -> None:
        """gates_passed is normalized to list of strings."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
            gates_passed=["gate1", "gate2"],
        )
        assert turn["gates_passed"] == ["gate1", "gate2"]

    def test_gates_passed_defaults_empty(self) -> None:
        """gates_passed defaults to empty list."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
        )
        assert turn["gates_passed"] == []

    def test_artifacts_normalized(self) -> None:
        """Artifacts are normalized."""
        turn = build_error_turn(
            agent="codex",
            milestone_id="M0",
            summary="Test",
            artifacts=[{"path": "src/foo.py", "description": "New file"}],
        )
        assert len(turn["artifacts"]) == 1
        assert turn["artifacts"][0]["path"] == "src/foo.py"


# -----------------------------------------------------------------------------
# error_turn alias tests
# -----------------------------------------------------------------------------


class TestErrorTurnAlias:
    """Tests for error_turn alias function."""

    def test_alias_produces_same_result(self) -> None:
        """error_turn produces same result as build_error_turn."""
        kwargs = {
            "agent": "codex",
            "milestone_id": "M0",
            "summary": "Test",
        }
        result1 = build_error_turn(**kwargs)
        result2 = error_turn(**kwargs)
        assert result1 == result2

    def test_alias_accepts_all_parameters(self) -> None:
        """error_turn accepts all parameters."""
        turn = error_turn(
            agent="claude",
            milestone_id="M1",
            summary="Summary",
            error_detail="Detail",
            next_agent="codex",
            next_prompt="Prompt",
            delegate_rationale="Rationale",
            stats_refs=["CL-1"],
            stats_id_set={"CL-1"},
            phase="verify",
            needs_write_access=False,
            work_completed=True,
            project_complete=False,
            gates_passed=["gate1"],
            requirement_progress={"covered_req_ids": ["REQ-001"]},
            artifacts=[{"path": "file.py", "description": "desc"}],
        )
        assert turn["agent"] == "claude"
        assert turn["phase"] == "verify"
        assert turn["work_completed"] is True


# -----------------------------------------------------------------------------
# Constants tests
# -----------------------------------------------------------------------------


class TestConstants:
    """Tests for module constants."""

    def test_agents_contains_codex_and_claude(self) -> None:
        """AGENTS contains 'codex' and 'claude'."""
        assert "codex" in AGENTS
        assert "claude" in AGENTS

    def test_valid_phases(self) -> None:
        """VALID_PHASES contains expected phases."""
        assert "plan" in VALID_PHASES
        assert "implement" in VALID_PHASES
        assert "verify" in VALID_PHASES
        assert "finalize" in VALID_PHASES

    def test_valid_phases_count(self) -> None:
        """VALID_PHASES has exactly 4 phases."""
        assert len(VALID_PHASES) == 4
