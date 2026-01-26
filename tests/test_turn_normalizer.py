# SPDX-License-Identifier: MIT
"""Unit tests for bridge/loop_pkg/turn_normalizer.py.

Tests the TurnNormalizer class that handles normalizing messy agent output
to the full Turn schema. Key functionality:
- JSON payload extraction from raw output with prose/markdown
- Invariant field auto-correction (agent, milestone_id, stats_refs)
- Required field validation
- Lenient turn validation with warnings
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.loop_pkg.turn_normalizer import (
    NormalizationResult,
    TurnNormalizer,
    normalize_agent_output,
    validate_turn_lenient,
)

# -----------------------------------------------------------------------------
# Test data fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def stats_id_set() -> set[str]:
    """Standard stats ID set for tests."""
    return {"CX-1", "CL-1"}


@pytest.fixture
def normalizer(stats_id_set: set[str]) -> TurnNormalizer:
    """Create a TurnNormalizer with standard settings."""
    return TurnNormalizer(
        expected_agent="claude",
        expected_milestone_id="M1",
        stats_id_set=stats_id_set,
        default_phase="implement",
    )


@pytest.fixture
def minimal_valid_payload() -> dict:
    """Minimal valid payload with required fields only."""
    return {
        "summary": "Did some work",
        "work_completed": True,
        "project_complete": False,
    }


@pytest.fixture
def full_valid_payload() -> dict:
    """Full valid payload with all fields."""
    return {
        "agent": "claude",
        "milestone_id": "M1",
        "phase": "implement",
        "work_completed": True,
        "project_complete": False,
        "summary": "Implemented feature X",
        "gates_passed": ["gate1", "gate2"],
        "requirement_progress": {
            "covered_req_ids": ["REQ-001"],
            "tests_added_or_modified": ["test_foo.py"],
            "commands_run": ["pytest"],
        },
        "next_agent": "codex",
        "next_prompt": "Review the changes",
        "delegate_rationale": "Need code review",
        "stats_refs": ["CL-1"],
        "needs_write_access": True,
        "artifacts": [{"path": "src/foo.py", "description": "New module"}],
    }


# -----------------------------------------------------------------------------
# TurnNormalizer: JSON extraction tests
# -----------------------------------------------------------------------------


class TestJsonExtraction:
    """Tests for extracting JSON from raw output."""

    def test_clean_json(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Clean JSON string parses directly."""
        raw = json.dumps(minimal_valid_payload)
        result = normalizer.normalize(raw)
        assert result.success
        assert result.turn is not None
        assert result.turn["summary"] == "Did some work"

    def test_json_with_markdown_fences(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """JSON wrapped in markdown code fences is extracted."""
        raw = f"```json\n{json.dumps(minimal_valid_payload)}\n```"
        result = normalizer.normalize(raw)
        assert result.success
        assert result.turn is not None

    def test_json_with_prose_before(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """JSON with prose before is extracted."""
        raw = f"Here is my response:\n\n{json.dumps(minimal_valid_payload)}"
        result = normalizer.normalize(raw)
        assert result.success
        assert result.turn is not None

    def test_json_with_prose_after(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """JSON with prose after is extracted."""
        raw = f"{json.dumps(minimal_valid_payload)}\n\nThat's my output."
        result = normalizer.normalize(raw)
        assert result.success
        assert result.turn is not None

    def test_json_with_prose_around(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """JSON with prose before and after is extracted."""
        raw = f"Starting:\n{json.dumps(minimal_valid_payload)}\nEnding."
        result = normalizer.normalize(raw)
        assert result.success
        assert result.turn is not None

    def test_empty_string_fails(self, normalizer: TurnNormalizer) -> None:
        """Empty string fails extraction."""
        result = normalizer.normalize("")
        assert not result.success
        assert "Cannot extract JSON" in (result.error or "")

    def test_no_json_fails(self, normalizer: TurnNormalizer) -> None:
        """String with no JSON fails extraction."""
        result = normalizer.normalize("This is just plain text with no JSON.")
        assert not result.success
        assert "Cannot extract JSON" in (result.error or "")

    def test_invalid_json_fails(self, normalizer: TurnNormalizer) -> None:
        """Invalid JSON structure fails extraction."""
        result = normalizer.normalize("{not valid json}")
        assert not result.success

    def test_nested_json_extracts_outer(self, normalizer: TurnNormalizer) -> None:
        """Nested JSON objects extract the first complete object."""
        inner = {"inner": True}
        outer = {
            "summary": "Test",
            "work_completed": True,
            "project_complete": False,
            "nested": inner,
        }
        raw = json.dumps(outer)
        result = normalizer.normalize(raw)
        assert result.success
        assert result.turn is not None
        assert result.turn["summary"] == "Test"


# -----------------------------------------------------------------------------
# TurnNormalizer: Required field validation
# -----------------------------------------------------------------------------


class TestRequiredFields:
    """Tests for required field validation."""

    def test_missing_summary_fails(self, normalizer: TurnNormalizer) -> None:
        """Missing summary field fails."""
        payload = {"work_completed": True, "project_complete": False}
        result = normalizer.normalize(json.dumps(payload))
        assert not result.success
        assert "summary" in (result.error or "")

    def test_missing_work_completed_fails(self, normalizer: TurnNormalizer) -> None:
        """Missing work_completed field fails."""
        payload = {"summary": "Did work", "project_complete": False}
        result = normalizer.normalize(json.dumps(payload))
        assert not result.success
        assert "work_completed" in (result.error or "")

    def test_missing_project_complete_fails(self, normalizer: TurnNormalizer) -> None:
        """Missing project_complete field fails."""
        payload = {"summary": "Did work", "work_completed": True}
        result = normalizer.normalize(json.dumps(payload))
        assert not result.success
        assert "project_complete" in (result.error or "")

    def test_all_required_present_succeeds(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """All required fields present succeeds."""
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success


# -----------------------------------------------------------------------------
# TurnNormalizer: Invariant field handling
# -----------------------------------------------------------------------------


class TestInvariantFields:
    """Tests for invariant field auto-correction."""

    def test_agent_mismatch_is_corrected(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Agent mismatch is auto-corrected with warning."""
        minimal_valid_payload["agent"] = "codex"  # Wrong agent
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["agent"] == "claude"  # Corrected to expected
        assert any("agent mismatch" in w for w in result.warnings)

    def test_milestone_mismatch_is_corrected(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Milestone mismatch is auto-corrected with warning."""
        minimal_valid_payload["milestone_id"] = "M2"  # Wrong milestone
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["milestone_id"] == "M1"  # Corrected to expected
        assert any("milestone_id mismatch" in w for w in result.warnings)

    def test_missing_stats_refs_defaults(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Missing stats_refs defaults to agent-appropriate value."""
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert "CL-1" in result.turn["stats_refs"]  # Default for claude
        assert any("stats_refs defaulted" in w for w in result.warnings)

    def test_invalid_stats_refs_defaults(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Invalid stats_refs defaults with warning."""
        minimal_valid_payload["stats_refs"] = ["INVALID-1"]
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["stats_refs"]  # Should have valid refs
        assert all(ref in {"CX-1", "CL-1"} for ref in result.turn["stats_refs"])

    def test_valid_stats_refs_preserved(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Valid stats_refs are preserved."""
        minimal_valid_payload["stats_refs"] = ["CL-1"]
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["stats_refs"] == ["CL-1"]


# -----------------------------------------------------------------------------
# TurnNormalizer: Phase handling
# -----------------------------------------------------------------------------


class TestPhaseHandling:
    """Tests for phase field handling."""

    def test_valid_phase_preserved(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Valid phase is preserved."""
        minimal_valid_payload["phase"] = "verify"
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["phase"] == "verify"

    def test_invalid_phase_defaults(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Invalid phase defaults to default_phase with warning."""
        minimal_valid_payload["phase"] = "invalid_phase"
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["phase"] == "implement"  # Default
        assert any("invalid phase" in w for w in result.warnings)

    def test_missing_phase_defaults(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Missing phase defaults to default_phase."""
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["phase"] == "implement"

    @pytest.mark.parametrize("phase", ["plan", "implement", "verify", "finalize"])
    def test_all_valid_phases(self, normalizer: TurnNormalizer, minimal_valid_payload: dict, phase: str) -> None:
        """All valid phases are accepted."""
        minimal_valid_payload["phase"] = phase
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["phase"] == phase


# -----------------------------------------------------------------------------
# TurnNormalizer: Optional fields
# -----------------------------------------------------------------------------


class TestOptionalFields:
    """Tests for optional field handling."""

    def test_missing_gates_passed_defaults_empty(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Missing gates_passed defaults to empty list."""
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["gates_passed"] == []

    def test_missing_requirement_progress_defaults(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Missing requirement_progress defaults correctly."""
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        rp = result.turn["requirement_progress"]
        assert rp["covered_req_ids"] == []
        assert rp["tests_added_or_modified"] == []
        assert rp["commands_run"] == []

    def test_missing_next_prompt_defaults_empty(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Missing next_prompt defaults to empty string."""
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["next_prompt"] == ""

    def test_missing_delegate_rationale_defaults_empty(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Missing delegate_rationale defaults to empty string."""
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["delegate_rationale"] == ""

    def test_missing_artifacts_defaults_empty(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Missing artifacts defaults to empty list."""
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["artifacts"] == []

    def test_needs_write_access_defaults_true(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Missing needs_write_access defaults to True."""
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["needs_write_access"] is True


# -----------------------------------------------------------------------------
# TurnNormalizer: Artifact normalization
# -----------------------------------------------------------------------------


class TestArtifactNormalization:
    """Tests for artifact list normalization."""

    def test_valid_artifacts_preserved(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Valid artifacts are preserved."""
        minimal_valid_payload["artifacts"] = [{"path": "src/foo.py", "description": "New file"}]
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert len(result.turn["artifacts"]) == 1
        assert result.turn["artifacts"][0]["path"] == "src/foo.py"

    def test_invalid_artifact_missing_path_filtered(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Artifacts missing path are filtered out."""
        minimal_valid_payload["artifacts"] = [{"description": "No path"}]
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["artifacts"] == []

    def test_invalid_artifact_missing_description_filtered(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Artifacts missing description are filtered out."""
        minimal_valid_payload["artifacts"] = [{"path": "src/foo.py"}]
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["artifacts"] == []

    def test_non_dict_artifacts_filtered(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Non-dict artifacts are filtered out."""
        minimal_valid_payload["artifacts"] = ["not a dict", 123]
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["artifacts"] == []

    def test_empty_path_artifact_filtered(self, normalizer: TurnNormalizer, minimal_valid_payload: dict) -> None:
        """Artifacts with empty path are filtered out."""
        minimal_valid_payload["artifacts"] = [
            {"path": "", "description": "Empty path"},
            {"path": "   ", "description": "Whitespace path"},
        ]
        result = normalizer.normalize(json.dumps(minimal_valid_payload))
        assert result.success
        assert result.turn is not None
        assert result.turn["artifacts"] == []


# -----------------------------------------------------------------------------
# normalize_agent_output convenience function
# -----------------------------------------------------------------------------


class TestNormalizeAgentOutput:
    """Tests for the convenience function."""

    def test_normalize_agent_output_basic(self, stats_id_set: set[str], minimal_valid_payload: dict) -> None:
        """Basic normalization via convenience function."""
        result = normalize_agent_output(
            raw_output=json.dumps(minimal_valid_payload),
            expected_agent="codex",
            expected_milestone_id="M0",
            stats_id_set=stats_id_set,
        )
        assert result.success
        assert result.turn is not None
        assert result.turn["agent"] == "codex"
        assert result.turn["milestone_id"] == "M0"

    def test_normalize_agent_output_with_custom_default_phase(self, stats_id_set: set[str], minimal_valid_payload: dict) -> None:
        """Default phase can be customized."""
        result = normalize_agent_output(
            raw_output=json.dumps(minimal_valid_payload),
            expected_agent="codex",
            expected_milestone_id="M0",
            stats_id_set=stats_id_set,
            default_phase="plan",
        )
        assert result.success
        assert result.turn is not None
        assert result.turn["phase"] == "plan"


# -----------------------------------------------------------------------------
# validate_turn_lenient function
# -----------------------------------------------------------------------------


class TestValidateTurnLenient:
    """Tests for lenient turn validation."""

    def test_valid_turn_passes(self, stats_id_set: set[str], full_valid_payload: dict) -> None:
        """Fully valid turn passes validation."""
        is_valid, msg, warnings = validate_turn_lenient(
            full_valid_payload,
            expected_agent="claude",
            expected_milestone_id="M1",
            stats_id_set=stats_id_set,
        )
        assert is_valid
        assert msg == "ok"
        assert len(warnings) == 0

    def test_agent_mismatch_corrected_with_warning(self, stats_id_set: set[str], full_valid_payload: dict) -> None:
        """Agent mismatch is corrected with warning."""
        full_valid_payload["agent"] = "codex"
        is_valid, msg, warnings = validate_turn_lenient(
            full_valid_payload,
            expected_agent="claude",
            expected_milestone_id="M1",
            stats_id_set=stats_id_set,
        )
        assert is_valid
        assert full_valid_payload["agent"] == "claude"  # Corrected in place
        assert any("agent mismatch" in w for w in warnings)

    def test_milestone_mismatch_corrected_with_warning(self, stats_id_set: set[str], full_valid_payload: dict) -> None:
        """Milestone mismatch is corrected with warning."""
        full_valid_payload["milestone_id"] = "M2"
        is_valid, msg, warnings = validate_turn_lenient(
            full_valid_payload,
            expected_agent="claude",
            expected_milestone_id="M1",
            stats_id_set=stats_id_set,
        )
        assert is_valid
        assert full_valid_payload["milestone_id"] == "M1"  # Corrected in place
        assert any("milestone_id mismatch" in w for w in warnings)

    def test_missing_key_fails(self, stats_id_set: set[str], full_valid_payload: dict) -> None:
        """Missing required key fails validation."""
        del full_valid_payload["summary"]
        is_valid, msg, warnings = validate_turn_lenient(
            full_valid_payload,
            expected_agent="claude",
            stats_id_set=stats_id_set,
        )
        assert not is_valid
        assert "summary" in msg

    def test_non_dict_fails(self, stats_id_set: set[str]) -> None:
        """Non-dict input fails validation."""
        is_valid, msg, warnings = validate_turn_lenient(
            "not a dict",
            expected_agent="claude",
            stats_id_set=stats_id_set,
        )
        assert not is_valid
        assert "not an object" in msg

    def test_invalid_phase_fails(self, stats_id_set: set[str], full_valid_payload: dict) -> None:
        """Invalid phase fails validation."""
        full_valid_payload["phase"] = "invalid"
        is_valid, msg, warnings = validate_turn_lenient(
            full_valid_payload,
            expected_agent="claude",
            stats_id_set=stats_id_set,
        )
        assert not is_valid
        assert "invalid phase" in msg

    def test_unknown_stats_refs_corrected(self, stats_id_set: set[str], full_valid_payload: dict) -> None:
        """Unknown stats_refs are removed and replaced."""
        full_valid_payload["stats_refs"] = ["UNKNOWN-1", "CL-1"]
        is_valid, msg, warnings = validate_turn_lenient(
            full_valid_payload,
            expected_agent="claude",
            stats_id_set=stats_id_set,
        )
        assert is_valid
        assert "CL-1" in full_valid_payload["stats_refs"]
        assert "UNKNOWN-1" not in full_valid_payload["stats_refs"]
        assert any("unknown stats_refs" in w for w in warnings)

    def test_extra_keys_fail(self, stats_id_set: set[str], full_valid_payload: dict) -> None:
        """Extra unexpected keys fail validation."""
        full_valid_payload["extra_key"] = "unexpected"
        is_valid, msg, warnings = validate_turn_lenient(
            full_valid_payload,
            expected_agent="claude",
            stats_id_set=stats_id_set,
        )
        assert not is_valid
        assert "unexpected keys" in msg


# -----------------------------------------------------------------------------
# NormalizationResult dataclass
# -----------------------------------------------------------------------------


class TestNormalizationResult:
    """Tests for NormalizationResult dataclass."""

    def test_successful_result(self) -> None:
        """Successful result has expected structure."""
        result = NormalizationResult(
            success=True,
            turn={"summary": "test"},
            warnings=["warning1"],
            error=None,
        )
        assert result.success
        assert result.turn == {"summary": "test"}
        assert result.warnings == ["warning1"]
        assert result.error is None

    def test_failure_result(self) -> None:
        """Failure result has expected structure."""
        result = NormalizationResult(
            success=False,
            turn=None,
            warnings=[],
            error="Something went wrong",
        )
        assert not result.success
        assert result.turn is None
        assert result.warnings == []
        assert result.error == "Something went wrong"
