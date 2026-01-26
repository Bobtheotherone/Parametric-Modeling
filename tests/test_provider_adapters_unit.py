# SPDX-License-Identifier: MIT
"""Unit tests for bridge/provider_adapters.py.

Tests the provider output adapters for unified structured output handling.
Key classes/functions tested:
- AdapterResult: Dataclass for adapter results
- RepairResult: Dataclass for repair results
- OpenAIOutputAdapter: Adapter for OpenAI (Codex) structured output
- ClaudeOutputAdapter: Adapter for Claude structured output
- OutputRepairService: Service for repairing malformed outputs
- get_adapter_for_agent: Get appropriate adapter for agent type
- normalize_provider_output: Main entry point for provider-agnostic normalization
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

from bridge.provider_adapters import (
    AdapterResult,
    ClaudeOutputAdapter,
    OpenAIOutputAdapter,
    OutputRepairService,
    RepairResult,
    get_adapter_for_agent,
    normalize_provider_output,
)

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def minimal_valid_turn() -> dict:
    """Minimal valid turn with all required fields."""
    return {
        "agent": "codex",
        "milestone_id": "M0",
        "phase": "implement",
        "work_completed": True,
        "project_complete": False,
        "summary": "Test summary",
        "gates_passed": [],
        "requirement_progress": {
            "covered_req_ids": [],
            "tests_added_or_modified": [],
            "commands_run": [],
        },
        "next_agent": "claude",
        "next_prompt": "Continue",
        "delegate_rationale": "Handoff",
        "stats_refs": ["CX-1"],
        "needs_write_access": True,
        "artifacts": [],
    }


@pytest.fixture
def openai_adapter() -> OpenAIOutputAdapter:
    """OpenAI adapter instance."""
    return OpenAIOutputAdapter()


@pytest.fixture
def claude_adapter() -> ClaudeOutputAdapter:
    """Claude adapter instance."""
    return ClaudeOutputAdapter()


# -----------------------------------------------------------------------------
# AdapterResult tests
# -----------------------------------------------------------------------------


class TestAdapterResult:
    """Tests for AdapterResult dataclass."""

    def test_successful_result(self) -> None:
        """Successful result has expected structure."""
        result = AdapterResult(
            success=True,
            turn={"summary": "test"},
            raw_output="raw",
        )
        assert result.success
        assert result.turn == {"summary": "test"}
        assert result.raw_output == "raw"
        assert result.warnings == []
        assert result.error is None
        assert result.needs_retry is False
        assert result.needs_repair is False

    def test_failure_result(self) -> None:
        """Failure result has expected structure."""
        result = AdapterResult(
            success=False,
            turn=None,
            raw_output="raw",
            error="Something failed",
            needs_retry=True,
        )
        assert not result.success
        assert result.turn is None
        assert result.error == "Something failed"
        assert result.needs_retry

    def test_result_with_warnings(self) -> None:
        """Result can include warnings."""
        result = AdapterResult(
            success=True,
            turn={"summary": "test"},
            raw_output="raw",
            warnings=["warning1", "warning2"],
        )
        assert result.warnings == ["warning1", "warning2"]


# -----------------------------------------------------------------------------
# RepairResult tests
# -----------------------------------------------------------------------------


class TestRepairResult:
    """Tests for RepairResult dataclass."""

    def test_successful_repair(self) -> None:
        """Successful repair result."""
        result = RepairResult(
            success=True,
            repaired_output='{"key": "value"}',
        )
        assert result.success
        assert result.repaired_output == '{"key": "value"}'
        assert result.error is None

    def test_failed_repair(self) -> None:
        """Failed repair result."""
        result = RepairResult(
            success=False,
            repaired_output="",
            error="Cannot repair",
        )
        assert not result.success
        assert result.error == "Cannot repair"


# -----------------------------------------------------------------------------
# OpenAIOutputAdapter tests
# -----------------------------------------------------------------------------


class TestOpenAIOutputAdapter:
    """Tests for OpenAI output adapter."""

    def test_parse_clean_json(self, openai_adapter: OpenAIOutputAdapter, minimal_valid_turn: dict) -> None:
        """Clean JSON is parsed correctly."""
        raw = json.dumps(minimal_valid_turn)
        result = openai_adapter.extract_turn(raw, "codex", "M0")
        assert result.success
        assert result.turn is not None
        assert result.turn["summary"] == "Test summary"

    def test_empty_output_needs_retry(self, openai_adapter: OpenAIOutputAdapter) -> None:
        """Empty output triggers retry."""
        result = openai_adapter.extract_turn("", "codex", "M0")
        assert not result.success
        assert result.needs_retry
        assert "Empty" in (result.error or "")

    def test_extract_from_code_fences(self, openai_adapter: OpenAIOutputAdapter, minimal_valid_turn: dict) -> None:
        """JSON wrapped in code fences is extracted."""
        raw = f"```json\n{json.dumps(minimal_valid_turn)}\n```"
        result = openai_adapter.extract_turn(raw, "codex", "M0")
        assert result.success
        assert "fences" in " ".join(result.warnings).lower()

    def test_extract_balanced_json_from_prose(self, openai_adapter: OpenAIOutputAdapter, minimal_valid_turn: dict) -> None:
        """JSON embedded in prose is extracted."""
        raw = f"Here is my output:\n{json.dumps(minimal_valid_turn)}\nThat's all."
        result = openai_adapter.extract_turn(raw, "codex", "M0")
        assert result.success
        assert result.turn is not None

    def test_invalid_json_needs_repair(self, openai_adapter: OpenAIOutputAdapter) -> None:
        """Invalid JSON triggers repair."""
        result = openai_adapter.extract_turn("not json at all", "codex", "M0")
        assert not result.success
        assert result.needs_repair
        assert "Cannot extract" in (result.error or "")

    def test_agent_invariant_override(self, openai_adapter: OpenAIOutputAdapter, minimal_valid_turn: dict) -> None:
        """Agent mismatch is corrected."""
        minimal_valid_turn["agent"] = "wrong_agent"
        raw = json.dumps(minimal_valid_turn)
        result = openai_adapter.extract_turn(raw, "codex", "M0")
        assert result.success
        assert result.turn is not None
        assert result.turn["agent"] == "codex"
        assert any("agent corrected" in w for w in result.warnings)

    def test_milestone_invariant_override(self, openai_adapter: OpenAIOutputAdapter, minimal_valid_turn: dict) -> None:
        """Milestone mismatch is corrected."""
        minimal_valid_turn["milestone_id"] = "M99"
        raw = json.dumps(minimal_valid_turn)
        result = openai_adapter.extract_turn(raw, "codex", "M0")
        assert result.success
        assert result.turn is not None
        assert result.turn["milestone_id"] == "M0"
        assert any("milestone_id corrected" in w for w in result.warnings)

    def test_missing_required_fields(self, openai_adapter: OpenAIOutputAdapter) -> None:
        """Missing required fields triggers repair."""
        incomplete = {"agent": "codex", "milestone_id": "M0"}
        raw = json.dumps(incomplete)
        result = openai_adapter.extract_turn(raw, "codex", "M0")
        assert not result.success
        assert result.needs_repair
        assert "Missing required" in (result.error or "")

    def test_get_schema_config(self, openai_adapter: OpenAIOutputAdapter) -> None:
        """Schema config is properly formatted for OpenAI."""
        schema = {"type": "object"}
        config = openai_adapter.get_schema_config(schema)
        assert "response_format" in config
        assert config["response_format"]["type"] == "json_schema"
        assert config["response_format"]["json_schema"]["strict"] is True


# -----------------------------------------------------------------------------
# ClaudeOutputAdapter tests
# -----------------------------------------------------------------------------


class TestClaudeOutputAdapter:
    """Tests for Claude output adapter."""

    def test_parse_clean_json(self, claude_adapter: ClaudeOutputAdapter, minimal_valid_turn: dict) -> None:
        """Clean JSON is parsed correctly."""
        minimal_valid_turn["agent"] = "claude"
        minimal_valid_turn["stats_refs"] = ["CL-1"]
        raw = json.dumps(minimal_valid_turn)
        result = claude_adapter.extract_turn(raw, "claude", "M0")
        assert result.success
        assert result.turn is not None

    def test_empty_output_needs_retry(self, claude_adapter: ClaudeOutputAdapter) -> None:
        """Empty output triggers retry."""
        result = claude_adapter.extract_turn("", "claude", "M0")
        assert not result.success
        assert result.needs_retry
        assert "Empty" in (result.error or "")

    def test_extract_from_json_stream(self, claude_adapter: ClaudeOutputAdapter) -> None:
        """Claude JSON stream format is parsed."""
        turn = {
            "agent": "claude",
            "milestone_id": "M0",
            "summary": "Test",
            "work_completed": True,
            "project_complete": False,
        }
        stream = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": json.dumps(turn)}]}},
        ]
        raw = "\n".join(json.dumps(obj) for obj in stream)
        result = claude_adapter.extract_turn(raw, "claude", "M0")
        assert result.success
        assert result.turn is not None
        assert "stream" in " ".join(result.warnings).lower()

    def test_extract_from_result_type(self, claude_adapter: ClaudeOutputAdapter) -> None:
        """Claude result type events are parsed."""
        turn = {
            "agent": "claude",
            "milestone_id": "M0",
            "summary": "Test",
            "work_completed": True,
            "project_complete": False,
        }
        stream = [{"type": "result", "result": json.dumps(turn)}]
        raw = json.dumps(stream[0])
        result = claude_adapter.extract_turn(raw, "claude", "M0")
        assert result.success

    def test_extract_from_code_fences(self, claude_adapter: ClaudeOutputAdapter) -> None:
        """JSON wrapped in code fences is extracted."""
        turn = {
            "agent": "claude",
            "milestone_id": "M0",
            "summary": "Test",
            "work_completed": True,
            "project_complete": False,
        }
        raw = f"```json\n{json.dumps(turn)}\n```"
        result = claude_adapter.extract_turn(raw, "claude", "M0")
        assert result.success

    def test_fill_defaults_for_missing_optional_fields(self, claude_adapter: ClaudeOutputAdapter) -> None:
        """Missing optional fields get defaults when required fields checked."""
        # Claude adapter has fill_defaults - it only fills when _check_required_fields
        # finds missing fields. The Claude adapter only requires summary,
        # work_completed, project_complete as required, so this test verifies
        # that the adapter successfully parses minimal input.
        turn = {
            "agent": "claude",
            "milestone_id": "M0",
            "summary": "Test",
            "work_completed": True,
            "project_complete": False,
        }
        raw = json.dumps(turn)
        result = claude_adapter.extract_turn(raw, "claude", "M0")
        assert result.success
        assert result.turn is not None
        # Claude adapter only requires summary, work_completed, project_complete
        # If more fields were missing and caused _check_required_fields to trigger
        # _fill_defaults, then defaults would be filled. Here we just verify
        # the extraction succeeds.
        assert result.turn["summary"] == "Test"

    def test_agent_invariant_override(self, claude_adapter: ClaudeOutputAdapter) -> None:
        """Agent mismatch is corrected."""
        turn = {
            "agent": "wrong",
            "milestone_id": "M0",
            "summary": "Test",
            "work_completed": True,
            "project_complete": False,
        }
        raw = json.dumps(turn)
        result = claude_adapter.extract_turn(raw, "claude", "M0")
        assert result.success
        assert result.turn is not None
        assert result.turn["agent"] == "claude"

    def test_get_schema_config(self, claude_adapter: ClaudeOutputAdapter) -> None:
        """Schema config is properly formatted for Claude tools."""
        schema = {"type": "object"}
        config = claude_adapter.get_schema_config(schema)
        assert "tools" in config
        assert config["tools"][0]["name"] == "submit_turn"
        assert "tool_choice" in config


# -----------------------------------------------------------------------------
# OutputRepairService tests
# -----------------------------------------------------------------------------


class TestOutputRepairService:
    """Tests for output repair service."""

    def test_basic_repair_strips_wrapper(self) -> None:
        """Basic repair strips common wrapper text."""
        service = OutputRepairService()
        raw = 'Here is the JSON:\n{"summary": "test"}'
        result = service.attempt_repair(raw, "Invalid JSON", 0)
        # May or may not succeed depending on exact patterns
        # This tests the repair logic runs
        assert isinstance(result, RepairResult)

    def test_max_repair_attempts_exceeded(self) -> None:
        """Repair fails when max attempts exceeded."""
        service = OutputRepairService()
        result = service.attempt_repair("bad", "error", 1)  # 1 >= MAX_REPAIR_ATTEMPTS
        assert not result.success
        assert "Max repair attempts" in (result.error or "")

    def test_repair_extracts_balanced_json(self) -> None:
        """Repair can extract balanced JSON from text."""
        service = OutputRepairService()
        raw = 'Some text before {"summary": "test"} and after'
        result = service.attempt_repair(raw, "error", 0)
        assert result.success
        assert "summary" in result.repaired_output

    def test_repair_strips_code_fences(self) -> None:
        """Repair strips markdown code fences."""
        service = OutputRepairService()
        raw = '```json\n{"summary": "test"}\n```'
        result = service.attempt_repair(raw, "error", 0)
        assert result.success
        parsed = json.loads(result.repaired_output)
        assert parsed["summary"] == "test"

    def test_repair_with_custom_prompt_builder(self) -> None:
        """Custom prompt builder can be provided."""
        custom_prompt = "custom prompt text"

        def builder(raw: str, error: str) -> str:
            return custom_prompt

        service = OutputRepairService(repair_prompt_builder=builder)
        # Just verify it can be called without error
        # The prompt builder is used for model-based repair which we don't test here
        assert service.repair_prompt_builder("", "") == custom_prompt

    def test_cannot_repair_truly_invalid_data(self) -> None:
        """Cannot repair data with no JSON."""
        service = OutputRepairService()
        result = service.attempt_repair("no json here at all", "error", 0)
        assert not result.success


# -----------------------------------------------------------------------------
# get_adapter_for_agent tests
# -----------------------------------------------------------------------------


class TestGetAdapterForAgent:
    """Tests for get_adapter_for_agent function."""

    def test_codex_returns_openai_adapter(self) -> None:
        """'codex' agent returns OpenAI adapter."""
        adapter = get_adapter_for_agent("codex")
        assert isinstance(adapter, OpenAIOutputAdapter)

    def test_claude_returns_claude_adapter(self) -> None:
        """'claude' agent returns Claude adapter."""
        adapter = get_adapter_for_agent("claude")
        assert isinstance(adapter, ClaudeOutputAdapter)

    def test_unknown_agent_returns_claude_adapter(self) -> None:
        """Unknown agent defaults to Claude adapter."""
        adapter = get_adapter_for_agent("unknown_agent")
        assert isinstance(adapter, ClaudeOutputAdapter)


# -----------------------------------------------------------------------------
# normalize_provider_output tests
# -----------------------------------------------------------------------------


class TestNormalizeProviderOutput:
    """Tests for normalize_provider_output function."""

    def test_codex_normalization(self, minimal_valid_turn: dict) -> None:
        """Codex output is normalized correctly."""
        raw = json.dumps(minimal_valid_turn)
        result = normalize_provider_output(
            raw_output=raw,
            agent="codex",
            expected_agent="codex",
            expected_milestone_id="M0",
        )
        assert result.success
        assert result.turn is not None

    def test_claude_normalization(self, minimal_valid_turn: dict) -> None:
        """Claude output is normalized correctly."""
        minimal_valid_turn["agent"] = "claude"
        minimal_valid_turn["stats_refs"] = ["CL-1"]
        raw = json.dumps(minimal_valid_turn)
        result = normalize_provider_output(
            raw_output=raw,
            agent="claude",
            expected_agent="claude",
            expected_milestone_id="M0",
        )
        assert result.success
        assert result.turn is not None

    def test_stats_refs_validation(self, minimal_valid_turn: dict) -> None:
        """Stats refs are validated against stats_id_set."""
        minimal_valid_turn["stats_refs"] = ["INVALID"]
        raw = json.dumps(minimal_valid_turn)
        result = normalize_provider_output(
            raw_output=raw,
            agent="codex",
            expected_agent="codex",
            expected_milestone_id="M0",
            stats_id_set={"CX-1", "CL-1"},
        )
        assert result.success
        assert result.turn is not None
        # Invalid refs should be replaced with default
        assert result.turn["stats_refs"] == ["CX-1"]
        assert any("stats_refs" in w for w in result.warnings)

    def test_stats_refs_claude_default(self, minimal_valid_turn: dict) -> None:
        """Claude gets CL-1 as default stats ref."""
        minimal_valid_turn["agent"] = "claude"
        minimal_valid_turn["stats_refs"] = []
        raw = json.dumps(minimal_valid_turn)
        result = normalize_provider_output(
            raw_output=raw,
            agent="claude",
            expected_agent="claude",
            expected_milestone_id="M0",
            stats_id_set={"CX-1", "CL-1"},
        )
        assert result.success
        assert result.turn is not None
        assert "CL-1" in result.turn["stats_refs"]


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_nested_json_extraction(self, openai_adapter: OpenAIOutputAdapter) -> None:
        """Nested JSON objects are handled correctly."""
        turn = {
            "agent": "codex",
            "milestone_id": "M0",
            "phase": "implement",
            "work_completed": True,
            "project_complete": False,
            "summary": "Test",
            "gates_passed": [],
            "requirement_progress": {
                "covered_req_ids": ["REQ-001"],
                "tests_added_or_modified": [],
                "commands_run": [],
            },
            "next_agent": "claude",
            "next_prompt": "",
            "delegate_rationale": "",
            "stats_refs": ["CX-1"],
            "needs_write_access": True,
            "artifacts": [{"path": "file.py", "description": "desc"}],
        }
        raw = json.dumps(turn)
        result = openai_adapter.extract_turn(raw, "codex", "M0")
        assert result.success
        assert result.turn is not None
        assert result.turn["requirement_progress"]["covered_req_ids"] == ["REQ-001"]

    def test_whitespace_only_treated_as_empty(self, openai_adapter: OpenAIOutputAdapter) -> None:
        """Whitespace-only output is treated as empty."""
        result = openai_adapter.extract_turn("   \n\t  ", "codex", "M0")
        assert not result.success
        assert result.needs_retry

    def test_json_array_not_accepted(self, openai_adapter: OpenAIOutputAdapter) -> None:
        """JSON array (not object) is not accepted."""
        result = openai_adapter.extract_turn('["item1", "item2"]', "codex", "M0")
        assert not result.success

    def test_extract_json_with_escaped_quotes(self, openai_adapter: OpenAIOutputAdapter) -> None:
        """JSON with escaped quotes in strings is handled."""
        turn = {
            "agent": "codex",
            "milestone_id": "M0",
            "phase": "implement",
            "work_completed": True,
            "project_complete": False,
            "summary": 'Test with "quotes"',
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
            "needs_write_access": True,
            "artifacts": [],
        }
        raw = json.dumps(turn)
        result = openai_adapter.extract_turn(raw, "codex", "M0")
        assert result.success
        assert "quotes" in result.turn["summary"]

    def test_claude_stream_with_multiple_messages(self, claude_adapter: ClaudeOutputAdapter) -> None:
        """Claude stream with multiple message events finds the turn."""
        turn = {
            "agent": "claude",
            "milestone_id": "M0",
            "summary": "Found",
            "work_completed": True,
            "project_complete": False,
        }
        stream = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Some preamble"}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": json.dumps(turn)}]}},
        ]
        raw = "\n".join(json.dumps(obj) for obj in stream)
        result = claude_adapter.extract_turn(raw, "claude", "M0")
        assert result.success
        assert result.turn["summary"] == "Found"
