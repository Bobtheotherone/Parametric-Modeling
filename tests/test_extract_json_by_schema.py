"""Tests for extract_json_by_schema.py helper.

These tests verify that the JSON extraction helper correctly parses Claude CLI
output streams and extracts valid JSON objects matching the requested schema.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the extraction script as a module
import importlib.util

extract_script_path = PROJECT_ROOT / "scripts" / "extract_json_by_schema.py"
spec = importlib.util.spec_from_file_location("extract_json_by_schema", extract_script_path)
extract_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_module)

# Get functions from the module
extract_from_claude_stream = extract_module.extract_from_claude_stream
validate_against_schema = extract_module.validate_against_schema
find_all_json_objects = extract_module.find_all_json_objects
strip_fences = extract_module.strip_fences


# -----------------------------
# Test fixtures
# -----------------------------

@pytest.fixture
def task_plan_schema() -> Dict[str, Any]:
    """Load the actual task_plan.schema.json for testing."""
    schema_path = PROJECT_ROOT / "bridge" / "task_plan.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


@pytest.fixture
def turn_schema() -> Dict[str, Any]:
    """Load the actual turn.schema.json for testing."""
    schema_path = PROJECT_ROOT / "bridge" / "turn.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


@pytest.fixture
def valid_task_plan() -> Dict[str, Any]:
    """A valid task plan object."""
    return {
        "milestone_id": "M2",
        "max_parallel_tasks": 4,
        "rationale": "Breaking down the work into parallel tasks",
        "tasks": [
            {
                "id": "M2-TASK-1",
                "title": "Implement feature A",
                "description": "Add feature A to the system",
                "preferred_agent": "codex",
                "estimated_intensity": "medium",
                "locks": [],
                "depends_on": [],
                "solo": False
            }
        ]
    }


@pytest.fixture
def valid_turn() -> Dict[str, Any]:
    """A valid turn object."""
    return {
        "agent": "claude",
        "milestone_id": "M1",
        "phase": "implement",
        "work_completed": True,
        "project_complete": False,
        "summary": "Implemented the feature",
        "gates_passed": ["ruff", "mypy"],
        "requirement_progress": {
            "covered_req_ids": ["REQ-1"],
            "tests_added_or_modified": ["test_feature.py"],
            "commands_run": ["pytest tests/"]
        },
        "next_agent": "codex",
        "next_prompt": "Continue with the next task",
        "delegate_rationale": "Handing off to codex for implementation",
        "stats_refs": ["CL-1"],
        "needs_write_access": True,
        "artifacts": []
    }


# -----------------------------
# Test strip_fences
# -----------------------------

class TestStripFences:
    """Test markdown code fence removal."""

    def test_no_fences(self) -> None:
        """Text without fences is returned as-is."""
        assert strip_fences('{"key": "value"}') == '{"key": "value"}'

    def test_json_fences(self) -> None:
        """JSON code fences are removed."""
        text = '```json\n{"key": "value"}\n```'
        assert strip_fences(text) == '{"key": "value"}'

    def test_plain_fences(self) -> None:
        """Plain code fences are removed."""
        text = '```\n{"key": "value"}\n```'
        assert strip_fences(text) == '{"key": "value"}'

    def test_whitespace_handling(self) -> None:
        """Whitespace is trimmed."""
        text = '  \n```json\n{"key": "value"}\n```  \n'
        assert strip_fences(text) == '{"key": "value"}'


# -----------------------------
# Test find_all_json_objects
# -----------------------------

class TestFindAllJsonObjects:
    """Test finding JSON objects in text."""

    def test_single_object(self) -> None:
        """Finds a single JSON object."""
        text = '{"key": "value"}'
        objects = find_all_json_objects(text)
        assert len(objects) == 1
        assert objects[0] == '{"key": "value"}'

    def test_multiple_objects(self) -> None:
        """Finds multiple JSON objects."""
        text = '{"a": 1}  {"b": 2}  {"c": 3}'
        objects = find_all_json_objects(text)
        assert len(objects) == 3

    def test_nested_objects(self) -> None:
        """Handles nested objects correctly."""
        text = '{"outer": {"inner": "value"}}'
        objects = find_all_json_objects(text)
        assert len(objects) == 1
        assert json.loads(objects[0]) == {"outer": {"inner": "value"}}

    def test_objects_with_strings_containing_braces(self) -> None:
        """Handles strings containing braces."""
        text = '{"code": "function() { return {}; }"}'
        objects = find_all_json_objects(text)
        assert len(objects) == 1
        parsed = json.loads(objects[0])
        assert parsed["code"] == "function() { return {}; }"


# -----------------------------
# Test validation
# -----------------------------

class TestValidateAgainstSchema:
    """Test JSON schema validation."""

    def test_valid_task_plan(
        self, task_plan_schema: Dict[str, Any], valid_task_plan: Dict[str, Any]
    ) -> None:
        """Valid task plan passes validation."""
        assert validate_against_schema(valid_task_plan, task_plan_schema)

    def test_invalid_task_plan_missing_key(
        self, task_plan_schema: Dict[str, Any]
    ) -> None:
        """Task plan missing required key fails validation."""
        invalid = {"milestone_id": "M2"}  # Missing other required keys
        assert not validate_against_schema(invalid, task_plan_schema)

    def test_valid_turn(
        self, turn_schema: Dict[str, Any], valid_turn: Dict[str, Any]
    ) -> None:
        """Valid turn passes validation."""
        assert validate_against_schema(valid_turn, turn_schema)

    def test_invalid_turn_extra_property(
        self, turn_schema: Dict[str, Any], valid_turn: Dict[str, Any]
    ) -> None:
        """Turn with extra property fails validation (additionalProperties: false)."""
        invalid = {**valid_turn, "extra_key": "not allowed"}
        # additionalProperties: false should cause this to fail
        assert not validate_against_schema(invalid, turn_schema)


# -----------------------------
# Test extraction from Claude stream
# -----------------------------

class TestExtractFromClaudeStream:
    """Test extraction from Claude CLI stream output."""

    def test_extract_direct_object(
        self, task_plan_schema: Dict[str, Any], valid_task_plan: Dict[str, Any]
    ) -> None:
        """Extracts a direct JSON object matching schema."""
        text = json.dumps(valid_task_plan)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"

    def test_extract_from_result_event(
        self, task_plan_schema: Dict[str, Any], valid_task_plan: Dict[str, Any]
    ) -> None:
        """Extracts JSON from a Claude CLI result event."""
        result_event = {
            "type": "result",
            "result": json.dumps(valid_task_plan)
        }
        text = json.dumps(result_event)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"

    def test_extract_from_assistant_message(
        self, task_plan_schema: Dict[str, Any], valid_task_plan: Dict[str, Any]
    ) -> None:
        """Extracts JSON from a Claude CLI assistant message."""
        assistant_event = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(valid_task_plan)
                    }
                ]
            }
        }
        text = json.dumps(assistant_event)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"

    def test_extract_from_code_fenced_result(
        self, task_plan_schema: Dict[str, Any], valid_task_plan: Dict[str, Any]
    ) -> None:
        """Extracts JSON from code-fenced content in result."""
        result_event = {
            "type": "result",
            "result": f"```json\n{json.dumps(valid_task_plan)}\n```"
        }
        text = json.dumps(result_event)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"

    def test_prefers_last_valid_candidate(
        self, task_plan_schema: Dict[str, Any], valid_task_plan: Dict[str, Any]
    ) -> None:
        """When multiple valid objects exist, prefers the last one."""
        plan1 = {**valid_task_plan, "rationale": "First plan"}
        plan2 = {**valid_task_plan, "rationale": "Second plan"}

        text = json.dumps(plan1) + "\n" + json.dumps(plan2)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["rationale"] == "Second plan"

    def test_ignores_invalid_objects(
        self, task_plan_schema: Dict[str, Any], valid_task_plan: Dict[str, Any]
    ) -> None:
        """Ignores objects that don't match schema."""
        invalid = {"not": "a task plan"}
        text = json.dumps(invalid) + "\n" + json.dumps(valid_task_plan)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"

    def test_returns_none_when_no_match(
        self, task_plan_schema: Dict[str, Any]
    ) -> None:
        """Returns None when no valid object is found."""
        text = '{"not": "matching"} {"also": "invalid"}'
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is None


# -----------------------------
# Test realistic Claude CLI output
# -----------------------------

class TestRealisticClaudeOutput:
    """Test with realistic Claude CLI output formats."""

    def test_jsonlines_stream_with_init_and_result(
        self, task_plan_schema: Dict[str, Any], valid_task_plan: Dict[str, Any]
    ) -> None:
        """Handles realistic JSON Lines stream with init and result events."""
        stream = (
            '{"type":"init","session_id":"abc123"}\n'
            '{"type":"assistant","message":{"content":[{"type":"text","text":"Let me plan..."}]}}\n'
            f'{{"type":"result","result":{json.dumps(json.dumps(valid_task_plan))}}}\n'
        )
        result = extract_from_claude_stream(stream, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"

    def test_handles_array_wrapper(
        self, task_plan_schema: Dict[str, Any], valid_task_plan: Dict[str, Any]
    ) -> None:
        """Handles output wrapped in an array."""
        events = [
            {"type": "init", "session_id": "abc"},
            {"type": "result", "result": json.dumps(valid_task_plan)}
        ]
        text = json.dumps(events)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"


# -----------------------------
# Integration tests
# -----------------------------

class TestScriptIntegration:
    """Test the script as a whole."""

    def test_main_function_success(
        self, task_plan_schema: Dict[str, Any], valid_task_plan: Dict[str, Any]
    ) -> None:
        """Test main() function with valid input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write test files
            raw_path = tmppath / "raw.txt"
            schema_path = tmppath / "schema.json"
            out_path = tmppath / "out.json"

            raw_path.write_text(json.dumps(valid_task_plan), encoding="utf-8")
            schema_path.write_text(json.dumps(task_plan_schema), encoding="utf-8")

            # Run extraction
            result = extract_from_claude_stream(
                raw_path.read_text(encoding="utf-8"),
                task_plan_schema
            )

            assert result is not None
            assert result["milestone_id"] == "M2"

    def test_turn_schema_extraction(
        self, turn_schema: Dict[str, Any], valid_turn: Dict[str, Any]
    ) -> None:
        """Test extraction works for turn schema too."""
        text = json.dumps(valid_turn)
        result = extract_from_claude_stream(text, turn_schema)
        assert result is not None
        assert result["agent"] == "claude"
        assert result["milestone_id"] == "M1"


# -----------------------------
# ORCH_SCHEMA_KIND environment variable tests
# -----------------------------

class TestOrchSchemaKindBehavior:
    """Test that ORCH_SCHEMA_KIND affects wrapper behavior.

    These tests verify the contract between loop.py and claude.sh:
    - ORCH_SCHEMA_KIND=task_plan should skip turn normalization
    - ORCH_SCHEMA_KIND=turn should use turn normalization
    """

    def test_task_plan_schema_has_different_keys_than_turn(
        self, task_plan_schema: Dict[str, Any], turn_schema: Dict[str, Any]
    ) -> None:
        """Verify task_plan and turn schemas have different required keys."""
        task_plan_required = set(task_plan_schema.get("required", []))
        turn_required = set(turn_schema.get("required", []))

        # They should have no overlap (different schemas for different purposes)
        overlap = task_plan_required & turn_required
        # milestone_id is the only expected overlap
        assert overlap == {"milestone_id"}, f"Unexpected overlap: {overlap}"

    def test_turn_normalized_output_fails_task_plan_schema(
        self, task_plan_schema: Dict[str, Any]
    ) -> None:
        """A turn-normalized output should fail task_plan schema validation.

        This is the root cause of the bug: turn normalization adds keys like
        'agent', 'phase', 'next_agent' which are not allowed in task_plan schema.
        """
        # This is what turn normalization produces (wrong for task_plan)
        turn_normalized = {
            "agent": "claude",
            "milestone_id": "M2",
            "phase": "plan",
            "work_completed": False,
            "project_complete": False,
            "summary": "Some summary",
            "gates_passed": [],
            "requirement_progress": {
                "covered_req_ids": [],
                "tests_added_or_modified": [],
                "commands_run": []
            },
            "next_agent": "codex",
            "next_prompt": "Continue",
            "delegate_rationale": "",
            "stats_refs": ["CL-1"],
            "needs_write_access": True,
            "artifacts": []
        }

        # This should FAIL task_plan validation (additional properties not allowed)
        assert not validate_against_schema(turn_normalized, task_plan_schema)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
