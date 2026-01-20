"""Tests for extract_json_by_schema.py helper.

These tests verify that the JSON extraction helper correctly parses Claude CLI
output streams and extracts valid JSON objects matching the requested schema.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

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
normalize_task_plan = extract_module.normalize_task_plan
is_task_plan_schema = extract_module.is_task_plan_schema


# -----------------------------
# Test fixtures
# -----------------------------


@pytest.fixture
def task_plan_schema() -> dict[str, Any]:
    """Load the actual task_plan.schema.json for testing."""
    schema_path = PROJECT_ROOT / "bridge" / "task_plan.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


@pytest.fixture
def turn_schema() -> dict[str, Any]:
    """Load the actual turn.schema.json for testing."""
    schema_path = PROJECT_ROOT / "bridge" / "turn.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


@pytest.fixture
def valid_task_plan() -> dict[str, Any]:
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
                "solo": False,
            }
        ],
    }


@pytest.fixture
def valid_turn() -> dict[str, Any]:
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
            "commands_run": ["pytest tests/"],
        },
        "next_agent": "codex",
        "next_prompt": "Continue with the next task",
        "delegate_rationale": "Handing off to codex for implementation",
        "stats_refs": ["CL-1"],
        "needs_write_access": True,
        "artifacts": [],
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

    def test_valid_task_plan(self, task_plan_schema: dict[str, Any], valid_task_plan: dict[str, Any]) -> None:
        """Valid task plan passes validation."""
        assert validate_against_schema(valid_task_plan, task_plan_schema)

    def test_invalid_task_plan_missing_key(self, task_plan_schema: dict[str, Any]) -> None:
        """Task plan missing required key fails validation."""
        invalid = {"milestone_id": "M2"}  # Missing other required keys
        assert not validate_against_schema(invalid, task_plan_schema)

    def test_valid_turn(self, turn_schema: dict[str, Any], valid_turn: dict[str, Any]) -> None:
        """Valid turn passes validation."""
        assert validate_against_schema(valid_turn, turn_schema)

    def test_invalid_turn_extra_property(self, turn_schema: dict[str, Any], valid_turn: dict[str, Any]) -> None:
        """Turn with extra property fails validation (additionalProperties: false)."""
        invalid = {**valid_turn, "extra_key": "not allowed"}
        # additionalProperties: false should cause this to fail
        assert not validate_against_schema(invalid, turn_schema)


# -----------------------------
# Test extraction from Claude stream
# -----------------------------


class TestExtractFromClaudeStream:
    """Test extraction from Claude CLI stream output."""

    def test_extract_direct_object(self, task_plan_schema: dict[str, Any], valid_task_plan: dict[str, Any]) -> None:
        """Extracts a direct JSON object matching schema."""
        text = json.dumps(valid_task_plan)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"

    def test_extract_from_result_event(self, task_plan_schema: dict[str, Any], valid_task_plan: dict[str, Any]) -> None:
        """Extracts JSON from a Claude CLI result event."""
        result_event = {"type": "result", "result": json.dumps(valid_task_plan)}
        text = json.dumps(result_event)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"

    def test_extract_from_assistant_message(self, task_plan_schema: dict[str, Any], valid_task_plan: dict[str, Any]) -> None:
        """Extracts JSON from a Claude CLI assistant message."""
        assistant_event = {"type": "assistant", "message": {"content": [{"type": "text", "text": json.dumps(valid_task_plan)}]}}
        text = json.dumps(assistant_event)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"

    def test_extract_from_code_fenced_result(self, task_plan_schema: dict[str, Any], valid_task_plan: dict[str, Any]) -> None:
        """Extracts JSON from code-fenced content in result."""
        result_event = {"type": "result", "result": f"```json\n{json.dumps(valid_task_plan)}\n```"}
        text = json.dumps(result_event)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"

    def test_prefers_last_valid_candidate(self, task_plan_schema: dict[str, Any], valid_task_plan: dict[str, Any]) -> None:
        """When multiple valid objects exist, prefers the last one."""
        plan1 = {**valid_task_plan, "rationale": "First plan"}
        plan2 = {**valid_task_plan, "rationale": "Second plan"}

        text = json.dumps(plan1) + "\n" + json.dumps(plan2)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["rationale"] == "Second plan"

    def test_ignores_invalid_objects(self, task_plan_schema: dict[str, Any], valid_task_plan: dict[str, Any]) -> None:
        """Ignores objects that don't match schema."""
        invalid = {"not": "a task plan"}
        text = json.dumps(invalid) + "\n" + json.dumps(valid_task_plan)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"

    def test_returns_none_when_no_match(self, task_plan_schema: dict[str, Any]) -> None:
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
        self, task_plan_schema: dict[str, Any], valid_task_plan: dict[str, Any]
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

    def test_handles_array_wrapper(self, task_plan_schema: dict[str, Any], valid_task_plan: dict[str, Any]) -> None:
        """Handles output wrapped in an array."""
        events = [{"type": "init", "session_id": "abc"}, {"type": "result", "result": json.dumps(valid_task_plan)}]
        text = json.dumps(events)
        result = extract_from_claude_stream(text, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"


# -----------------------------
# Integration tests
# -----------------------------


class TestScriptIntegration:
    """Test the script as a whole."""

    def test_main_function_success(self, task_plan_schema: dict[str, Any], valid_task_plan: dict[str, Any]) -> None:
        """Test main() function with valid input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write test files
            raw_path = tmppath / "raw.txt"
            schema_path = tmppath / "schema.json"
            tmppath / "out.json"

            raw_path.write_text(json.dumps(valid_task_plan), encoding="utf-8")
            schema_path.write_text(json.dumps(task_plan_schema), encoding="utf-8")

            # Run extraction
            result = extract_from_claude_stream(raw_path.read_text(encoding="utf-8"), task_plan_schema)

            assert result is not None
            assert result["milestone_id"] == "M2"

    def test_turn_schema_extraction(self, turn_schema: dict[str, Any], valid_turn: dict[str, Any]) -> None:
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
        self, task_plan_schema: dict[str, Any], turn_schema: dict[str, Any]
    ) -> None:
        """Verify task_plan and turn schemas have different required keys."""
        task_plan_required = set(task_plan_schema.get("required", []))
        turn_required = set(turn_schema.get("required", []))

        # They should have no overlap (different schemas for different purposes)
        overlap = task_plan_required & turn_required
        # milestone_id is the only expected overlap
        assert overlap == {"milestone_id"}, f"Unexpected overlap: {overlap}"

    def test_turn_normalized_output_fails_task_plan_schema(self, task_plan_schema: dict[str, Any]) -> None:
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
            "requirement_progress": {"covered_req_ids": [], "tests_added_or_modified": [], "commands_run": []},
            "next_agent": "codex",
            "next_prompt": "Continue",
            "delegate_rationale": "",
            "stats_refs": ["CL-1"],
            "needs_write_access": True,
            "artifacts": [],
        }

        # This should FAIL task_plan validation (additional properties not allowed)
        assert not validate_against_schema(turn_normalized, task_plan_schema)


# -----------------------------
# Task Plan Normalization tests
# -----------------------------


class TestTaskPlanNormalization:
    """Test task plan normalization for common model drift issues."""

    def test_is_task_plan_schema_by_title(self) -> None:
        """Detect task plan schema by title."""
        schema = {"title": "ParallelTaskPlan", "required": []}
        assert is_task_plan_schema(schema)

    def test_is_task_plan_schema_by_required_keys(self) -> None:
        """Detect task plan schema by required keys."""
        schema = {"required": ["milestone_id", "max_parallel_tasks", "tasks", "rationale"]}
        assert is_task_plan_schema(schema)

    def test_normalize_intensity_to_estimated_intensity(self) -> None:
        """Normalize 'intensity' to 'estimated_intensity'."""
        plan = {
            "milestone_id": "M1",
            "max_parallel_tasks": 4,
            "rationale": "Test",
            "tasks": [
                {
                    "id": "T1",
                    "title": "Task 1",
                    "description": "Desc",
                    "preferred_agent": "codex",
                    "intensity": "low",  # Wrong key
                    "locks": [],
                    "solo": False,
                    # Note: depends_on is missing
                }
            ],
        }
        normalized = normalize_task_plan(plan)
        task = normalized["tasks"][0]
        assert "estimated_intensity" in task
        assert task["estimated_intensity"] == "low"
        assert "intensity" not in task

    def test_normalize_dependencies_to_depends_on(self) -> None:
        """Normalize 'dependencies' to 'depends_on'."""
        plan = {
            "milestone_id": "M1",
            "max_parallel_tasks": 4,
            "rationale": "Test",
            "tasks": [
                {
                    "id": "T1",
                    "title": "Task 1",
                    "description": "Desc",
                    "preferred_agent": "codex",
                    "estimated_intensity": "medium",
                    "locks": [],
                    "dependencies": ["T0"],  # Wrong key
                    "solo": False,
                }
            ],
        }
        normalized = normalize_task_plan(plan)
        task = normalized["tasks"][0]
        assert "depends_on" in task
        assert task["depends_on"] == ["T0"]
        assert "dependencies" not in task

    def test_normalize_agent_to_preferred_agent(self) -> None:
        """Normalize 'agent' to 'preferred_agent'."""
        plan = {
            "milestone_id": "M1",
            "max_parallel_tasks": 4,
            "rationale": "Test",
            "tasks": [
                {
                    "id": "T1",
                    "title": "Task 1",
                    "description": "Desc",
                    "agent": "claude",  # Wrong key
                    "estimated_intensity": "high",
                    "locks": [],
                    "depends_on": [],
                    "solo": True,
                }
            ],
        }
        normalized = normalize_task_plan(plan)
        task = normalized["tasks"][0]
        assert "preferred_agent" in task
        assert task["preferred_agent"] == "claude"
        assert "agent" not in task

    def test_normalize_adds_missing_defaults(self) -> None:
        """Add default values for missing required fields."""
        plan = {
            "milestone_id": "M1",
            "max_parallel_tasks": 4,
            "rationale": "Test",
            "tasks": [
                {
                    "id": "T1",
                    "title": "Task 1",
                    "description": "Desc",
                    "preferred_agent": "codex",
                    "estimated_intensity": "low",
                    # Missing: locks, depends_on, solo
                }
            ],
        }
        normalized = normalize_task_plan(plan)
        task = normalized["tasks"][0]
        assert task["locks"] == []
        assert task["depends_on"] == []
        assert task["solo"] is False

    def test_normalize_removes_unknown_keys(self) -> None:
        """Remove keys not allowed by schema (additionalProperties=false)."""
        plan = {
            "milestone_id": "M1",
            "max_parallel_tasks": 4,
            "rationale": "Test",
            "extra_top_key": "should be removed",
            "tasks": [
                {
                    "id": "T1",
                    "title": "Task 1",
                    "description": "Desc",
                    "preferred_agent": "codex",
                    "estimated_intensity": "low",
                    "locks": [],
                    "depends_on": [],
                    "solo": False,
                    "extra_task_key": "should be removed",
                }
            ],
        }
        normalized = normalize_task_plan(plan)
        assert "extra_top_key" not in normalized
        assert "extra_task_key" not in normalized["tasks"][0]

    def test_normalize_unwraps_task_plan_wrapper(self) -> None:
        """Unwrap {"task_plan": {...}} wrapper."""
        inner_plan = {
            "milestone_id": "M1",
            "max_parallel_tasks": 4,
            "rationale": "Test",
            "tasks": [
                {
                    "id": "T1",
                    "title": "Task 1",
                    "description": "Desc",
                    "preferred_agent": "codex",
                    "estimated_intensity": "low",
                    "locks": [],
                    "depends_on": [],
                    "solo": False,
                }
            ],
        }
        wrapped = {"task_plan": inner_plan}
        normalized = normalize_task_plan(wrapped)
        assert "task_plan" not in normalized
        assert normalized["milestone_id"] == "M1"

    def test_extraction_normalizes_drifted_plan(self, task_plan_schema: dict[str, Any]) -> None:
        """Full extraction test: plan with wrong keys gets normalized and validates."""
        # This simulates what Claude might output with model drift
        drifted_plan = {
            "milestone_id": "M2",
            "max_parallel_tasks": 4,
            "rationale": "Breaking down the work",
            "tasks": [
                {
                    "id": "M2-TASK-1",
                    "title": "Implement feature",
                    "description": "Add feature to system",
                    "agent": "codex",  # Wrong: should be preferred_agent
                    "intensity": "medium",  # Wrong: should be estimated_intensity
                    "locks": ["M2-core"],
                    # Missing: depends_on, solo
                }
            ],
        }

        # The raw plan should NOT validate
        assert not validate_against_schema(drifted_plan, task_plan_schema)

        # But extraction should normalize it and return a valid plan
        stream = json.dumps(drifted_plan)
        result = extract_from_claude_stream(stream, task_plan_schema)

        assert result is not None
        assert result["milestone_id"] == "M2"

        # Check normalized task
        task = result["tasks"][0]
        assert task["preferred_agent"] == "codex"
        assert task["estimated_intensity"] == "medium"
        assert task["depends_on"] == []
        assert task["solo"] is False
        assert "agent" not in task
        assert "intensity" not in task

        # Final result should validate
        assert validate_against_schema(result, task_plan_schema)

    def test_extraction_normalizes_wrapped_plan(self, task_plan_schema: dict[str, Any]) -> None:
        """Extraction handles {"task_plan": {...}} wrapper."""
        inner_plan = {
            "milestone_id": "M3",
            "max_parallel_tasks": 2,
            "rationale": "Wrapped plan test",
            "tasks": [
                {
                    "id": "M3-T1",
                    "title": "Task",
                    "description": "Description",
                    "preferred_agent": "claude",
                    "intensity": "high",  # Wrong key
                    "locks": [],
                    # Missing depends_on, solo
                }
            ],
        }
        wrapped = {"task_plan": inner_plan}

        stream = json.dumps(wrapped)
        result = extract_from_claude_stream(stream, task_plan_schema)

        assert result is not None
        assert result["milestone_id"] == "M3"
        assert validate_against_schema(result, task_plan_schema)

    def test_extraction_from_claude_assistant_message_normalizes(self, task_plan_schema: dict[str, Any]) -> None:
        """Test normalization works for plan embedded in Claude assistant message."""
        drifted_plan = {
            "milestone_id": "M1",
            "max_parallel_tasks": 4,
            "rationale": "Test from assistant message",
            "tasks": [
                {
                    "id": "M1-T1",
                    "title": "Task 1",
                    "description": "Desc",
                    "agent": "either",  # Wrong key
                    "intensity": "low",  # Wrong key
                    "dependencies": [],  # Wrong key
                    "locks": [],
                    "solo": False,
                }
            ],
        }

        # Wrap in Claude CLI assistant message format
        assistant_event = {"type": "assistant", "message": {"content": [{"type": "text", "text": json.dumps(drifted_plan)}]}}

        stream = json.dumps(assistant_event)
        result = extract_from_claude_stream(stream, task_plan_schema)

        assert result is not None
        assert validate_against_schema(result, task_plan_schema)
        task = result["tasks"][0]
        assert task["preferred_agent"] == "either"
        assert task["estimated_intensity"] == "low"
        assert task["depends_on"] == []


# -----------------------------
# Regression tests for specific failure patterns
# -----------------------------


class TestRegressionProseWithFencedJson:
    """Regression test for run 20260120T100520Z failure.

    The failure pattern was:
    1. Raw stream contains multiple JSON arrays (one per line)
    2. Each array contains Claude CLI events
    3. The "result" event has a "result" field containing:
       - Prose text explaining the analysis
       - A JSON code fence (```json...```) with the actual task plan
       - More prose after the fence

    The original extractor failed because:
    - strip_fences() only worked when the string STARTED with ```
    - When prose precedes the fence, strip_fences returned the original string
    - json.loads() then failed on the prose+json mix
    """

    def test_strip_fences_with_prose_before_and_after(self) -> None:
        """strip_fences should extract fenced content even with surrounding prose."""
        text = 'Here is my analysis of the situation.\n\n```json\n{"key": "value"}\n```\n\nThat concludes my response.'
        result = strip_fences(text)
        assert result == '{"key": "value"}'

    def test_strip_fences_with_prose_before_fence(self) -> None:
        """strip_fences handles prose before code fence."""
        text = 'Some explanation text\n```json\n{"a": 1}\n```'
        result = strip_fences(text)
        assert result == '{"a": 1}'

    def test_extract_from_result_with_prose_and_fenced_json(
        self, task_plan_schema: dict[str, Any], valid_task_plan: dict[str, Any]
    ) -> None:
        """Extracts JSON from result event where plan is fenced but surrounded by prose.

        This is the exact pattern that caused run 20260120T100520Z to fail.
        """
        # Simulate the result field content with prose + fenced JSON
        result_text = (
            "Now I see the situation more clearly. The 5 root failure tasks have "
            "**commits** on their task branches, but those branches were never "
            "**merged** into `agent-run/20260120T091327Z`.\n\n"
            "Based on my analysis:\n"
            "1. The 5 root failure tasks are **implemented** but have merge conflicts\n"
            "2. These need to be resolved first\n\n"
            f"```json\n{json.dumps(valid_task_plan)}\n```"
        )

        result_event = {"type": "result", "subtype": "success", "result": result_text, "session_id": "test-session"}

        # Wrap in an array like Claude CLI does
        stream = json.dumps([result_event])

        result = extract_from_claude_stream(stream, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"
        assert validate_against_schema(result, task_plan_schema)

    def test_extract_from_multiline_stream_with_tool_events(
        self, task_plan_schema: dict[str, Any], valid_task_plan: dict[str, Any]
    ) -> None:
        """Handles multi-line stream with tool_use/tool_result events.

        Simulates the actual run 20260120T100520Z stream structure:
        - Line 1: Array of events including tool_use, tool_result, and a result
        - Line 2: Another array of events with the final result
        """
        # Line 1: First session with tool events and partial result
        line1_events = [
            {"type": "system", "subtype": "init", "session_id": "session1"},
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "id": "toolu_123", "name": "Read", "input": {"file_path": "/some/file.md"}}]
                },
            },
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "toolu_123", "content": "File content here"}]},
            },
            {"type": "result", "subtype": "success", "result": "Analysis from first session without a valid plan"},
        ]

        # Line 2: Second session with the actual task plan in fenced JSON
        result_with_fenced_plan = (
            f"After thorough analysis, here is the task plan:\n\n```json\n{json.dumps(valid_task_plan)}\n```"
        )
        line2_events = [
            {"type": "system", "subtype": "init", "session_id": "session2"},
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "id": "toolu_456", "name": "Grep", "input": {"pattern": "milestone"}}]
                },
            },
            {"type": "result", "subtype": "success", "result": result_with_fenced_plan},
        ]

        # Combine as two lines (NDJSON-like format)
        stream = json.dumps(line1_events) + "\n" + json.dumps(line2_events)

        result = extract_from_claude_stream(stream, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"
        assert validate_against_schema(result, task_plan_schema)

    def test_extract_json_embedded_in_prose_without_fences(
        self, task_plan_schema: dict[str, Any], valid_task_plan: dict[str, Any]
    ) -> None:
        """Extracts JSON embedded in prose even without code fences.

        Sometimes Claude outputs JSON inline without fences.
        """
        result_text = (
            f"Based on my analysis, here is the task plan: {json.dumps(valid_task_plan)} This should handle all the requirements."
        )

        result_event = {"type": "result", "result": result_text}

        stream = json.dumps([result_event])
        result = extract_from_claude_stream(stream, task_plan_schema)
        assert result is not None
        assert result["milestone_id"] == "M2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
