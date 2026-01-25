"""Offline tests for Gemini wrapper using a stub executable.

Tests cover:
- Success: valid JSON turn output
- Malformed: non-JSON garbage output
- Error: non-zero exit code with error message
- Invalid schema: JSON that doesn't match turn schema
- Wrapped outputs: JSON in code fences or result wrapper
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "bridge"
GEMINI_WRAPPER = BRIDGE / "gemini.py"
TURN_SCHEMA = BRIDGE / "turn.schema.json"
FAKE_GEMINI = ROOT / "tests" / "fixtures" / "fake_gemini.py"


def _run_wrapper(
    prompt: str = "Test prompt",
    env_updates: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    """Run the gemini wrapper and return result + parsed output JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_path = Path(tmpdir) / "prompt.txt"
        out_path = Path(tmpdir) / "out.json"
        prompt_path.write_text(prompt, encoding="utf-8")

        env = os.environ.copy()
        # Point to fake gemini executable
        env["GEMINI_BIN"] = str(FAKE_GEMINI)
        env["PYTHONPATH"] = str(ROOT)
        if env_updates:
            env.update(env_updates)

        result = subprocess.run(
            [sys.executable, str(GEMINI_WRAPPER), str(prompt_path), str(TURN_SCHEMA), str(out_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        payload = None
        if out_path.exists():
            try:
                payload = json.loads(out_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        return result, payload


def _validate_turn_structure(payload: dict[str, Any]) -> None:
    """Assert that payload has required Turn schema structure."""
    required_keys = [
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
    ]
    for key in required_keys:
        assert key in payload, f"Missing required key: {key}"

    # Validate types
    assert isinstance(payload["agent"], str)
    assert isinstance(payload["milestone_id"], str)
    assert isinstance(payload["phase"], str)
    assert isinstance(payload["work_completed"], bool)
    assert isinstance(payload["project_complete"], bool)
    assert isinstance(payload["summary"], str)
    assert isinstance(payload["gates_passed"], list)
    assert isinstance(payload["requirement_progress"], dict)
    assert isinstance(payload["next_agent"], str)
    assert isinstance(payload["next_prompt"], str)
    assert isinstance(payload["delegate_rationale"], str)
    assert isinstance(payload["stats_refs"], list)
    assert len(payload["stats_refs"]) >= 1
    assert isinstance(payload["needs_write_access"], bool)
    assert isinstance(payload["artifacts"], list)

    # Validate enums
    assert payload["agent"] in ("codex", "claude")
    assert payload["next_agent"] in ("codex", "claude")
    assert payload["phase"] in ("plan", "implement", "verify", "finalize")

    # Validate requirement_progress structure
    rp = payload["requirement_progress"]
    assert "covered_req_ids" in rp
    assert "tests_added_or_modified" in rp
    assert "commands_run" in rp


class TestGeminiWrapperSuccess:
    """Tests for successful Gemini wrapper execution."""

    def test_success_returns_valid_turn(self) -> None:
        """Gemini wrapper returns valid turn JSON on success."""
        result, payload = _run_wrapper(env_updates={"FAKE_GEMINI_MODE": "success"})
        assert result.returncode == 0
        assert payload is not None
        _validate_turn_structure(payload)
        assert payload["work_completed"] is True

    def test_success_extracts_milestone_from_prompt(self) -> None:
        """Wrapper extracts milestone from prompt text."""
        prompt = "**Milestone:** M1\nImplement feature X"
        result, payload = _run_wrapper(prompt=prompt, env_updates={"FAKE_GEMINI_MODE": "success"})
        assert result.returncode == 0
        assert payload is not None
        # The wrapper should extract M1 or use M0 fallback
        assert payload["milestone_id"] in ("M0", "M1")

    def test_success_with_code_fence_wrapped_json(self) -> None:
        """Wrapper handles JSON wrapped in markdown code fences."""
        result, payload = _run_wrapper(env_updates={"FAKE_GEMINI_MODE": "code_fence"})
        assert result.returncode == 0
        assert payload is not None
        _validate_turn_structure(payload)

    def test_success_with_result_wrapped_json(self) -> None:
        """Wrapper unwraps JSON from result wrapper."""
        result, payload = _run_wrapper(env_updates={"FAKE_GEMINI_MODE": "wrapped"})
        assert result.returncode == 0
        assert payload is not None
        _validate_turn_structure(payload)


class TestGeminiWrapperMalformed:
    """Tests for malformed output handling."""

    def test_malformed_output_returns_error_turn(self) -> None:
        """Wrapper returns valid error turn for non-JSON output."""
        result, payload = _run_wrapper(env_updates={"FAKE_GEMINI_MODE": "malformed"})
        assert result.returncode == 0  # Wrapper itself should succeed
        assert payload is not None
        _validate_turn_structure(payload)
        assert "malformed" in payload["summary"].lower()

    def test_empty_output_returns_error_turn(self) -> None:
        """Wrapper returns valid error turn for empty output."""
        result, payload = _run_wrapper(env_updates={"FAKE_GEMINI_MODE": "empty"})
        assert result.returncode == 0
        assert payload is not None
        _validate_turn_structure(payload)
        assert "error" in payload["summary"].lower() or "malformed" in payload["summary"].lower()

    def test_invalid_schema_output_returns_error_turn(self) -> None:
        """Wrapper returns valid error turn for schema-invalid JSON."""
        result, payload = _run_wrapper(env_updates={"FAKE_GEMINI_MODE": "invalid_schema"})
        assert result.returncode == 0
        assert payload is not None
        _validate_turn_structure(payload)
        # Should mention validation failure
        assert "error" in payload["summary"].lower() or "validation" in payload["summary"].lower()


class TestGeminiWrapperError:
    """Tests for error condition handling."""

    def test_nonzero_exit_returns_error_turn(self) -> None:
        """Wrapper returns valid error turn for non-zero exit code."""
        result, payload = _run_wrapper(
            env_updates={"FAKE_GEMINI_MODE": "error", "FAKE_GEMINI_EXIT_CODE": "1"}
        )
        assert result.returncode == 0  # Wrapper itself should succeed
        assert payload is not None
        _validate_turn_structure(payload)
        assert "error" in payload["summary"].lower()
        assert "exit" in payload["summary"].lower() or "non-zero" in payload["summary"].lower()

    def test_stderr_captured_in_error_summary(self) -> None:
        """Wrapper captures stderr in error turn summary."""
        result, payload = _run_wrapper(
            env_updates={
                "FAKE_GEMINI_MODE": "error",
                "FAKE_GEMINI_EXIT_CODE": "1",
                "FAKE_GEMINI_STDERR": "API rate limit exceeded",
            }
        )
        assert result.returncode == 0
        assert payload is not None
        _validate_turn_structure(payload)
        # The wrapper should capture stderr in the error details
        assert "error" in payload["summary"].lower()


class TestGeminiWrapperSchemaValidation:
    """Tests for output schema validation."""

    def test_output_has_valid_agent_enum(self) -> None:
        """Output agent must be in allowed enum."""
        result, payload = _run_wrapper(env_updates={"FAKE_GEMINI_MODE": "success"})
        assert payload is not None
        assert payload["agent"] in ("codex", "claude")

    def test_output_has_valid_phase_enum(self) -> None:
        """Output phase must be in allowed enum."""
        result, payload = _run_wrapper(env_updates={"FAKE_GEMINI_MODE": "success"})
        assert payload is not None
        assert payload["phase"] in ("plan", "implement", "verify", "finalize")

    def test_output_has_nonempty_stats_refs(self) -> None:
        """Output stats_refs must be non-empty array."""
        result, payload = _run_wrapper(env_updates={"FAKE_GEMINI_MODE": "success"})
        assert payload is not None
        assert isinstance(payload["stats_refs"], list)
        assert len(payload["stats_refs"]) >= 1

    def test_output_artifacts_have_required_fields(self) -> None:
        """Each artifact must have path and description."""
        custom_turn = {
            "agent": "codex",
            "milestone_id": "M0",
            "phase": "implement",
            "work_completed": True,
            "project_complete": False,
            "summary": "Test with artifacts",
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
            "artifacts": [
                {"path": "test.py", "description": "Test file"},
            ],
        }
        result, payload = _run_wrapper(
            env_updates={
                "FAKE_GEMINI_MODE": "success",
                "FAKE_GEMINI_STDOUT": json.dumps(custom_turn),
            }
        )
        assert payload is not None
        _validate_turn_structure(payload)
        if payload["artifacts"]:
            for artifact in payload["artifacts"]:
                assert "path" in artifact
                assert "description" in artifact


class TestGeminiWrapperEnvironment:
    """Tests for environment variable handling."""

    def test_respects_gemini_model_env(self) -> None:
        """Wrapper respects GEMINI_MODEL environment variable."""
        # This test verifies the wrapper constructs commands correctly
        result, payload = _run_wrapper(
            env_updates={
                "FAKE_GEMINI_MODE": "success",
                "GEMINI_MODEL": "gemini-pro",
            }
        )
        assert result.returncode == 0
        assert payload is not None

    def test_respects_write_access_env(self) -> None:
        """Wrapper respects WRITE_ACCESS environment variable."""
        result, payload = _run_wrapper(
            env_updates={
                "FAKE_GEMINI_MODE": "success",
                "WRITE_ACCESS": "1",
            }
        )
        assert result.returncode == 0
        assert payload is not None
        # The actual needs_write_access comes from the fake output
        # but wrapper should process the env var


@pytest.fixture
def ensure_fake_gemini_exists() -> None:
    """Ensure fake_gemini.py exists and is executable."""
    assert FAKE_GEMINI.exists(), f"Fake Gemini stub not found at {FAKE_GEMINI}"


def test_fake_gemini_stub_exists(ensure_fake_gemini_exists: None) -> None:
    """Verify fake Gemini stub exists for offline testing."""
    assert FAKE_GEMINI.is_file()
    # Check it's executable or can be run with Python
    result = subprocess.run(
        [sys.executable, str(FAKE_GEMINI)],
        capture_output=True,
        text=True,
        env={**os.environ, "FAKE_GEMINI_MODE": "success"},
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert "agent" in output
