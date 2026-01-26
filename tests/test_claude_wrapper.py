from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "bridge" / "turn.schema.json"
CLAUDE_WRAPPER = ROOT / "bridge" / "agents" / "claude.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR)


def _base_env(claude_bin: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CLAUDE_BIN"] = str(claude_bin)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("CLAUDE_API_KEY", None)
    return env


def _validate_turn(payload: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=payload, schema=schema)


def test_claude_wrapper_emits_schema_valid_turn_on_success(tmp_path: Path) -> None:
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        """#!/usr/bin/env bash
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
  exit 0
fi
if [[ "$1" == "--version" ]]; then
  echo "claude 0.0.0"
  exit 0
fi

python3 - <<'PY'
import json
turn = {
    "agent": "claude",
    "milestone_id": "M0",
    "phase": "plan",
    "work_completed": False,
    "project_complete": False,
    "summary": "ok",
    "gates_passed": [],
    "requirement_progress": {
        "covered_req_ids": [],
        "tests_added_or_modified": [],
        "commands_run": [],
    },
    "next_agent": "codex",
    "next_prompt": "",
    "delegate_rationale": "",
    "stats_refs": ["CL-1"],
    "needs_write_access": True,
    "artifacts": [],
}
print(json.dumps({"result": json.dumps(turn)}))
PY
""",
    )

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = _base_env(claude_stub)

    subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        check=True,
        env=env,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"


def test_claude_wrapper_emits_schema_valid_turn_on_parse_failure(tmp_path: Path) -> None:
    """Test that wrapper emits schema-valid error turn on unparseable output."""
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        """#!/usr/bin/env bash
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
  exit 0
fi
if [[ "$1" == "--version" ]]; then
  echo "claude 0.0.0"
  exit 0
fi

echo "not-json" >&2
exit 1
""",
    )

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = _base_env(claude_stub)

    result = subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"
    assert "wrapper_status=error" in payload["summary"].lower()
    assert "ERROR" in result.stderr


def test_claude_wrapper_missing_binary_emits_error_turn(tmp_path: Path) -> None:
    """Test that missing Claude CLI still yields a schema-valid error turn."""
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = os.environ.copy()
    env["CLAUDE_BIN"] = str(tmp_path / "missing-claude")
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("CLAUDE_API_KEY", None)

    result = subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"
    assert "wrapper_status=error" in payload["summary"].lower()


def test_claude_wrapper_fallback_when_help_lacks_flags(tmp_path: Path) -> None:
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        """#!/usr/bin/env bash
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: claude"
  exit 0
fi
if [[ "$1" == "--version" ]]; then
  echo "claude 0.0.0"
  exit 0
fi

python3 - <<'PY'
import json
turn = {
    "agent": "claude",
    "milestone_id": "M0",
    "phase": "plan",
    "work_completed": False,
    "project_complete": False,
    "summary": "ok",
    "gates_passed": [],
    "requirement_progress": {
        "covered_req_ids": [],
        "tests_added_or_modified": [],
        "commands_run": [],
    },
    "next_agent": "codex",
    "next_prompt": "",
    "delegate_rationale": "",
    "stats_refs": ["CL-1"],
    "needs_write_access": True,
    "artifacts": [],
}
print(json.dumps({"result": json.dumps(turn)}))
PY
""",
    )

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = _base_env(claude_stub)

    subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        check=True,
        env=env,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"


def test_claude_wrapper_invokes_cli_without_api_key(tmp_path: Path) -> None:
    called_marker = tmp_path / "called.txt"
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        f"""#!/usr/bin/env bash
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
  exit 0
fi
if [[ "$1" == "--version" ]]; then
  echo "claude 0.0.0"
  exit 0
fi

echo "called" > "{called_marker}"
python3 - <<'PY'
import json
turn = {{
    "agent": "claude",
    "milestone_id": "M0",
    "phase": "plan",
    "work_completed": False,
    "project_complete": False,
    "summary": "ok",
    "gates_passed": [],
    "requirement_progress": {{
        "covered_req_ids": [],
        "tests_added_or_modified": [],
        "commands_run": [],
    }},
    "next_agent": "codex",
    "next_prompt": "",
    "delegate_rationale": "",
    "stats_refs": ["CL-1"],
    "needs_write_access": True,
    "artifacts": [],
}}
print(json.dumps({{"result": json.dumps(turn)}}))
PY
""",
    )

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = _base_env(claude_stub)

    subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        check=True,
        env=env,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"
    assert called_marker.exists()


def test_claude_help_timeout_fallback(tmp_path: Path) -> None:
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        """#!/usr/bin/env bash
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  sleep 2
  exit 0
fi
if [[ "$1" == "--version" ]]; then
  echo "claude 0.0.0"
  exit 0
fi

python3 - <<'PY'
import json
turn = {
    "agent": "claude",
    "milestone_id": "M0",
    "phase": "plan",
    "work_completed": False,
    "project_complete": False,
    "summary": "ok",
    "gates_passed": [],
    "requirement_progress": {
        "covered_req_ids": [],
        "tests_added_or_modified": [],
        "commands_run": [],
    },
    "next_agent": "codex",
    "next_prompt": "",
    "delegate_rationale": "",
    "stats_refs": ["CL-1"],
    "needs_write_access": True,
    "artifacts": [],
}
print(json.dumps({"result": json.dumps(turn)}))
PY
""",
    )

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = _base_env(claude_stub)
    env["CLAUDE_HELP_TIMEOUT_S"] = "1"

    subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        check=True,
        env=env,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"


def test_claude_wrapper_warns_when_api_key_set(tmp_path: Path) -> None:
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        """#!/usr/bin/env bash
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
  exit 0
fi
if [[ "$1" == "--version" ]]; then
  echo "claude 0.0.0"
  exit 0
fi

python3 - <<'PY'
import json
turn = {
    "agent": "claude",
    "milestone_id": "M0",
    "phase": "plan",
    "work_completed": False,
    "project_complete": False,
    "summary": "ok",
    "gates_passed": [],
    "requirement_progress": {
        "covered_req_ids": [],
        "tests_added_or_modified": [],
        "commands_run": [],
    },
    "next_agent": "codex",
    "next_prompt": "",
    "delegate_rationale": "",
    "stats_refs": ["CL-1"],
    "needs_write_access": True,
    "artifacts": [],
}
print(json.dumps({"result": json.dumps(turn)}))
PY
""",
    )

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = _base_env(claude_stub)
    env["ANTHROPIC_API_KEY"] = "dummy"

    subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        check=True,
        env=env,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    summary = payload["summary"]
    assert "WARNING:" in summary
    assert "api billing" in summary.lower()
    assert "auth_mode=api_key" in summary.lower()


# =============================================================================
# Regression tests for robust JSON stream parsing
# =============================================================================


def test_claude_wrapper_json_sequence_concatenated(tmp_path: Path) -> None:
    """Test parsing of concatenated JSON values (two arrays back-to-back).

    This simulates the real Claude CLI output format where multiple JSON
    values are emitted without newlines between them.
    """
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        """#!/usr/bin/env bash
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
  exit 0
fi
if [[ "$1" == "--version" ]]; then
  echo "claude 0.0.0"
  exit 0
fi

python3 - <<'PY'
import json

# Simulate Claude CLI output: init array + result event concatenated
init_array = [{"type": "system", "subtype": "init", "session_id": "abc123"}]
turn = {
    "agent": "claude",
    "milestone_id": "M0",
    "phase": "implement",
    "work_completed": True,
    "project_complete": False,
    "summary": "Completed task",
    "gates_passed": ["ruff", "mypy"],
    "requirement_progress": {
        "covered_req_ids": ["REQ-1"],
        "tests_added_or_modified": [],
        "commands_run": [],
    },
    "next_agent": "codex",
    "next_prompt": "Run verify",
    "delegate_rationale": "Handing off for verification",
    "stats_refs": ["CL-1"],
    "needs_write_access": True,
    "artifacts": [],
}
result_event = {"type": "result", "result": json.dumps(turn)}

# Print two JSON values concatenated (no newline between them)
print(json.dumps(init_array) + json.dumps(result_event))
PY
""",
    )

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = _base_env(claude_stub)

    result = subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, f"Wrapper failed: {result.stderr}"
    assert out_path.exists(), "Output file not created"

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"
    assert "synthesized" not in out_path.read_text(encoding="utf-8").lower()


def test_claude_wrapper_pretty_printed_multiline_json(tmp_path: Path) -> None:
    """Test parsing of pretty-printed multi-line JSON arrays.

    Line-by-line parsing would fail on this format.
    """
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        """#!/usr/bin/env bash
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
  exit 0
fi
if [[ "$1" == "--version" ]]; then
  echo "claude 0.0.0"
  exit 0
fi

python3 - <<'PY'
import json

turn = {
    "agent": "claude",
    "milestone_id": "M0",
    "phase": "verify",
    "work_completed": True,
    "project_complete": False,
    "summary": "Verified",
    "gates_passed": [],
    "requirement_progress": {
        "covered_req_ids": [],
        "tests_added_or_modified": [],
        "commands_run": [],
    },
    "next_agent": "codex",
    "next_prompt": "",
    "delegate_rationale": "",
    "stats_refs": ["CL-1"],
    "needs_write_access": True,
    "artifacts": [],
}

# Pretty-print the array (multi-line format that line-by-line parsing would break)
events = [
    {"type": "system", "subtype": "init"},
    {"type": "result", "result": json.dumps(turn)},
]
print(json.dumps(events, indent=2))
PY
""",
    )

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = _base_env(claude_stub)

    result = subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, f"Wrapper failed: {result.stderr}"
    assert out_path.exists(), "Output file not created"

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"
    assert "synthesized" not in out_path.read_text(encoding="utf-8").lower()


def test_claude_wrapper_assistant_message_format(tmp_path: Path) -> None:
    """Test parsing of assistant message format (fallback when no result event)."""
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        """#!/usr/bin/env bash
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
  exit 0
fi
if [[ "$1" == "--version" ]]; then
  echo "claude 0.0.0"
  exit 0
fi

python3 - <<'PY'
import json

turn = {
    "agent": "claude",
    "milestone_id": "M0",
    "phase": "plan",
    "work_completed": False,
    "project_complete": False,
    "summary": "From assistant message",
    "gates_passed": [],
    "requirement_progress": {
        "covered_req_ids": [],
        "tests_added_or_modified": [],
        "commands_run": [],
    },
    "next_agent": "codex",
    "next_prompt": "",
    "delegate_rationale": "",
    "stats_refs": ["CL-1"],
    "needs_write_access": True,
    "artifacts": [],
}

# Use assistant message format (no result event)
events = [
    {"type": "system", "subtype": "init"},
    {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": json.dumps(turn)}
            ]
        }
    },
]
print(json.dumps(events))
PY
""",
    )

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = _base_env(claude_stub)

    result = subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, f"Wrapper failed: {result.stderr}"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"


def test_claude_wrapper_emits_error_turn_on_invalid_json(tmp_path: Path) -> None:
    """Test that wrapper emits a schema-valid error turn on unparseable content."""
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        """#!/usr/bin/env bash
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
  exit 0
fi
if [[ "$1" == "--version" ]]; then
  echo "claude 0.0.0"
  exit 0
fi

# Output garbage that's not valid JSON
echo "This is not JSON at all"
echo "Neither is this"
exit 0
""",
    )

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = _base_env(claude_stub)

    result = subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"
    assert "wrapper_status=error" in payload["summary"].lower()
    assert "ERROR" in result.stderr


def test_claude_wrapper_result_event_priority(tmp_path: Path) -> None:
    """Test that result event takes priority over assistant message."""
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        """#!/usr/bin/env bash
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
  exit 0
fi
if [[ "$1" == "--version" ]]; then
  echo "claude 0.0.0"
  exit 0
fi

python3 - <<'PY'
import json

# Two different turns: one in assistant, one in result
# The result should take priority
turn_assistant = {
    "agent": "claude",
    "milestone_id": "M0",
    "phase": "plan",
    "work_completed": False,
    "project_complete": False,
    "summary": "FROM ASSISTANT - WRONG",
    "gates_passed": [],
    "requirement_progress": {"covered_req_ids": [], "tests_added_or_modified": [], "commands_run": []},
    "next_agent": "codex",
    "next_prompt": "",
    "delegate_rationale": "",
    "stats_refs": ["CL-1"],
    "needs_write_access": True,
    "artifacts": [],
}

turn_result = {
    "agent": "claude",
    "milestone_id": "M0",
    "phase": "implement",
    "work_completed": True,
    "project_complete": False,
    "summary": "FROM RESULT - CORRECT",
    "gates_passed": [],
    "requirement_progress": {"covered_req_ids": [], "tests_added_or_modified": [], "commands_run": []},
    "next_agent": "codex",
    "next_prompt": "",
    "delegate_rationale": "",
    "stats_refs": ["CL-1"],
    "needs_write_access": True,
    "artifacts": [],
}

events = [
    {"type": "system", "subtype": "init"},
    {"type": "assistant", "message": {"content": [{"type": "text", "text": json.dumps(turn_assistant)}]}},
    {"type": "result", "result": json.dumps(turn_result)},
]
print(json.dumps(events))
PY
""",
    )

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = _base_env(claude_stub)

    result = subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, f"Wrapper failed: {result.stderr}"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"
    # Result event should take priority
    assert "FROM RESULT - CORRECT" in payload["summary"]
    assert payload["phase"] == "implement"
    assert payload["work_completed"] is True


# =============================================================================
# Tests using the FakeClaudeCLI fixture
# =============================================================================


def test_fake_claude_fixture_success_scenario(tmp_path: Path) -> None:
    """Test that FakeClaudeCLI generates working stub for success scenario."""
    from tests.fixtures.fake_claude import FakeClaudeCLI, create_fake_claude_env

    stub = FakeClaudeCLI(scenario="success")
    stub.write_to(tmp_path / "claude")

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = create_fake_claude_env(stub)

    subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        check=True,
        env=env,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"


def test_fake_claude_fixture_parse_failure_scenario(tmp_path: Path) -> None:
    """Test that FakeClaudeCLI parse_failure scenario produces error turn."""
    from tests.fixtures.fake_claude import FakeClaudeCLI, create_fake_claude_env

    stub = FakeClaudeCLI(scenario="parse_failure")
    stub.write_to(tmp_path / "claude")

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = create_fake_claude_env(stub)

    result = subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"
    assert "wrapper_status=error" in payload["summary"].lower()


def test_fake_claude_fixture_concatenated_json_scenario(tmp_path: Path) -> None:
    """Test that FakeClaudeCLI concatenated_json scenario parses correctly."""
    from tests.fixtures.fake_claude import FakeClaudeCLI, create_fake_claude_env

    stub = FakeClaudeCLI(scenario="concatenated_json")
    stub.write_to(tmp_path / "claude")

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = create_fake_claude_env(stub)

    result = subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, f"Wrapper failed: {result.stderr}"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"


def test_fake_claude_fixture_with_turn_overrides(tmp_path: Path) -> None:
    """Test that FakeClaudeCLI turn_overrides work correctly."""
    from tests.fixtures.fake_claude import FakeClaudeCLI, create_fake_claude_env

    stub = FakeClaudeCLI(
        scenario="success",
        turn_overrides={
            "phase": "implement",
            "work_completed": True,
            "summary": "Custom summary from override",
            "gates_passed": ["ruff", "mypy"],
        },
    )
    stub.write_to(tmp_path / "claude")

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = create_fake_claude_env(stub)

    subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        check=True,
        env=env,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["phase"] == "implement"
    assert payload["work_completed"] is True
    assert "Custom summary from override" in payload["summary"]
    assert payload["gates_passed"] == ["ruff", "mypy"]


def test_fake_claude_fixture_api_warning_scenario(tmp_path: Path) -> None:
    """Test that FakeClaudeCLI with API key triggers warning in output."""
    from tests.fixtures.fake_claude import FakeClaudeCLI, create_fake_claude_env

    stub = FakeClaudeCLI(scenario="api_warning")
    stub.write_to(tmp_path / "claude")

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nCL-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = create_fake_claude_env(stub, include_api_key=True)

    subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        check=True,
        env=env,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    summary = payload["summary"]
    assert "WARNING:" in summary
    assert "api billing" in summary.lower()


def test_make_valid_turn_helper() -> None:
    """Test that make_valid_turn produces schema-valid output."""
    from tests.fixtures.fake_claude import make_valid_turn

    turn = make_valid_turn()

    # Validate against schema
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=turn, schema=schema)

    # Check defaults
    assert turn["agent"] == "claude"
    assert turn["milestone_id"] == "M0"
    assert turn["phase"] == "plan"
    assert turn["work_completed"] is False
    assert turn["stats_refs"] == ["CL-1"]


def test_make_valid_turn_with_custom_values() -> None:
    """Test make_valid_turn with custom parameter values."""
    from tests.fixtures.fake_claude import make_valid_turn

    turn = make_valid_turn(
        phase="verify",
        work_completed=True,
        project_complete=True,
        summary="All tests passed",
        gates_passed=["ruff", "mypy", "pytest"],
        covered_req_ids=["REQ-001", "REQ-002"],
        next_agent="claude",
        stats_refs=["CL-1", "CX-1"],
    )

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=turn, schema=schema)

    assert turn["phase"] == "verify"
    assert turn["work_completed"] is True
    assert turn["project_complete"] is True
    assert turn["summary"] == "All tests passed"
    assert turn["gates_passed"] == ["ruff", "mypy", "pytest"]
    assert turn["requirement_progress"]["covered_req_ids"] == ["REQ-001", "REQ-002"]
    assert turn["next_agent"] == "claude"
    assert turn["stats_refs"] == ["CL-1", "CX-1"]
