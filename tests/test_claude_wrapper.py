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
if [[ "$1" == "--help" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
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


def test_claude_wrapper_exits_nonzero_on_parse_failure(tmp_path: Path) -> None:
    """Test that wrapper exits non-zero when Claude outputs unparseable garbage.

    Previously the wrapper would synthesize a fallback response; now it must
    exit non-zero to signal failure to the orchestrator.
    """
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        """#!/usr/bin/env bash
if [[ "$1" == "--help" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
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

    # Wrapper must exit non-zero when Claude output is unparseable
    assert result.returncode != 0, "Wrapper should fail on unparseable output"
    # Should emit ERROR to stderr
    assert "ERROR" in result.stderr


def test_claude_wrapper_fallback_when_help_lacks_flags(tmp_path: Path) -> None:
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        """#!/usr/bin/env bash
if [[ "$1" == "--help" ]]; then
  echo "Usage: claude"
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
if [[ "$1" == "--help" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
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
if [[ "$1" == "--help" ]]; then
  sleep 2
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
if [[ "$1" == "--help" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
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
        '''#!/usr/bin/env bash
if [[ "$1" == "--help" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
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
''',
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
        '''#!/usr/bin/env bash
if [[ "$1" == "--help" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
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
''',
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
        '''#!/usr/bin/env bash
if [[ "$1" == "--help" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
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
''',
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


def test_claude_wrapper_nonzero_exit_on_invalid_json(tmp_path: Path) -> None:
    """Test that wrapper exits non-zero when Claude outputs unparseable content."""
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        """#!/usr/bin/env bash
if [[ "$1" == "--help" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
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

    # Wrapper should exit non-zero when it can't parse JSON
    assert result.returncode != 0, "Wrapper should fail on unparseable output"
    assert "ERROR" in result.stderr or not out_path.exists() or out_path.stat().st_size == 0


def test_claude_wrapper_result_event_priority(tmp_path: Path) -> None:
    """Test that result event takes priority over assistant message."""
    claude_stub = tmp_path / "claude"
    _write_executable(
        claude_stub,
        '''#!/usr/bin/env bash
if [[ "$1" == "--help" ]]; then
  echo "Usage: claude --prompt --output-format --json-schema --model --no-session-persistence --permission-mode --tools --json"
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
''',
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
