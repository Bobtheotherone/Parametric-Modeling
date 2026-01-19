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


def test_claude_wrapper_emits_schema_valid_turn_on_error(tmp_path: Path) -> None:
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

    subprocess.run(
        [str(CLAUDE_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        check=True,
        env=env,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "claude"
    assert "schema_attempt" in payload["summary"]
    assert "rc=" in payload["summary"]
    assert "not-json" in payload["summary"]


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
