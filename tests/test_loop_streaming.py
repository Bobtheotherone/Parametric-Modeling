from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR)


def test_streamed_agent_output_is_logged_and_prefixed(tmp_path: Path) -> None:
    wrapper = tmp_path / "wrapper.sh"
    _write_executable(
        wrapper,
        """#!/usr/bin/env bash
set -euo pipefail

echo "stub-stdout-1"
echo "stub-stdout-2"
echo "stub-stderr-1" >&2

cat > "$3" <<'JSON'
{
  "agent": "codex",
  "milestone_id": "M0",
  "phase": "plan",
  "work_completed": false,
  "project_complete": false,
  "summary": "mock",
  "gates_passed": [],
  "requirement_progress": {
    "covered_req_ids": [],
    "tests_added_or_modified": [],
    "commands_run": []
  },
  "next_agent": "codex",
  "next_prompt": "",
  "delegate_rationale": "",
  "stats_refs": ["CX-1"],
  "needs_write_access": false,
  "artifacts": []
}
JSON

if [[ "${FF_EMIT_JSON_STDOUT:-1}" == "1" ]]; then
  cat "$3"
fi
""",
    )

    config = _read_json(ROOT / "bridge" / "config.json")
    config["limits"]["max_total_calls"] = 1
    config["limits"]["max_calls_per_agent"] = 1
    config["limits"]["quota_retry_attempts"] = 1
    for agent in config["agents"].values():
        agent["script"] = str(wrapper)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    env = os.environ.copy()
    env["FF_SKIP_VERIFY"] = "1"
    env["FF_EMIT_JSON_STDOUT"] = "0"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bridge" / "loop.py"),
            "--project-root",
            str(ROOT),
            "--config",
            str(config_path),
            "--mode",
            "live",
            "--start-agent",
            "codex",
            "--no-agent-branch",
            "--stream-agent-output",
            "both",
        ],
        text=True,
        capture_output=True,
        env=env,
    )

    assert proc.returncode == 6
    assert "[codex][stdout] stub-stdout-1" in proc.stdout
    assert "[codex][stdout] stub-stdout-2" in proc.stdout
    assert "[codex][stderr] stub-stderr-1" in proc.stderr

    match = re.search(r"run_id=([0-9A-Za-zTZ]+)", proc.stdout)
    assert match is not None
    run_id = match.group(1)
    call_dir = ROOT / "runs" / run_id / "calls" / "call_0001"
    stdout_log = call_dir / "agent_stdout.log"
    stderr_log = call_dir / "agent_stderr.log"

    assert stdout_log.exists()
    assert stderr_log.exists()
    assert "stub-stdout-1" in stdout_log.read_text(encoding="utf-8")
    assert "stub-stdout-2" in stdout_log.read_text(encoding="utf-8")
    assert "stub-stderr-1" in stderr_log.read_text(encoding="utf-8")
