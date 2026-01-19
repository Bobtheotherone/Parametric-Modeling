from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _milestone_id() -> str:
    text = (ROOT / "DESIGN_DOCUMENT.md").read_text(encoding="utf-8")
    match = re.search(r"\*\*Milestone:\*\*\s*(M\d+)\b", text)
    assert match is not None
    return match.group(1)


def test_smoke_route_forces_sequence(tmp_path: Path) -> None:
    config = _read_json(ROOT / "bridge" / "config.json")
    config["limits"]["max_total_calls"] = 3
    config["limits"]["max_calls_per_agent"] = 3
    config["limits"]["quota_retry_attempts"] = 1
    config["agents"]["gemini"]["model"] = "gemini-test-model"
    config["agents"]["codex"]["model"] = "codex-test-model"
    config["agents"]["claude"]["model"] = "claude-test-model"

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    milestone_id = _milestone_id()
    scenario = {
        "agents": {
            "gemini": [
                {
                    "type": "ok",
                    "response": {
                        "agent": "gemini",
                        "milestone_id": milestone_id,
                        "phase": "plan",
                        "work_completed": False,
                        "project_complete": False,
                        "summary": "mock",
                        "gates_passed": [],
                        "requirement_progress": {
                            "covered_req_ids": [],
                            "tests_added_or_modified": [],
                            "commands_run": [],
                        },
                        "next_agent": "gemini",
                        "next_prompt": "",
                        "delegate_rationale": "",
                        "stats_refs": ["GM-1"],
                        "needs_write_access": False,
                        "artifacts": [],
                    },
                }
            ],
            "codex": [
                {
                    "type": "ok",
                    "response": {
                        "agent": "codex",
                        "milestone_id": milestone_id,
                        "phase": "plan",
                        "work_completed": False,
                        "project_complete": False,
                        "summary": "mock",
                        "gates_passed": [],
                        "requirement_progress": {
                            "covered_req_ids": [],
                            "tests_added_or_modified": [],
                            "commands_run": [],
                        },
                        "next_agent": "codex",
                        "next_prompt": "",
                        "delegate_rationale": "",
                        "stats_refs": ["CX-1"],
                        "needs_write_access": False,
                        "artifacts": [],
                    },
                }
            ],
            "claude": [
                {
                    "type": "ok",
                    "response": {
                        "agent": "claude",
                        "milestone_id": milestone_id,
                        "phase": "plan",
                        "work_completed": False,
                        "project_complete": False,
                        "summary": "mock",
                        "gates_passed": [],
                        "requirement_progress": {
                            "covered_req_ids": [],
                            "tests_added_or_modified": [],
                            "commands_run": [],
                        },
                        "next_agent": "claude",
                        "next_prompt": "",
                        "delegate_rationale": "",
                        "stats_refs": ["CL-1"],
                        "needs_write_access": False,
                        "artifacts": [],
                    },
                }
            ],
        }
    }

    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")

    env = os.environ.copy()
    env["FF_SKIP_VERIFY"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bridge" / "loop.py"),
            "--project-root",
            str(ROOT),
            "--config",
            str(config_path),
            "--mode",
            "mock",
            "--mock-scenario",
            str(scenario_path),
            "--smoke-route",
            "gemini,codex,claude",
            "--no-agent-branch",
        ],
        text=True,
        capture_output=True,
        env=env,
    )

    assert proc.returncode == 6
    calls = re.findall(r"CALL \d+ \| agent=([a-z]+) \| model=([^\s|]+)", proc.stdout)
    assert calls == [
        ("gemini", "gemini-test-model"),
        ("codex", "codex-test-model"),
        ("claude", "claude-test-model"),
    ]
