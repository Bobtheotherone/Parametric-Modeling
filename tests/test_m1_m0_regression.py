from __future__ import annotations

import sys
from pathlib import Path

import pytest
from bridge import loop
from tools.completion_gates import verify_args_for_completion


def test_verify_include_m0_required_for_completion() -> None:
    assert "--include-m0" not in verify_args_for_completion("M0")
    assert "--include-m0" in verify_args_for_completion("M1")
    assert "--include-m0" in verify_args_for_completion("M2")


def test_orchestrator_completion_gate_includes_m0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    design_doc = tmp_path / "DESIGN_DOCUMENT.md"
    design_doc.write_text("**Milestone:** M1 — test\n", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str, str]:
        calls.append(cmd)
        if cmd[:3] == [sys.executable, "-m", "tools.verify"]:
            return 0, "", ""
        if cmd[:3] == ["git", "status", "--porcelain=v1"]:
            return 0, "", ""
        if cmd[:2] == ["git", "rev-parse"]:
            return 0, "abc", ""
        return 0, "", ""

    monkeypatch.setattr(loop, "_run_cmd", fake_run_cmd)
    ok, _ = loop._completion_gates_ok(tmp_path, "M1")  # noqa: SLF001
    assert ok
    verify_calls = [cmd for cmd in calls if cmd[:3] == [sys.executable, "-m", "tools.verify"]]
    assert verify_calls
    assert any("--include-m0" in cmd for cmd in verify_calls)
