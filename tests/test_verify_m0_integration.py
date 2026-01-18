from __future__ import annotations

from pathlib import Path

import pytest
from tools import verify


def _write_design_doc(path: Path, milestone_id: str) -> None:
    path.write_text(f"**Milestone:** {milestone_id} — test\n", encoding="utf-8")


def _has_m0_call(calls: list[dict[str, object]], command: str) -> bool:
    for call in calls:
        cmd = call.get("cmd")
        if isinstance(cmd, list) and "tools.m0" in cmd and command in cmd:
            return True
    return False


def test_verify_runs_m0_gates_for_m0(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    design_doc = tmp_path / "DESIGN_DOCUMENT.md"
    _write_design_doc(design_doc, "M0")

    calls: list[dict[str, object]] = []

    def fake_run(cmd: list[str], cwd: Path, *, timeout_s: int | None = None) -> verify.GateResult:
        calls.append({"cmd": cmd, "timeout_s": timeout_s, "cwd": cwd})
        return verify.GateResult(name="fake", passed=True, cmd=cmd)

    monkeypatch.setattr(verify, "_run", fake_run)

    rc = verify.main(
        [
            "--project-root",
            str(tmp_path),
            "--skip-pytest",
            "--skip-quality",
            "--skip-git",
        ]
    )
    assert rc == 0
    assert _has_m0_call(calls, "smoke")
    assert _has_m0_call(calls, "repro-check")

    calls.clear()
    _write_design_doc(design_doc, "M1")
    rc = verify.main(
        [
            "--project-root",
            str(tmp_path),
            "--skip-pytest",
            "--skip-quality",
            "--skip-git",
        ]
    )
    assert rc == 0
    assert not _has_m0_call(calls, "smoke")
    assert not _has_m0_call(calls, "repro-check")

    calls.clear()
    rc = verify.main(
        [
            "--project-root",
            str(tmp_path),
            "--skip-pytest",
            "--skip-quality",
            "--skip-git",
            "--include-m0",
        ]
    )
    assert rc == 0
    assert _has_m0_call(calls, "smoke")
    assert _has_m0_call(calls, "repro-check")


def test_verify_time_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    design_doc = tmp_path / "DESIGN_DOCUMENT.md"
    _write_design_doc(design_doc, "M0")

    calls: list[dict[str, object]] = []

    def fake_run(cmd: list[str], cwd: Path, *, timeout_s: int | None = None) -> verify.GateResult:
        calls.append({"cmd": cmd, "timeout_s": timeout_s, "cwd": cwd})
        return verify.GateResult(name="fake", passed=True, cmd=cmd)

    monkeypatch.setattr(verify, "_run", fake_run)

    rc = verify.main(
        [
            "--project-root",
            str(tmp_path),
            "--skip-pytest",
            "--skip-quality",
            "--skip-git",
        ]
    )
    assert rc == 0
    m0_calls: list[dict[str, object]] = []
    for call in calls:
        cmd = call.get("cmd")
        if isinstance(cmd, list) and "tools.m0" in cmd:
            m0_calls.append(call)
    assert m0_calls
    for call in m0_calls:
        timeout_s = call.get("timeout_s")
        assert isinstance(timeout_s, int)
        assert timeout_s <= verify.M0_GATE_TIMEOUT_S
