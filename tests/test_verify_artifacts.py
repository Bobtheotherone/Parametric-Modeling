from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools import verify


def _write_design_doc(path: Path, milestone_id: str) -> None:
    path.write_text(f"**Milestone:** {milestone_id} — test\n", encoding="utf-8")


def test_verify_writes_failure_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_design_doc(tmp_path / "DESIGN_DOCUMENT.md", "M1")

    def fake_run(cmd: list[str], cwd: Path, *, timeout_s: int | None = None) -> verify.GateResult:
        return verify.GateResult(
            name="fake",
            passed=False,
            cmd=cmd,
            stdout="gate-stdout",
            stderr="gate-stderr",
            note="rc=2",
        )

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

    assert rc == 2
    captured = capsys.readouterr()
    assert "gate-stdout" in captured.out
    assert "gate-stderr" in captured.err

    artifacts_root = tmp_path / "artifacts" / "verify"
    run_dirs = [path for path in artifacts_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    logs_dir = run_dir / "logs"
    assert (logs_dir / "spec_lint.stdout.log").read_text(encoding="utf-8") == "gate-stdout"
    assert (logs_dir / "spec_lint.stderr.log").read_text(encoding="utf-8") == "gate-stderr"

    failure_path = run_dir / "failures" / "spec_lint.json"
    assert failure_path.exists()
    failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure_payload["name"] == "spec_lint"
    assert failure_payload["note"] == "rc=2"
