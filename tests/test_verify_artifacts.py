from __future__ import annotations

import json
import subprocess
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

    verify_log = (run_dir / "verify.log").read_text(encoding="utf-8")
    assert "gate-stdout" in verify_log
    assert "gate-stderr" in verify_log

    failure_path = run_dir / "failures" / "spec_lint.json"
    assert failure_path.exists()
    failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure_payload["name"] == "spec_lint"
    assert failure_payload["note"] == "rc=2"


def test_run_injects_deterministic_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifacts = verify._init_verify_artifacts(tmp_path)
    env_overrides = verify._build_verify_env(artifacts.tmp_dir)
    monkeypatch.setattr(verify, "_VERIFY_ENV", env_overrides)

    captured: dict[str, dict[str, str]] = {}

    def fake_run(
        cmd: list[str],
        cwd: str,
        text: bool,
        capture_output: bool,
        timeout: float | None,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        captured["env"] = env
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = verify._run(["echo", "ok"], tmp_path)
    assert result.passed is True

    env = captured["env"]
    for key, value in verify.DETERMINISTIC_ENV.items():
        assert env.get(key) == value

    expected_tmp = str(artifacts.tmp_dir)
    assert env.get("TMPDIR") == expected_tmp
    assert env.get("TEMP") == expected_tmp
    assert env.get("TMP") == expected_tmp
    assert artifacts.tmp_dir.exists()
