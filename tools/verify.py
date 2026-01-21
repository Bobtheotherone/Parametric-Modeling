"""Project verification gates.

This tool is the single entrypoint the agents should run to prove a milestone is complete.
It is intentionally opinionated and fast.

Gates (default):
- spec_lint: DESIGN_DOCUMENT contract + requirement->test mapping
- pytest: unit tests
- ruff: style + import sorting (if installed)
- mypy: type checking (if installed)
- git_guard: secret scan + working tree status (non-fatal by default)

Usage:
  python -m tools.verify [--project-root .] [--json out.json] [--strict-git]

Optional skips (useful for unit tests of this tool itself):
  --skip-pytest
  --skip-quality   # ruff+mypy
  --skip-git

Exit codes:
  0 = all required gates passed
  2 = gate failure(s)
  3 = tool error
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools import spec_lint


@dataclass
class GateResult:
    name: str
    passed: bool
    cmd: list[str] | None = None
    stdout: str = ""
    stderr: str = ""
    note: str = ""


DEFAULT_M0_GATE_TIMEOUT_S = 90
DETERMINISTIC_ENV = {
    "LC_ALL": "C.UTF-8",
    "LANG": "C.UTF-8",
    "TZ": "UTC",
    "PYTHONHASHSEED": "0",
}
VERIFY_ARTIFACTS_DIR = Path("artifacts") / "verify"
_VERIFY_ENV: dict[str, str] | None = None


def _run(cmd: list[str], cwd: Path, *, timeout_s: int | None = None) -> GateResult:
    env = os.environ.copy()
    if _VERIFY_ENV:
        env.update(_VERIFY_ENV)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=env,
        )
        return GateResult(
            name="",
            passed=(proc.returncode == 0),
            cmd=cmd,
            stdout=proc.stdout,
            stderr=proc.stderr,
            note=f"rc={proc.returncode}",
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return GateResult(
            name="",
            passed=False,
            cmd=cmd,
            stdout=stdout,
            stderr=stderr,
            note=f"timeout after {timeout_s}s",
        )
    except OSError as exc:
        return GateResult(
            name="",
            passed=False,
            cmd=cmd,
            stdout="",
            stderr=str(exc),
            note="os error",
        )


def _gate_spec_lint(project_root: Path) -> GateResult:
    cmd = [sys.executable, "-m", "tools.spec_lint", "DESIGN_DOCUMENT.md", "--collect"]
    res = _run(cmd, project_root)
    res.name = "spec_lint"
    return res


def _gate_pytest(project_root: Path) -> GateResult:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    res = _run(cmd, project_root)
    res.name = "pytest"
    return res


def _gate_optional_tool(project_root: Path, name: str, cmd: list[str]) -> GateResult:
    if shutil.which(cmd[0]) is None:
        return GateResult(name=name, passed=True, cmd=cmd, note="skipped (not installed)")
    res = _run(cmd, project_root)
    res.name = name
    return res


def _gate_git_guard(project_root: Path, strict: bool) -> GateResult:
    cmd = [sys.executable, "-m", "tools.git_guard"]
    res = _run(cmd, project_root)
    res.name = "git_guard"
    if strict:
        return res
    # Non-strict mode: warnings do not fail the build.
    if not res.passed:
        res.note = (res.note + "; non-strict: treated as warning").lstrip(";")
        res.passed = True
    return res


def _detect_milestone_id(project_root: Path) -> str | None:
    doc_path = project_root / "DESIGN_DOCUMENT.md"
    if not doc_path.exists():
        return None
    text = doc_path.read_text(encoding="utf-8")
    match = spec_lint.MILESTONE_RE.search(text)
    return match.group(1) if match else None


def _gate_m0_smoke(project_root: Path, *, timeout_s: int) -> GateResult:
    cmd = [sys.executable, "-m", "tools.m0", "smoke"]
    res = _run(cmd, project_root, timeout_s=timeout_s)
    res.name = "m0_smoke"
    return res


def _gate_m0_repro_check(project_root: Path, *, timeout_s: int) -> GateResult:
    cmd = [sys.executable, "-m", "tools.m0", "repro-check"]
    res = _run(cmd, project_root, timeout_s=timeout_s)
    res.name = "m0_repro_check"
    return res


def _resolve_m0_timeout(args: argparse.Namespace) -> int:
    if isinstance(getattr(args, "m0_timeout_s", None), int):
        return int(args.m0_timeout_s)
    env_val = os.environ.get("FF_M0_GATE_TIMEOUT_S")
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            return DEFAULT_M0_GATE_TIMEOUT_S
    return DEFAULT_M0_GATE_TIMEOUT_S


@dataclass(frozen=True)
class VerifyArtifacts:
    run_id: str
    run_dir: Path
    logs_dir: Path
    failures_dir: Path
    tmp_dir: Path


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _init_verify_artifacts(project_root: Path) -> VerifyArtifacts:
    run_id = _timestamp_utc()
    run_dir = project_root / VERIFY_ARTIFACTS_DIR / run_id
    logs_dir = run_dir / "logs"
    failures_dir = run_dir / "failures"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="tmp-", dir=run_dir))
    return VerifyArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        logs_dir=logs_dir,
        failures_dir=failures_dir,
        tmp_dir=tmp_dir,
    )


def _build_verify_env(tmp_dir: Path) -> dict[str, str]:
    env = dict(DETERMINISTIC_ENV)
    tmp_path = str(tmp_dir)
    env["TMPDIR"] = tmp_path
    env["TEMP"] = tmp_path
    env["TMP"] = tmp_path
    return env


def _gate_slug(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name.strip())
    return cleaned or "gate"


def _write_verify_artifacts(
    artifacts: VerifyArtifacts,
    results: list[GateResult],
    payload: dict[str, Any],
    env_overrides: dict[str, str],
) -> None:
    log_lines = [
        f"verify_run={artifacts.run_id}",
        f"tmp_dir={artifacts.tmp_dir}",
        "env:",
    ]
    for key in sorted(env_overrides):
        log_lines.append(f"  {key}={env_overrides[key]}")
    log_lines.append("")

    results_path = artifacts.run_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    env_payload = {
        "run_id": artifacts.run_id,
        "timestamp_utc": artifacts.run_id,
        "env": env_overrides,
    }
    (artifacts.run_dir / "env.json").write_text(
        json.dumps(env_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        log_lines.append(f"[{status}] {result.name} {result.note}".rstrip())
        if result.cmd:
            log_lines.append(f"cmd: {' '.join(result.cmd)}")
        if result.stdout.strip():
            log_lines.append("stdout:")
            log_lines.append(result.stdout.rstrip())
        if result.stderr.strip():
            log_lines.append("stderr:")
            log_lines.append(result.stderr.rstrip())
        log_lines.append("")

        slug = _gate_slug(result.name)
        stdout_path = artifacts.logs_dir / f"{slug}.stdout.log"
        stderr_path = artifacts.logs_dir / f"{slug}.stderr.log"
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        if not result.passed:
            failure_payload = {
                "name": result.name,
                "cmd": result.cmd,
                "note": result.note,
                "stdout_path": str(stdout_path.relative_to(artifacts.run_dir)),
                "stderr_path": str(stderr_path.relative_to(artifacts.run_dir)),
            }
            (artifacts.failures_dir / f"{slug}.json").write_text(
                json.dumps(failure_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    verify_log_path = artifacts.run_dir / "verify.log"
    verify_log_path.write_text("\n".join(log_lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--json", dest="json_path", default="")
    ap.add_argument("--strict-git", action="store_true", help="Fail verify if git_guard fails")
    ap.add_argument("--skip-pytest", action="store_true")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-git", action="store_true")
    ap.add_argument(
        "--include-m0",
        action="store_true",
        help="Run M0 substrate gates even when milestone is not M0.",
    )
    ap.add_argument(
        "--m0-timeout-s",
        type=int,
        default=None,
        help="Override M0 gate timeout in seconds (or set FF_M0_GATE_TIMEOUT_S).",
    )
    args = ap.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    artifacts = _init_verify_artifacts(project_root)
    global _VERIFY_ENV
    _VERIFY_ENV = _build_verify_env(artifacts.tmp_dir)
    os.environ.update(_VERIFY_ENV)

    results: list[GateResult] = []
    results.append(_gate_spec_lint(project_root))

    milestone_id = _detect_milestone_id(project_root)
    run_m0_gates = milestone_id == "M0" or args.include_m0
    if run_m0_gates:
        m0_timeout_s = _resolve_m0_timeout(args)
        results.append(_gate_m0_smoke(project_root, timeout_s=m0_timeout_s))
        results.append(_gate_m0_repro_check(project_root, timeout_s=m0_timeout_s))

    if not args.skip_pytest:
        results.append(_gate_pytest(project_root))

    # Optional quality gates.
    if not args.skip_quality:
        results.append(_gate_optional_tool(project_root, "ruff", ["ruff", "check", "."]))
        results.append(_gate_optional_tool(project_root, "mypy", ["mypy", "."]))

    if not args.skip_git:
        results.append(_gate_git_guard(project_root, strict=args.strict_git))

    required_failures = [r for r in results if not r.passed]

    payload: dict[str, Any] = {
        "ok": len(required_failures) == 0,
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "cmd": r.cmd,
                "note": r.note,
                "stdout": r.stdout[-20000:],
                "stderr": r.stderr[-20000:],
            }
            for r in results
        ],
    }

    if args.json_path:
        out_path = project_root / args.json_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    _write_verify_artifacts(artifacts, results, payload, _VERIFY_ENV)

    # Human-friendly summary
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name} {('(' + r.note + ')') if r.note else ''}")
        if not r.passed:
            if r.stdout.strip():
                print(r.stdout.strip())
            if r.stderr.strip():
                print(r.stderr.strip(), file=sys.stderr)

    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
