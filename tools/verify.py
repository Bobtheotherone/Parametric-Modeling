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
import shutil
import subprocess
import sys
from dataclasses import dataclass
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


M0_GATE_TIMEOUT_S = 90


def _run(cmd: list[str], cwd: Path, *, timeout_s: int | None = None) -> GateResult:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_s,
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


def _gate_m0_smoke(project_root: Path) -> GateResult:
    cmd = [sys.executable, "-m", "tools.m0", "smoke"]
    res = _run(cmd, project_root, timeout_s=M0_GATE_TIMEOUT_S)
    res.name = "m0_smoke"
    return res


def _gate_m0_repro_check(project_root: Path) -> GateResult:
    cmd = [sys.executable, "-m", "tools.m0", "repro-check"]
    res = _run(cmd, project_root, timeout_s=M0_GATE_TIMEOUT_S)
    res.name = "m0_repro_check"
    return res


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
    args = ap.parse_args(argv)

    project_root = Path(args.project_root).resolve()

    results: list[GateResult] = []
    results.append(_gate_spec_lint(project_root))

    milestone_id = _detect_milestone_id(project_root)
    run_m0_gates = milestone_id == "M0" or args.include_m0
    if run_m0_gates:
        results.append(_gate_m0_smoke(project_root))
        results.append(_gate_m0_repro_check(project_root))

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
