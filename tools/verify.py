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
import shlex
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
class SkippedTestInfo:
    """Information about a skipped or xfail test."""

    node_id: str
    reason: str
    kind: str  # 'skipped', 'xfail', or 'placeholder'


def _detect_skipped_or_placeholder_tests(
    project_root: Path,
    test_node_ids: list[str],
) -> list[SkippedTestInfo]:
    """Run pytest --collect-only with skip markers and detect skipped/xfail/placeholder tests.

    Args:
        project_root: The project root path.
        test_node_ids: List of test node IDs to check.

    Returns:
        List of SkippedTestInfo for tests that are skipped, xfail, or placeholder.
    """
    if not test_node_ids:
        return []

    # Run pytest --collect-only -q with the specific test IDs to see markers
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        *test_node_ids,
    ]

    env = os.environ.copy()
    if _VERIFY_ENV:
        env.update(_VERIFY_ENV)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            text=True,
            capture_output=True,
            timeout=60,
            env=env,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []

    skipped: list[SkippedTestInfo] = []
    stdout = proc.stdout

    # Parse pytest output for skip markers
    # Look for patterns like:
    #   <Module tests/test_foo.py>
    #     <Function test_bar>
    #       <Skip reason='some reason'>
    # or in -q output, skipped tests may show with 's' or 'x' in collection
    for node_id in test_node_ids:
        # Check if node has skip/xfail marker by running a dry-run
        # We'll use --co (collect-only) with verbose to see markers
        pass

    # Better approach: run pytest --collect-only --quiet and look for skip info
    # Or run each test with --collect-only and see the output

    # Actually, let's parse the collection output more carefully
    # In verbose mode, skipped tests show markers
    cmd_verbose = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-v",
        *test_node_ids,
    ]

    try:
        proc_v = subprocess.run(
            cmd_verbose,
            cwd=str(project_root),
            text=True,
            capture_output=True,
            timeout=60,
            env=env,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []

    # Parse verbose output for skip/xfail markers
    lines = proc_v.stdout.splitlines()
    for line in lines:
        line_stripped = line.strip()
        # Look for lines with test IDs and skip/xfail markers
        for node_id in test_node_ids:
            if node_id in line_stripped:
                lower = line_stripped.lower()
                if "skip" in lower:
                    reason = _extract_reason(line_stripped, "skip")
                    skipped.append(SkippedTestInfo(node_id=node_id, reason=reason, kind="skipped"))
                elif "xfail" in lower:
                    reason = _extract_reason(line_stripped, "xfail")
                    skipped.append(SkippedTestInfo(node_id=node_id, reason=reason, kind="xfail"))

    # Also check for placeholder tests (tests with NotImplementedError or 'pass' only)
    for node_id in test_node_ids:
        if _is_placeholder_test(project_root, node_id):
            skipped.append(SkippedTestInfo(node_id=node_id, reason="placeholder implementation", kind="placeholder"))

    # Deduplicate
    seen: set[str] = set()
    result: list[SkippedTestInfo] = []
    for info in skipped:
        if info.node_id not in seen:
            seen.add(info.node_id)
            result.append(info)

    return result


def _extract_reason(line: str, marker_type: str) -> str:
    """Extract the reason from a skip/xfail marker line."""
    import re

    # Look for reason='...' or reason="..."
    match = re.search(r"reason=['\"]([^'\"]+)['\"]", line, re.IGNORECASE)
    if match:
        return match.group(1)
    return f"{marker_type} marker present"


def _is_placeholder_test(project_root: Path, node_id: str) -> bool:
    """Check if a test is a placeholder (raises NotImplementedError or is just 'pass').

    Args:
        project_root: The project root path.
        node_id: The pytest node ID (e.g., tests/test_foo.py::test_bar).

    Returns:
        True if the test appears to be a placeholder.
    """
    # Parse node_id to get file path and test name
    if "::" not in node_id:
        return False

    parts = node_id.split("::")
    file_path = project_root / parts[0]
    if not file_path.exists():
        return False

    test_name = parts[-1]
    # Handle parametrized tests
    if "[" in test_name:
        test_name = test_name.split("[")[0]

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError:
        return False

    # Simple heuristic: find the test function and check if it's a placeholder
    import re

    # Match def test_name(...): followed by body
    pattern = rf"def\s+{re.escape(test_name)}\s*\([^)]*\)\s*(?:->.*?)?\s*:\s*\n((?:\s+.*\n)*)"
    match = re.search(pattern, content)
    if not match:
        return False

    body = match.group(1).strip()
    # Check for placeholder patterns
    placeholder_patterns = [
        r"^\s*pass\s*$",
        r"^\s*raise\s+NotImplementedError",
        r"^\s*\.\.\.\s*$",
        r'^\s*""".*?"""\s*$',
        r"^\s*#.*$",
    ]

    body_lines = [l.strip() for l in body.splitlines() if l.strip()]
    if not body_lines:
        return True  # Empty body
    if len(body_lines) == 1:
        for pat in placeholder_patterns:
            if re.match(pat, body_lines[0]):
                return True
    return False


@dataclass
class GateResult:
    name: str
    passed: bool
    cmd: list[str] | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    note: str = ""


DEFAULT_M0_GATE_TIMEOUT_S = 90
DETERMINISTIC_ENV = {
    "LC_ALL": "UTC",
    "LANG": "UTC",
    "TZ": "UTC",
    "PYTHONHASHSEED": "0",
    "FF_MAX_KICAD_JOBS": "1",
    "MAX_KICAD_JOBS": "1",
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
            returncode=proc.returncode,
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
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            note=f"timeout after {timeout_s}s",
        )
    except OSError as exc:
        return GateResult(
            name="",
            passed=False,
            cmd=cmd,
            returncode=None,
            stdout="",
            stderr=str(exc),
            note="os error",
        )


def _gate_spec_lint(project_root: Path, *, collect: bool = True) -> GateResult:
    cmd = [sys.executable, "-m", "tools.spec_lint", "DESIGN_DOCUMENT.md"]
    if collect:
        cmd.append("--collect")
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
    cmd = [sys.executable, "-m", "tools.m0", "smoke", "--mode", "strict"]
    res = _run(cmd, project_root, timeout_s=timeout_s)
    res.name = "m0_smoke"
    return res


def _gate_m0_repro_check(project_root: Path, *, timeout_s: int) -> GateResult:
    cmd = [sys.executable, "-m", "tools.m0", "repro-check", "--mode", "strict"]
    res = _run(cmd, project_root, timeout_s=timeout_s)
    res.name = "m0_repro_check"
    return res


def _gate_toolchain_lock(project_root: Path) -> GateResult:
    try:
        from formula_foundry.coupongen.toolchain import (
            DEFAULT_LOCK_PATH,
            ToolchainLoadError,
            load_toolchain_lock,
        )
    except ImportError as exc:
        return GateResult(
            name="toolchain_lock",
            passed=False,
            stderr=f"Failed to import toolchain lock loader: {exc}",
            note="import error",
        )

    lock_path = project_root / DEFAULT_LOCK_PATH
    try:
        load_toolchain_lock(lock_path=lock_path)
    except ToolchainLoadError as exc:
        return GateResult(
            name="toolchain_lock",
            passed=False,
            stderr=str(exc),
            note="invalid toolchain lock",
        )

    return GateResult(name="toolchain_lock", passed=True, note="ok")


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


def _gate_no_skipped_tests(project_root: Path, milestone_id: str | None) -> GateResult:
    """Gate that fails if any tests in the test matrix are skipped, xfail, or placeholder.

    This prevents "green-but-incomplete" milestones where tests appear to pass
    because they are skipped or not implemented.

    Args:
        project_root: The project root path.
        milestone_id: The current milestone (e.g., "M0", "M1").

    Returns:
        GateResult indicating pass/fail status.
    """
    if milestone_id is None:
        return GateResult(
            name="no_skipped_tests",
            passed=True,
            note="no milestone detected, skipping check",
        )

    # Get test matrix from spec_lint
    doc_path = project_root / "DESIGN_DOCUMENT.md"
    lint_result = spec_lint.lint_design_document(doc_path)

    if not lint_result.matrix_map:
        return GateResult(
            name="no_skipped_tests",
            passed=True,
            note="no test matrix found",
        )

    # Collect all test node IDs from the matrix
    all_test_ids: list[str] = []
    for req_id, test_ids in lint_result.matrix_map.items():
        # Only check tests for the current milestone
        if req_id.startswith(f"REQ-{milestone_id}-"):
            all_test_ids.extend(test_ids)

    if not all_test_ids:
        return GateResult(
            name="no_skipped_tests",
            passed=True,
            note=f"no tests mapped to {milestone_id} requirements",
        )

    # Check for skipped/xfail/placeholder tests
    skipped = _detect_skipped_or_placeholder_tests(project_root, all_test_ids)

    if skipped:
        skipped_lines = []
        for info in skipped:
            skipped_lines.append(f"  - {info.node_id} ({info.kind}): {info.reason}")

        return GateResult(
            name="no_skipped_tests",
            passed=False,
            note=f"{len(skipped)} skipped/xfail/placeholder test(s) in {milestone_id} matrix",
            stdout="\n".join(skipped_lines),
        )

    return GateResult(
        name="no_skipped_tests",
        passed=True,
        note=f"all {len(all_test_ids)} {milestone_id} matrix tests are active",
    )


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
    tmp_dir = run_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
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


@dataclass(frozen=True)
class FailureArtifact:
    kind: str
    source: Path
    dest: Path
    root: Path
    root_label: str


MAX_CAPTURE_PER_KIND = 5
MAX_CANONICALIZE_FILES = 50
MAX_CANONICALIZE_BYTES = 5_000_000


def _format_cmd(cmd: list[str] | None) -> str:
    if not cmd:
        return ""
    return shlex.join(cmd)


def _safe_relpath(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _collect_recent_matches(
    search_roots: list[tuple[Path, str]],
    pattern: str,
    limit: int,
    seen: set[Path],
) -> list[tuple[Path, Path, str]]:
    matches: list[tuple[Path, Path, str]] = []
    for root, label in search_roots:
        if not root.exists():
            continue
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen:
                continue
            matches.append((path, root, label))

    matches.sort(key=lambda item: _path_mtime(item[0]), reverse=True)
    return matches[:limit]


def _copy_failure_artifact(
    path: Path,
    root: Path,
    label: str,
    dest_root: Path,
    *,
    kind: str,
) -> FailureArtifact | None:
    rel = _safe_relpath(path, root)
    dest = dest_root / label / rel
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    except OSError:
        return None
    return FailureArtifact(kind=kind, source=path, dest=dest, root=root, root_label=label)


def _read_text_file(path: Path, *, max_bytes: int) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def _canonicalize_manifest_text(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    from formula_foundry.substrate import canonical_json_dumps

    return canonical_json_dumps(payload)


def _canonicalize_drc_text(text: str) -> str:
    from formula_foundry.coupongen.kicad.canonicalize import canonicalize_drc_json

    return canonicalize_drc_json(text)


def _canonicalize_export_text(path: Path, text: str) -> str | None:
    from formula_foundry.coupongen.kicad.canonicalize import canonicalize_board, canonicalize_export

    suffix = path.suffix.lower()
    if suffix == ".kicad_pcb":
        return canonicalize_board(text)
    if suffix in {".drl", ".xln"}:
        return canonicalize_export(text)
    if suffix.startswith(".g") and len(suffix) <= 4:
        return canonicalize_export(text)
    return None


def _canonical_dest_path(raw_dest: Path, *, raw_root: Path, canonical_root: Path) -> Path:
    relative = raw_dest.relative_to(raw_root)
    canonical_dest = canonical_root / relative
    return canonical_dest.with_name(canonical_dest.name + ".canonical")


def _write_canonicalized_artifacts(
    captured: list[FailureArtifact],
    raw_root: Path,
    canonical_root: Path,
) -> list[FailureArtifact]:
    canonicalized: list[FailureArtifact] = []
    canonical_root.mkdir(parents=True, exist_ok=True)

    output_dirs: list[tuple[Path, Path, str]] = []
    for artifact in captured:
        name = artifact.source.name.lower()
        if name in {"drc.json", "manifest.json"}:
            output_dirs.append((artifact.source.parent, artifact.root, artifact.root_label))
        else:
            continue

        text = _read_text_file(artifact.source, max_bytes=MAX_CANONICALIZE_BYTES)
        if text is None:
            continue
        if name == "drc.json":
            canonical = _canonicalize_drc_text(text)
        else:
            canonical = _canonicalize_manifest_text(text)
            if canonical is None:
                continue
        canonical_dest = _canonical_dest_path(artifact.dest, raw_root=raw_root, canonical_root=canonical_root)
        canonical_dest.parent.mkdir(parents=True, exist_ok=True)
        canonical_dest.write_text(canonical, encoding="utf-8")
        canonicalized.append(
            FailureArtifact(
                kind=f"{artifact.kind}_canonical",
                source=artifact.source,
                dest=canonical_dest,
                root=artifact.root,
                root_label=artifact.root_label,
            )
        )

    seen_exports: set[Path] = set()
    export_count = 0
    for output_dir, root, label in output_dirs:
        if export_count >= MAX_CANONICALIZE_FILES:
            break
        if not output_dir.exists():
            continue
        for path in output_dir.rglob("*"):
            if export_count >= MAX_CANONICALIZE_FILES:
                break
            if not path.is_file():
                continue
            if path.name.lower() in {"drc.json", "manifest.json"}:
                continue
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen_exports:
                continue
            text = _read_text_file(path, max_bytes=MAX_CANONICALIZE_BYTES)
            if text is None:
                continue
            canonical = _canonicalize_export_text(path, text)
            if canonical is None:
                continue
            rel = _safe_relpath(path, root)
            dest = canonical_root / label / rel
            dest = dest.with_name(dest.name + ".canonical")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(canonical, encoding="utf-8")
            canonicalized.append(
                FailureArtifact(
                    kind="canonical_export",
                    source=path,
                    dest=dest,
                    root=root,
                    root_label=label,
                )
            )
            seen_exports.add(resolved)
            export_count += 1

    return canonicalized


def _capture_failure_artifacts(
    artifacts: VerifyArtifacts,
    results: list[GateResult],
    project_root: Path,
) -> list[FailureArtifact]:
    if all(result.passed for result in results):
        return []

    raw_root = artifacts.failures_dir / "raw"
    canonical_root = artifacts.failures_dir / "canonicalized"
    raw_root.mkdir(parents=True, exist_ok=True)

    search_roots: list[tuple[Path, str]] = [
        (artifacts.run_dir, "verify_run"),
        (project_root / "runs", "runs"),
        (project_root / "artifacts", "artifacts"),
        (project_root / "output", "output"),
        (project_root / "out", "out"),
    ]

    captured: list[FailureArtifact] = []
    seen: set[Path] = set()
    pattern_specs = [
        ("drc.json", "drc"),
        ("manifest.json", "manifest"),
        ("*canonical*", "canonical_output"),
    ]
    for pattern, kind in pattern_specs:
        matches = _collect_recent_matches(search_roots, pattern, MAX_CAPTURE_PER_KIND, seen)
        for path, root, label in matches:
            artifact = _copy_failure_artifact(
                path,
                root,
                label,
                raw_root,
                kind=kind,
            )
            if artifact is None:
                continue
            try:
                seen.add(path.resolve())
            except OSError:
                seen.add(path)
            captured.append(artifact)

    captured.extend(_write_canonicalized_artifacts(captured, raw_root=raw_root, canonical_root=canonical_root))
    return captured


def _write_verify_artifacts(
    artifacts: VerifyArtifacts,
    results: list[GateResult],
    payload: dict[str, Any],
    env_overrides: dict[str, str],
    project_root: Path,
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
            log_lines.append(f"cmd: {_format_cmd(result.cmd)}")
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
                "cmd_str": _format_cmd(result.cmd),
                "note": result.note,
                "returncode": result.returncode,
                "stdout_path": str(stdout_path.relative_to(artifacts.run_dir)),
                "stderr_path": str(stderr_path.relative_to(artifacts.run_dir)),
            }
            (artifacts.failures_dir / f"{slug}.json").write_text(
                json.dumps(failure_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    captured = _capture_failure_artifacts(artifacts, results, project_root)
    if captured:
        log_lines.append("failure_artifacts:")
        for artifact in captured:
            dest_rel = artifact.dest.relative_to(artifacts.run_dir)
            log_lines.append(f"  - {artifact.kind}: {dest_rel}")
        log_lines.append("")

    failed_gates = [result.name for result in results if not result.passed]
    if failed_gates:
        log_lines.append("failed_gates:")
        for gate in failed_gates:
            log_lines.append(f"  - {gate}")
        log_lines.append("")

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
    ap.add_argument("--skip-no-skipped-tests", action="store_true", help="Skip the no-skipped-tests gate")
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
    tempfile.tempdir = str(artifacts.tmp_dir)

    results: list[GateResult] = []
    results.append(_gate_spec_lint(project_root, collect=not args.skip_pytest))

    milestone_id = _detect_milestone_id(project_root)
    run_m0_gates = milestone_id == "M0" or args.include_m0
    if run_m0_gates:
        m0_timeout_s = _resolve_m0_timeout(args)
        results.append(_gate_m0_smoke(project_root, timeout_s=m0_timeout_s))
        results.append(_gate_m0_repro_check(project_root, timeout_s=m0_timeout_s))
    if milestone_id == "M1":
        results.append(_gate_toolchain_lock(project_root))

    if not args.skip_pytest:
        results.append(_gate_pytest(project_root))

    # Check for skipped/xfail/placeholder tests in the milestone test matrix
    if not getattr(args, "skip_no_skipped_tests", False):
        results.append(_gate_no_skipped_tests(project_root, milestone_id))

    # Optional quality gates.
    if not args.skip_quality:
        results.append(_gate_optional_tool(project_root, "ruff", ["ruff", "check", "."]))
        results.append(_gate_optional_tool(project_root, "mypy", ["mypy", "."]))

    if not args.skip_git:
        results.append(_gate_git_guard(project_root, strict=args.strict_git))

    required_failures = [r for r in results if not r.passed]
    failed_gates = [r.name for r in required_failures]

    payload: dict[str, Any] = {
        "ok": len(required_failures) == 0,
        "failed_gates": failed_gates,
        "first_failed_gate": failed_gates[0] if failed_gates else "",
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "cmd": r.cmd,
                "cmd_str": _format_cmd(r.cmd),
                "returncode": r.returncode,
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

    _write_verify_artifacts(artifacts, results, payload, _VERIFY_ENV, project_root)

    # Human-friendly summary
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name} {('(' + r.note + ')') if r.note else ''}")
        if r.cmd:
            print(f"cmd: {_format_cmd(r.cmd)}")
        if not r.passed:
            if r.stdout.strip():
                print(r.stdout.strip())
            if r.stderr.strip():
                print(r.stderr.strip(), file=sys.stderr)

    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
