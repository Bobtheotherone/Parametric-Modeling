"""Design-document linter.

This repository treats DESIGN_DOCUMENT.md as *normative*. The core contract:
- A milestone identifier exists (e.g., "**Milestone:** M3 — ...").
- A "Normative Requirements" section lists stable requirement IDs.
- A "Definition of Done" section exists.
- A "Test Matrix" maps each requirement ID to >=1 pytest node id.

The linter is intentionally strict so that agentic workflows can prove compliance via pytest.

Usage:
  python -m tools.spec_lint DESIGN_DOCUMENT.md [--collect]

Exit codes:
  0 = pass
  2 = lint failures
  3 = tool error
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REQ_RE = re.compile(r"^\s*-\s*\[(REQ-[A-Z0-9]+-\d{3,})\]\s+(.+?)\s*$")
MILESTONE_RE = re.compile(r"\*\*Milestone:\*\*\s*(M\d+)\b")


@dataclass(frozen=True)
class SpecLintResult:
    ok: bool
    milestone_id: str | None
    requirement_ids: list[str]
    matrix_map: dict[str, list[str]]
    issues: list[str]


def _collect_pytest_node_ids(project_root: Path) -> set[str]:
    # NOTE: `--collect-only -q` output changed across pytest versions (some versions
    # print file counts instead of node IDs). We intentionally avoid `-q` and parse
    # the explicit node id listing.
    cmd = [sys.executable, "-m", "pytest", "--collect-only"]
    proc = subprocess.run(cmd, cwd=str(project_root), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"pytest collection failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    node_ids: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        if "::" in line and (line.startswith("tests/") or line.startswith("test/")):
            node_ids.add(line)
    return node_ids


def _find_section(lines: list[str], header: str) -> int | None:
    h = header.strip().lower()
    for i, line in enumerate(lines):
        if line.strip().lower() == h:
            return i
    return None


def _parse_test_matrix(lines: list[str]) -> dict[str, list[str]]:
    table_start = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("| requirement") and "pytest" in line.lower():
            table_start = i
            break
    if table_start is None:
        return {}

    mapping: dict[str, list[str]] = {}
    for row in lines[table_start + 2 :]:
        if not row.strip().startswith("|"):
            break
        cols = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cols) < 2:
            continue
        req = cols[0].strip()
        tests = cols[1].strip()
        test_ids = [t.strip() for t in re.split(r"[,;]", tests) if t.strip()]
        if req:
            mapping[req] = test_ids
    return mapping


def lint_design_document(doc_path: Path) -> SpecLintResult:
    issues: list[str] = []
    if not doc_path.exists():
        return SpecLintResult(
            ok=False,
            milestone_id=None,
            requirement_ids=[],
            matrix_map={},
            issues=[f"DESIGN document not found: {doc_path}"],
        )

    text = doc_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    milestone_id: str | None = None
    m = MILESTONE_RE.search(text)
    if m:
        milestone_id = m.group(1)
    else:
        issues.append("Missing milestone line like '**Milestone:** M# — ...'.")

    req_ids: list[str] = []
    for line in lines:
        m = REQ_RE.match(line)
        if m:
            req_ids.append(m.group(1))

    if not req_ids:
        issues.append(
            "No normative requirements found. Expected bullet lines like: - [REQ-M3-001] ..."
        )

    if _find_section(lines, "## Normative Requirements (must)") is None and _find_section(
        lines, "## Normative Requirements"
    ) is None:
        issues.append("Missing '## Normative Requirements' section header.")

    if _find_section(lines, "## Definition of Done") is None:
        issues.append("Missing '## Definition of Done' section header.")

    matrix = _parse_test_matrix(lines)
    if not matrix:
        issues.append("Missing or malformed '## Test Matrix' markdown table.")

    if req_ids and matrix:
        missing = [r for r in req_ids if r not in matrix]
        if missing:
            issues.append(f"Test Matrix missing requirement IDs: {missing}")
        empty = [r for r, ts in matrix.items() if not ts]
        if empty:
            issues.append(f"Test Matrix has empty test list for: {empty}")

    return SpecLintResult(
        ok=(len(issues) == 0),
        milestone_id=milestone_id,
        requirement_ids=req_ids,
        matrix_map=matrix,
        issues=issues,
    )


def verify_test_matrix_exists_in_pytest(project_root: Path, mapping: dict[str, list[str]]) -> list[str]:
    issues: list[str] = []
    node_ids = _collect_pytest_node_ids(project_root)
    for req, tests in mapping.items():
        for t in tests:
            if t not in node_ids:
                issues.append(f"Requirement {req} maps to missing pytest node id: {t}")
    return issues


def _print_issues(issues: Iterable[str]) -> None:
    for it in issues:
        print(f"- {it}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("doc", type=str, nargs="?", default="DESIGN_DOCUMENT.md")
    ap.add_argument("--collect", action="store_true", help="Verify that mapped pytest node ids exist")
    args = ap.parse_args()

    doc_path = Path(args.doc).resolve()
    res = lint_design_document(doc_path)

    issues = list(res.issues)

    if args.collect and res.matrix_map:
        try:
            issues.extend(verify_test_matrix_exists_in_pytest(doc_path.parent, res.matrix_map))
        except Exception as e:
            print(f"spec_lint tool error: {e}", file=sys.stderr)
            return 3

    if issues:
        print("DESIGN_DOCUMENT lint FAILED:\n")
        _print_issues(issues)
        return 2

    print(
        f"DESIGN_DOCUMENT lint PASSED (milestone={res.milestone_id}, requirements={len(res.requirement_ids)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
