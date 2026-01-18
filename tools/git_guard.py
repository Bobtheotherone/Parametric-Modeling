"""Lightweight git + secret hygiene checks.

This is *not* a replacement for a proper secret scanner. It's a fast, local guardrail
for agentic workflows so you don't accidentally commit credentials while iterating.

Checks:
- Warn if .env exists and is not ignored.
- Scan tracked files for common credential patterns.
- Print git status summary.

Exit code:
- 0 if clean / no hits
- 2 if potential secrets found
- 3 if tool error
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9\-]{10,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b")),
    (
        "Generic API_KEY assignment",
        re.compile(r"\b[A-Z0-9_]*API_KEY\s*=\s*['\"][^'\"]{8,}['\"]"),
    ),
    ("Bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9\._\-]{20,}\b")),
]


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def _git_root() -> Path:
    rc, out, err = _run(["git", "rev-parse", "--show-toplevel"], Path.cwd())
    if rc != 0:
        raise RuntimeError(f"Not a git repo (or git not available): {err.strip()}")
    return Path(out.strip())


def _tracked_files(root: Path) -> list[Path]:
    rc, out, err = _run(["git", "ls-files"], root)
    if rc != 0:
        raise RuntimeError(err.strip())
    files: list[Path] = []
    for line in out.splitlines():
        p = (root / line.strip()).resolve()
        if p.is_file():
            files.append(p)
    return files


def _scan_file(path: Path) -> list[str]:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    hits: list[str] = []
    for label, pat in SECRET_PATTERNS:
        if pat.search(data):
            hits.append(label)
    return hits


def main() -> int:
    try:
        root = _git_root()
    except Exception as e:
        print(f"git_guard error: {e}", file=sys.stderr)
        return 3

    findings: list[str] = []

    env_path = root / ".env"
    if env_path.exists():
        rc, _, _ = _run(["git", "check-ignore", ".env"], root)
        if rc != 0:
            findings.append(".env exists but is not ignored by git")

    for f in _tracked_files(root):
        hits = _scan_file(f)
        for h in hits:
            findings.append(f"{f.relative_to(root)}: {h}")

    rc, status, err = _run(["git", "status", "--porcelain=v1"], root)
    if rc != 0:
        print(err, file=sys.stderr)
        return 3

    if status.strip():
        print("git status --porcelain reports uncommitted changes:")
        print(status.strip())

    if findings:
        print("Potential secret/hygiene issues detected:")
        for f in findings:
            print(f"- {f}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
