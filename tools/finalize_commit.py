"""Finalize an agent run by staging and committing.

This is used by the orchestrator's completion gates. It intentionally refuses to commit
unless verification passes.

Usage:
  python -m tools.finalize_commit --milestone M3

Exit codes:
  0 success
  2 preconditions failed
  3 tool error
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", required=True, help="Milestone id like M0")
    ap.add_argument("--message", default="", help="Override commit message")
    args = ap.parse_args()

    root = Path.cwd()

    # 1) Verify gates.
    rc, out, err = _run([sys.executable, "-m", "tools.verify", "--strict-git"], root)
    if rc != 0:
        print("Refusing to commit: verification failed.")
        if out.strip():
            print(out.strip())
        if err.strip():
            print(err.strip(), file=sys.stderr)
        return 2

    # 2) Stage everything.
    rc, _, err = _run(["git", "add", "-A"], root)
    if rc != 0:
        print(err.strip(), file=sys.stderr)
        return 3

    # 3) Ensure no unstaged changes remain.
    rc, _, err = _run(["git", "diff", "--quiet"], root)
    if rc not in (0, 1):
        print(err.strip(), file=sys.stderr)
        return 3
    if rc == 1:
        print("Refusing to commit: unstaged changes remain after 'git add -A'.")
        return 3

    # 4) Commit if there are staged changes.
    rc, _, err = _run(["git", "diff", "--cached", "--quiet"], root)
    if rc not in (0, 1):
        print(err.strip(), file=sys.stderr)
        return 3

    if rc == 0:
        print("Nothing to commit (index clean).")
        return 0

    msg = args.message.strip() or f"milestone({args.milestone}): satisfy DESIGN_DOCUMENT"
    rc, out, err = _run(["git", "commit", "-m", msg], root)
    if rc != 0:
        print(out.strip())
        print(err.strip(), file=sys.stderr)
        return 3

    print(out.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
