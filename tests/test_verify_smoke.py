from __future__ import annotations

import subprocess
import sys


def test_verify_script_exists_and_runs() -> None:
    # Avoid recursion by skipping pytest itself.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.verify",
            "--skip-pytest",
            "--skip-quality",
            "--skip-git",
        ],
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, f"verify smoke failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
