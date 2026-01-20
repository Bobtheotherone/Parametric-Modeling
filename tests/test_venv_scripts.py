"""Regression tests ensuring run scripts prefer the venv interpreter.

These tests prevent drift where shell scripts start calling system python3
directly, which can run outside the locked environment and cause missing
deps or nondeterministic failures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script_name",
    [
        "run_live.sh",
        "run_mock.sh",
    ],
)
def test_script_prefers_venv_interpreter(script_name: str) -> None:
    """Verify that run scripts contain venv-preference logic."""
    script_path = REPO_ROOT / script_name
    assert script_path.exists(), f"{script_name} not found at {script_path}"

    content = script_path.read_text(encoding="utf-8")

    # Must check for venv python
    assert ".venv/bin/python" in content, f"{script_name} must contain '.venv/bin/python' check for venv-first logic"

    # Must NOT directly call bare python3 as the primary interpreter
    # (OK if python3 is used as fallback after venv check)
    lines = content.splitlines()
    non_comment_lines = [line for line in lines if not line.strip().startswith("#")]
    direct_python3_calls = [
        line for line in non_comment_lines if "python3 -u bridge/loop.py" in line or "python3 bridge/loop.py" in line
    ]
    assert not direct_python3_calls, (
        f"{script_name} should not directly call 'python3 bridge/loop.py'; "
        f"use a venv-checked $PYTHON variable instead. Found: {direct_python3_calls}"
    )
