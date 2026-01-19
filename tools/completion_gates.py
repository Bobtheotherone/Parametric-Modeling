"""Completion gate helpers.

The orchestrator refuses to stop unless these gates pass:
- strict verification (`python -m tools.verify --strict-git`)
- clean working tree (`git status --porcelain` empty)
- at least one commit exists (`git rev-parse HEAD` succeeds)

We expose a pure predicate so pytest can validate the logic without relying on git.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionGateInputs:
    verify_rc: int
    git_porcelain: str
    head_sha: str


@dataclass(frozen=True)
class CompletionGateResult:
    ok: bool
    reason: str


def evaluate_completion_gates(inp: CompletionGateInputs) -> CompletionGateResult:
    if inp.verify_rc != 0:
        return CompletionGateResult(False, "tools.verify failed")
    if inp.git_porcelain.strip():
        return CompletionGateResult(False, "git status not clean")
    if not inp.head_sha.strip():
        return CompletionGateResult(False, "no commit found")
    return CompletionGateResult(True, "ok")


def verify_args_for_completion(milestone_id: str) -> list[str]:
    cmd = [sys.executable, "-m", "tools.verify", "--strict-git"]
    if _milestone_requires_m0(milestone_id):
        cmd.append("--include-m0")
    return cmd


def _milestone_requires_m0(milestone_id: str) -> bool:
    if not milestone_id.startswith("M"):
        return False
    try:
        value = int(milestone_id[1:])
    except ValueError:
        return False
    return value >= 1
