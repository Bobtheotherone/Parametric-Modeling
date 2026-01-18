from __future__ import annotations

from tools.completion_gates import CompletionGateInputs, evaluate_completion_gates


def test_orchestrator_completion_gates() -> None:
    assert not evaluate_completion_gates(
        CompletionGateInputs(verify_rc=2, git_porcelain="", head_sha="abc")
    ).ok
    assert not evaluate_completion_gates(
        CompletionGateInputs(verify_rc=0, git_porcelain=" M foo.py\n", head_sha="abc")
    ).ok
    assert not evaluate_completion_gates(
        CompletionGateInputs(verify_rc=0, git_porcelain="", head_sha="")
    ).ok
    assert evaluate_completion_gates(
        CompletionGateInputs(verify_rc=0, git_porcelain="", head_sha="abc")
    ).ok
