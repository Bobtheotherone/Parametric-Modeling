from __future__ import annotations

from pathlib import Path

from tools.spec_lint import lint_design_document


def test_design_document_contract() -> None:
    res = lint_design_document(Path("DESIGN_DOCUMENT.md"))
    assert res.ok, f"DESIGN_DOCUMENT.md failed lint: {res.issues}"
