from __future__ import annotations

from pathlib import Path

from tools.spec_lint import lint_design_document


def test_spec_lint_detects_missing_sections(tmp_path: Path) -> None:
    doc = tmp_path / "DESIGN_DOCUMENT.md"
    doc.write_text("# Doc\n\n**Milestone:** M1 — X\n\n## Normative Requirements (must)\n\n- [REQ-M1-001] hi\n", encoding="utf-8")

    res = lint_design_document(doc)
    assert not res.ok
    # Should complain about missing DoD and Test Matrix.
    joined = "\n".join(res.issues)
    assert "Definition of Done" in joined
    assert "Test Matrix" in joined
