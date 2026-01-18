from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_reproduction_docs_present() -> None:
    doc_path = ROOT / "docs" / "reproduction.md"
    assert doc_path.is_file()

    content = doc_path.read_text(encoding="utf-8")
    required_snippets = [
        "git clone",
        "bootstrap_venv.sh",
        "tools.m0 smoke",
        "tools.m0 repro-check",
        "tools.verify --strict-git",
    ]
    for snippet in required_snippets:
        assert snippet in content


def test_determinism_limitations_documented() -> None:
    doc_path = ROOT / "docs" / "determinism.md"
    assert doc_path.is_file()

    content = doc_path.read_text(encoding="utf-8").lower()
    assert "strict mode" in content
    assert "guarantee" in content
    assert "nondeterminism" in content
    assert "cublas" in content
    assert "cudnn" in content
