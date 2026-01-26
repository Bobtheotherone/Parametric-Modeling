"""Tests to verify that required documentation files exist and contain required sections.

This module ensures that critical documentation for reproduction and determinism
does not regress. It validates both the existence of documentation files and
the presence of required sections within them.

Done when: tests assert reproduction and determinism docs exist and include
required sections. Why: prevents docs regressions from passing verify.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"


class TestReproductionDocsExist:
    """Test that reproduction documentation exists and has required content."""

    def test_reproduction_md_exists(self) -> None:
        """The reproduction.md file must exist."""
        doc_path = DOCS_DIR / "reproduction.md"
        assert doc_path.is_file(), f"Missing required documentation: {doc_path}"

    def test_reproduction_md_has_prerequisites_section(self) -> None:
        """reproduction.md must have a Prerequisites section."""
        doc_path = DOCS_DIR / "reproduction.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "## Prerequisites" in content, "reproduction.md must contain a '## Prerequisites' section"

    def test_reproduction_md_has_clone_instructions(self) -> None:
        """reproduction.md must include git clone instructions."""
        doc_path = DOCS_DIR / "reproduction.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "git clone" in content, "reproduction.md must include git clone instructions"

    def test_reproduction_md_has_bootstrap_instructions(self) -> None:
        """reproduction.md must include bootstrap instructions."""
        doc_path = DOCS_DIR / "reproduction.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "bootstrap_venv.sh" in content, "reproduction.md must reference bootstrap_venv.sh"

    def test_reproduction_md_has_smoke_test_instructions(self) -> None:
        """reproduction.md must include smoke test instructions."""
        doc_path = DOCS_DIR / "reproduction.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "tools.m0 smoke" in content, "reproduction.md must include smoke test command"

    def test_reproduction_md_has_repro_check_instructions(self) -> None:
        """reproduction.md must include repro-check instructions."""
        doc_path = DOCS_DIR / "reproduction.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "tools.m0 repro-check" in content or "repro-check" in content, "reproduction.md must include repro-check command"

    def test_reproduction_md_has_verify_instructions(self) -> None:
        """reproduction.md must include verify instructions."""
        doc_path = DOCS_DIR / "reproduction.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "tools.verify" in content, "reproduction.md must include verify command"

    def test_reproduction_md_has_gates_summary(self) -> None:
        """reproduction.md must have a gates summary section."""
        doc_path = DOCS_DIR / "reproduction.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "Gates" in content or "gates" in content.lower(), "reproduction.md must document M0 gates"

    def test_reproduction_md_has_toolchain_pinning_section(self) -> None:
        """reproduction.md must have toolchain pinning documentation."""
        doc_path = DOCS_DIR / "reproduction.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "Toolchain" in content or "toolchain" in content.lower(), "reproduction.md must document toolchain pinning"

    def test_reproduction_md_has_troubleshooting_section(self) -> None:
        """reproduction.md must have a troubleshooting section."""
        doc_path = DOCS_DIR / "reproduction.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "Troubleshooting" in content or "troubleshoot" in content.lower(), (
            "reproduction.md must have a troubleshooting section"
        )


class TestDeterminismDocsExist:
    """Test that determinism documentation exists and has required content."""

    def test_determinism_md_exists(self) -> None:
        """The determinism.md file must exist."""
        doc_path = DOCS_DIR / "determinism.md"
        assert doc_path.is_file(), f"Missing required documentation: {doc_path}"

    def test_determinism_md_has_overview_section(self) -> None:
        """determinism.md must have an overview section."""
        doc_path = DOCS_DIR / "determinism.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "## 1. Overview" in content or "Overview" in content, "determinism.md must have an overview section"

    def test_determinism_md_has_guarantees_section(self) -> None:
        """determinism.md must document determinism guarantees."""
        doc_path = DOCS_DIR / "determinism.md"
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "guarantee" in content, "determinism.md must document determinism guarantees"

    def test_determinism_md_has_strict_mode_documentation(self) -> None:
        """determinism.md must document strict mode."""
        doc_path = DOCS_DIR / "determinism.md"
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "strict mode" in content or "strict" in content, "determinism.md must document strict mode"

    def test_determinism_md_has_limitations_section(self) -> None:
        """determinism.md must document limitations."""
        doc_path = DOCS_DIR / "determinism.md"
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "limitation" in content or "nondeterminism" in content, (
            "determinism.md must document limitations and nondeterminism sources"
        )

    def test_determinism_md_has_gpu_cpu_section(self) -> None:
        """determinism.md must document GPU/CPU differences."""
        doc_path = DOCS_DIR / "determinism.md"
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "gpu" in content and "cpu" in content, "determinism.md must document GPU/CPU behavior differences"

    def test_determinism_md_has_cublas_documentation(self) -> None:
        """determinism.md must document cuBLAS configuration."""
        doc_path = DOCS_DIR / "determinism.md"
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "cublas" in content, "determinism.md must document cuBLAS workspace configuration"

    def test_determinism_md_has_cudnn_documentation(self) -> None:
        """determinism.md must document cuDNN configuration."""
        doc_path = DOCS_DIR / "determinism.md"
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "cudnn" in content, "determinism.md must document cuDNN deterministic settings"

    def test_determinism_md_has_toolchain_pinning_section(self) -> None:
        """determinism.md must document toolchain pinning."""
        doc_path = DOCS_DIR / "determinism.md"
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "toolchain" in content and "pinning" in content, "determinism.md must document toolchain pinning"

    def test_determinism_md_has_hashing_documentation(self) -> None:
        """determinism.md must document hashing and provenance."""
        doc_path = DOCS_DIR / "determinism.md"
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "hash" in content and "provenance" in content, "determinism.md must document hashing and provenance"

    def test_determinism_md_has_verification_section(self) -> None:
        """determinism.md must document verification and auditing."""
        doc_path = DOCS_DIR / "determinism.md"
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "verification" in content or "audit" in content, "determinism.md must document verification and auditing"

    def test_determinism_md_has_recommendations_section(self) -> None:
        """determinism.md must have recommendations for users."""
        doc_path = DOCS_DIR / "determinism.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "Recommendation" in content or "recommendation" in content.lower(), (
            "determinism.md must have recommendations for users"
        )


class TestDeterminismPolicyDocsExist:
    """Test that DETERMINISM_POLICY.md exists and has required content."""

    def test_determinism_policy_md_exists(self) -> None:
        """The DETERMINISM_POLICY.md file must exist."""
        doc_path = DOCS_DIR / "DETERMINISM_POLICY.md"
        assert doc_path.is_file(), f"Missing required documentation: {doc_path}"

    def test_determinism_policy_has_hash_documentation(self) -> None:
        """DETERMINISM_POLICY.md must document hash computation."""
        doc_path = DOCS_DIR / "DETERMINISM_POLICY.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "design_hash" in content, "DETERMINISM_POLICY.md must document design_hash"
        assert "toolchain_hash" in content, "DETERMINISM_POLICY.md must document toolchain_hash"

    def test_determinism_policy_has_canonicalization_section(self) -> None:
        """DETERMINISM_POLICY.md must document canonicalization."""
        doc_path = DOCS_DIR / "DETERMINISM_POLICY.md"
        content = doc_path.read_text(encoding="utf-8").lower()
        assert "canonical" in content, "DETERMINISM_POLICY.md must document canonicalization"

    def test_determinism_policy_has_manifest_requirements(self) -> None:
        """DETERMINISM_POLICY.md must document manifest requirements."""
        doc_path = DOCS_DIR / "DETERMINISM_POLICY.md"
        content = doc_path.read_text(encoding="utf-8")
        assert "manifest" in content.lower(), "DETERMINISM_POLICY.md must document manifest requirements"


class TestDocsRequiredSectionsIntegrity:
    """Integration tests for documentation section cross-references."""

    def test_reproduction_references_determinism_docs(self) -> None:
        """reproduction.md should reference determinism documentation."""
        reproduction_path = DOCS_DIR / "reproduction.md"
        content = reproduction_path.read_text(encoding="utf-8").lower()
        # Either directly references determinism.md or mentions determinism concepts
        assert "determinism" in content or "deterministic" in content, "reproduction.md should reference determinism concepts"

    def test_determinism_references_policy_docs(self) -> None:
        """determinism.md should reference DETERMINISM_POLICY.md."""
        determinism_path = DOCS_DIR / "determinism.md"
        content = determinism_path.read_text(encoding="utf-8")
        assert "DETERMINISM_POLICY.md" in content, "determinism.md should reference DETERMINISM_POLICY.md"

    @pytest.mark.parametrize(
        "doc_name",
        [
            "reproduction.md",
            "determinism.md",
            "DETERMINISM_POLICY.md",
        ],
    )
    def test_required_docs_are_not_empty(self, doc_name: str) -> None:
        """Required documentation files must not be empty."""
        doc_path = DOCS_DIR / doc_name
        content = doc_path.read_text(encoding="utf-8")
        # Minimum reasonable size for documentation
        assert len(content) > 500, f"{doc_name} appears to be too short (< 500 chars)"

    @pytest.mark.parametrize(
        "doc_name",
        [
            "reproduction.md",
            "determinism.md",
            "DETERMINISM_POLICY.md",
        ],
    )
    def test_required_docs_have_title(self, doc_name: str) -> None:
        """Required documentation files must have a title."""
        doc_path = DOCS_DIR / doc_name
        content = doc_path.read_text(encoding="utf-8")
        # Check for markdown title (# Title)
        assert content.strip().startswith("#"), f"{doc_name} must start with a markdown title"
