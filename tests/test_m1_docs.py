# SPDX-License-Identifier: MIT
"""Documentation tests for M1 compliance.

This module verifies that M1-related documentation exists and contains
required sections with key assertions about:
- Hash inputs (what is included in design_hash, toolchain_hash)
- Exclusions (what is NOT hashed)
- Digest pinning (Docker image pinning requirements)
- Determinism gates (G1-G5)

Satisfies: Task m0-6 - Determinism and toolchain documentation with verifiable assertions
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"


class TestM1DeterminismPolicyDoc:
    """Tests for docs/m1_determinism_policy.md existence and required sections."""

    DOC_PATH = DOCS_DIR / "m1_determinism_policy.md"

    def test_file_exists(self) -> None:
        """m1_determinism_policy.md must exist."""
        assert self.DOC_PATH.is_file(), f"Missing: {self.DOC_PATH}"

    def test_has_hash_inputs_section(self) -> None:
        """Document must have a Hash Inputs section."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "hash inputs" in content, "Missing 'Hash Inputs' section"

    def test_documents_design_hash(self) -> None:
        """Document must explain design_hash."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "design_hash" in content, "Missing design_hash documentation"
        assert "sha256" in content, "Missing SHA256 reference for design_hash"

    def test_documents_toolchain_hash(self) -> None:
        """Document must explain toolchain_hash."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "toolchain_hash" in content, "Missing toolchain_hash documentation"

    def test_has_exclusions_section(self) -> None:
        """Document must have an Exclusions section."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "exclusion" in content, "Missing 'Exclusions' section"

    def test_documents_timestamp_exclusions(self) -> None:
        """Document must explain what timestamps are excluded from hashes."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "timestamp" in content, "Missing timestamp exclusion documentation"
        assert "excluded" in content or "exclusion" in content, "Missing exclusion language"

    def test_documents_uuid_exclusions(self) -> None:
        """Document must explain UUID exclusions."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "uuid" in content, "Missing UUID exclusion documentation"

    def test_has_digest_pinning_section(self) -> None:
        """Document must have a Digest Pinning section."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "digest pinning" in content or "docker_digest" in content, "Missing 'Digest Pinning' section"

    def test_documents_docker_digest_requirement(self) -> None:
        """Document must explain docker_digest requirements."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "docker" in content, "Missing Docker documentation"
        assert "sha256" in content, "Missing SHA256 digest format"
        assert "placeholder" in content, "Missing PLACEHOLDER discussion"

    def test_has_determinism_gates_section(self) -> None:
        """Document must have a Determinism Gates section."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "determinism gate" in content or "g1" in content, "Missing 'Determinism Gates' section"

    def test_documents_all_gates(self) -> None:
        """Document must reference gates G1-G5."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        for gate in ["g1", "g2", "g3", "g4", "g5"]:
            assert gate in content, f"Missing gate {gate.upper()} documentation"

    def test_has_canonicalization_section(self) -> None:
        """Document must have a Canonicalization section."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "canonicalization" in content or "canonical" in content, "Missing 'Canonicalization' section"


class TestM1ToolchainDoc:
    """Tests for docs/m1_toolchain.md existence and required sections."""

    DOC_PATH = DOCS_DIR / "m1_toolchain.md"

    def test_file_exists(self) -> None:
        """m1_toolchain.md must exist."""
        assert self.DOC_PATH.is_file(), f"Missing: {self.DOC_PATH}"

    def test_has_toolchain_lock_section(self) -> None:
        """Document must have a Toolchain Lock File section."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "toolchain lock" in content or "lock file" in content, "Missing 'Toolchain Lock File' section"

    def test_documents_lock_file_path(self) -> None:
        """Document must specify lock file path."""
        content = self.DOC_PATH.read_text(encoding="utf-8")
        assert "toolchain/kicad.lock.json" in content, "Missing lock file path documentation"

    def test_documents_lock_file_schema(self) -> None:
        """Document must show lock file schema."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "schema_version" in content, "Missing schema_version field"
        assert "kicad_version" in content, "Missing kicad_version field"
        assert "docker_image" in content, "Missing docker_image field"
        assert "docker_digest" in content, "Missing docker_digest field"

    def test_has_docker_pinning_section(self) -> None:
        """Document must have a Docker Image Pinning section."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "docker" in content and "pin" in content, "Missing 'Docker Image Pinning' section"

    def test_documents_pinning_rationale(self) -> None:
        """Document must explain why Docker images are pinned."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "reproducib" in content, "Missing reproducibility rationale"

    def test_documents_placeholder_digest(self) -> None:
        """Document must explain PLACEHOLDER digest."""
        content = self.DOC_PATH.read_text(encoding="utf-8")
        assert "PLACEHOLDER" in content, "Missing PLACEHOLDER digest documentation"

    def test_has_hash_computation_section(self) -> None:
        """Document must have a Toolchain Hash Computation section."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "toolchain_hash" in content, "Missing toolchain_hash documentation"
        assert "computation" in content or "computed" in content or "algorithm" in content, (
            "Missing hash computation documentation"
        )

    def test_documents_kicad_cli(self) -> None:
        """Document must reference KiCad CLI commands."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "kicad-cli" in content, "Missing kicad-cli documentation"

    def test_has_api_reference_section(self) -> None:
        """Document must have an API Reference section."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "api" in content, "Missing 'API Reference' section"

    def test_documents_toolchainconfig(self) -> None:
        """Document must reference ToolchainConfig class."""
        content = self.DOC_PATH.read_text(encoding="utf-8")
        assert "ToolchainConfig" in content, "Missing ToolchainConfig documentation"

    def test_documents_load_function(self) -> None:
        """Document must reference load_toolchain_lock function."""
        content = self.DOC_PATH.read_text(encoding="utf-8")
        assert "load_toolchain_lock" in content, "Missing load_toolchain_lock documentation"

    def test_has_error_handling_section(self) -> None:
        """Document must have an Error Handling section."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "error" in content, "Missing 'Error Handling' section"

    def test_documents_toolchainloaderror(self) -> None:
        """Document must reference ToolchainLoadError."""
        content = self.DOC_PATH.read_text(encoding="utf-8")
        assert "ToolchainLoadError" in content, "Missing ToolchainLoadError documentation"


class TestM1DocsConsistency:
    """Cross-document consistency tests."""

    DETERMINISM_DOC = DOCS_DIR / "m1_determinism_policy.md"
    TOOLCHAIN_DOC = DOCS_DIR / "m1_toolchain.md"
    GENERAL_POLICY_DOC = DOCS_DIR / "DETERMINISM_POLICY.md"

    def test_determinism_doc_references_toolchain_doc(self) -> None:
        """m1_determinism_policy.md should reference m1_toolchain.md."""
        content = self.DETERMINISM_DOC.read_text(encoding="utf-8")
        assert "m1_toolchain.md" in content, "m1_determinism_policy.md should reference m1_toolchain.md"

    def test_toolchain_doc_references_determinism_doc(self) -> None:
        """m1_toolchain.md should reference m1_determinism_policy.md."""
        content = self.TOOLCHAIN_DOC.read_text(encoding="utf-8")
        assert "m1_determinism_policy.md" in content, "m1_toolchain.md should reference m1_determinism_policy.md"

    def test_both_docs_reference_eco(self) -> None:
        """Both docs should reference the ECO."""
        det_content = self.DETERMINISM_DOC.read_text(encoding="utf-8")
        tool_content = self.TOOLCHAIN_DOC.read_text(encoding="utf-8")

        assert "ECO" in det_content, "m1_determinism_policy.md should reference ECO"
        assert "ECO" in tool_content, "m1_toolchain.md should reference ECO"

    def test_kicad_version_consistent(self) -> None:
        """KiCad version should be consistent across docs."""
        det_content = self.DETERMINISM_DOC.read_text(encoding="utf-8")
        tool_content = self.TOOLCHAIN_DOC.read_text(encoding="utf-8")

        # Both should mention 9.0.7 as the pinned version
        assert "9.0.7" in det_content, "m1_determinism_policy.md should mention KiCad 9.0.7"
        assert "9.0.7" in tool_content, "m1_toolchain.md should mention KiCad 9.0.7"


class TestDeterminismPolicyDocCompleteness:
    """Tests for the general DETERMINISM_POLICY.md completeness."""

    DOC_PATH = DOCS_DIR / "DETERMINISM_POLICY.md"

    def test_file_exists(self) -> None:
        """DETERMINISM_POLICY.md must exist."""
        assert self.DOC_PATH.is_file(), f"Missing: {self.DOC_PATH}"

    def test_documents_hash_inputs(self) -> None:
        """Document must explain what is hashed."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "hashed" in content or "hash" in content, "Missing hash documentation"

    def test_documents_exclusions(self) -> None:
        """Document must explain exclusions."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "excluded" in content or "exclusion" in content, "Missing exclusion documentation"

    def test_documents_digest_pinning(self) -> None:
        """Document must explain digest pinning."""
        content = self.DOC_PATH.read_text(encoding="utf-8").lower()
        assert "docker_digest" in content, "Missing docker_digest documentation"
        assert "sha256" in content, "Missing SHA256 format"
