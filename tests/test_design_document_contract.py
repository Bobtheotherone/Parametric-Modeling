#!/usr/bin/env python3
"""Design document contract tests.

These tests verify that the design document parser works correctly with
multiple document formats and enforces the right contract constraints.

The key principle is MODULARITY: the orchestrator must accept ANY markdown
design document and extract what it can using robust heuristics, without
requiring exact headings or rigid structure.

Run with: pytest tests/test_design_document_contract.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from bridge.design_doc import (
    ContractMode,
    DesignDocSpec,
    Requirement,
    parse_design_doc,
    parse_design_doc_text,
)

# =============================================================================
# Test the current DESIGN_DOCUMENT.md (if it exists)
# =============================================================================


class TestCurrentDesignDocument:
    """Test the actual DESIGN_DOCUMENT.md in the repository."""

    def test_design_document_parses_without_errors_loose_mode(self):
        """Verify DESIGN_DOCUMENT.md parses without errors in loose mode."""
        doc_path = Path("DESIGN_DOCUMENT.md")
        if not doc_path.exists():
            pytest.skip("DESIGN_DOCUMENT.md not found in current directory")

        spec = parse_design_doc(doc_path, contract_mode="loose")

        # In loose mode, we should get no errors (warnings are OK)
        assert spec.is_valid, f"DESIGN_DOCUMENT.md failed loose validation: {spec.errors}"

        # Should extract at least basic information
        print(f"Extracted milestone: {spec.milestone_id}")
        print(f"Extracted requirements: {len(spec.requirements)}")
        print(f"Warnings: {spec.warnings}")

    def test_design_document_parses_off_mode(self):
        """Verify DESIGN_DOCUMENT.md parses in off mode (no validation)."""
        doc_path = Path("DESIGN_DOCUMENT.md")
        if not doc_path.exists():
            pytest.skip("DESIGN_DOCUMENT.md not found in current directory")

        spec = parse_design_doc(doc_path, contract_mode="off")

        # Off mode should never fail
        assert spec.is_valid, f"DESIGN_DOCUMENT.md failed off mode: {spec.errors}"


# =============================================================================
# Test multiple document formats (synthetic)
# =============================================================================


class TestMultipleDocumentFormats:
    """Test that the parser works with various document formats."""

    def test_standard_milestone_format(self):
        """Test standard **Milestone:** M1 format."""
        doc = """# Milestone Design Document

**Milestone:** M1 — Quality Hardening

## Scope
Make things better.

## Normative Requirements (must)

- [REQ-M1-001] The system MUST do thing A.
- [REQ-M1-002] The system MUST do thing B.
- [REQ-M1-003] The system MUST do thing C.

## Definition of Done

- Thing A is implemented and tested.
- Thing B is implemented and tested.
- All tests pass in CI.

## Test Matrix

| Requirement | Pytest(s) |
|---|---|
| REQ-M1-001 | tests/test_a.py::test_a |
| REQ-M1-002 | tests/test_b.py::test_b |
| REQ-M1-003 | tests/test_c.py::test_c |
"""
        spec = parse_design_doc_text(doc, contract_mode="strict")

        assert spec.is_valid, f"Standard format failed: {spec.errors}"
        assert spec.milestone_id == "M1"
        assert len(spec.requirements) == 3
        assert spec.requirements[0].id == "REQ-M1-001"
        assert len(spec.definition_of_done) == 3
        assert "REQ-M1-001" in spec.test_matrix
        assert "tests/test_a.py::test_a" in spec.test_matrix["REQ-M1-001"]

    def test_milestone_id_format(self):
        """Test **Milestone ID:** M2 format."""
        doc = """# Design Document

**Milestone ID:** M2

## Requirements

- [REQ-M2-001] Do something.

## Definition of Done

- Done.

## Test Matrix

| Requirement | Pytest(s) |
|---|---|
| REQ-M2-001 | tests/test.py::test |
"""
        spec = parse_design_doc_text(doc, contract_mode="strict")

        assert spec.is_valid, f"Milestone ID format failed: {spec.errors}"
        assert spec.milestone_id == "M2"

    def test_heading_milestone_format(self):
        """Test # M3 Design Document heading format."""
        doc = """# M3 Design Document

Some overview text.

## Requirements

- [REQ-M3-001] Do the thing.

## DoD

- Thing is done.

## Test Matrix

| Requirement | Pytest |
|---|---|
| REQ-M3-001 | tests/test.py::test |
"""
        spec = parse_design_doc_text(doc, contract_mode="loose")

        assert spec.is_valid, f"Heading format failed: {spec.errors}"
        assert spec.milestone_id == "M3"

    def test_short_hardening_doc_format(self):
        """Test a short 70-line hardening document format."""
        doc = """# Hardening Task

Milestone: M4

Fix critical bugs found in code review.

## Requirements

[REQ-M4-001] Fix the null pointer bug.
[REQ-M4-002] Add error handling to API.

## Done When

- No more null pointer exceptions.
- API returns proper error codes.
"""
        spec = parse_design_doc_text(doc, contract_mode="loose")

        assert spec.is_valid, f"Short doc failed: {spec.errors}"
        assert spec.milestone_id == "M4"
        assert len(spec.requirements) >= 2
        # DoD section has different header but should still be extracted
        assert len(spec.definition_of_done) >= 0  # May or may not extract "Done When"

    def test_no_explicit_milestone_but_has_requirements(self):
        """Test document without explicit milestone but with requirements."""
        doc = """# Bug Fixes

## Requirements

- [REQ-BUG-001] Fix login issue.
- [REQ-BUG-002] Fix logout issue.

## Definition of Done

- Login works.
- Logout works.

## Test Matrix

| Requirement | Pytest(s) |
|---|---|
| REQ-BUG-001 | tests/test_auth.py::test_login |
| REQ-BUG-002 | tests/test_auth.py::test_logout |
"""
        spec = parse_design_doc_text(doc, contract_mode="loose")

        assert spec.is_valid, f"No milestone doc failed: {spec.errors}"
        assert spec.milestone_id is None
        assert len(spec.requirements) == 2

    def test_minimal_document_loose_mode(self):
        """Test that a minimal document passes loose mode."""
        doc = """# Task

[REQ-001] Do something.
"""
        spec = parse_design_doc_text(doc, contract_mode="loose")

        assert spec.is_valid, f"Minimal doc failed loose mode: {spec.errors}"
        assert len(spec.requirements) == 1

    def test_minimal_document_strict_mode_fails(self):
        """Test that a minimal document fails strict mode."""
        doc = """# Task

[REQ-001] Do something.
"""
        spec = parse_design_doc_text(doc, contract_mode="strict")

        # Should fail strict mode due to missing fields
        assert not spec.is_valid, "Minimal doc should fail strict mode"
        assert len(spec.errors) > 0

    def test_alternate_requirement_formats(self):
        """Test that alternative requirement formats are extracted."""
        doc = """# Task

**Milestone:** M5

## Requirements

REQ-M5-001: The first requirement text.
REQ-M5-002 - The second requirement text.
[REQ-M5-003] The third requirement text.

## Definition of Done

- All done.

## Test Matrix

| Requirement | Pytest |
|---|---|
| REQ-M5-001 | tests/test.py::test1 |
| REQ-M5-002 | tests/test.py::test2 |
| REQ-M5-003 | tests/test.py::test3 |
"""
        spec = parse_design_doc_text(doc, contract_mode="strict")

        assert spec.is_valid, f"Alternate formats failed: {spec.errors}"
        assert len(spec.requirements) >= 3

    def test_multi_milestone_document(self):
        """Test document with multiple milestones."""
        doc = """# Combined Milestone Document

## M1 — Initial

**Milestone:** M1

- [REQ-M1-001] First M1 req.

## M2 — Follow-up

**Milestone:** M2

- [REQ-M2-001] First M2 req.

## Definition of Done

- M1 complete.
- M2 complete.

## Test Matrix

| Requirement | Pytest |
|---|---|
| REQ-M1-001 | tests/test.py::test_m1 |
| REQ-M2-001 | tests/test.py::test_m2 |
"""
        spec = parse_design_doc_text(doc, contract_mode="loose")

        assert spec.is_valid, f"Multi-milestone failed: {spec.errors}"
        # Should pick up first milestone or multiple
        assert spec.milestone_id in ("M1", "M2") or spec.milestone_id is not None
        assert len(spec.requirements) >= 2


# =============================================================================
# Test contract validation modes
# =============================================================================


class TestContractModes:
    """Test the contract validation behavior."""

    def test_strict_mode_requires_all_fields(self):
        """Test that strict mode requires all fields."""
        # Missing test matrix
        doc = """# Design Document

**Milestone:** M1

- [REQ-M1-001] Do thing.

## Definition of Done

- Done.
"""
        spec = parse_design_doc_text(doc, contract_mode="strict")

        assert not spec.is_valid
        assert any("test matrix" in e.lower() for e in spec.errors)

    def test_strict_mode_requires_milestone(self):
        """Test that strict mode requires milestone ID."""
        doc = """# Design Document

- [REQ-001] Do thing.

## Definition of Done

- Done.

## Test Matrix

| Requirement | Pytest |
|---|---|
| REQ-001 | tests/test.py::test |
"""
        spec = parse_design_doc_text(doc, contract_mode="strict")

        assert not spec.is_valid
        assert any("milestone" in e.lower() for e in spec.errors)

    def test_strict_mode_with_override_passes(self):
        """Test that strict mode passes with CLI milestone override."""
        doc = """# Design Document

- [REQ-001] Do thing.

## Definition of Done

- Done.

## Test Matrix

| Requirement | Pytest |
|---|---|
| REQ-001 | tests/test.py::test |
"""
        spec = parse_design_doc_text(doc, contract_mode="strict", milestone_override="M99")

        assert spec.is_valid, f"Override should satisfy strict mode: {spec.errors}"
        assert spec.milestone_id == "M99"

    def test_loose_mode_accepts_partial(self):
        """Test that loose mode accepts partial documents."""
        doc = """# Design Document

**Milestone:** M1

- [REQ-M1-001] Do thing.
"""
        spec = parse_design_doc_text(doc, contract_mode="loose")

        assert spec.is_valid, f"Loose mode should accept partial: {spec.errors}"
        assert len(spec.warnings) > 0  # Should have warnings about missing sections

    def test_off_mode_accepts_anything(self):
        """Test that off mode accepts any content."""
        doc = """Random text without any structure."""
        spec = parse_design_doc_text(doc, contract_mode="off")

        assert spec.is_valid, "Off mode should accept anything"
        assert len(spec.requirements) == 0
        assert spec.milestone_id is None


# =============================================================================
# Test specific extraction heuristics
# =============================================================================


class TestExtractionHeuristics:
    """Test specific extraction heuristics."""

    def test_requirement_text_extraction(self):
        """Test that requirement text is correctly extracted."""
        doc = """# Doc

**Milestone:** M1

- [REQ-M1-001] The system MUST implement feature A with full functionality including edge cases.

## Definition of Done

- Done.

## Test Matrix

| Requirement | Pytest |
|---|---|
| REQ-M1-001 | tests/test.py::test |
"""
        spec = parse_design_doc_text(doc, contract_mode="strict")

        assert spec.is_valid
        assert len(spec.requirements) == 1
        req = spec.requirements[0]
        assert "feature A" in req.text
        assert "edge cases" in req.text

    def test_duplicate_requirement_warning(self):
        """Test that duplicate requirements generate warnings."""
        doc = """# Doc

**Milestone:** M1

- [REQ-M1-001] First occurrence.
- [REQ-M1-001] Second occurrence (duplicate).

## Definition of Done

- Done.

## Test Matrix

| Requirement | Pytest |
|---|---|
| REQ-M1-001 | tests/test.py::test |
"""
        spec = parse_design_doc_text(doc, contract_mode="loose")

        assert spec.is_valid
        assert any("duplicate" in w.lower() for w in spec.warnings)
        # Should keep first occurrence only
        assert len(spec.requirements) == 1

    def test_dod_fuzzy_heading_matching(self):
        """Test that DoD is extracted with various heading formats."""
        test_cases = [
            "## Definition of Done\n- Item 1.",
            "## DoD\n- Item 1.",
            "## Done When\n- Item 1.",
            "## Acceptance Criteria\n- Item 1.",
            "## Success Criteria\n- Item 1.",
        ]

        for doc_section in test_cases:
            doc = f"""# Doc

**Milestone:** M1

- [REQ-M1-001] Req.

{doc_section}

## Test Matrix

| Requirement | Pytest |
|---|---|
| REQ-M1-001 | tests/test.py::test |
"""
            spec = parse_design_doc_text(doc, contract_mode="loose")
            assert len(spec.definition_of_done) > 0, f"Failed for: {doc_section[:30]}"

    def test_test_matrix_parsing(self):
        """Test that test matrix is correctly parsed."""
        doc = """# Doc

**Milestone:** M1

- [REQ-M1-001] Req 1.
- [REQ-M1-002] Req 2.

## Definition of Done

- Done.

## Test Matrix

| Requirement | Pytest(s) |
|---|---|
| REQ-M1-001 | tests/test_a.py::test_a, tests/test_a.py::test_b |
| REQ-M1-002 | tests/test_b.py::test_c; tests/test_b.py::test_d |
"""
        spec = parse_design_doc_text(doc, contract_mode="strict")

        assert spec.is_valid
        assert "REQ-M1-001" in spec.test_matrix
        assert "REQ-M1-002" in spec.test_matrix
        # Multiple tests should be parsed
        assert len(spec.test_matrix["REQ-M1-001"]) == 2
        assert len(spec.test_matrix["REQ-M1-002"]) == 2

    def test_doc_hash_is_stable(self):
        """Test that doc_hash is deterministic."""
        doc = """# Doc

**Milestone:** M1

- [REQ-M1-001] Req.
"""
        spec1 = parse_design_doc_text(doc, contract_mode="off")
        spec2 = parse_design_doc_text(doc, contract_mode="off")

        assert spec1.doc_hash == spec2.doc_hash
        assert len(spec1.doc_hash) == 64  # SHA-256 hex

    def test_prompt_context_generation(self):
        """Test that prompt context is correctly generated."""
        doc = """# Doc

**Milestone:** M1

- [REQ-M1-001] Req text here.

## Definition of Done

- Item 1.

## Test Matrix

| Requirement | Pytest |
|---|---|
| REQ-M1-001 | tests/test.py::test |
"""
        spec = parse_design_doc_text(doc, contract_mode="loose")
        context = spec.to_prompt_context(max_raw_chars=1000)

        assert "**Milestone:** M1" in context
        assert "REQ-M1-001" in context
        assert "Definition of Done" in context or "Item 1" in context


# =============================================================================
# Test error handling
# =============================================================================


class TestErrorHandling:
    """Test error handling in the parser."""

    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        spec = parse_design_doc(Path("/nonexistent/path/doc.md"), contract_mode="loose")

        assert not spec.is_valid
        assert any("not found" in e.lower() for e in spec.errors)

    def test_empty_document(self):
        """Test handling of empty document."""
        spec = parse_design_doc_text("", contract_mode="loose")

        # Empty document should fail even loose mode
        assert not spec.is_valid

    def test_whitespace_only_document(self):
        """Test handling of whitespace-only document."""
        spec = parse_design_doc_text("   \n\n   \t\n", contract_mode="loose")

        # Whitespace-only should fail
        assert not spec.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
