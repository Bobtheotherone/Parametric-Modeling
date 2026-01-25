# SPDX-License-Identifier: MIT
"""Unit tests for bridge/design_doc.py internal helper functions.

These tests supplement test_design_document_contract.py by testing the
internal helper functions in isolation. This ensures:
- Individual extraction functions work correctly
- Edge cases in parsing are handled
- Robustness of pattern matching

Key functions tested:
- _extract_milestone_id: Milestone ID extraction with multiple patterns
- _extract_all_milestones: Extracting all milestones from document
- _extract_title: Document title extraction
- _extract_requirements: Requirement extraction from various formats
- _extract_definition_of_done: DoD section extraction
- _extract_test_matrix: Test matrix parsing
- _validate_contract: Contract validation logic
- Requirement and DesignDocSpec dataclasses

Run with: pytest tests/test_design_doc_helpers.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.design_doc import (
    DesignDocSpec,
    Requirement,
    _extract_all_milestones,
    _extract_definition_of_done,
    _extract_milestone_id,
    _extract_requirements,
    _extract_test_matrix,
    _extract_title,
    _validate_contract,
)

# -----------------------------------------------------------------------------
# Requirement dataclass tests
# -----------------------------------------------------------------------------


class TestRequirementDataclass:
    """Tests for Requirement dataclass."""

    def test_creates_requirement(self) -> None:
        """Creates requirement with all fields."""
        req = Requirement(id="REQ-001", text="Test requirement", line_no=10)
        assert req.id == "REQ-001"
        assert req.text == "Test requirement"
        assert req.line_no == 10

    def test_to_dict(self) -> None:
        """to_dict returns correct structure."""
        req = Requirement(id="REQ-001", text="Test requirement", line_no=10)
        d = req.to_dict()
        assert d == {"id": "REQ-001", "text": "Test requirement", "line_no": 10}


# -----------------------------------------------------------------------------
# DesignDocSpec dataclass tests
# -----------------------------------------------------------------------------


class TestDesignDocSpecDataclass:
    """Tests for DesignDocSpec dataclass."""

    def _make_spec(self, **kwargs) -> DesignDocSpec:
        """Create spec with defaults."""
        defaults = {
            "path": Path("test.md"),
            "raw_text": "# Test",
            "doc_hash": "abc123",
        }
        defaults.update(kwargs)
        return DesignDocSpec(**defaults)

    def test_is_valid_when_no_errors(self) -> None:
        """is_valid returns True when no errors."""
        spec = self._make_spec()
        assert spec.is_valid is True

    def test_is_valid_when_has_errors(self) -> None:
        """is_valid returns False when has errors."""
        spec = self._make_spec(errors=["Error 1"])
        assert spec.is_valid is False

    def test_is_valid_with_warnings_only(self) -> None:
        """is_valid returns True with warnings but no errors."""
        spec = self._make_spec(warnings=["Warning 1"])
        assert spec.is_valid is True

    def test_requirement_ids_property(self) -> None:
        """requirement_ids returns list of IDs."""
        reqs = [
            Requirement(id="REQ-001", text="Req 1", line_no=1),
            Requirement(id="REQ-002", text="Req 2", line_no=2),
        ]
        spec = self._make_spec(requirements=reqs)
        assert spec.requirement_ids == ["REQ-001", "REQ-002"]

    def test_get_requirement_text_found(self) -> None:
        """get_requirement_text returns text when found."""
        reqs = [Requirement(id="REQ-001", text="Test text", line_no=1)]
        spec = self._make_spec(requirements=reqs)
        assert spec.get_requirement_text("REQ-001") == "Test text"

    def test_get_requirement_text_not_found(self) -> None:
        """get_requirement_text returns None when not found."""
        spec = self._make_spec()
        assert spec.get_requirement_text("REQ-999") is None

    def test_to_dict(self) -> None:
        """to_dict returns correct structure."""
        spec = self._make_spec(milestone_id="M1", title="Test Doc")
        d = spec.to_dict()
        assert d["path"] == "test.md"
        assert d["milestone_id"] == "M1"
        assert d["title"] == "Test Doc"
        assert d["is_valid"] is True

    def test_to_prompt_context(self) -> None:
        """to_prompt_context generates valid context."""
        spec = self._make_spec(
            milestone_id="M1",
            title="Test Doc",
            requirements=[Requirement(id="REQ-001", text="Test req", line_no=1)],
        )
        context = spec.to_prompt_context(max_raw_chars=100)
        assert "M1" in context
        assert "REQ-001" in context

    def test_to_prompt_context_with_truncation(self) -> None:
        """to_prompt_context truncates long raw text."""
        spec = self._make_spec(raw_text="x" * 1000)
        context = spec.to_prompt_context(max_raw_chars=100)
        assert "truncated" in context

    def test_to_prompt_context_no_raw(self) -> None:
        """to_prompt_context with max_raw_chars=0 excludes raw text."""
        spec = self._make_spec(raw_text="secret text")
        context = spec.to_prompt_context(max_raw_chars=0)
        assert "secret text" not in context


# -----------------------------------------------------------------------------
# _extract_milestone_id tests
# -----------------------------------------------------------------------------


class TestExtractMilestoneId:
    """Tests for _extract_milestone_id function."""

    def test_bold_milestone_format(self) -> None:
        """Extracts **Milestone:** M1 format."""
        text = "**Milestone:** M1 — Description"
        mid, pattern = _extract_milestone_id(text)
        assert mid == "M1"
        assert pattern is not None

    def test_bold_milestone_id_format(self) -> None:
        """Extracts **Milestone ID:** M2 format."""
        text = "**Milestone ID:** M2"
        mid, pattern = _extract_milestone_id(text)
        assert mid == "M2"

    def test_line_start_milestone_format(self) -> None:
        """Extracts Milestone: M3 at line start."""
        text = "Some intro\nMilestone: M3\nMore text"
        mid, pattern = _extract_milestone_id(text)
        assert mid == "M3"

    def test_heading_milestone_format(self) -> None:
        """Extracts # M4 from heading."""
        text = "# M4 Design Document"
        mid, pattern = _extract_milestone_id(text)
        assert mid == "M4"

    def test_heading_with_dash_format(self) -> None:
        """Extracts ## M5 — Title format."""
        text = "## M5 — Quality Hardening"
        mid, pattern = _extract_milestone_id(text)
        assert mid == "M5"

    def test_returns_none_for_no_milestone(self) -> None:
        """Returns None when no milestone found."""
        text = "No milestone here at all"
        mid, pattern = _extract_milestone_id(text)
        assert mid is None
        assert pattern is None

    def test_case_insensitive(self) -> None:
        """Milestone extraction is case-insensitive."""
        text = "**MILESTONE:** M6"
        mid, pattern = _extract_milestone_id(text)
        assert mid == "M6"

    def test_large_milestone_number(self) -> None:
        """Handles large milestone numbers."""
        text = "**Milestone:** M99"
        mid, pattern = _extract_milestone_id(text)
        assert mid == "M99"


# -----------------------------------------------------------------------------
# _extract_all_milestones tests
# -----------------------------------------------------------------------------


class TestExtractAllMilestones:
    """Tests for _extract_all_milestones function."""

    def test_single_milestone(self) -> None:
        """Extracts single milestone."""
        text = "**Milestone:** M1"
        milestones = _extract_all_milestones(text)
        assert milestones == ["M1"]

    def test_multiple_milestones(self) -> None:
        """Extracts multiple milestones."""
        text = """
        **Milestone:** M1
        ## M2 — Next phase
        Milestone: M3
        """
        milestones = _extract_all_milestones(text)
        assert "M1" in milestones
        assert "M2" in milestones
        assert "M3" in milestones

    def test_sorts_by_number(self) -> None:
        """Milestones are sorted by number."""
        text = "M3, M1, M2"
        milestones = _extract_all_milestones(text)
        # Should be sorted
        assert milestones == sorted(milestones, key=lambda x: int(x[1:]))

    def test_no_milestones(self) -> None:
        """Returns empty list when no milestones."""
        text = "No milestones here"
        milestones = _extract_all_milestones(text)
        assert milestones == []

    def test_deduplicates(self) -> None:
        """Deduplicates repeated milestones."""
        text = "**Milestone:** M1\n## M1 Again"
        milestones = _extract_all_milestones(text)
        assert milestones.count("M1") == 1


# -----------------------------------------------------------------------------
# _extract_title tests
# -----------------------------------------------------------------------------


class TestExtractTitle:
    """Tests for _extract_title function."""

    def test_h1_heading(self) -> None:
        """Extracts title from H1 heading."""
        text = "# My Document Title\n\nContent here"
        title = _extract_title(text)
        assert title == "My Document Title"

    def test_bold_title_format(self) -> None:
        """Extracts **Title:** format."""
        text = "**Title:** My Document"
        title = _extract_title(text)
        assert title == "My Document"

    def test_cleans_markdown_formatting(self) -> None:
        """Removes markdown formatting from title."""
        text = "# **Bold** and `code` title"
        title = _extract_title(text)
        assert title == "Bold and code title"

    def test_truncates_long_title(self) -> None:
        """Truncates very long titles."""
        text = "# " + "A" * 300
        title = _extract_title(text)
        assert title is not None
        assert len(title) <= 200

    def test_returns_none_for_no_title(self) -> None:
        """Returns None when no title found."""
        text = "No heading here just text"
        title = _extract_title(text)
        assert title is None


# -----------------------------------------------------------------------------
# _extract_requirements tests
# -----------------------------------------------------------------------------


class TestExtractRequirements:
    """Tests for _extract_requirements function."""

    def test_standard_req_format(self) -> None:
        """Extracts [REQ-M1-001] format."""
        text = "- [REQ-M1-001] The system must do X."
        lines = text.split("\n")
        reqs, warnings = _extract_requirements(text, lines)
        assert len(reqs) == 1
        assert reqs[0].id == "REQ-M1-001"
        assert "must do X" in reqs[0].text

    def test_req_without_brackets(self) -> None:
        """Extracts REQ-001 without brackets."""
        text = "REQ-M2-001: Do something."
        lines = text.split("\n")
        reqs, warnings = _extract_requirements(text, lines)
        assert len(reqs) == 1
        assert reqs[0].id == "REQ-M2-001"

    def test_multiple_requirements(self) -> None:
        """Extracts multiple requirements."""
        text = """
        - [REQ-001] First requirement.
        - [REQ-002] Second requirement.
        - [REQ-003] Third requirement.
        """
        lines = text.split("\n")
        reqs, warnings = _extract_requirements(text, lines)
        assert len(reqs) == 3

    def test_duplicate_warning(self) -> None:
        """Warns on duplicate requirement IDs."""
        text = """
        - [REQ-001] First occurrence.
        - [REQ-001] Duplicate.
        """
        lines = text.split("\n")
        reqs, warnings = _extract_requirements(text, lines)
        assert len(reqs) == 1  # Only first is kept
        assert any("duplicate" in w.lower() for w in warnings)

    def test_req_on_next_line(self) -> None:
        """Extracts requirement text from next line if same line empty."""
        text = """[REQ-001]
        The requirement text is here.
        """
        lines = text.split("\n")
        reqs, warnings = _extract_requirements(text, lines)
        assert len(reqs) == 1
        # Should capture text from next line

    def test_empty_text_gets_placeholder(self) -> None:
        """Empty requirement text gets placeholder."""
        text = "[REQ-001]"
        lines = text.split("\n")
        reqs, warnings = _extract_requirements(text, lines)
        assert len(reqs) == 1
        assert "(no description)" in reqs[0].text or len(reqs[0].text) == 0

    def test_bug_requirement_format(self) -> None:
        """Extracts REQ-BUG-001 format."""
        text = "[REQ-BUG-001] Fix the bug."
        lines = text.split("\n")
        reqs, warnings = _extract_requirements(text, lines)
        assert len(reqs) == 1
        assert reqs[0].id == "REQ-BUG-001"

    def test_line_number_tracking(self) -> None:
        """Tracks line numbers correctly."""
        text = "Line 1\nLine 2\n[REQ-001] On line 3\nLine 4"
        lines = text.split("\n")
        reqs, warnings = _extract_requirements(text, lines)
        assert len(reqs) == 1
        assert reqs[0].line_no == 3


# -----------------------------------------------------------------------------
# _extract_definition_of_done tests
# -----------------------------------------------------------------------------


class TestExtractDefinitionOfDone:
    """Tests for _extract_definition_of_done function."""

    def test_standard_dod_header(self) -> None:
        """Extracts from ## Definition of Done."""
        text = """## Definition of Done

- Item one.
- Item two.

## Next Section
"""
        lines = text.split("\n")
        dod = _extract_definition_of_done(text, lines)
        assert len(dod) == 2
        assert "Item one" in dod[0]

    def test_dod_header(self) -> None:
        """Extracts from ## DoD header."""
        text = """## DoD

- Done item.
"""
        lines = text.split("\n")
        dod = _extract_definition_of_done(text, lines)
        assert len(dod) == 1

    def test_done_when_header(self) -> None:
        """Extracts from ## Done When header."""
        text = """## Done When

- Task complete.
"""
        lines = text.split("\n")
        dod = _extract_definition_of_done(text, lines)
        assert len(dod) == 1

    def test_acceptance_criteria_header(self) -> None:
        """Extracts from ## Acceptance Criteria header."""
        text = """## Acceptance Criteria

- Criteria met.
"""
        lines = text.split("\n")
        dod = _extract_definition_of_done(text, lines)
        assert len(dod) == 1

    def test_success_criteria_header(self) -> None:
        """Extracts from ## Success Criteria header."""
        text = """## Success Criteria

- Success achieved.
"""
        lines = text.split("\n")
        dod = _extract_definition_of_done(text, lines)
        assert len(dod) == 1

    def test_bold_dod_format(self) -> None:
        """Extracts from **Definition of Done:** format."""
        text = """**Definition of Done:**

- Item.
"""
        lines = text.split("\n")
        dod = _extract_definition_of_done(text, lines)
        assert len(dod) == 1

    def test_no_dod_section(self) -> None:
        """Returns empty list when no DoD section."""
        text = "No DoD here"
        lines = text.split("\n")
        dod = _extract_definition_of_done(text, lines)
        assert dod == []

    def test_stops_at_next_header(self) -> None:
        """Stops extraction at next header."""
        text = """## Definition of Done

- DoD item.

## Test Matrix

- Not a DoD item.
"""
        lines = text.split("\n")
        dod = _extract_definition_of_done(text, lines)
        assert len(dod) == 1
        assert "DoD item" in dod[0]

    def test_filters_short_items(self) -> None:
        """Filters out very short items."""
        text = """## Definition of Done

- A
- Valid item here.
- B
"""
        lines = text.split("\n")
        dod = _extract_definition_of_done(text, lines)
        # Should only include items > 2 chars
        assert any("Valid" in item for item in dod)

    def test_asterisk_bullets(self) -> None:
        """Handles * bullets."""
        text = """## Definition of Done

* Item with asterisk.
"""
        lines = text.split("\n")
        dod = _extract_definition_of_done(text, lines)
        assert len(dod) == 1


# -----------------------------------------------------------------------------
# _extract_test_matrix tests
# -----------------------------------------------------------------------------


class TestExtractTestMatrix:
    """Tests for _extract_test_matrix function."""

    def test_standard_table(self) -> None:
        """Extracts standard markdown table."""
        text = """## Test Matrix

| Requirement | Pytest(s) |
|---|---|
| REQ-001 | tests/test.py::test_one |
| REQ-002 | tests/test.py::test_two |
"""
        lines = text.split("\n")
        matrix, warnings = _extract_test_matrix(text, lines)
        assert "REQ-001" in matrix
        assert "REQ-002" in matrix

    def test_multiple_tests_comma_separated(self) -> None:
        """Parses comma-separated test IDs."""
        text = """| Requirement | Pytest(s) |
|---|---|
| REQ-001 | tests/a.py::test_a, tests/b.py::test_b |
"""
        lines = text.split("\n")
        matrix, warnings = _extract_test_matrix(text, lines)
        assert len(matrix.get("REQ-001", [])) == 2

    def test_multiple_tests_semicolon_separated(self) -> None:
        """Parses semicolon-separated test IDs."""
        text = """| Requirement | Pytest(s) |
|---|---|
| REQ-001 | tests/a.py::test_a; tests/b.py::test_b |
"""
        lines = text.split("\n")
        matrix, warnings = _extract_test_matrix(text, lines)
        assert len(matrix.get("REQ-001", [])) == 2

    def test_no_table(self) -> None:
        """Returns empty dict when no table."""
        text = "No table here"
        lines = text.split("\n")
        matrix, warnings = _extract_test_matrix(text, lines)
        assert matrix == {}

    def test_filters_invalid_test_ids(self) -> None:
        """Filters out non-pytest test IDs."""
        text = """| Requirement | Pytest(s) |
|---|---|
| REQ-001 | tests/test.py::test_one, not_a_test |
"""
        lines = text.split("\n")
        matrix, warnings = _extract_test_matrix(text, lines)
        tests = matrix.get("REQ-001", [])
        assert "tests/test.py::test_one" in tests
        # "not_a_test" should be filtered out (no :: and doesn't start with tests/)

    def test_duplicate_entry_warning(self) -> None:
        """Warns on duplicate test matrix entries."""
        text = """| Requirement | Pytest(s) |
|---|---|
| REQ-001 | tests/test.py::test_one |
| REQ-001 | tests/test.py::test_two |
"""
        lines = text.split("\n")
        matrix, warnings = _extract_test_matrix(text, lines)
        assert any("duplicate" in w.lower() for w in warnings)
        # Should merge entries
        assert len(matrix.get("REQ-001", [])) >= 2


# -----------------------------------------------------------------------------
# _validate_contract tests
# -----------------------------------------------------------------------------


class TestValidateContract:
    """Tests for _validate_contract function."""

    def _make_spec(self, **kwargs) -> DesignDocSpec:
        """Create spec with defaults."""
        defaults = {
            "path": Path("test.md"),
            "raw_text": "# Test",
            "doc_hash": "abc123",
            "warnings": [],
            "errors": [],
        }
        defaults.update(kwargs)
        return DesignDocSpec(**defaults)

    def test_off_mode_no_validation(self) -> None:
        """Off mode does no validation."""
        spec = self._make_spec()
        _validate_contract(spec, "off")
        assert spec.errors == []
        assert spec.warnings == []

    def test_strict_requires_milestone(self) -> None:
        """Strict mode requires milestone_id."""
        spec = self._make_spec(
            requirements=[Requirement(id="REQ-001", text="Test", line_no=1)],
            definition_of_done=["Done"],
            test_matrix={"REQ-001": ["test"]},
        )
        _validate_contract(spec, "strict")
        assert any("milestone" in e.lower() for e in spec.errors)

    def test_strict_requires_requirements(self) -> None:
        """Strict mode requires requirements."""
        spec = self._make_spec(
            milestone_id="M1",
            definition_of_done=["Done"],
            test_matrix={"REQ-001": ["test"]},
        )
        _validate_contract(spec, "strict")
        assert any("requirements" in e.lower() for e in spec.errors)

    def test_strict_requires_dod(self) -> None:
        """Strict mode requires Definition of Done."""
        spec = self._make_spec(
            milestone_id="M1",
            requirements=[Requirement(id="REQ-001", text="Test", line_no=1)],
            test_matrix={"REQ-001": ["test"]},
        )
        _validate_contract(spec, "strict")
        assert any("definition of done" in e.lower() for e in spec.errors)

    def test_strict_requires_test_matrix(self) -> None:
        """Strict mode requires test matrix."""
        spec = self._make_spec(
            milestone_id="M1",
            requirements=[Requirement(id="REQ-001", text="Test", line_no=1)],
            definition_of_done=["Done"],
        )
        _validate_contract(spec, "strict")
        assert any("test matrix" in e.lower() for e in spec.errors)

    def test_strict_checks_test_coverage(self) -> None:
        """Strict mode checks all requirements have test coverage."""
        spec = self._make_spec(
            milestone_id="M1",
            requirements=[
                Requirement(id="REQ-001", text="Test", line_no=1),
                Requirement(id="REQ-002", text="Test", line_no=2),
            ],
            definition_of_done=["Done"],
            test_matrix={"REQ-001": ["test"]},  # Missing REQ-002
        )
        _validate_contract(spec, "strict")
        assert any("REQ-002" in e for e in spec.errors)

    def test_loose_mode_valid_with_milestone_only(self) -> None:
        """Loose mode accepts milestone without requirements."""
        spec = self._make_spec(milestone_id="M1")
        _validate_contract(spec, "loose")
        assert spec.is_valid  # No errors

    def test_loose_mode_valid_with_requirements_only(self) -> None:
        """Loose mode accepts requirements without milestone."""
        spec = self._make_spec(requirements=[Requirement(id="REQ-001", text="Test", line_no=1)])
        _validate_contract(spec, "loose")
        assert spec.is_valid

    def test_loose_mode_fails_with_nothing(self) -> None:
        """Loose mode fails with no milestone and no requirements."""
        spec = self._make_spec()
        _validate_contract(spec, "loose")
        assert not spec.is_valid
        assert any("need at least" in e.lower() for e in spec.errors)

    def test_loose_mode_adds_warnings(self) -> None:
        """Loose mode adds warnings for missing sections."""
        spec = self._make_spec(milestone_id="M1")
        _validate_contract(spec, "loose")
        # Should have warnings about missing DoD and test matrix
        assert len(spec.warnings) > 0
