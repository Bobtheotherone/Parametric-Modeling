#!/usr/bin/env python3
"""Design document parser for modular orchestration.

This module provides robust, format-agnostic parsing of design documents.
The orchestrator MUST NOT be tightly coupled to a specific design doc format.
It accepts arbitrary markdown structure and extracts what it can using heuristics.

Key design principles:
- Graceful degradation: If something can't be extracted, continue with warnings
- Multiple extraction strategies: Try various patterns for each field
- Contract modes: strict (CI gate), loose (development), off (bypass)

Usage:
    from bridge.design_doc import parse_design_doc, DesignDocSpec

    spec = parse_design_doc(Path("DESIGN_DOCUMENT.md"), contract_mode="loose")
    if not spec.is_valid:
        print(f"Warnings: {spec.warnings}")
        print(f"Errors: {spec.errors}")
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ContractMode = Literal["strict", "loose", "off"]


@dataclass
class Requirement:
    """A parsed requirement from the design document."""
    id: str
    text: str
    line_no: int

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "line_no": self.line_no}


@dataclass
class DesignDocSpec:
    """Parsed design document specification.

    This dataclass contains all extracted data from a design document,
    with warnings for anything that couldn't be extracted.
    """
    # Source information
    path: Path
    raw_text: str
    doc_hash: str  # SHA-256 of raw_text

    # Extracted fields (may be None/empty if not found)
    milestone_id: str | None = None
    title: str | None = None
    requirements: list[Requirement] = field(default_factory=list)
    definition_of_done: list[str] = field(default_factory=list)
    test_matrix: dict[str, list[str]] = field(default_factory=dict)

    # Validation results
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    contract_mode_used: ContractMode = "loose"

    @property
    def is_valid(self) -> bool:
        """Return True if no errors (warnings are acceptable)."""
        return len(self.errors) == 0

    @property
    def requirement_ids(self) -> list[str]:
        """Return list of requirement IDs."""
        return [r.id for r in self.requirements]

    def get_requirement_text(self, req_id: str) -> str | None:
        """Get the text for a specific requirement ID."""
        for r in self.requirements:
            if r.id == req_id:
                return r.text
        return None

    def to_prompt_context(self, max_raw_chars: int = 20000) -> str:
        """Generate prompt context string for agent consumption.

        This provides structured data that agents can use without
        needing to parse the raw document themselves.
        """
        parts = []

        parts.append(f"## Design Document Context")
        parts.append(f"**Document Hash:** {self.doc_hash[:16]}")

        if self.milestone_id:
            parts.append(f"**Milestone:** {self.milestone_id}")

        if self.title:
            parts.append(f"**Title:** {self.title}")

        if self.requirements:
            parts.append(f"\n### Requirements ({len(self.requirements)} total)")
            for r in self.requirements:
                parts.append(f"- **{r.id}**: {r.text[:200]}{'...' if len(r.text) > 200 else ''}")

        if self.definition_of_done:
            parts.append(f"\n### Definition of Done")
            for item in self.definition_of_done:
                parts.append(f"- {item}")

        if self.test_matrix:
            parts.append(f"\n### Test Matrix")
            for req_id, tests in self.test_matrix.items():
                parts.append(f"- {req_id}: {', '.join(tests)}")

        # Include truncated raw text for additional context
        if max_raw_chars > 0:
            truncated = self.raw_text[:max_raw_chars]
            if len(self.raw_text) > max_raw_chars:
                truncated += f"\n\n[... truncated {len(self.raw_text) - max_raw_chars} chars ...]"
            parts.append(f"\n### Raw Document Excerpt\n```markdown\n{truncated}\n```")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": str(self.path),
            "doc_hash": self.doc_hash,
            "milestone_id": self.milestone_id,
            "title": self.title,
            "requirements": [r.to_dict() for r in self.requirements],
            "definition_of_done": self.definition_of_done,
            "test_matrix": self.test_matrix,
            "warnings": self.warnings,
            "errors": self.errors,
            "contract_mode_used": self.contract_mode_used,
            "is_valid": self.is_valid,
        }


# =============================================================================
# Milestone ID Extraction
# =============================================================================

# Patterns to try for milestone extraction (in priority order)
MILESTONE_PATTERNS = [
    # **Milestone:** M1 — description
    re.compile(r"\*\*Milestone:\*\*\s*(M\d+)\b", re.IGNORECASE),
    # **Milestone ID:** M1
    re.compile(r"\*\*Milestone\s+ID:\*\*\s*(M\d+)\b", re.IGNORECASE),
    # Milestone: M1
    re.compile(r"^Milestone:\s*(M\d+)\b", re.MULTILINE | re.IGNORECASE),
    # # M1 Design Document
    re.compile(r"^#\s+(M\d+)\s+", re.MULTILINE),
    # M1 — at the start of a heading
    re.compile(r"^#+\s*(M\d+)\s*[—\-–:]", re.MULTILINE),
    # ## M1: Title
    re.compile(r"^#+\s*(M\d+):", re.MULTILINE),
]


def _extract_milestone_id(text: str) -> tuple[str | None, str | None]:
    """Extract milestone ID using multiple patterns.

    Returns:
        Tuple of (milestone_id, pattern_name_used) or (None, None)
    """
    for i, pattern in enumerate(MILESTONE_PATTERNS):
        match = pattern.search(text)
        if match:
            return match.group(1), f"pattern_{i}"
    return None, None


def _extract_all_milestones(text: str) -> list[str]:
    """Extract all milestone IDs found in the document."""
    milestones = set()
    for pattern in MILESTONE_PATTERNS:
        for match in pattern.finditer(text):
            milestones.add(match.group(1))
    return sorted(milestones, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)


# =============================================================================
# Title Extraction
# =============================================================================

TITLE_PATTERNS = [
    # # Title (first H1)
    re.compile(r"^#\s+(.+?)(?:\n|$)", re.MULTILINE),
    # **Title:** Something
    re.compile(r"\*\*Title:\*\*\s*(.+?)(?:\n|$)", re.IGNORECASE),
]


def _extract_title(text: str) -> str | None:
    """Extract document title."""
    for pattern in TITLE_PATTERNS:
        match = pattern.search(text)
        if match:
            title = match.group(1).strip()
            # Clean up markdown formatting
            title = re.sub(r"\*\*|\*|`", "", title)
            return title[:200] if title else None
    return None


# =============================================================================
# Requirements Extraction
# =============================================================================

# Primary pattern for requirements:
# - [REQ-M1-001] - with milestone prefix (most common)
# - [REQ-ABC-123] - with arbitrary prefix
# - [REQ-001] - without prefix (simple format)
# - REQ-BUG-001 - for bug requirements
# The pattern: REQ- followed by optional prefix (letters/numbers/hyphens) and then 3+ digits
REQ_ID_PATTERN = re.compile(r"\[?(REQ-(?:[A-Z0-9]+-)*\d{3,})\]?")

# Alternative patterns for requirement-like structures
ALT_REQ_PATTERNS = [
    # R1, R2, etc.
    re.compile(r"^\s*-\s*\[?(R\d+)\]?\s+(.+)", re.MULTILINE),
    # REQ1, REQ2, etc.
    re.compile(r"^\s*-\s*\[?(REQ\d+)\]?\s+(.+)", re.MULTILINE),
]


def _extract_requirements(text: str, lines: list[str]) -> tuple[list[Requirement], list[str]]:
    """Extract requirements from the document.

    Returns:
        Tuple of (requirements, warnings)
    """
    requirements: list[Requirement] = []
    seen_ids: set[str] = set()
    warnings: list[str] = []

    # Strategy 1: Look for REQ-* pattern with following text
    for line_no, line in enumerate(lines, 1):
        # Find all REQ IDs in this line
        matches = list(REQ_ID_PATTERN.finditer(line))
        for match in matches:
            req_id = match.group(1)

            if req_id in seen_ids:
                warnings.append(f"Duplicate requirement ID {req_id} at line {line_no}")
                continue

            seen_ids.add(req_id)

            # Extract the requirement text (everything after the ID on the same line)
            text_start = match.end()
            req_text = line[text_start:].strip()

            # Clean up leading punctuation/brackets
            req_text = re.sub(r"^[\]\s:\-–—]+", "", req_text).strip()

            # If no text on same line, try to get the next non-empty line
            if not req_text and line_no < len(lines):
                for next_line in lines[line_no:line_no + 3]:
                    next_line = next_line.strip()
                    if next_line and not next_line.startswith("#") and not next_line.startswith("-"):
                        req_text = next_line
                        break

            requirements.append(Requirement(
                id=req_id,
                text=req_text or "(no description)",
                line_no=line_no,
            ))

    # Strategy 2: If no REQ-* found, try alternative patterns
    if not requirements:
        for pattern in ALT_REQ_PATTERNS:
            for match in pattern.finditer(text):
                req_id = match.group(1)
                req_text = match.group(2).strip() if match.lastindex >= 2 else ""

                if req_id not in seen_ids:
                    seen_ids.add(req_id)
                    requirements.append(Requirement(
                        id=req_id,
                        text=req_text or "(no description)",
                        line_no=text[:match.start()].count("\n") + 1,
                    ))

    return requirements, warnings


# =============================================================================
# Definition of Done Extraction
# =============================================================================

# Headers that might indicate DoD section (case-insensitive fuzzy matching)
DOD_HEADER_PATTERNS = [
    re.compile(r"^#+\s*Definition\s+of\s+Done\b", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^#+\s*DoD\b", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^#+\s*Done\s+When\b", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^#+\s*Acceptance\s+Criteria\b", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^#+\s*Success\s+Criteria\b", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\*\*Definition\s+of\s+Done[:\*]*\b", re.MULTILINE | re.IGNORECASE),
]


def _extract_definition_of_done(text: str, lines: list[str]) -> list[str]:
    """Extract Definition of Done bullets."""
    dod_items: list[str] = []

    # Find the start of the DoD section
    dod_start = None
    for pattern in DOD_HEADER_PATTERNS:
        match = pattern.search(text)
        if match:
            dod_start = match.end()
            break

    if dod_start is None:
        return []

    # Find where the section ends (next header or end of document)
    section_text = text[dod_start:]
    next_header_match = re.search(r"^#+\s", section_text, re.MULTILINE)
    if next_header_match:
        section_text = section_text[:next_header_match.start()]

    # Extract bullet points
    bullet_pattern = re.compile(r"^\s*[-*]\s+(.+)", re.MULTILINE)
    for match in bullet_pattern.finditer(section_text):
        item = match.group(1).strip()
        if item and len(item) > 2:  # Filter out very short items (e.g., just punctuation)
            dod_items.append(item)

    return dod_items


# =============================================================================
# Test Matrix Extraction
# =============================================================================


def _extract_test_matrix(text: str, lines: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Extract test matrix mapping requirements to pytest node IDs.

    Returns:
        Tuple of (test_matrix dict, warnings)
    """
    test_matrix: dict[str, list[str]] = {}
    warnings: list[str] = []

    # Look for markdown table with "Requirement" and "pytest" headers
    # Pattern: | Requirement | Pytest | or similar
    table_header_pattern = re.compile(
        r"^\s*\|\s*(?:Requirement|REQ).*\|.*(?:pytest|test|coverage)",
        re.MULTILINE | re.IGNORECASE
    )

    table_start = None
    for i, line in enumerate(lines):
        if table_header_pattern.match(line):
            table_start = i
            break

    if table_start is None:
        return {}, []

    # Skip header and separator rows
    data_start = table_start + 2
    if data_start >= len(lines):
        return {}, []

    # Parse table rows
    for line in lines[data_start:]:
        line = line.strip()
        if not line.startswith("|"):
            break

        # Split by | and get columns
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 2:
            continue

        req_col = cols[0].strip()
        tests_col = cols[1].strip() if len(cols) > 1 else ""

        # Extract requirement ID from first column
        req_match = REQ_ID_PATTERN.search(req_col)
        if not req_match:
            continue

        req_id = req_match.group(1)

        # Parse test node IDs (comma or semicolon separated)
        test_ids = [t.strip() for t in re.split(r"[,;]", tests_col) if t.strip()]
        # Filter to likely pytest node IDs
        test_ids = [t for t in test_ids if "::" in t or t.startswith("tests/") or t.startswith("test_")]

        if test_ids:
            if req_id in test_matrix:
                warnings.append(f"Duplicate test matrix entry for {req_id}")
                test_matrix[req_id].extend(test_ids)
            else:
                test_matrix[req_id] = test_ids

    return test_matrix, warnings


# =============================================================================
# Contract Validation
# =============================================================================


def _validate_contract(spec: DesignDocSpec, mode: ContractMode) -> None:
    """Validate the spec against the contract mode.

    Adds errors/warnings to the spec in place.
    """
    if mode == "off":
        return

    if mode == "strict":
        # Strict mode: must have milestone_id, requirements, DoD, and full test matrix
        if not spec.milestone_id:
            spec.errors.append("STRICT: Missing milestone_id (use --milestone-id to override)")

        if not spec.requirements:
            spec.errors.append("STRICT: No requirements found in document")

        if not spec.definition_of_done:
            spec.errors.append("STRICT: No Definition of Done section found")

        if not spec.test_matrix:
            spec.errors.append("STRICT: No test matrix found")
        elif spec.requirements:
            # Check all requirements have test coverage
            missing_coverage = [r.id for r in spec.requirements if r.id not in spec.test_matrix]
            if missing_coverage:
                spec.errors.append(f"STRICT: Requirements missing test coverage: {missing_coverage}")

    elif mode == "loose":
        # Loose mode: must have at least milestone_id OR some requirements
        if not spec.milestone_id and not spec.requirements:
            spec.errors.append("LOOSE: Need at least milestone_id or some requirements")

        # Warn but don't fail on missing sections
        if not spec.milestone_id:
            spec.warnings.append("Could not extract milestone_id")

        if not spec.definition_of_done:
            spec.warnings.append("No Definition of Done section found")

        if not spec.test_matrix:
            spec.warnings.append("No test matrix found")
        elif spec.requirements:
            missing_coverage = [r.id for r in spec.requirements if r.id not in spec.test_matrix]
            if missing_coverage:
                spec.warnings.append(f"Requirements missing test coverage: {missing_coverage}")


# =============================================================================
# Main Parser Function
# =============================================================================


def parse_design_doc(
    path: Path | str,
    contract_mode: ContractMode = "loose",
    milestone_override: str | None = None,
) -> DesignDocSpec:
    """Parse a design document and extract structured data.

    Args:
        path: Path to the design document
        contract_mode: Validation strictness level
            - "strict": Fail if any required field is missing
            - "loose": Warn on missing fields, fail only if nothing found
            - "off": No validation, just extract what's possible
        milestone_override: Override the extracted milestone_id (CLI flag)

    Returns:
        DesignDocSpec with extracted data and any warnings/errors
    """
    path = Path(path)

    # Handle non-existent file
    if not path.exists():
        return DesignDocSpec(
            path=path,
            raw_text="",
            doc_hash="",
            contract_mode_used=contract_mode,
            errors=[f"Design document not found: {path}"],
        )

    # Read the document
    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception as e:
        return DesignDocSpec(
            path=path,
            raw_text="",
            doc_hash="",
            contract_mode_used=contract_mode,
            errors=[f"Failed to read design document: {e}"],
        )

    # Compute hash
    doc_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    # Split into lines for line-number tracking
    lines = raw_text.splitlines()

    # Extract all fields
    milestone_id, _ = _extract_milestone_id(raw_text)
    title = _extract_title(raw_text)
    requirements, req_warnings = _extract_requirements(raw_text, lines)
    definition_of_done = _extract_definition_of_done(raw_text, lines)
    test_matrix, matrix_warnings = _extract_test_matrix(raw_text, lines)

    # Apply milestone override if provided
    if milestone_override:
        milestone_id = milestone_override

    # Build the spec
    spec = DesignDocSpec(
        path=path,
        raw_text=raw_text,
        doc_hash=doc_hash,
        milestone_id=milestone_id,
        title=title,
        requirements=requirements,
        definition_of_done=definition_of_done,
        test_matrix=test_matrix,
        warnings=req_warnings + matrix_warnings,
        errors=[],
        contract_mode_used=contract_mode,
    )

    # Validate against contract
    _validate_contract(spec, contract_mode)

    return spec


def parse_design_doc_text(
    text: str,
    contract_mode: ContractMode = "loose",
    milestone_override: str | None = None,
    path: Path | str = Path("<inline>"),
) -> DesignDocSpec:
    """Parse design document from raw text (for testing).

    Args:
        text: Raw markdown text
        contract_mode: Validation strictness level
        milestone_override: Override the extracted milestone_id
        path: Path to use in the spec (for identification)

    Returns:
        DesignDocSpec with extracted data
    """
    path = Path(path)
    doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    lines = text.splitlines()

    milestone_id, _ = _extract_milestone_id(text)
    title = _extract_title(text)
    requirements, req_warnings = _extract_requirements(text, lines)
    definition_of_done = _extract_definition_of_done(text, lines)
    test_matrix, matrix_warnings = _extract_test_matrix(text, lines)

    if milestone_override:
        milestone_id = milestone_override

    spec = DesignDocSpec(
        path=path,
        raw_text=text,
        doc_hash=doc_hash,
        milestone_id=milestone_id,
        title=title,
        requirements=requirements,
        definition_of_done=definition_of_done,
        test_matrix=test_matrix,
        warnings=req_warnings + matrix_warnings,
        errors=[],
        contract_mode_used=contract_mode,
    )

    _validate_contract(spec, contract_mode)
    return spec


# =============================================================================
# CLI for standalone testing
# =============================================================================


def main() -> int:
    """CLI entry point for testing the parser."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Parse a design document")
    parser.add_argument("doc", type=str, help="Path to design document")
    parser.add_argument(
        "--contract",
        choices=["strict", "loose", "off"],
        default="loose",
        help="Contract validation mode",
    )
    parser.add_argument(
        "--milestone-id",
        type=str,
        help="Override milestone ID",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    spec = parse_design_doc(
        Path(args.doc),
        contract_mode=args.contract,
        milestone_override=args.milestone_id,
    )

    if args.json:
        print(json.dumps(spec.to_dict(), indent=2))
    else:
        print(f"Path: {spec.path}")
        print(f"Hash: {spec.doc_hash[:16]}...")
        print(f"Milestone: {spec.milestone_id or '(not found)'}")
        print(f"Title: {spec.title or '(not found)'}")
        print(f"Requirements: {len(spec.requirements)}")
        for r in spec.requirements[:5]:
            print(f"  - {r.id}: {r.text[:60]}...")
        if len(spec.requirements) > 5:
            print(f"  ... and {len(spec.requirements) - 5} more")
        print(f"DoD items: {len(spec.definition_of_done)}")
        print(f"Test matrix entries: {len(spec.test_matrix)}")
        print(f"Warnings: {len(spec.warnings)}")
        for w in spec.warnings:
            print(f"  - {w}")
        print(f"Errors: {len(spec.errors)}")
        for e in spec.errors:
            print(f"  - {e}")
        print(f"Valid: {spec.is_valid}")

    return 0 if spec.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
