"""Tests for canonical case layout validation.

This module tests REQ-M2-003: Validator enforces canonical case layout
and required fab files.

Test coverage:
- Required files validation
- Required fab set validation
- Naming convention enforcement
- Fail-fast behavior
- Extra/forbidden file detection
- Error reporting
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestCaseLayoutValidation:
    """Tests for validate_case_layout function."""

    def test_valid_case_layout_passes(self, tmp_path: Path) -> None:
        """A case directory with all required files validates successfully."""
        from formula_foundry.validation.layout import validate_case_layout

        # Create valid case structure
        case_dir = tmp_path / "valid_case"
        case_dir.mkdir()

        # Create oracle_case.json
        oracle_case = {
            "case_id": "CAL_THRU_001",
            "format_version": "1.0",
            "frequency": {"start_hz": 1e6, "stop_hz": 10e9},
        }
        (case_dir / "oracle_case.json").write_text(json.dumps(oracle_case))

        # Create fab/gerbers directory with a gerber file
        gerbers_dir = case_dir / "fab" / "gerbers"
        gerbers_dir.mkdir(parents=True)
        (gerbers_dir / "test-F_Cu.gtl").write_text("G04 Test*")

        result = validate_case_layout(case_dir, fail_fast=False)
        assert result.valid is True
        assert result.case_id == "CAL_THRU_001"
        assert result.format_version == "1.0"
        assert len(result.errors) == 0

    def test_missing_oracle_case_json_fails(self, tmp_path: Path) -> None:
        """Missing oracle_case.json causes validation failure."""
        from formula_foundry.validation.layout import (
            CaseLayoutError,
            validate_case_layout,
        )

        case_dir = tmp_path / "missing_oracle"
        case_dir.mkdir()

        # Create fab structure but no oracle_case.json
        gerbers_dir = case_dir / "fab" / "gerbers"
        gerbers_dir.mkdir(parents=True)
        (gerbers_dir / "test-F_Cu.gtl").write_text("G04 Test*")

        with pytest.raises(CaseLayoutError) as exc_info:
            validate_case_layout(case_dir)

        assert "oracle_case.json" in str(exc_info.value)
        assert exc_info.value.result.valid is False
        assert "oracle_case.json" in exc_info.value.result.missing_required

    def test_invalid_oracle_case_json_fails(self, tmp_path: Path) -> None:
        """Invalid JSON in oracle_case.json causes validation failure."""
        from formula_foundry.validation.layout import (
            CaseLayoutError,
            validate_case_layout,
        )

        case_dir = tmp_path / "invalid_json"
        case_dir.mkdir()

        # Create invalid JSON
        (case_dir / "oracle_case.json").write_text("{ invalid json }")

        with pytest.raises(CaseLayoutError) as exc_info:
            validate_case_layout(case_dir)

        assert "not valid JSON" in str(exc_info.value)

    def test_missing_case_id_fails(self, tmp_path: Path) -> None:
        """Missing case_id in oracle_case.json causes validation failure."""
        from formula_foundry.validation.layout import (
            CaseLayoutError,
            validate_case_layout,
        )

        case_dir = tmp_path / "missing_case_id"
        case_dir.mkdir()

        oracle_case = {"format_version": "1.0"}
        (case_dir / "oracle_case.json").write_text(json.dumps(oracle_case))

        with pytest.raises(CaseLayoutError) as exc_info:
            validate_case_layout(case_dir)

        assert "case_id" in str(exc_info.value)

    def test_invalid_case_id_naming_fails(self, tmp_path: Path) -> None:
        """Invalid case_id naming pattern causes validation failure."""
        from formula_foundry.validation.layout import (
            CaseLayoutError,
            validate_case_layout,
        )

        case_dir = tmp_path / "invalid_name"
        case_dir.mkdir()

        # case_id must start with uppercase letter or digit
        oracle_case = {"case_id": "lowercase_invalid", "format_version": "1.0"}
        (case_dir / "oracle_case.json").write_text(json.dumps(oracle_case))

        with pytest.raises(CaseLayoutError) as exc_info:
            validate_case_layout(case_dir)

        assert "naming pattern" in str(exc_info.value)

    def test_missing_fab_gerbers_fails(self, tmp_path: Path) -> None:
        """Missing fab/gerbers directory causes validation failure when require_fab=True."""
        from formula_foundry.validation.layout import (
            CaseLayoutError,
            validate_case_layout,
        )

        case_dir = tmp_path / "missing_gerbers"
        case_dir.mkdir()

        oracle_case = {"case_id": "TEST_CASE", "format_version": "1.0"}
        (case_dir / "oracle_case.json").write_text(json.dumps(oracle_case))

        with pytest.raises(CaseLayoutError) as exc_info:
            validate_case_layout(case_dir, require_fab=True)

        assert "fab/" in str(exc_info.value) or "gerbers" in str(exc_info.value)

    def test_empty_gerbers_directory_fails(self, tmp_path: Path) -> None:
        """Empty fab/gerbers directory causes validation failure."""
        from formula_foundry.validation.layout import (
            CaseLayoutError,
            validate_case_layout,
        )

        case_dir = tmp_path / "empty_gerbers"
        case_dir.mkdir()

        oracle_case = {"case_id": "TEST_CASE", "format_version": "1.0"}
        (case_dir / "oracle_case.json").write_text(json.dumps(oracle_case))

        gerbers_dir = case_dir / "fab" / "gerbers"
        gerbers_dir.mkdir(parents=True)
        # No files in gerbers directory

        with pytest.raises(CaseLayoutError) as exc_info:
            validate_case_layout(case_dir, require_fab=True)

        assert "empty" in str(exc_info.value).lower()

    def test_invalid_gerber_naming_flagged(self, tmp_path: Path) -> None:
        """Gerber files with invalid naming are flagged."""
        from formula_foundry.validation.layout import validate_case_layout

        case_dir = tmp_path / "invalid_gerber_names"
        case_dir.mkdir()

        oracle_case = {"case_id": "TEST_CASE", "format_version": "1.0"}
        (case_dir / "oracle_case.json").write_text(json.dumps(oracle_case))

        gerbers_dir = case_dir / "fab" / "gerbers"
        gerbers_dir.mkdir(parents=True)
        # Create a gerber file with invalid naming (contains spaces)
        (gerbers_dir / "invalid file name.txt").write_text("invalid")
        # Also create a valid one
        (gerbers_dir / "valid-F_Cu.gtl").write_text("G04 Test*")

        result = validate_case_layout(case_dir, fail_fast=False, strict=False)
        assert "invalid file name.txt" in " ".join(result.invalid_names)

    def test_forbidden_files_cause_failure_in_strict_mode(self, tmp_path: Path) -> None:
        """Forbidden files (.env, etc.) cause failure in strict mode."""
        from formula_foundry.validation.layout import (
            CaseLayoutError,
            validate_case_layout,
        )

        case_dir = tmp_path / "forbidden_files"
        case_dir.mkdir()

        oracle_case = {"case_id": "TEST_CASE", "format_version": "1.0"}
        (case_dir / "oracle_case.json").write_text(json.dumps(oracle_case))

        gerbers_dir = case_dir / "fab" / "gerbers"
        gerbers_dir.mkdir(parents=True)
        (gerbers_dir / "valid-F_Cu.gtl").write_text("G04 Test*")

        # Create a forbidden file
        (case_dir / ".env").write_text("SECRET=value")

        with pytest.raises(CaseLayoutError) as exc_info:
            validate_case_layout(case_dir, strict=True)

        assert ".env" in str(exc_info.value) or "Forbidden" in str(exc_info.value)

    def test_fail_fast_raises_on_first_error(self, tmp_path: Path) -> None:
        """fail_fast=True raises CaseLayoutError immediately."""
        from formula_foundry.validation.layout import (
            CaseLayoutError,
            validate_case_layout,
        )

        case_dir = tmp_path / "fail_fast_test"
        case_dir.mkdir()
        # No oracle_case.json - should fail

        with pytest.raises(CaseLayoutError):
            validate_case_layout(case_dir, fail_fast=True)

    def test_fail_fast_false_collects_all_errors(self, tmp_path: Path) -> None:
        """fail_fast=False collects all errors without raising."""
        from formula_foundry.validation.layout import validate_case_layout

        case_dir = tmp_path / "collect_errors"
        case_dir.mkdir()

        # Create invalid oracle_case.json (missing case_id)
        oracle_case = {"format_version": "1.0"}
        (case_dir / "oracle_case.json").write_text(json.dumps(oracle_case))

        # No fab directory

        result = validate_case_layout(case_dir, fail_fast=False, require_fab=True)
        assert result.valid is False
        # Should have multiple errors: missing case_id and missing fab
        assert len(result.errors) >= 2

    def test_require_fab_false_skips_fab_validation(self, tmp_path: Path) -> None:
        """require_fab=False allows cases without fab directory."""
        from formula_foundry.validation.layout import validate_case_layout

        case_dir = tmp_path / "no_fab_required"
        case_dir.mkdir()

        oracle_case = {"case_id": "TEST_CASE", "format_version": "1.0"}
        (case_dir / "oracle_case.json").write_text(json.dumps(oracle_case))

        # No fab directory, but require_fab=False
        result = validate_case_layout(case_dir, require_fab=False)
        assert result.valid is True

    def test_nonexistent_directory_raises_file_not_found(self) -> None:
        """Nonexistent case directory raises FileNotFoundError."""
        from formula_foundry.validation.layout import validate_case_layout

        with pytest.raises(FileNotFoundError):
            validate_case_layout("/nonexistent/path/to/case")

    def test_path_is_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Passing a file path instead of directory raises FileNotFoundError."""
        from formula_foundry.validation.layout import validate_case_layout

        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content")

        with pytest.raises(FileNotFoundError):
            validate_case_layout(file_path)

    def test_result_to_dict_serializable(self, tmp_path: Path) -> None:
        """CaseLayoutResult.to_dict() returns JSON-serializable dict."""
        from formula_foundry.validation.layout import validate_case_layout

        case_dir = tmp_path / "serializable_test"
        case_dir.mkdir()

        oracle_case = {"case_id": "TEST_CASE", "format_version": "1.0"}
        (case_dir / "oracle_case.json").write_text(json.dumps(oracle_case))

        gerbers_dir = case_dir / "fab" / "gerbers"
        gerbers_dir.mkdir(parents=True)
        (gerbers_dir / "valid-F_Cu.gtl").write_text("G04 Test*")

        result = validate_case_layout(case_dir, fail_fast=False)
        result_dict = result.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(result_dict)
        assert json_str is not None
        assert result_dict["valid"] is True
        assert result_dict["case_id"] == "TEST_CASE"


class TestCaseLayoutQuickValidation:
    """Tests for validate_case_layout_quick function."""

    def test_quick_validation_valid_case(self, tmp_path: Path) -> None:
        """Quick validation passes for valid case."""
        from formula_foundry.validation.layout import validate_case_layout_quick

        case_dir = tmp_path / "quick_valid"
        case_dir.mkdir()
        (case_dir / "oracle_case.json").write_text('{"case_id": "X"}')

        assert validate_case_layout_quick(case_dir) is True

    def test_quick_validation_missing_oracle(self, tmp_path: Path) -> None:
        """Quick validation fails for missing oracle_case.json."""
        from formula_foundry.validation.layout import validate_case_layout_quick

        case_dir = tmp_path / "quick_invalid"
        case_dir.mkdir()

        assert validate_case_layout_quick(case_dir) is False

    def test_quick_validation_nonexistent_dir(self, tmp_path: Path) -> None:
        """Quick validation fails for nonexistent directory."""
        from formula_foundry.validation.layout import validate_case_layout_quick

        assert validate_case_layout_quick(tmp_path / "nonexistent") is False


class TestCaseFileEnumeration:
    """Tests for list_case_files function."""

    def test_list_case_files_categorizes_correctly(self, tmp_path: Path) -> None:
        """list_case_files correctly categorizes files."""
        from formula_foundry.validation.layout import list_case_files

        case_dir = tmp_path / "enum_test"
        case_dir.mkdir()

        # Create various files
        (case_dir / "oracle_case.json").write_text("{}")
        (case_dir / "meta.json").write_text("{}")
        (case_dir / "other_file.txt").write_text("content")

        gerbers_dir = case_dir / "fab" / "gerbers"
        gerbers_dir.mkdir(parents=True)
        (gerbers_dir / "test.gtl").write_text("gerber")

        result = list_case_files(case_dir)

        assert "oracle_case.json" in result["required"]
        assert "fab/gerbers/test.gtl" in result["fab"]
        assert "meta.json" in result["optional"]
        assert "other_file.txt" in result["other"]

    def test_list_case_files_empty_dir(self, tmp_path: Path) -> None:
        """list_case_files returns empty lists for empty directory."""
        from formula_foundry.validation.layout import list_case_files

        case_dir = tmp_path / "empty_enum"
        case_dir.mkdir()

        result = list_case_files(case_dir)

        assert result["required"] == []
        assert result["fab"] == []
        assert result["optional"] == []
        assert result["other"] == []


class TestCaseLayoutNamingPatterns:
    """Tests for naming pattern validation."""

    def test_valid_case_ids(self, tmp_path: Path) -> None:
        """Valid case_id patterns are accepted."""
        from formula_foundry.validation.layout import validate_case_layout

        valid_ids = [
            "A",
            "A1",
            "CAL_THRU",
            "CAL-THRU-001",
            "X123_ABC_DEF",
            "1TEST",
            "0CASE",
        ]

        for case_id in valid_ids:
            case_dir = tmp_path / f"case_{case_id}"
            case_dir.mkdir()
            oracle_case = {"case_id": case_id, "format_version": "1.0"}
            (case_dir / "oracle_case.json").write_text(json.dumps(oracle_case))

            gerbers_dir = case_dir / "fab" / "gerbers"
            gerbers_dir.mkdir(parents=True)
            (gerbers_dir / "valid-F_Cu.gtl").write_text("G04 Test*")

            result = validate_case_layout(case_dir, fail_fast=False)
            assert result.valid is True, f"case_id '{case_id}' should be valid"

    def test_invalid_case_ids(self, tmp_path: Path) -> None:
        """Invalid case_id patterns are rejected."""
        from formula_foundry.validation.layout import validate_case_layout

        invalid_ids = [
            "lowercase",
            "_UNDERSCORE_START",
            "-DASH-START",
            "has spaces",
            "has.dots",
            "",
        ]

        for case_id in invalid_ids:
            case_dir = tmp_path / f"case_invalid_{hash(case_id) % 10000}"
            case_dir.mkdir()
            oracle_case = {"case_id": case_id, "format_version": "1.0"}
            (case_dir / "oracle_case.json").write_text(json.dumps(oracle_case))

            result = validate_case_layout(case_dir, fail_fast=False, require_fab=False)
            assert result.valid is False, f"case_id '{case_id}' should be invalid"


class TestValidatorEnforcesCanonicalCaseLayoutAndRequiredFabFiles:
    """REQ-M2-003: Test validator enforces canonical case layout and required fab files.

    This test class directly addresses the requirement in the test matrix.
    """

    def test_validator_enforces_canonical_case_layout_and_required_fab_files(self, tmp_path: Path) -> None:
        """REQ-M2-003: Canonical case layout validation with required fab files."""
        from formula_foundry.validation.layout import (
            REQUIRED_FILES,
            CaseLayoutError,
            validate_case_layout,
        )

        # Part 1: Valid case passes validation
        valid_case = tmp_path / "valid_canonical"
        valid_case.mkdir()

        oracle_case = {
            "case_id": "CAL_0_THRU",
            "format_version": "1.0",
            "frequency": {"start_hz": 1e6, "stop_hz": 20e9, "npoints": 201},
            "solver_policy": {"boundary": "PEC", "pml_cells": 8},
        }
        (valid_case / "oracle_case.json").write_text(json.dumps(oracle_case))

        gerbers_dir = valid_case / "fab" / "gerbers"
        gerbers_dir.mkdir(parents=True)
        # Create required Gerber files per layer_sets.json
        (gerbers_dir / "case-F_Cu.gtl").write_text("G04 F.Cu*")
        (gerbers_dir / "case-B_Cu.gbl").write_text("G04 B.Cu*")

        result = validate_case_layout(valid_case, fail_fast=False)
        assert result.valid is True
        assert result.case_id == "CAL_0_THRU"
        assert len(result.errors) == 0

        # Part 2: Missing oracle_case.json fails
        missing_oracle = tmp_path / "missing_oracle"
        missing_oracle.mkdir()
        (missing_oracle / "fab" / "gerbers").mkdir(parents=True)
        (missing_oracle / "fab" / "gerbers" / "test.gtl").write_text("G04*")

        with pytest.raises(CaseLayoutError) as exc:
            validate_case_layout(missing_oracle, fail_fast=True)
        assert "oracle_case.json" in str(exc.value)

        # Part 3: Missing fab files fails when required
        missing_fab = tmp_path / "missing_fab"
        missing_fab.mkdir()
        (missing_fab / "oracle_case.json").write_text(json.dumps({"case_id": "TEST", "format_version": "1.0"}))

        with pytest.raises(CaseLayoutError) as exc:
            validate_case_layout(missing_fab, require_fab=True, fail_fast=True)
        assert "fab" in str(exc.value).lower()

        # Part 4: Extra/forbidden files detected in strict mode
        extra_files = tmp_path / "extra_files"
        extra_files.mkdir()
        (extra_files / "oracle_case.json").write_text(json.dumps({"case_id": "TEST", "format_version": "1.0"}))
        (extra_files / "fab" / "gerbers").mkdir(parents=True)
        (extra_files / "fab" / "gerbers" / "test.gtl").write_text("G04*")
        (extra_files / ".env").write_text("SECRET=bad")

        with pytest.raises(CaseLayoutError) as exc:
            validate_case_layout(extra_files, strict=True, fail_fast=True)
        assert "Forbidden" in str(exc.value) or ".env" in str(exc.value)

        # Part 5: Verify REQUIRED_FILES contains expected entries
        assert "oracle_case.json" in REQUIRED_FILES
