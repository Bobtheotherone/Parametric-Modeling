# SPDX-License-Identifier: MIT
"""Tests for the gate_runner CLI module (audit-m1).

This module tests:
- Gate selection parsing (comma-separated, ranges)
- Pytest marker expression generation
- JUnit XML parsing
- Audit report generation
- Summary output formatting
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.coupongen.gate_runner import (
    GATE_DESCRIPTIONS,
    GATES,
    build_audit_report,
    build_pytest_marker_expr,
    categorize_tests_by_gate,
    compute_gate_status,
    parse_gates,
    parse_junit_xml,
    run_gates,
)


class TestParseGates:
    """Tests for parse_gates function."""

    def test_single_gate(self) -> None:
        """Parse single gate ID."""
        assert parse_gates("G1") == ["G1"]
        assert parse_gates("g2") == ["G2"]  # case insensitive
        assert parse_gates("G5") == ["G5"]

    def test_comma_separated_gates(self) -> None:
        """Parse comma-separated gate IDs."""
        assert parse_gates("G1,G2,G3") == ["G1", "G2", "G3"]
        assert parse_gates("g1, g3, g5") == ["G1", "G3", "G5"]
        assert parse_gates("G2,G4") == ["G2", "G4"]

    def test_range_syntax(self) -> None:
        """Parse range syntax (e.g., G1-G5)."""
        assert parse_gates("G1-G5") == ["G1", "G2", "G3", "G4", "G5"]
        assert parse_gates("G2-G4") == ["G2", "G3", "G4"]
        assert parse_gates("g1-g3") == ["G1", "G2", "G3"]

    def test_whitespace_handling(self) -> None:
        """Handle whitespace in gate specifications."""
        assert parse_gates(" G1 ") == ["G1"]
        assert parse_gates("G1 , G2 , G3") == ["G1", "G2", "G3"]
        assert parse_gates("  G1-G3  ") == ["G1", "G2", "G3"]

    def test_invalid_gate_raises(self) -> None:
        """Invalid gate IDs raise ValueError."""
        with pytest.raises(ValueError, match="Unknown gate"):
            parse_gates("G0")
        with pytest.raises(ValueError, match="Unknown gate"):
            parse_gates("G6")
        with pytest.raises(ValueError, match="Unknown gate"):
            parse_gates("INVALID")

    def test_empty_string(self) -> None:
        """Empty string returns empty list."""
        assert parse_gates("") == []
        assert parse_gates("   ") == []


class TestBuildPytestMarkerExpr:
    """Tests for build_pytest_marker_expr function."""

    def test_single_gate(self) -> None:
        """Single gate produces single marker."""
        assert build_pytest_marker_expr(["G1"]) == "gate_g1"
        assert build_pytest_marker_expr(["G5"]) == "gate_g5"

    def test_multiple_gates(self) -> None:
        """Multiple gates produce OR expression."""
        expr = build_pytest_marker_expr(["G1", "G2", "G3"])
        assert expr == "gate_g1 or gate_g2 or gate_g3"

    def test_all_gates(self) -> None:
        """All gates produce full OR expression."""
        expr = build_pytest_marker_expr(["G1", "G2", "G3", "G4", "G5"])
        assert "gate_g1" in expr
        assert "gate_g2" in expr
        assert "gate_g3" in expr
        assert "gate_g4" in expr
        assert "gate_g5" in expr
        assert expr.count(" or ") == 4


class TestGateConstants:
    """Tests for gate constant definitions."""

    def test_all_gates_have_markers(self) -> None:
        """All gates have associated pytest markers."""
        for gate_id in ["G1", "G2", "G3", "G4", "G5"]:
            assert gate_id in GATES
            assert GATES[gate_id].startswith("gate_g")

    def test_all_gates_have_descriptions(self) -> None:
        """All gates have descriptions."""
        for gate_id in ["G1", "G2", "G3", "G4", "G5"]:
            assert gate_id in GATE_DESCRIPTIONS
            assert len(GATE_DESCRIPTIONS[gate_id]) > 0

    def test_marker_format(self) -> None:
        """Markers follow expected format."""
        assert GATES["G1"] == "gate_g1"
        assert GATES["G2"] == "gate_g2"
        assert GATES["G3"] == "gate_g3"
        assert GATES["G4"] == "gate_g4"
        assert GATES["G5"] == "gate_g5"


class TestParseJunitXml:
    """Tests for parse_junit_xml function."""

    def test_parse_valid_junit_xml(self, tmp_path: Path) -> None:
        """Parse valid JUnit XML with mixed results."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="pytest" tests="5" errors="1" failures="1" skipped="1">
    <testcase classname="tests.gates.test_g1_determinism.TestG1" name="test_determinism" time="0.1"/>
    <testcase classname="tests.gates.test_g2_constraints.TestG2" name="test_constraints" time="0.2">
        <failure message="assertion failed">AssertionError: expected 1, got 2</failure>
    </testcase>
    <testcase classname="tests.gates.test_g3_drc.TestG3" name="test_drc" time="0.3">
        <error message="unexpected error">RuntimeError: connection failed</error>
    </testcase>
    <testcase classname="tests.gates.test_g4_export.TestG4" name="test_export" time="0.1">
        <skipped message="docker not available"/>
    </testcase>
    <testcase classname="tests.gates.test_g5_hash.TestG5" name="test_hash" time="0.15"/>
</testsuite>
"""
        xml_path = tmp_path / "junit.xml"
        xml_path.write_text(xml_content)

        result = parse_junit_xml(xml_path)

        assert result["summary"]["total"] == 5
        assert result["summary"]["failed"] == 1
        assert result["summary"]["errors"] == 1
        assert result["summary"]["skipped"] == 1
        assert result["summary"]["passed"] == 2

        assert len(result["tests"]) == 5

    def test_parse_all_passed(self, tmp_path: Path) -> None:
        """Parse JUnit XML where all tests passed."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="pytest" tests="3" errors="0" failures="0" skipped="0">
    <testcase classname="test_g1" name="test1" time="0.1"/>
    <testcase classname="test_g1" name="test2" time="0.2"/>
    <testcase classname="test_g1" name="test3" time="0.3"/>
</testsuite>
"""
        xml_path = tmp_path / "junit.xml"
        xml_path.write_text(xml_content)

        result = parse_junit_xml(xml_path)

        assert result["summary"]["total"] == 3
        assert result["summary"]["passed"] == 3
        assert result["summary"]["failed"] == 0

    def test_missing_file(self, tmp_path: Path) -> None:
        """Handle missing JUnit XML file gracefully."""
        xml_path = tmp_path / "nonexistent.xml"

        result = parse_junit_xml(xml_path)

        assert "error" in result
        assert result["tests"] == []

    def test_invalid_xml(self, tmp_path: Path) -> None:
        """Handle invalid XML gracefully."""
        xml_path = tmp_path / "invalid.xml"
        xml_path.write_text("not valid xml <<<<")

        result = parse_junit_xml(xml_path)

        assert "error" in result


class TestCategorizeTestsByGate:
    """Tests for categorize_tests_by_gate function."""

    def test_categorize_by_classname(self) -> None:
        """Categorize tests by classname patterns."""
        tests = [
            {"classname": "tests.gates.test_g1_determinism.TestG1", "name": "test_foo"},
            {"classname": "tests.gates.test_g2_constraints.TestG2", "name": "test_bar"},
            {"classname": "tests.gates.test_g3_drc.TestG3", "name": "test_baz"},
        ]

        result = categorize_tests_by_gate(tests)

        assert len(result["G1"]) == 1
        assert len(result["G2"]) == 1
        assert len(result["G3"]) == 1
        assert len(result["G4"]) == 0
        assert len(result["G5"]) == 0

    def test_categorize_by_keyword(self) -> None:
        """Categorize tests by keyword patterns in classname."""
        tests = [
            {"classname": "TestDeterminism", "name": "test_foo"},
            {"classname": "TestConstraints", "name": "test_bar"},
            {"classname": "TestDRC", "name": "test_baz"},
            {"classname": "TestExportCompleteness", "name": "test_qux"},
            {"classname": "TestHashStability", "name": "test_quux"},
        ]

        result = categorize_tests_by_gate(tests)

        assert len(result["G1"]) == 1  # determinism
        assert len(result["G2"]) == 1  # constraint
        assert len(result["G3"]) == 1  # drc
        assert len(result["G4"]) == 1  # export/completeness
        assert len(result["G5"]) == 1  # hash/stability


class TestComputeGateStatus:
    """Tests for compute_gate_status function."""

    def test_all_passed(self) -> None:
        """All tests passed -> gate passed."""
        tests = [
            {"status": "passed"},
            {"status": "passed"},
            {"status": "passed"},
        ]
        assert compute_gate_status(tests) == "passed"

    def test_any_failed(self) -> None:
        """Any test failed -> gate failed."""
        tests = [
            {"status": "passed"},
            {"status": "failed"},
            {"status": "passed"},
        ]
        assert compute_gate_status(tests) == "failed"

    def test_any_error(self) -> None:
        """Any test error -> gate failed."""
        tests = [
            {"status": "passed"},
            {"status": "error"},
        ]
        assert compute_gate_status(tests) == "failed"

    def test_all_skipped(self) -> None:
        """All tests skipped -> gate skipped."""
        tests = [
            {"status": "skipped"},
            {"status": "skipped"},
        ]
        assert compute_gate_status(tests) == "skipped"

    def test_empty_tests(self) -> None:
        """No tests -> no_tests."""
        assert compute_gate_status([]) == "no_tests"

    def test_passed_and_skipped(self) -> None:
        """Mixed passed and skipped -> passed (skipped don't fail gate)."""
        tests = [
            {"status": "passed"},
            {"status": "skipped"},
        ]
        assert compute_gate_status(tests) == "passed"


class TestBuildAuditReport:
    """Tests for build_audit_report function."""

    def test_build_report_structure(self) -> None:
        """Audit report has required structure."""
        junit_results = {
            "tests": [
                {"classname": "test_g1", "name": "test1", "status": "passed"},
                {"classname": "test_g2", "name": "test2", "status": "passed"},
            ],
            "summary": {
                "total": 2,
                "passed": 2,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
            },
        }

        report = build_audit_report(
            gates_requested=["G1", "G2"],
            junit_results=junit_results,
            return_code=0,
        )

        # Check required keys
        assert "schema_version" in report
        assert "timestamp" in report
        assert "gates_requested" in report
        assert "overall_status" in report
        assert "pytest_returncode" in report
        assert "summary" in report
        assert "gates" in report
        assert "per_spec" in report

        # Check gates
        assert "G1" in report["gates"]
        assert "G2" in report["gates"]

    def test_overall_status_passed(self) -> None:
        """Overall status is passed when all gates pass."""
        junit_results = {
            "tests": [
                {"classname": "test_g1", "name": "test1", "status": "passed"},
            ],
            "summary": {"total": 1, "passed": 1, "failed": 0, "skipped": 0, "errors": 0},
        }

        report = build_audit_report(
            gates_requested=["G1"],
            junit_results=junit_results,
            return_code=0,
        )

        assert report["overall_status"] == "passed"

    def test_overall_status_failed(self) -> None:
        """Overall status is failed when any gate fails."""
        junit_results = {
            "tests": [
                {"classname": "test_g1", "name": "test1", "status": "failed"},
            ],
            "summary": {"total": 1, "passed": 0, "failed": 1, "skipped": 0, "errors": 0},
        }

        report = build_audit_report(
            gates_requested=["G1"],
            junit_results=junit_results,
            return_code=1,
        )

        assert report["overall_status"] == "failed"

    def test_report_is_json_serializable(self) -> None:
        """Audit report can be serialized to JSON."""
        junit_results = {
            "tests": [],
            "summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0},
        }

        report = build_audit_report(
            gates_requested=["G1", "G2", "G3", "G4", "G5"],
            junit_results=junit_results,
            return_code=0,
        )

        # Should not raise
        json_str = json.dumps(report)
        assert isinstance(json_str, str)

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["gates_requested"] == ["G1", "G2", "G3", "G4", "G5"]


class TestMarkerDeclaration:
    """Tests that verify markers are properly declared in pyproject.toml."""

    def test_markers_declared(self) -> None:
        """All gate markers should be declared in pyproject.toml."""
        # Read pyproject.toml
        repo_root = Path(__file__).resolve().parents[1]
        pyproject_path = repo_root / "pyproject.toml"

        if not pyproject_path.exists():
            pytest.skip("pyproject.toml not found")

        content = pyproject_path.read_text()

        # Check all gate markers are declared
        for gate_id, marker in GATES.items():
            assert marker in content, f"Marker {marker} for {gate_id} not declared in pyproject.toml"


class TestArtifactPath:
    """Tests for artifact path computation."""

    def test_compute_artifact_path_default_root(self) -> None:
        """compute_artifact_path uses default artifacts root."""
        from formula_foundry.coupongen.gate_runner import compute_artifact_path

        design_hash = "abcd1234567890ef"
        path = compute_artifact_path(design_hash)

        assert path.name == "audit_report.json"
        assert path.parent.name == design_hash
        assert path.parent.parent.name == "audit_m1"
        assert "artifacts" in str(path)

    def test_compute_artifact_path_custom_root(self, tmp_path: Path) -> None:
        """compute_artifact_path uses custom artifacts root."""
        from formula_foundry.coupongen.gate_runner import compute_artifact_path

        design_hash = "test_hash_123"
        path = compute_artifact_path(design_hash, artifact_root=tmp_path)

        expected = tmp_path / "audit_m1" / design_hash / "audit_report.json"
        assert path == expected


class TestSubprocessMocking:
    """Tests for gate runner with subprocess mocking.

    These tests verify report structure without actually running pytest.
    """

    def test_run_gates_mocked_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test run_gates with mocked subprocess returning success."""
        from unittest.mock import MagicMock

        # Create mock JUnit XML
        junit_xml_path = tmp_path / "junit.xml"
        junit_content = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="pytest" tests="3" errors="0" failures="0" skipped="0">
    <testcase classname="test_g1" name="test_determinism" time="0.1"/>
    <testcase classname="test_g2" name="test_constraints" time="0.2"/>
    <testcase classname="test_g3" name="test_drc" time="0.3"/>
</testsuite>
"""
        junit_xml_path.write_text(junit_content)

        # Mock subprocess.run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "3 passed"
        mock_result.stderr = ""

        import subprocess

        def mock_subprocess_run(cmd, **kwargs):
            # Write the JUnit XML to the specified path
            for _i, arg in enumerate(cmd):
                if isinstance(arg, str) and arg.startswith("--junit-xml="):
                    path = Path(arg.split("=", 1)[1])
                    path.write_text(junit_content)
                    break
            return mock_result

        monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

        # Run gates with mocked subprocess
        return_code, result_xml = run_gates(
            gates=["G1", "G2", "G3"],
            test_dir=tmp_path,
            verbose=False,
        )

        assert return_code == 0
        assert result_xml is not None
        assert result_xml.exists()

    def test_run_gates_mocked_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test run_gates with mocked subprocess returning failure."""
        from unittest.mock import MagicMock

        junit_content = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="pytest" tests="2" errors="0" failures="1" skipped="0">
    <testcase classname="test_g1" name="test_pass" time="0.1"/>
    <testcase classname="test_g1" name="test_fail" time="0.2">
        <failure message="assertion failed">AssertionError</failure>
    </testcase>
</testsuite>
"""

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "1 passed, 1 failed"
        mock_result.stderr = ""

        import subprocess

        def mock_subprocess_run(cmd, **kwargs):
            for arg in cmd:
                if isinstance(arg, str) and arg.startswith("--junit-xml="):
                    path = Path(arg.split("=", 1)[1])
                    path.write_text(junit_content)
                    break
            return mock_result

        monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

        return_code, result_xml = run_gates(
            gates=["G1"],
            test_dir=tmp_path,
            verbose=False,
        )

        assert return_code == 1
        assert result_xml is not None

    def test_report_structure_with_mocked_results(self, tmp_path: Path) -> None:
        """Test audit report structure with mocked JUnit results."""
        junit_results = {
            "tests": [
                {"classname": "test_g1", "name": "test_a", "status": "passed", "time": 0.1},
                {"classname": "test_g1", "name": "test_b", "status": "passed", "time": 0.2},
                {"classname": "test_g2", "name": "test_c", "status": "failed", "message": "error"},
                {"classname": "test_g3", "name": "test_d", "status": "skipped"},
            ],
            "summary": {
                "total": 4,
                "passed": 2,
                "failed": 1,
                "skipped": 1,
                "errors": 0,
            },
        }

        report = build_audit_report(
            gates_requested=["G1", "G2", "G3"],
            junit_results=junit_results,
            return_code=1,
            junit_xml_path=tmp_path / "junit.xml",
        )

        # Verify required report structure
        assert "schema_version" in report
        assert report["schema_version"] == "1.0.0"
        assert "timestamp" in report
        assert "gates_requested" in report
        assert report["gates_requested"] == ["G1", "G2", "G3"]
        assert "overall_status" in report
        assert report["overall_status"] == "failed"  # G2 failed
        assert "pytest_returncode" in report
        assert report["pytest_returncode"] == 1
        assert "summary" in report
        assert "gates" in report

        # Verify gate structure
        for gate_id in ["G1", "G2", "G3"]:
            assert gate_id in report["gates"]
            gate = report["gates"][gate_id]
            assert "description" in gate
            assert "marker" in gate
            assert "status" in gate
            assert "tests_count" in gate
            assert "passed" in gate
            assert "failed" in gate
            assert "skipped" in gate
            assert "tests" in gate

        # Verify G1 passed (both tests passed)
        assert report["gates"]["G1"]["status"] == "passed"
        assert report["gates"]["G1"]["passed"] == 2

        # Verify G2 failed
        assert report["gates"]["G2"]["status"] == "failed"
        assert report["gates"]["G2"]["failed"] == 1

        # Verify G3 skipped
        assert report["gates"]["G3"]["status"] == "skipped"

    def test_report_with_design_hash_artifact_structure(self, tmp_path: Path) -> None:
        """Test that report with design_hash creates proper artifact structure."""
        from formula_foundry.coupongen.gate_runner import compute_artifact_path

        design_hash = "abc123def456"
        artifact_root = tmp_path / "artifacts"

        output_path = compute_artifact_path(design_hash, artifact_root=artifact_root)

        expected_path = artifact_root / "audit_m1" / design_hash / "audit_report.json"
        assert output_path == expected_path

        # Verify structure
        assert output_path.parent.name == design_hash
        assert output_path.parent.parent.name == "audit_m1"

    def test_report_json_serialization(self) -> None:
        """Test that report is properly JSON serializable."""
        junit_results = {
            "tests": [
                {"classname": "test_g1", "name": "test_a", "status": "passed"},
            ],
            "summary": {"total": 1, "passed": 1, "failed": 0, "skipped": 0, "errors": 0},
        }

        report = build_audit_report(
            gates_requested=["G1", "G2", "G3", "G4", "G5"],
            junit_results=junit_results,
            return_code=0,
        )

        # Serialize to JSON
        json_str = json.dumps(report, indent=2, sort_keys=True)

        # Parse back
        parsed = json.loads(json_str)

        # Verify round-trip
        assert parsed["gates_requested"] == ["G1", "G2", "G3", "G4", "G5"]
        assert parsed["overall_status"] in ("passed", "failed", "partial")
        assert "summary" in parsed
        assert "gates" in parsed

        # All gates should be present
        for gate_id in ["G1", "G2", "G3", "G4", "G5"]:
            assert gate_id in parsed["gates"]

    def test_report_per_spec_summary_includes_paths(self, tmp_path: Path) -> None:
        """Per-spec summary should include spec path and status."""
        repo_root = tmp_path
        spec_dir = repo_root / "tests" / "golden_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / "f0_cal_example.yaml"
        spec_path.write_text("invalid: true\n", encoding="utf-8")

        junit_results = {
            "tests": [
                {
                    "classname": "tests.gates.test_g1_determinism.TestG1",
                    "name": "test_determinism[f0_cal_example.yaml]",
                    "status": "passed",
                    "time": 0.1,
                },
                {
                    "classname": "tests.gates.test_g2_constraints.TestG2",
                    "name": "test_constraints[f0_cal_example.yaml]",
                    "status": "failed",
                    "time": 0.2,
                },
            ],
            "summary": {"total": 2, "passed": 1, "failed": 1, "skipped": 0, "errors": 0},
        }

        report = build_audit_report(
            gates_requested=["G1", "G2"],
            junit_results=junit_results,
            return_code=1,
            repo_root=repo_root,
        )

        assert "per_spec" in report
        spec_entry = report["per_spec"]["f0_cal_example"]
        assert spec_entry["status"] == "failed"
        assert spec_entry["artifact_paths"]["spec"] == str(spec_path)
