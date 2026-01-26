# SPDX-License-Identifier: MIT
"""Gate runner CLI for M1 compliance verification (audit-m1).

This module provides a CLI for running G1-G5 gate tests with pytest markers,
generating machine-readable audit_report.json output, and summary statistics.

Per ECO-M1-ALIGN-0001, the gates verify:
- G1: Resolved design determinism
- G2: Constraint proof completeness and reject/repair behavior
- G3: DRC clean (KiCad DRC passes with zero violations)
- G4: Export completeness (all required layers + drill files present)
- G5: Stable canonical hashes across repeated runs

Usage:
    audit-m1 run --gates G1,G2,G3,G4,G5 --output audit_report.json
    audit-m1 run --gates G1,G2 --verbose
    audit-m1 list  # List available gates
    audit-m1 summary --input audit_report.json  # Summarize report
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# Gate marker names and descriptions
GATES: dict[str, str] = {
    "G1": "gate_g1",  # Resolved design determinism
    "G2": "gate_g2",  # Constraint proof completeness and reject/repair behavior
    "G3": "gate_g3",  # DRC clean
    "G4": "gate_g4",  # Export completeness
    "G5": "gate_g5",  # Stable canonical hashes
}

GATE_DESCRIPTIONS: dict[str, str] = {
    "G1": "Resolved design determinism",
    "G2": "Constraint proof completeness and reject/repair behavior",
    "G3": "DRC clean (KiCad DRC passes with zero violations)",
    "G4": "Export completeness (all required layers + drill files present)",
    "G5": "Stable canonical hashes across repeated runs",
}

KICAD_INTEGRATION_GATES = {"G3", "G4", "G5"}
SPEC_TOKEN_RE = re.compile(r"\b(f[01]_[A-Za-z0-9_.-]+)\b", re.IGNORECASE)


def parse_gates(gates_str: str) -> list[str]:
    """Parse comma-separated gate specification.

    Args:
        gates_str: Comma-separated gate IDs like "G1,G2,G3" or "G1-G5" for range

    Returns:
        List of gate IDs (e.g., ["G1", "G2", "G3"])

    Raises:
        ValueError: If gate ID is invalid
    """
    gates_str = gates_str.upper().strip()

    # Handle range syntax (e.g., "G1-G5")
    if "-" in gates_str and "," not in gates_str:
        parts = gates_str.split("-")
        if len(parts) == 2:
            start = parts[0].strip()
            end = parts[1].strip()
            if start.startswith("G") and end.startswith("G"):
                try:
                    start_num = int(start[1:])
                    end_num = int(end[1:])
                    return [f"G{i}" for i in range(start_num, end_num + 1)]
                except ValueError:
                    pass

    # Handle comma-separated (e.g., "G1,G2,G3")
    result = []
    for gate in gates_str.split(","):
        gate = gate.strip().upper()
        if not gate:
            continue
        if gate not in GATES:
            valid = ", ".join(sorted(GATES.keys()))
            raise ValueError(f"Unknown gate '{gate}'. Valid gates: {valid}")
        result.append(gate)

    return result


def build_pytest_marker_expr(gates: list[str]) -> str:
    """Build pytest -m marker expression for selected gates.

    Args:
        gates: List of gate IDs (e.g., ["G1", "G2"])

    Returns:
        Pytest marker expression (e.g., "gate_g1 or gate_g2")
    """
    markers = [GATES[g] for g in gates]
    return " or ".join(markers)


def run_gates(
    gates: list[str],
    test_dir: Path | list[Path] | None = None,
    junit_xml: Path | None = None,
    verbose: bool = False,
    extra_args: list[str] | None = None,
) -> tuple[int, Path | None]:
    """Run selected gate tests using pytest.

    Args:
        gates: List of gate IDs to run (e.g., ["G1", "G2", "G3"])
        test_dir: Directory containing gate tests (default: tests/gates/)
        junit_xml: Path for JUnit XML output (default: temp file)
        verbose: Enable verbose pytest output
        extra_args: Additional pytest arguments

    Returns:
        Tuple of (return_code, junit_xml_path)
    """
    import tempfile

    # Determine test directory
    test_dirs = _resolve_test_dirs(gates, test_dir)
    if not test_dirs:
        sys.stderr.write("Error: No test directories resolved\n")
        return 1, None
    for path in test_dirs:
        if not path.exists():
            sys.stderr.write(f"Error: Test directory not found: {path}\n")
            return 1, None

    # Create temp JUnit XML if not specified
    if junit_xml is None:
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_fd:
            junit_xml = Path(tmp_fd.name)

    # Build pytest command
    marker_expr = build_pytest_marker_expr(gates)
    pytest_args = [
        sys.executable,
        "-m",
        "pytest",
        *[str(path) for path in test_dirs],
        "-m",
        marker_expr,
        f"--junit-xml={junit_xml}",
    ]

    if verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-q")

    if extra_args:
        pytest_args.extend(extra_args)

    # Run pytest
    result = subprocess.run(
        pytest_args,
        capture_output=not verbose,
        text=True,
    )

    if verbose and result.stdout:
        sys.stdout.write(result.stdout)
    if verbose and result.stderr:
        sys.stderr.write(result.stderr)

    return result.returncode, junit_xml


def _resolve_test_dirs(gates: list[str], test_dir: Path | list[Path] | None) -> list[Path]:
    """Resolve which test directories to include based on gates requested."""
    if test_dir is not None:
        if isinstance(test_dir, (list, tuple)):
            return [Path(path) for path in test_dir]
        return [Path(test_dir)]

    repo_root = Path(__file__).resolve().parents[4]
    test_dirs = [repo_root / "tests" / "gates"]

    if any(gate in KICAD_INTEGRATION_GATES for gate in gates):
        integration_dir = repo_root / "tests" / "integration"
        if integration_dir.exists():
            test_dirs.append(integration_dir)

    return test_dirs


def parse_junit_xml(xml_path: Path) -> dict[str, Any]:
    """Parse JUnit XML to extract test results.

    Args:
        xml_path: Path to JUnit XML file

    Returns:
        Dictionary with parsed test results
    """
    if not xml_path.exists():
        return {"error": "JUnit XML not found", "tests": [], "summary": {}}

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        return {"error": f"Failed to parse JUnit XML: {e}", "tests": [], "summary": {}}

    # Parse testsuite element
    testsuite = root if root.tag == "testsuite" else root.find("testsuite")
    if testsuite is None:
        # Try testsuites container
        testsuites = root.find("testsuites")
        if testsuites is not None:
            testsuite = testsuites.find("testsuite")

    tests: list[dict[str, Any]] = []
    summary = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
    }

    if testsuite is not None:
        summary["total"] = int(testsuite.get("tests", 0))
        summary["failed"] = int(testsuite.get("failures", 0))
        summary["errors"] = int(testsuite.get("errors", 0))
        summary["skipped"] = int(testsuite.get("skipped", 0))
        summary["passed"] = summary["total"] - summary["failed"] - summary["errors"] - summary["skipped"]

        # Parse individual testcases
        for testcase in testsuite.findall(".//testcase"):
            test_info: dict[str, Any] = {
                "name": testcase.get("name", ""),
                "classname": testcase.get("classname", ""),
                "time": float(testcase.get("time", 0)),
                "status": "passed",
            }

            # Check for failure
            failure = testcase.find("failure")
            if failure is not None:
                test_info["status"] = "failed"
                test_info["message"] = failure.get("message", "")
                test_info["details"] = failure.text or ""

            # Check for error
            error = testcase.find("error")
            if error is not None:
                test_info["status"] = "error"
                test_info["message"] = error.get("message", "")
                test_info["details"] = error.text or ""

            # Check for skip
            skipped = testcase.find("skipped")
            if skipped is not None:
                test_info["status"] = "skipped"
                test_info["message"] = skipped.get("message", "")

            tests.append(test_info)

    return {"tests": tests, "summary": summary}


def categorize_tests_by_gate(tests: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Categorize test results by gate.

    Args:
        tests: List of test results from parse_junit_xml

    Returns:
        Dictionary mapping gate ID to tests for that gate
    """
    gate_tests: dict[str, list[dict[str, Any]]] = {f"G{i}": [] for i in range(1, 6)}

    for test in tests:
        gate_id = _infer_gate_id(test)
        if gate_id and gate_id in gate_tests:
            gate_tests[gate_id].append(test)

    return gate_tests


def _infer_gate_id(test: dict[str, Any]) -> str | None:
    classname = test.get("classname", "").lower()
    name = test.get("name", "").lower()
    haystack = f"{classname} {name}"

    if "g1" in haystack or "determinism" in haystack or "resolve" in haystack:
        return "G1"
    if "g2" in haystack or "constraint" in haystack or "repair" in haystack:
        return "G2"
    if "g3" in haystack or "drc" in haystack:
        return "G3"
    if "g4" in haystack or "export" in haystack or "completeness" in haystack:
        return "G4"
    if "g5" in haystack or "hash" in haystack or "stability" in haystack:
        return "G5"
    return None


def compute_gate_status(tests: list[dict[str, Any]]) -> Literal["passed", "failed", "skipped", "no_tests"]:
    """Compute overall status for a gate based on its tests.

    Args:
        tests: List of test results for a gate

    Returns:
        Gate status: "passed", "failed", "skipped", or "no_tests"
    """
    if not tests:
        return "no_tests"

    statuses = [t.get("status", "unknown") for t in tests]

    if any(s in ("failed", "error") for s in statuses):
        return "failed"
    if all(s == "skipped" for s in statuses):
        return "skipped"
    return "passed"


def build_audit_report(
    gates_requested: list[str],
    junit_results: dict[str, Any],
    return_code: int,
    junit_xml_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build machine-readable audit report.

    Args:
        gates_requested: List of gate IDs that were requested
        junit_results: Parsed JUnit XML results
        return_code: pytest return code
        junit_xml_path: Path to JUnit XML (for reference)

    Returns:
        Audit report dictionary
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    tests = junit_results.get("tests", [])
    summary = junit_results.get("summary", {})
    gate_tests = categorize_tests_by_gate(tests)

    # Build gate results
    gate_results: dict[str, dict[str, Any]] = {}
    for gate_id in gates_requested:
        tests_for_gate = gate_tests.get(gate_id, [])
        gate_status = compute_gate_status(tests_for_gate)
        gate_results[gate_id] = {
            "description": GATE_DESCRIPTIONS.get(gate_id, ""),
            "marker": GATES.get(gate_id, ""),
            "status": gate_status,
            "tests_count": len(tests_for_gate),
            "passed": sum(1 for t in tests_for_gate if t.get("status") == "passed"),
            "failed": sum(1 for t in tests_for_gate if t.get("status") in ("failed", "error")),
            "skipped": sum(1 for t in tests_for_gate if t.get("status") == "skipped"),
            "tests": tests_for_gate,
        }

    # Compute overall status
    all_passed = all(g["status"] == "passed" for g in gate_results.values())
    any_failed = any(g["status"] == "failed" for g in gate_results.values())
    overall_status: Literal["passed", "failed", "partial"] = "passed" if all_passed else "failed" if any_failed else "partial"

    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "timestamp": timestamp,
        "gates_requested": gates_requested,
        "overall_status": overall_status,
        "pytest_returncode": return_code,
        "summary": {
            "total_tests": summary.get("total", 0),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "skipped": summary.get("skipped", 0),
            "errors": summary.get("errors", 0),
            "gates_passed": sum(1 for g in gate_results.values() if g["status"] == "passed"),
            "gates_failed": sum(1 for g in gate_results.values() if g["status"] == "failed"),
            "gates_skipped": sum(1 for g in gate_results.values() if g["status"] in ("skipped", "no_tests")),
        },
        "gates": gate_results,
    }

    if junit_xml_path:
        report["junit_xml_path"] = str(junit_xml_path)

    if tests:
        report["per_spec"] = _build_per_spec_results(tests, repo_root)
    else:
        report["per_spec"] = {}

    return report


def _build_per_spec_results(tests: list[dict[str, Any]], repo_root: Path | None) -> dict[str, Any]:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]

    per_spec: dict[str, dict[str, Any]] = {}
    for test in tests:
        spec_id = _extract_spec_id(test)
        if not spec_id:
            continue

        if spec_id not in per_spec:
            per_spec[spec_id] = _init_spec_entry(spec_id, repo_root)

        entry = per_spec[spec_id]
        status = test.get("status", "unknown")

        entry["summary"]["total"] += 1
        if status == "passed":
            entry["summary"]["passed"] += 1
        elif status == "skipped":
            entry["summary"]["skipped"] += 1
        elif status in ("failed", "error"):
            entry["summary"]["failed"] += 1
        else:
            entry["summary"]["errors"] += 1

        gate_id = _infer_gate_id(test)
        if gate_id:
            gate_entry = entry["gates"].setdefault(
                gate_id,
                {
                    "tests": [],
                    "tests_count": 0,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                },
            )
            gate_entry["tests"].append(_minimal_test_ref(test, gate_id))
            gate_entry["tests_count"] += 1
            if status == "passed":
                gate_entry["passed"] += 1
            elif status == "skipped":
                gate_entry["skipped"] += 1
            elif status in ("failed", "error"):
                gate_entry["failed"] += 1

        entry["tests"].append(_minimal_test_ref(test, gate_id))

    for _spec_id, entry in per_spec.items():
        statuses = [t["status"] for t in entry["tests"]]
        if not statuses:
            entry["status"] = "no_tests"
        elif any(status in ("failed", "error") for status in statuses):
            entry["status"] = "failed"
        elif all(status == "skipped" for status in statuses):
            entry["status"] = "skipped"
        elif any(status == "skipped" for status in statuses):
            entry["status"] = "partial"
        else:
            entry["status"] = "passed"

        for _gate_id, gate_entry in entry["gates"].items():
            gate_entry["status"] = compute_gate_status(gate_entry["tests"])

    return per_spec


def _minimal_test_ref(test: dict[str, Any], gate_id: str | None) -> dict[str, Any]:
    return {
        "name": test.get("name", ""),
        "classname": test.get("classname", ""),
        "status": test.get("status", "unknown"),
        "gate": gate_id or "",
    }


def _extract_spec_id(test: dict[str, Any]) -> str | None:
    name = test.get("name", "")
    classname = test.get("classname", "")

    param_id = _extract_param_id(name)
    for candidate in (param_id, name, classname):
        spec_id = _extract_spec_token(candidate)
        if spec_id:
            return spec_id
    return None


def _extract_param_id(name: str) -> str | None:
    if "[" in name and name.endswith("]"):
        return name[name.rfind("[") + 1 : -1]
    return None


def _extract_spec_token(candidate: str | None) -> str | None:
    if not candidate:
        return None
    match = SPEC_TOKEN_RE.search(candidate)
    if match:
        return _normalize_spec_id(match.group(1))
    return None


def _normalize_spec_id(candidate: str) -> str | None:
    candidate = candidate.strip()
    if not candidate:
        return None
    for ext in (".yaml", ".yml", ".json"):
        if ext in candidate:
            candidate = candidate.split(ext, 1)[0] + ext
            break
    candidate = candidate.split("/")[-1].split("\\")[-1]
    if candidate.endswith((".yaml", ".yml", ".json")):
        candidate = Path(candidate).stem
    return candidate or None


def _init_spec_entry(spec_id: str, repo_root: Path) -> dict[str, Any]:
    metadata = _resolve_spec_metadata(spec_id, repo_root)
    return {
        "status": "no_tests",
        "summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0},
        "tests": [],
        "gates": {},
        **metadata,
    }


def _resolve_spec_metadata(spec_id: str, repo_root: Path) -> dict[str, Any]:
    artifact_paths: dict[str, str] = {}
    spec_path = _find_spec_path(spec_id, repo_root)
    if spec_path:
        artifact_paths["spec"] = str(spec_path)

    metadata: dict[str, Any] = {"artifact_paths": artifact_paths}
    if spec_path:
        try:
            from formula_foundry.coupongen import coupon_id_from_design_hash, design_hash, load_spec, resolve_spec

            spec = load_spec(spec_path)
            resolved = resolve_spec(spec)
            design_hash_value = design_hash(resolved)
            metadata["design_hash"] = design_hash_value
            metadata["coupon_id"] = coupon_id_from_design_hash(design_hash_value)
        except Exception as exc:  # pragma: no cover - best-effort metadata
            metadata["design_hash_error"] = str(exc)

    return metadata


def _find_spec_path(spec_id: str, repo_root: Path) -> Path | None:
    golden_dir = repo_root / "tests" / "golden_specs"
    direct = golden_dir / spec_id
    if direct.exists():
        return direct
    for ext in (".yaml", ".yml", ".json"):
        candidate = golden_dir / f"{spec_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def print_summary(report: dict[str, Any], verbose: bool = False) -> None:
    """Print human-readable summary of audit report.

    Args:
        report: Audit report dictionary
        verbose: Include detailed test results
    """
    summary = report.get("summary", {})
    gates = report.get("gates", {})

    status_emoji = {
        "passed": "✓",
        "failed": "✗",
        "skipped": "○",
        "no_tests": "?",
    }

    print("\n" + "=" * 60)
    print("M1 AUDIT REPORT")
    print("=" * 60)
    print(f"Timestamp: {report.get('timestamp', 'N/A')}")
    print(f"Overall Status: {report.get('overall_status', 'unknown').upper()}")
    print()

    print("Gate Results:")
    print("-" * 60)
    for gate_id in sorted(gates.keys()):
        gate = gates[gate_id]
        status = gate.get("status", "unknown")
        emoji = status_emoji.get(status, "?")
        desc = gate.get("description", "")
        passed = gate.get("passed", 0)
        total = gate.get("tests_count", 0)
        print(f"  {emoji} {gate_id}: {desc}")
        print(f"       Status: {status.upper()} ({passed}/{total} tests passed)")

        if verbose and gate.get("tests"):
            for test in gate["tests"]:
                test_status = test.get("status", "unknown")
                test_emoji = status_emoji.get(test_status, "?")
                print(f"         {test_emoji} {test.get('name', 'unknown')}")
                if test_status == "failed" and test.get("message"):
                    print(f"            {test.get('message', '')[:80]}")

    print()
    print("Summary:")
    print("-" * 60)
    print(f"  Total Tests: {summary.get('total_tests', 0)}")
    print(f"  Passed: {summary.get('passed', 0)}")
    print(f"  Failed: {summary.get('failed', 0)}")
    print(f"  Skipped: {summary.get('skipped', 0)}")
    print(f"  Gates Passed: {summary.get('gates_passed', 0)}/{len(gates)}")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for audit-m1 CLI."""
    parser = argparse.ArgumentParser(
        prog="audit-m1",
        description="M1 compliance gate runner - execute G1-G5 gate tests and generate audit reports",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Run selected gate tests",
    )
    run_parser.add_argument(
        "--gates",
        type=str,
        default="G1-G5",
        help="Gates to run (comma-separated or range, e.g., 'G1,G2,G3' or 'G1-G5'). Default: G1-G5",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for audit_report.json (default: stdout)",
    )
    run_parser.add_argument(
        "--junit-xml",
        type=Path,
        default=None,
        help="Path for JUnit XML output (default: temp file)",
    )
    run_parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Directory containing gate tests (default: tests/gates/)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    run_parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing human-readable summary",
    )
    run_parser.add_argument(
        "--design-hash",
        type=str,
        default=None,
        help="Design hash for artifact directory structure (artifacts/audit_m1/<design_hash>/)",
    )
    run_parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Override artifacts root directory (default: repo_root/artifacts)",
    )

    # list subcommand
    subparsers.add_parser(
        "list",
        help="List available gates",
    )

    # summary subcommand
    summary_parser = subparsers.add_parser(
        "summary",
        help="Print summary from existing audit report",
    )
    summary_parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to audit_report.json",
    )
    summary_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Include detailed test results",
    )

    return parser


def cmd_list() -> int:
    """Handle 'list' subcommand."""
    print("\nAvailable M1 Compliance Gates:")
    print("-" * 60)
    for gate_id in sorted(GATES.keys()):
        marker = GATES[gate_id]
        desc = GATE_DESCRIPTIONS[gate_id]
        print(f"  {gate_id}: {desc}")
        print(f"       pytest marker: {marker}")
    print()
    return 0


def compute_artifact_path(design_hash: str, artifact_root: Path | None = None) -> Path:
    """Compute the artifact path for a given design hash.

    Per ECO-M1-ALIGN-0001, audit reports are stored at:
        artifacts/audit_m1/<design_hash>/audit_report.json

    Args:
        design_hash: The design hash to use as directory name
        artifact_root: Override the artifacts root directory (default: repo_root/artifacts)

    Returns:
        Path to the audit_report.json file
    """
    if artifact_root is None:
        repo_root = Path(__file__).resolve().parents[4]
        artifact_root = repo_root / "artifacts"

    return artifact_root / "audit_m1" / design_hash / "audit_report.json"


def cmd_run(args: argparse.Namespace) -> int:
    """Handle 'run' subcommand."""
    # Parse gates
    try:
        gates = parse_gates(args.gates)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1

    if not gates:
        sys.stderr.write("Error: No gates specified\n")
        return 1

    if args.verbose:
        print(f"Running gates: {', '.join(gates)}")
        print(f"Marker expression: {build_pytest_marker_expr(gates)}")

    # Run tests
    return_code, junit_xml = run_gates(
        gates=gates,
        test_dir=args.test_dir,
        junit_xml=args.junit_xml,
        verbose=args.verbose,
    )

    # Parse results
    junit_results = {}
    if junit_xml and junit_xml.exists():
        junit_results = parse_junit_xml(junit_xml)

    # Build report
    report = build_audit_report(
        gates_requested=gates,
        junit_results=junit_results,
        return_code=return_code,
        junit_xml_path=junit_xml,
    )

    # Add design_hash to report if provided
    design_hash = getattr(args, "design_hash", None)
    if design_hash:
        report["design_hash"] = design_hash

    # Determine output path
    output_path = args.output
    if output_path is None and design_hash:
        # Use artifact path structure when design_hash is provided
        output_path = compute_artifact_path(design_hash, args.artifact_root)

    # Output report
    report_json = json.dumps(report, indent=2, sort_keys=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_json, encoding="utf-8")
        if args.verbose:
            print(f"Audit report written to: {output_path}")

        # Also write JUnit XML to artifact directory if using artifact structure
        if design_hash and junit_xml and junit_xml.exists():
            artifact_junit_path = output_path.parent / "junit.xml"
            artifact_junit_path.write_bytes(junit_xml.read_bytes())
            report["artifact_paths"] = {
                "audit_report": str(output_path),
                "junit_xml": str(artifact_junit_path),
            }
            # Re-write report with artifact paths
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(report_json)

    # Print summary unless disabled
    if not args.no_summary and output_path:
        print_summary(report, verbose=args.verbose)

    # Return non-zero if any gates failed
    if report.get("overall_status") == "failed":
        return 1
    return return_code


def cmd_summary(args: argparse.Namespace) -> int:
    """Handle 'summary' subcommand."""
    if not args.input.exists():
        sys.stderr.write(f"Error: Audit report not found: {args.input}\n")
        return 1

    try:
        report = json.loads(args.input.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Error: Invalid JSON in audit report: {e}\n")
        return 1

    print_summary(report, verbose=args.verbose)

    if report.get("overall_status") == "failed":
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for audit-m1 CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return cmd_list()
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "summary":
        return cmd_summary(args)
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
