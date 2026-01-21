# SPDX-License-Identifier: MIT
"""Tests for M1 compliance report generation."""

from __future__ import annotations

import json
from pathlib import Path

from tools import m1_compliance_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_report_generation(tmp_path: Path) -> None:
    verify_dir = tmp_path / "artifacts" / "verify" / "20260101T000000Z"
    verify_dir.mkdir(parents=True)
    _write_json(
        verify_dir / "results.json",
        {
            "ok": True,
            "results": [
                {"name": "spec_lint", "passed": True, "note": "rc=0"},
                {"name": "pytest", "passed": True, "note": "rc=0"},
            ],
        },
    )
    _write_json(verify_dir / "env.json", {"run_id": "20260101T000000Z", "env": {"TZ": "UTC"}})

    audit_report_path = tmp_path / "artifacts" / "audit" / "audit_report.json"
    _write_json(
        audit_report_path,
        {
            "schema_version": "1.0.0",
            "timestamp": "2026-01-01T00:00:00Z",
            "gates_requested": ["G1", "G2"],
            "overall_status": "passed",
            "pytest_returncode": 0,
            "summary": {"total_tests": 2, "passed": 2, "failed": 0, "skipped": 0, "errors": 0},
            "gates": {
                "G1": {
                    "description": "Resolved design determinism",
                    "status": "passed",
                    "tests_count": 1,
                    "passed": 1,
                    "failed": 0,
                    "skipped": 0,
                },
                "G2": {
                    "description": "Constraint proof completeness",
                    "status": "passed",
                    "tests_count": 1,
                    "passed": 1,
                    "failed": 0,
                    "skipped": 0,
                },
            },
        },
    )

    manifest_path = tmp_path / "artifacts" / "sample_run" / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "1.0.0",
            "coupon_family": "F0",
            "design_hash": "abc123",
            "toolchain_hash": "def456",
            "toolchain": {
                "kicad": {"version": "9.0.7", "cli_version_output": "9.0.7"},
                "docker": {"image_ref": "kicad/kicad:9.0.7@sha256:deadbeef"},
                "mode": "docker",
                "generator_git_sha": "1234567890abcdef",
                "lock_file_toolchain_hash": "lockhash",
            },
            "exports": [
                {"path": "F0/gerbers/file1.gbr", "hash": "hash1"},
                {"path": "F0/gerbers/file2.gbr", "hash": "hash2"},
            ],
            "verification": {
                "drc": {
                    "returncode": 0,
                    "report_path": "drc.json",
                    "summary": {"violations": 0},
                    "canonical_hash": "hash",
                }
            },
        },
    )

    drc_path = tmp_path / "artifacts" / "sample_run" / "drc.json"
    _write_json(
        drc_path,
        {
            "source": "board.kicad_pcb",
            "violations": [],
            "warnings": [{"id": "W1"}],
            "exclusions": [],
        },
    )

    lock_path = tmp_path / "toolchain" / "kicad.lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": "1.0",
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:deadbeef",
            "toolchain_hash": "lockhash",
        },
    )

    output_path = tmp_path / "artifacts" / "m1_compliance_report.md"
    rc = m1_compliance_report.main(
        [
            "--project-root",
            str(tmp_path),
            "--verify-dir",
            str(verify_dir),
            "--audit-report",
            str(audit_report_path),
            "--manifest",
            str(manifest_path),
            "--drc-report",
            str(drc_path),
            "--toolchain-lock",
            str(lock_path),
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")

    required_sections = [
        "# M1 Compliance Report",
        "## Inputs",
        "## Verify Summary",
        "## M1 Gate Results (audit-m1)",
        "## Toolchain Versions",
        "## Sample DRC Report Excerpt",
        "## Sample Manifest Excerpt",
    ]
    for section in required_sections:
        assert section in content

    assert "spec_lint" in content
    assert "G1" in content
    assert "- manifest: artifacts/sample_run/manifest.json" in content
    assert "- drc_report: artifacts/sample_run/drc.json" in content
    assert "kicad/kicad:9.0.7@sha256:deadbeef" in content
    assert "lock.kicad_version: 9.0.7" in content
    assert "lock.docker_digest: sha256:deadbeef" in content
    assert "design_hash" in content
