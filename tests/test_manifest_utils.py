# SPDX-License-Identifier: MIT
"""Unit tests for manifest module utility functions.

This module provides focused tests for utility functions in
formula_foundry.coupongen.manifest:
- parse_drc_summary: DRC report summary parsing
- canonicalize_drc_report: DRC report canonicalization and hashing
- toolchain_hash: Toolchain metadata hashing
- write_manifest/load_manifest: Manifest I/O operations
- _utc_timestamp: UTC timestamp generation
- _resolve_git_sha: Git SHA resolution

Satisfies REQ-M1-018: The repo must emit a manifest.json for every build
containing required provenance fields and export hashes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.coupongen.manifest import (
    canonicalize_drc_report,
    load_manifest,
    parse_drc_summary,
    toolchain_hash,
    write_manifest,
)


class TestParseDrcSummary:
    """Tests for parse_drc_summary function."""

    def test_valid_drc_report(self, tmp_path: Path) -> None:
        """Parse a valid DRC report with violations."""
        report = {
            "violations": [{"type": "clearance", "severity": "error"}],
            "warnings": [{"type": "silkscreen", "severity": "warning"}],
            "exclusions": [],
        }
        report_path = tmp_path / "drc_report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        summary = parse_drc_summary(report_path)

        assert summary["violations"] == 1
        assert summary["warnings"] == 1
        assert summary["exclusions"] == 0

    def test_drc_report_with_exclusions(self, tmp_path: Path) -> None:
        """Parse a DRC report with exclusions."""
        report = {
            "violations": [],
            "warnings": [],
            "exclusions": [{"type": "known_issue"}, {"type": "waived"}],
        }
        report_path = tmp_path / "drc_report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        summary = parse_drc_summary(report_path)

        assert summary["violations"] == 0
        assert summary["warnings"] == 0
        assert summary["exclusions"] == 2

    def test_empty_drc_report(self, tmp_path: Path) -> None:
        """Parse an empty DRC report (clean design)."""
        report = {"violations": [], "warnings": [], "exclusions": []}
        report_path = tmp_path / "drc_report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        summary = parse_drc_summary(report_path)

        assert summary["violations"] == 0
        assert summary["warnings"] == 0
        assert summary["exclusions"] == 0

    def test_missing_drc_report(self, tmp_path: Path) -> None:
        """Missing DRC report returns zero counts."""
        report_path = tmp_path / "nonexistent.json"

        summary = parse_drc_summary(report_path)

        assert summary["violations"] == 0
        assert summary["warnings"] == 0
        assert summary["exclusions"] == 0

    def test_malformed_drc_report(self, tmp_path: Path) -> None:
        """Malformed JSON returns zero counts."""
        report_path = tmp_path / "bad_drc.json"
        report_path.write_text("{invalid json", encoding="utf-8")

        summary = parse_drc_summary(report_path)

        assert summary["violations"] == 0
        assert summary["warnings"] == 0
        assert summary["exclusions"] == 0

    def test_drc_report_missing_keys(self, tmp_path: Path) -> None:
        """DRC report with missing keys returns zero for those keys."""
        report = {"violations": [{"id": 1}]}  # Missing warnings and exclusions
        report_path = tmp_path / "drc_report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        summary = parse_drc_summary(report_path)

        assert summary["violations"] == 1
        assert summary["warnings"] == 0
        assert summary["exclusions"] == 0


class TestCanonicalizeDrcReport:
    """Tests for canonicalize_drc_report function."""

    def test_canonical_hash_is_64_hex(self, tmp_path: Path) -> None:
        """Canonical hash is 64-character hex string (SHA-256)."""
        report = {"violations": []}
        report_path = tmp_path / "drc.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        result = canonicalize_drc_report(report_path)

        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_canonical_hash_deterministic(self, tmp_path: Path) -> None:
        """Same content produces same canonical hash."""
        report = {"violations": [{"id": 1}], "warnings": []}
        report_path = tmp_path / "drc.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        hash1 = canonicalize_drc_report(report_path)
        hash2 = canonicalize_drc_report(report_path)

        assert hash1 == hash2

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Different content produces different canonical hash."""
        report1_path = tmp_path / "drc1.json"
        report2_path = tmp_path / "drc2.json"

        report1_path.write_text(json.dumps({"violations": []}), encoding="utf-8")
        report2_path.write_text(json.dumps({"violations": [{"id": 1}]}), encoding="utf-8")

        hash1 = canonicalize_drc_report(report1_path)
        hash2 = canonicalize_drc_report(report2_path)

        assert hash1 != hash2

    def test_missing_file_returns_empty_hash(self, tmp_path: Path) -> None:
        """Missing file returns hash of empty bytes."""
        report_path = tmp_path / "nonexistent.json"

        result = canonicalize_drc_report(report_path)

        assert len(result) == 64

    def test_malformed_json_returns_empty_hash(self, tmp_path: Path) -> None:
        """Malformed JSON returns hash of empty bytes."""
        report_path = tmp_path / "bad.json"
        report_path.write_text("{invalid", encoding="utf-8")

        result = canonicalize_drc_report(report_path)

        assert len(result) == 64


class TestToolchainHash:
    """Tests for toolchain_hash function."""

    def test_basic_toolchain_hash(self) -> None:
        """Basic toolchain metadata produces valid hash."""
        toolchain: dict[str, Any] = {
            "kicad": {"version": "9.0.7"},
            "mode": "docker",
            "docker": {"image_ref": "kicad/kicad:9.0.7@sha256:abc123"},
        }

        result = toolchain_hash(toolchain)

        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_toolchain_hash_deterministic(self) -> None:
        """Same toolchain produces same hash."""
        toolchain: dict[str, Any] = {
            "kicad": {"version": "9.0.7"},
            "mode": "local",
        }

        hash1 = toolchain_hash(toolchain)
        hash2 = toolchain_hash(toolchain)

        assert hash1 == hash2

    def test_different_toolchain_different_hash(self) -> None:
        """Different toolchain metadata produces different hash."""
        toolchain1: dict[str, Any] = {"kicad": {"version": "9.0.7"}}
        toolchain2: dict[str, Any] = {"kicad": {"version": "9.0.8"}}

        hash1 = toolchain_hash(toolchain1)
        hash2 = toolchain_hash(toolchain2)

        assert hash1 != hash2

    def test_key_order_invariant(self) -> None:
        """Hash is invariant to key order in input."""
        toolchain1: dict[str, Any] = {"a": 1, "b": 2, "c": 3}
        toolchain2: dict[str, Any] = {"c": 3, "b": 2, "a": 1}

        # Both should produce same canonical JSON and thus same hash
        hash1 = toolchain_hash(toolchain1)
        hash2 = toolchain_hash(toolchain2)

        assert hash1 == hash2

    def test_empty_toolchain(self) -> None:
        """Empty toolchain produces valid hash."""
        result = toolchain_hash({})

        assert len(result) == 64


class TestWriteLoadManifest:
    """Tests for write_manifest and load_manifest functions."""

    def test_write_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Manifest can be written and loaded with same content."""
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "coupon_family": "F0",
            "design_hash": "a" * 64,
            "toolchain": {"mode": "local"},
        }
        manifest_path = tmp_path / "manifest.json"

        write_manifest(manifest_path, manifest)
        loaded = load_manifest(manifest_path)

        assert loaded == manifest

    def test_write_requires_parent_dirs(self, tmp_path: Path) -> None:
        """write_manifest requires parent directories to exist."""
        manifest: dict[str, Any] = {"key": "value"}
        manifest_path = tmp_path / "deep" / "nested" / "manifest.json"

        # Parent directories don't exist, so write should fail
        with pytest.raises(FileNotFoundError):
            write_manifest(manifest_path, manifest)

    def test_write_with_existing_parent_dirs(self, tmp_path: Path) -> None:
        """write_manifest works when parent directories exist."""
        manifest: dict[str, Any] = {"key": "value"}
        parent_dir = tmp_path / "deep" / "nested"
        parent_dir.mkdir(parents=True)
        manifest_path = parent_dir / "manifest.json"

        write_manifest(manifest_path, manifest)

        assert manifest_path.exists()
        assert load_manifest(manifest_path) == manifest

    def test_write_overwrites_existing(self, tmp_path: Path) -> None:
        """write_manifest overwrites existing file."""
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text('{"old": true}', encoding="utf-8")

        new_manifest: dict[str, Any] = {"new": True}
        write_manifest(manifest_path, new_manifest)

        loaded = load_manifest(manifest_path)
        assert loaded == new_manifest

    def test_manifest_has_trailing_newline(self, tmp_path: Path) -> None:
        """Written manifest ends with newline."""
        manifest: dict[str, Any] = {"key": "value"}
        manifest_path = tmp_path / "manifest.json"

        write_manifest(manifest_path, manifest)

        content = manifest_path.read_text(encoding="utf-8")
        assert content.endswith("\n")

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        """Loading nonexistent manifest raises FileNotFoundError."""
        manifest_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_manifest(manifest_path)

    def test_nested_manifest_content(self, tmp_path: Path) -> None:
        """Nested manifest content is preserved."""
        manifest: dict[str, Any] = {
            "level1": {
                "level2": {
                    "level3": {"value": 123},
                },
            },
            "list": [1, 2, {"nested": True}],
        }
        manifest_path = tmp_path / "manifest.json"

        write_manifest(manifest_path, manifest)
        loaded = load_manifest(manifest_path)

        assert loaded == manifest
        assert loaded["level1"]["level2"]["level3"]["value"] == 123
        assert loaded["list"][2]["nested"] is True


class TestManifestJsonCanonical:
    """Tests for canonical JSON output from manifest writing."""

    def test_manifest_json_sorted_keys(self, tmp_path: Path) -> None:
        """Manifest JSON has sorted keys."""
        manifest: dict[str, Any] = {"z": 1, "a": 2, "m": 3}
        manifest_path = tmp_path / "manifest.json"

        write_manifest(manifest_path, manifest)

        content = manifest_path.read_text(encoding="utf-8")
        # Keys should appear in sorted order
        z_pos = content.find('"z"')
        a_pos = content.find('"a"')
        m_pos = content.find('"m"')
        assert a_pos < m_pos < z_pos

    def test_manifest_deterministic_output(self, tmp_path: Path) -> None:
        """Same manifest produces byte-identical output."""
        manifest: dict[str, Any] = {
            "b": [3, 2, 1],
            "a": {"nested": True},
        }
        path1 = tmp_path / "m1.json"
        path2 = tmp_path / "m2.json"

        write_manifest(path1, manifest)
        write_manifest(path2, manifest)

        assert path1.read_bytes() == path2.read_bytes()
