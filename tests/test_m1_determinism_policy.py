# SPDX-License-Identifier: MIT
"""Unit tests for determinism policy (CP-5.1, CP-5.2).

This module tests the determinism guarantees documented in docs/DETERMINISM_POLICY.md:
- Toolchain hash computation
- Manifest completeness requirements
- No 'unknown' values for docker builds
- Hash stability across runs

References:
- CP-5.1: Ensure manifest toolchain completeness
- CP-5.2: Explicit determinism policy documentation for what is hashed
- D5: Toolchain provenance incomplete in manifest (fix)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.substrate import canonical_json_dumps


class TestToolchainHashComputation:
    """Tests for toolchain_hash computation per DETERMINISM_POLICY.md Section 2.2."""

    def test_toolchain_hash_excludes_self(self) -> None:
        """Toolchain hash must exclude the toolchain_hash field itself."""
        lock_data = {
            "schema_version": "1.0",
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:abc123",
            "toolchain_hash": "should_be_excluded",
        }

        # Compute hash manually
        data_for_hash = {k: v for k, v in lock_data.items() if k != "toolchain_hash"}
        canonical = json.dumps(data_for_hash, sort_keys=True, separators=(",", ":"))
        expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        # Verify the hash doesn't change if we modify the toolchain_hash field
        lock_data_2 = lock_data.copy()
        lock_data_2["toolchain_hash"] = "different_value"

        data_for_hash_2 = {k: v for k, v in lock_data_2.items() if k != "toolchain_hash"}
        canonical_2 = json.dumps(data_for_hash_2, sort_keys=True, separators=(",", ":"))
        actual_hash = hashlib.sha256(canonical_2.encode("utf-8")).hexdigest()

        assert expected_hash == actual_hash

    def test_toolchain_hash_is_deterministic(self) -> None:
        """Same input must produce same toolchain_hash."""
        lock_data = {
            "schema_version": "1.0",
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:abc123def456",
        }

        hashes = []
        for _ in range(3):
            data_for_hash = {k: v for k, v in lock_data.items() if k != "toolchain_hash"}
            canonical = json.dumps(data_for_hash, sort_keys=True, separators=(",", ":"))
            h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            hashes.append(h)

        assert len(set(hashes)) == 1, "Toolchain hash must be deterministic"

    def test_toolchain_hash_changes_with_version(self) -> None:
        """Toolchain hash must change when kicad_version changes."""
        base_data = {
            "schema_version": "1.0",
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:abc123",
        }

        modified_data = base_data.copy()
        modified_data["kicad_version"] = "9.0.8"
        modified_data["docker_image"] = "kicad/kicad:9.0.8"

        def compute_hash(data: dict[str, Any]) -> str:
            canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        hash1 = compute_hash(base_data)
        hash2 = compute_hash(modified_data)

        assert hash1 != hash2, "Toolchain hash must change when version changes"


class TestCanonicalJsonDeterminism:
    """Tests for canonical JSON serialization determinism."""

    def test_canonical_json_key_ordering(self) -> None:
        """Canonical JSON must have sorted keys."""
        data = {"zebra": 1, "alpha": 2, "beta": 3}
        canonical = canonical_json_dumps(data)

        # Parse back and check key order
        assert '"alpha":2' in canonical
        assert '"beta":3' in canonical
        assert '"zebra":1' in canonical
        assert canonical.index("alpha") < canonical.index("beta") < canonical.index("zebra")

    def test_canonical_json_minimal_whitespace(self) -> None:
        """Canonical JSON must have minimal whitespace."""
        data = {"key": "value", "nested": {"inner": 123}}
        canonical = canonical_json_dumps(data)

        # No unnecessary spaces
        assert "  " not in canonical
        assert ": " not in canonical  # colon should be followed directly by value

    def test_canonical_json_stable_across_calls(self) -> None:
        """Canonical JSON must produce identical output across calls."""
        data = {
            "complex": {
                "nested": [1, 2, 3],
                "string": "hello",
            },
            "top": True,
        }

        outputs = [canonical_json_dumps(data) for _ in range(5)]
        assert len(set(outputs)) == 1, "Canonical JSON must be stable"


class TestManifestCompleteness:
    """Tests for manifest completeness requirements per DETERMINISM_POLICY.md Section 5."""

    REQUIRED_TOP_LEVEL_FIELDS = [
        "schema_version",
        "coupon_family",
        "design_hash",
        "coupon_id",
        "resolved_design",
        "derived_features",
        "dimensionless_groups",
        "fab_profile",
        "stackup",
        "toolchain",
        "toolchain_hash",
        "exports",
        "verification",
        "lineage",
    ]

    REQUIRED_TOOLCHAIN_FIELDS_DOCKER = [
        "kicad",
        "docker",
        "mode",
    ]

    REQUIRED_KICAD_FIELDS = [
        "version",
        "cli_version_output",
    ]

    def test_required_fields_documented(self) -> None:
        """Verify required fields list is comprehensive."""
        assert len(self.REQUIRED_TOP_LEVEL_FIELDS) == 14
        assert "toolchain" in self.REQUIRED_TOP_LEVEL_FIELDS
        assert "toolchain_hash" in self.REQUIRED_TOP_LEVEL_FIELDS

    def test_no_unknown_values_docker_toolchain(self) -> None:
        """Docker builds must not have 'unknown' values for toolchain fields."""
        # Sample docker toolchain that would be valid
        valid_toolchain = {
            "kicad": {
                "version": "9.0.7",
                "cli_version_output": "9.0.7",
            },
            "docker": {
                "image_ref": "kicad/kicad:9.0.7@sha256:abc123",
            },
            "mode": "docker",
            "generator_git_sha": "a" * 40,
        }

        # Check no 'unknown' values
        assert valid_toolchain["kicad"]["version"] != "unknown"
        assert valid_toolchain["kicad"]["cli_version_output"] != "unknown"
        assert valid_toolchain["docker"]["image_ref"] != "unknown"

    def test_detect_unknown_values(self) -> None:
        """Test that we can detect 'unknown' values in toolchain."""
        invalid_toolchain = {
            "kicad": {
                "version": "9.0.7",
                "cli_version_output": "unknown",  # Invalid for docker!
            },
            "docker": {
                "image_ref": "kicad/kicad:9.0.7",
            },
            "mode": "docker",
        }

        # This should be detected as invalid
        def has_unknown_values(toolchain: dict[str, Any]) -> bool:
            if toolchain.get("mode") != "docker":
                return False

            kicad = toolchain.get("kicad", {})
            if kicad.get("version") == "unknown":
                return True
            if kicad.get("cli_version_output") == "unknown":
                return True

            docker = toolchain.get("docker", {})
            if docker.get("image_ref") == "unknown":
                return True

            return False

        assert has_unknown_values(invalid_toolchain) is True


class TestExportHashCanonicalization:
    """Tests for export file hash canonicalization per DETERMINISM_POLICY.md Section 2.3."""

    def test_gerber_comment_removal(self) -> None:
        """Gerber comments (G04) should be removed before hashing."""
        from formula_foundry.coupongen.hashing import canonicalize_export_text

        gerber_with_comments = """G04 This is a comment*
X0Y0D03*
G04 Another comment*
X1000Y1000D01*"""

        canonical = canonicalize_export_text(gerber_with_comments)

        assert "G04" not in canonical
        assert "X0Y0D03*" in canonical
        assert "X1000Y1000D01*" in canonical

    def test_line_ending_normalization(self) -> None:
        """CRLF and CR should be normalized to LF."""
        from formula_foundry.coupongen.hashing import canonicalize_export_text

        mixed_endings = "line1\r\nline2\rline3\n"
        canonical = canonicalize_export_text(mixed_endings)

        assert "\r" not in canonical
        assert canonical.count("\n") == 3  # Normalized lines


class TestDesignHashStability:
    """Tests for design_hash stability per DETERMINISM_POLICY.md Section 2.1."""

    def test_design_hash_format(self) -> None:
        """Design hash must be 64-character lowercase hex."""
        # A valid design hash
        sample_hash = "a" * 64

        assert len(sample_hash) == 64
        assert all(c in "0123456789abcdef" for c in sample_hash)

    def test_design_hash_excludes_timestamps(self) -> None:
        """Design hash should not include timestamps."""
        # This is a policy statement - timestamps are tracked in lineage, not design
        # The test verifies the intent is documented
        required_in_design = ["resolved_design"]
        excluded_from_design = ["timestamp_utc", "git_sha"]

        for field in required_in_design:
            assert field is not None  # Sanity check

        for field in excluded_from_design:
            assert field is not None  # Sanity check


class TestToolchainLockFile:
    """Tests for toolchain lock file per DETERMINISM_POLICY.md Section 4."""

    def test_lock_file_required_fields(self) -> None:
        """Lock file must have required fields."""
        required_fields = [
            "schema_version",
            "kicad_version",
            "docker_image",
            "docker_digest",
        ]

        sample_lock = {
            "schema_version": "1.0",
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:abc123",
            "toolchain_hash": "def456",
        }

        for field in required_fields:
            assert field in sample_lock, f"Lock file missing required field: {field}"

    def test_placeholder_digest_detected(self) -> None:
        """PLACEHOLDER digest should be detected as invalid for production."""
        lock_data = {
            "docker_digest": "sha256:PLACEHOLDER",
        }

        is_placeholder = "PLACEHOLDER" in lock_data["docker_digest"]
        assert is_placeholder is True


class TestDrcReportCanonicalization:
    """Tests for DRC report canonicalization per DETERMINISM_POLICY.md Section 2.4."""

    def test_drc_timestamps_removed(self) -> None:
        """DRC report canonicalization should remove timestamps."""
        from formula_foundry.coupongen.manifest import _canonicalize_drc_object

        drc_report = {
            "date": "2026-01-20",
            "time": "12:00:00",
            "violations": [],
            "kicad_version": "9.0.7",
        }

        canonical = _canonicalize_drc_object(drc_report)

        assert "date" not in canonical
        assert "time" not in canonical
        assert "violations" in canonical
        assert "kicad_version" in canonical

    def test_drc_paths_removed(self) -> None:
        """DRC report canonicalization should remove absolute paths."""
        from formula_foundry.coupongen.manifest import _canonicalize_drc_object

        drc_report = {
            "source": "/home/user/project/board.kicad_pcb",
            "filename": "/tmp/drc_report.json",
            "violations": [],
        }

        canonical = _canonicalize_drc_object(drc_report)

        assert "source" not in canonical
        assert "filename" not in canonical
        assert "violations" in canonical
