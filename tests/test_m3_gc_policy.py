# SPDX-License-Identifier: MIT
"""Unit tests for M3 garbage collection retention policies.

Tests the retention policy dataclasses and helper functions for garbage
collection. These are lightweight unit tests that don't require the full
artifact store infrastructure. Key functionality:
- RetentionPolicy creation and serialization
- PinnedArtifact creation and serialization
- GCCandidate evaluation logic
- GCResult serialization
- Built-in policy definitions
- Helper functions (age calculation, byte formatting)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from formula_foundry.m3.gc import (
    BUILTIN_POLICIES,
    DEFAULT_POLICY,
    GCCandidate,
    GCError,
    GCResult,
    PinnedArtifact,
    PolicyNotFoundError,
    RetentionPolicy,
    RetentionUnit,
    _get_age_days,
    _now_utc_iso,
    _parse_iso_datetime,
    format_bytes,
    load_pins_from_file,
    save_pins_to_file,
)


class TestRetentionPolicy:
    """Tests for RetentionPolicy dataclass."""

    def test_default_values(self) -> None:
        """RetentionPolicy has sensible defaults."""
        policy = RetentionPolicy(name="test")

        assert policy.name == "test"
        assert policy.keep_min_age_days == 30
        assert policy.keep_min_count == 1
        assert policy.keep_pinned is True
        assert policy.keep_with_descendants is True
        assert policy.keep_ancestors_of_pinned is True
        assert policy.keep_max_count is None
        assert policy.space_budget_bytes is None
        assert policy.keep_artifact_types == []
        assert policy.keep_roles == []
        assert policy.dvc_gc_flags == []

    def test_custom_values(self) -> None:
        """RetentionPolicy accepts custom values."""
        policy = RetentionPolicy(
            name="custom",
            description="A custom policy",
            keep_min_age_days=7,
            keep_max_count=100,
            keep_min_count=5,
            keep_pinned=False,
            keep_with_descendants=False,
            keep_ancestors_of_pinned=False,
            keep_artifact_types=["model", "dataset"],
            keep_roles=["final_output"],
            space_budget_bytes=1024 * 1024 * 1024,
            dvc_gc_flags=["--cloud"],
        )

        assert policy.name == "custom"
        assert policy.description == "A custom policy"
        assert policy.keep_min_age_days == 7
        assert policy.keep_max_count == 100
        assert policy.keep_min_count == 5
        assert policy.keep_pinned is False
        assert policy.keep_with_descendants is False
        assert policy.keep_ancestors_of_pinned is False
        assert policy.keep_artifact_types == ["model", "dataset"]
        assert policy.keep_roles == ["final_output"]
        assert policy.space_budget_bytes == 1024 * 1024 * 1024
        assert policy.dvc_gc_flags == ["--cloud"]

    def test_to_dict(self) -> None:
        """to_dict produces JSON-serializable dict."""
        policy = RetentionPolicy(
            name="test",
            description="Test policy",
            keep_min_age_days=14,
            keep_max_count=50,
            keep_artifact_types=["dataset"],
            space_budget_bytes=10_000_000,
        )

        data = policy.to_dict()

        assert data["name"] == "test"
        assert data["description"] == "Test policy"
        assert data["keep_min_age_days"] == 14
        assert data["keep_max_count"] == 50
        assert data["keep_artifact_types"] == ["dataset"]
        assert data["space_budget_bytes"] == 10_000_000

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert "test" in json_str

    def test_to_dict_omits_none_and_empty(self) -> None:
        """to_dict omits None values and empty lists."""
        policy = RetentionPolicy(name="minimal")

        data = policy.to_dict()

        assert "description" not in data
        assert "keep_max_count" not in data
        assert "space_budget_bytes" not in data
        assert "keep_artifact_types" not in data
        assert "keep_roles" not in data
        assert "dvc_gc_flags" not in data

    def test_from_dict(self) -> None:
        """from_dict creates policy from dict."""
        data = {
            "name": "from_dict_test",
            "description": "Created from dict",
            "keep_min_age_days": 21,
            "keep_max_count": 25,
            "keep_min_count": 3,
            "keep_pinned": False,
            "keep_artifact_types": ["checkpoint"],
            "space_budget_bytes": 5_000_000_000,
        }

        policy = RetentionPolicy.from_dict(data)

        assert policy.name == "from_dict_test"
        assert policy.description == "Created from dict"
        assert policy.keep_min_age_days == 21
        assert policy.keep_max_count == 25
        assert policy.keep_min_count == 3
        assert policy.keep_pinned is False
        assert policy.keep_artifact_types == ["checkpoint"]
        assert policy.space_budget_bytes == 5_000_000_000

    def test_from_dict_uses_defaults(self) -> None:
        """from_dict uses defaults for missing keys."""
        data = {"name": "minimal"}

        policy = RetentionPolicy.from_dict(data)

        assert policy.name == "minimal"
        assert policy.keep_min_age_days == 30  # Default
        assert policy.keep_min_count == 1  # Default
        assert policy.keep_pinned is True  # Default

    def test_roundtrip_to_from_dict(self) -> None:
        """to_dict and from_dict are inverse operations."""
        original = RetentionPolicy(
            name="roundtrip",
            description="Roundtrip test",
            keep_min_age_days=10,
            keep_max_count=20,
            keep_min_count=2,
            keep_artifact_types=["a", "b"],
            keep_roles=["x", "y"],
            space_budget_bytes=1_000_000,
            dvc_gc_flags=["--flag1"],
        )

        data = original.to_dict()
        restored = RetentionPolicy.from_dict(data)

        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.keep_min_age_days == original.keep_min_age_days
        assert restored.keep_max_count == original.keep_max_count
        assert restored.keep_artifact_types == original.keep_artifact_types


class TestPinnedArtifact:
    """Tests for PinnedArtifact dataclass."""

    def test_pin_by_artifact_id(self) -> None:
        """Pin by specific artifact ID."""
        pin = PinnedArtifact(artifact_id="abc123", reason="Critical artifact")

        assert pin.artifact_id == "abc123"
        assert pin.run_id is None
        assert pin.dataset_id is None
        assert pin.reason == "Critical artifact"

    def test_pin_by_run_id(self) -> None:
        """Pin all artifacts from a run."""
        pin = PinnedArtifact(run_id="run-001", reason="Production run")

        assert pin.artifact_id is None
        assert pin.run_id == "run-001"
        assert pin.reason == "Production run"

    def test_pin_by_dataset_id(self) -> None:
        """Pin all artifacts in a dataset."""
        pin = PinnedArtifact(dataset_id="dataset-v1", reason="Release dataset")

        assert pin.artifact_id is None
        assert pin.run_id is None
        assert pin.dataset_id == "dataset-v1"

    def test_to_dict(self) -> None:
        """to_dict produces correct output."""
        pin = PinnedArtifact(
            artifact_id="art123",
            reason="Important",
            pinned_utc="2024-01-15T12:00:00Z",
        )

        data = pin.to_dict()

        assert data["artifact_id"] == "art123"
        assert data["reason"] == "Important"
        assert data["pinned_utc"] == "2024-01-15T12:00:00Z"
        assert "run_id" not in data
        assert "dataset_id" not in data

    def test_from_dict(self) -> None:
        """from_dict creates pin from dict."""
        data = {
            "artifact_id": "xyz789",
            "reason": "Test pin",
            "pinned_utc": "2024-02-20T08:30:00Z",
        }

        pin = PinnedArtifact.from_dict(data)

        assert pin.artifact_id == "xyz789"
        assert pin.reason == "Test pin"
        assert pin.pinned_utc == "2024-02-20T08:30:00Z"


class TestGCCandidate:
    """Tests for GCCandidate dataclass."""

    def test_should_delete_when_only_delete_reasons(self) -> None:
        """should_delete is True when only delete reasons exist."""
        candidate = GCCandidate(
            artifact_id="art1",
            content_hash_digest="abc123",
            byte_size=1000,
            created_utc="2024-01-01T00:00:00Z",
            artifact_type="intermediate",
            run_id="run1",
            storage_path="/path/to/art1",
            reasons_to_delete=["age:45.0d >= 30d"],
            reasons_to_keep=[],
        )

        assert candidate.should_delete is True

    def test_should_not_delete_when_keep_reasons_exist(self) -> None:
        """should_delete is False when keep reasons exist."""
        candidate = GCCandidate(
            artifact_id="art2",
            content_hash_digest="def456",
            byte_size=2000,
            created_utc="2024-01-01T00:00:00Z",
            artifact_type="dataset",
            run_id="run2",
            storage_path="/path/to/art2",
            reasons_to_delete=["age:45.0d >= 30d"],
            reasons_to_keep=["pinned"],
        )

        assert candidate.should_delete is False

    def test_should_not_delete_when_no_delete_reasons(self) -> None:
        """should_delete is False when no delete reasons."""
        candidate = GCCandidate(
            artifact_id="art3",
            content_hash_digest="ghi789",
            byte_size=3000,
            created_utc="2024-06-01T00:00:00Z",
            artifact_type="model",
            run_id="run3",
            storage_path="/path/to/art3",
            reasons_to_delete=[],
            reasons_to_keep=["age:5.0d < 30d"],
        )

        assert candidate.should_delete is False


class TestGCResult:
    """Tests for GCResult dataclass."""

    def test_to_dict(self) -> None:
        """to_dict produces complete dictionary."""
        result = GCResult(
            policy_name="test_policy",
            started_utc="2024-01-01T10:00:00Z",
            finished_utc="2024-01-01T10:05:00Z",
            dry_run=True,
            artifacts_scanned=100,
            artifacts_deleted=15,
            bytes_freed=1_500_000,
            bytes_total_before=10_000_000,
            bytes_total_after=8_500_000,
            pinned_protected=5,
            descendant_protected=3,
            ancestor_protected=2,
            dvc_gc_ran=False,
            dvc_gc_output=None,
            errors=[],
            deleted_artifacts=["a1", "a2"],
            protected_artifacts=["p1", "p2"],
        )

        data = result.to_dict()

        assert data["policy_name"] == "test_policy"
        assert data["dry_run"] is True
        assert data["artifacts_scanned"] == 100
        assert data["artifacts_deleted"] == 15
        assert data["bytes_freed"] == 1_500_000
        assert data["pinned_protected"] == 5
        assert data["ancestor_protected"] == 2
        assert data["deleted_artifacts"] == ["a1", "a2"]

    def test_to_json(self) -> None:
        """to_json produces valid JSON string."""
        result = GCResult(
            policy_name="json_test",
            started_utc="2024-01-01T00:00:00Z",
            finished_utc="2024-01-01T00:01:00Z",
            dry_run=False,
            artifacts_scanned=50,
            artifacts_deleted=10,
            bytes_freed=500_000,
            bytes_total_before=5_000_000,
            bytes_total_after=4_500_000,
            pinned_protected=2,
            descendant_protected=1,
            dvc_gc_ran=True,
            dvc_gc_output="DVC gc completed",
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["policy_name"] == "json_test"
        assert parsed["dry_run"] is False


class TestBuiltinPolicies:
    """Tests for built-in retention policies."""

    def test_builtin_policies_exist(self) -> None:
        """Built-in policies are defined."""
        assert "laptop_default" in BUILTIN_POLICIES
        assert "ci_aggressive" in BUILTIN_POLICIES
        assert "archive" in BUILTIN_POLICIES
        assert "dev_minimal" in BUILTIN_POLICIES

    def test_default_policy_exists(self) -> None:
        """DEFAULT_POLICY references an existing policy."""
        assert DEFAULT_POLICY in BUILTIN_POLICIES

    def test_laptop_default_policy(self) -> None:
        """laptop_default policy has expected values."""
        policy = BUILTIN_POLICIES["laptop_default"]

        assert policy.name == "laptop_default"
        assert policy.keep_min_age_days == 14
        assert policy.keep_min_count == 5
        assert policy.keep_pinned is True
        assert policy.space_budget_bytes == 50 * 1024 * 1024 * 1024  # 50 GB

    def test_ci_aggressive_policy(self) -> None:
        """ci_aggressive policy has shorter retention."""
        policy = BUILTIN_POLICIES["ci_aggressive"]

        assert policy.keep_min_age_days == 7
        assert policy.keep_min_count == 2
        assert policy.keep_with_descendants is False

    def test_archive_policy(self) -> None:
        """archive policy has longer retention."""
        policy = BUILTIN_POLICIES["archive"]

        assert policy.keep_min_age_days == 365
        assert policy.keep_min_count == 10
        assert policy.space_budget_bytes == 500 * 1024 * 1024 * 1024  # 500 GB


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_now_utc_iso(self) -> None:
        """_now_utc_iso returns ISO 8601 format."""
        iso_str = _now_utc_iso()

        assert iso_str.endswith("Z")
        # Should be parseable
        dt = _parse_iso_datetime(iso_str)
        assert dt.tzinfo is not None

    def test_parse_iso_datetime_with_z(self) -> None:
        """_parse_iso_datetime handles Z suffix."""
        iso_str = "2024-01-15T14:30:00Z"
        dt = _parse_iso_datetime(iso_str)

        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 14
        assert dt.minute == 30

    def test_get_age_days(self) -> None:
        """_get_age_days calculates correct age."""
        # Create a timestamp from 10 days ago
        ten_days_ago = datetime.now(timezone.utc) - timedelta(days=10)
        iso_str = ten_days_ago.strftime("%Y-%m-%dT%H:%M:%SZ")

        age = _get_age_days(iso_str)

        # Should be approximately 10 days
        assert 9.9 < age < 10.1

    def test_format_bytes_bytes(self) -> None:
        """format_bytes formats small values as bytes."""
        assert "B" in format_bytes(500)

    def test_format_bytes_kb(self) -> None:
        """format_bytes formats KB."""
        result = format_bytes(2048)
        assert "KB" in result

    def test_format_bytes_mb(self) -> None:
        """format_bytes formats MB."""
        result = format_bytes(5 * 1024 * 1024)
        assert "MB" in result

    def test_format_bytes_gb(self) -> None:
        """format_bytes formats GB."""
        result = format_bytes(10 * 1024 * 1024 * 1024)
        assert "GB" in result


class TestPinFilePersistence:
    """Tests for pin file loading and saving."""

    def test_save_and_load_pins(self, tmp_path: Path) -> None:
        """Pins can be saved and loaded from file."""
        pins_file = tmp_path / "pins.json"

        pins = [
            PinnedArtifact(artifact_id="art1", reason="Important"),
            PinnedArtifact(run_id="run1", reason="Production"),
        ]

        save_pins_to_file(pins_file, pins)
        assert pins_file.exists()

        loaded = load_pins_from_file(pins_file)

        assert len(loaded) == 2
        assert loaded[0].artifact_id == "art1"
        assert loaded[1].run_id == "run1"

    def test_load_pins_from_nonexistent_file(self, tmp_path: Path) -> None:
        """Loading from nonexistent file returns empty list."""
        pins_file = tmp_path / "does_not_exist.json"

        loaded = load_pins_from_file(pins_file)

        assert loaded == []

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_pins_to_file creates parent directories."""
        pins_file = tmp_path / "deep" / "nested" / "pins.json"

        pins = [PinnedArtifact(artifact_id="art1")]
        save_pins_to_file(pins_file, pins)

        assert pins_file.exists()
        assert pins_file.parent.exists()


class TestRetentionUnit:
    """Tests for RetentionUnit enum."""

    def test_retention_units(self) -> None:
        """RetentionUnit has expected values."""
        assert RetentionUnit.HOURS.value == "hours"
        assert RetentionUnit.DAYS.value == "days"
        assert RetentionUnit.WEEKS.value == "weeks"
        assert RetentionUnit.MONTHS.value == "months"


class TestGCErrorTypes:
    """Tests for GC error types."""

    def test_gc_error_is_exception(self) -> None:
        """GCError is an Exception."""
        error = GCError("Test error")
        assert isinstance(error, Exception)

    def test_policy_not_found_error_is_gc_error(self) -> None:
        """PolicyNotFoundError is a GCError."""
        error = PolicyNotFoundError("Policy not found")
        assert isinstance(error, GCError)
        assert isinstance(error, Exception)
