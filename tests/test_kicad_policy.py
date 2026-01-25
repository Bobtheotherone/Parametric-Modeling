"""Unit tests for KiCad zone policy module.

Tests the zone policy configuration used for DRC and export operations.

Satisfies REQ-M1-006: If CPWG uses zones, DRC MUST be run with zone refill
enabled and exports MUST be run with zone checks enabled (KiCad CLI
flags/policy pinned in code and recorded in manifest).

Satisfies REQ-M1-013: The manifest MUST include an explicit zone policy
record (refill/check behavior and toolchain versioning).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Direct import to avoid broken import chain in formula_foundry.__init__
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"

# Load policy module directly - register in sys.modules before exec
_policy_spec = importlib.util.spec_from_file_location(
    "formula_foundry.coupongen.kicad.policy",
    _SRC_DIR / "formula_foundry" / "coupongen" / "kicad" / "policy.py"
)
_policy = importlib.util.module_from_spec(_policy_spec)  # type: ignore[arg-type]
sys.modules["formula_foundry.coupongen.kicad.policy"] = _policy
_policy_spec.loader.exec_module(_policy)  # type: ignore[union-attr]

DEFAULT_ZONE_POLICY = _policy.DEFAULT_ZONE_POLICY
ZonePolicy = _policy.ZonePolicy
get_zone_policy_record = _policy.get_zone_policy_record


class TestZonePolicyDataclass:
    """Tests for ZonePolicy dataclass."""

    def test_policy_is_frozen(self) -> None:
        """ZonePolicy instances are immutable."""
        policy = ZonePolicy(
            policy_id="test",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=True,
            export_check_flag="--check-zones",
        )
        with pytest.raises(AttributeError):
            policy.policy_id = "modified"  # type: ignore[misc]

    def test_policy_attributes(self) -> None:
        """ZonePolicy has expected attributes."""
        policy = ZonePolicy(
            policy_id="test-policy-v1",
            drc_refill_zones=True,
            drc_refill_flag="--my-refill-flag",
            export_check_zones=False,
            export_check_flag="--my-check-flag",
            kicad_cli_version="9.0.7",
        )

        assert policy.policy_id == "test-policy-v1"
        assert policy.drc_refill_zones is True
        assert policy.drc_refill_flag == "--my-refill-flag"
        assert policy.export_check_zones is False
        assert policy.export_check_flag == "--my-check-flag"
        assert policy.kicad_cli_version == "9.0.7"

    def test_kicad_cli_version_optional(self) -> None:
        """kicad_cli_version is optional and defaults to None."""
        policy = ZonePolicy(
            policy_id="test",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=True,
            export_check_flag="--check-zones",
        )
        assert policy.kicad_cli_version is None


class TestZonePolicyToDict:
    """Tests for ZonePolicy.to_dict method."""

    def test_to_dict_structure(self) -> None:
        """to_dict returns expected structure for manifest."""
        policy = ZonePolicy(
            policy_id="test-v1",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=True,
            export_check_flag="--check-zones",
        )
        result = policy.to_dict()

        assert "policy_id" in result
        assert "drc" in result
        assert "export" in result
        assert result["policy_id"] == "test-v1"

    def test_to_dict_drc_section(self) -> None:
        """to_dict includes proper DRC section."""
        policy = ZonePolicy(
            policy_id="test",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=True,
            export_check_flag="--check-zones",
        )
        result = policy.to_dict()

        assert result["drc"]["refill_zones"] is True
        assert result["drc"]["flag"] == "--refill-zones"

    def test_to_dict_export_section(self) -> None:
        """to_dict includes proper export section."""
        policy = ZonePolicy(
            policy_id="test",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=True,
            export_check_flag="--check-zones",
        )
        result = policy.to_dict()

        assert result["export"]["check_zones"] is True
        assert result["export"]["flag"] == "--check-zones"

    def test_to_dict_without_version(self) -> None:
        """to_dict excludes kicad_cli_version when None."""
        policy = ZonePolicy(
            policy_id="test",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=True,
            export_check_flag="--check-zones",
        )
        result = policy.to_dict()

        assert "kicad_cli_version" not in result

    def test_to_dict_with_version(self) -> None:
        """to_dict includes kicad_cli_version when set."""
        policy = ZonePolicy(
            policy_id="test",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=True,
            export_check_flag="--check-zones",
            kicad_cli_version="9.0.7",
        )
        result = policy.to_dict()

        assert result["kicad_cli_version"] == "9.0.7"

    def test_to_dict_json_serializable(self) -> None:
        """to_dict result is JSON serializable."""
        import json

        policy = ZonePolicy(
            policy_id="test",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=True,
            export_check_flag="--check-zones",
            kicad_cli_version="9.0.7",
        )
        result = policy.to_dict()

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)


class TestZonePolicyWithKicadCliVersion:
    """Tests for ZonePolicy.with_kicad_cli_version method."""

    def test_with_version_returns_new_policy(self) -> None:
        """with_kicad_cli_version returns a new policy instance."""
        original = ZonePolicy(
            policy_id="test",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=True,
            export_check_flag="--check-zones",
        )
        updated = original.with_kicad_cli_version("9.0.7")

        assert original is not updated
        assert original.kicad_cli_version is None
        assert updated.kicad_cli_version == "9.0.7"

    def test_with_same_version_returns_self(self) -> None:
        """with_kicad_cli_version returns self if version unchanged."""
        original = ZonePolicy(
            policy_id="test",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=True,
            export_check_flag="--check-zones",
            kicad_cli_version="9.0.7",
        )
        updated = original.with_kicad_cli_version("9.0.7")

        assert original is updated

    def test_with_none_version_clears(self) -> None:
        """with_kicad_cli_version can set to None."""
        original = ZonePolicy(
            policy_id="test",
            drc_refill_zones=True,
            drc_refill_flag="--refill-zones",
            export_check_zones=True,
            export_check_flag="--check-zones",
            kicad_cli_version="9.0.7",
        )
        updated = original.with_kicad_cli_version(None)

        assert updated.kicad_cli_version is None

    def test_preserves_other_attributes(self) -> None:
        """with_kicad_cli_version preserves all other attributes."""
        original = ZonePolicy(
            policy_id="test-v2",
            drc_refill_zones=False,
            drc_refill_flag="--custom-refill",
            export_check_zones=False,
            export_check_flag="--custom-check",
        )
        updated = original.with_kicad_cli_version("10.0.0")

        assert updated.policy_id == original.policy_id
        assert updated.drc_refill_zones == original.drc_refill_zones
        assert updated.drc_refill_flag == original.drc_refill_flag
        assert updated.export_check_zones == original.export_check_zones
        assert updated.export_check_flag == original.export_check_flag


class TestDefaultZonePolicy:
    """Tests for DEFAULT_ZONE_POLICY constant."""

    def test_policy_id_is_stable(self) -> None:
        """DEFAULT_ZONE_POLICY has stable policy ID."""
        assert DEFAULT_ZONE_POLICY.policy_id == "kicad-cli-zones-v1"

    def test_drc_refill_enabled(self) -> None:
        """REQ-M1-006: DRC zone refill is enabled."""
        assert DEFAULT_ZONE_POLICY.drc_refill_zones is True

    def test_drc_refill_flag_correct(self) -> None:
        """DRC refill flag is standard --refill-zones."""
        assert DEFAULT_ZONE_POLICY.drc_refill_flag == "--refill-zones"

    def test_export_check_enabled(self) -> None:
        """REQ-M1-006: Export zone check is enabled."""
        assert DEFAULT_ZONE_POLICY.export_check_zones is True

    def test_export_check_flag_correct(self) -> None:
        """Export check flag is standard --check-zones."""
        assert DEFAULT_ZONE_POLICY.export_check_flag == "--check-zones"

    def test_no_version_by_default(self) -> None:
        """DEFAULT_ZONE_POLICY has no kicad_cli_version by default."""
        assert DEFAULT_ZONE_POLICY.kicad_cli_version is None


class TestGetZonePolicyRecord:
    """Tests for get_zone_policy_record function."""

    def test_returns_dict(self) -> None:
        """get_zone_policy_record returns a dictionary."""
        record = get_zone_policy_record()
        assert isinstance(record, dict)

    def test_uses_default_policy(self) -> None:
        """Without arguments, uses DEFAULT_ZONE_POLICY."""
        record = get_zone_policy_record()
        assert record["policy_id"] == DEFAULT_ZONE_POLICY.policy_id

    def test_includes_version_when_provided(self) -> None:
        """kicad_cli_version is included when provided."""
        record = get_zone_policy_record(kicad_cli_version="9.0.7")
        assert record["kicad_cli_version"] == "9.0.7"

    def test_excludes_version_when_not_provided(self) -> None:
        """kicad_cli_version is excluded when not provided."""
        record = get_zone_policy_record()
        assert "kicad_cli_version" not in record

    def test_custom_policy(self) -> None:
        """Custom policy can be used."""
        custom_policy = ZonePolicy(
            policy_id="custom-v2",
            drc_refill_zones=False,
            drc_refill_flag="--no-refill",
            export_check_zones=False,
            export_check_flag="--no-check",
        )
        record = get_zone_policy_record(policy=custom_policy)

        assert record["policy_id"] == "custom-v2"
        assert record["drc"]["refill_zones"] is False
        assert record["export"]["check_zones"] is False

    def test_custom_policy_with_version(self) -> None:
        """Custom policy can be combined with version."""
        custom_policy = ZonePolicy(
            policy_id="custom-v3",
            drc_refill_zones=True,
            drc_refill_flag="--refill",
            export_check_zones=True,
            export_check_flag="--check",
        )
        record = get_zone_policy_record(
            kicad_cli_version="10.0.0",
            policy=custom_policy,
        )

        assert record["policy_id"] == "custom-v3"
        assert record["kicad_cli_version"] == "10.0.0"


class TestReqM1006PolicyCompliance:
    """Tests for REQ-M1-006 compliance through policy.

    REQ-M1-006: If CPWG uses zones, DRC MUST be run with zone refill enabled
    and exports MUST be run with zone checks enabled.
    """

    def test_default_policy_satisfies_req_m1_006_drc(self) -> None:
        """Default policy satisfies REQ-M1-006 for DRC."""
        assert DEFAULT_ZONE_POLICY.drc_refill_zones is True
        assert DEFAULT_ZONE_POLICY.drc_refill_flag == "--refill-zones"

    def test_default_policy_satisfies_req_m1_006_export(self) -> None:
        """Default policy satisfies REQ-M1-006 for exports."""
        assert DEFAULT_ZONE_POLICY.export_check_zones is True
        assert DEFAULT_ZONE_POLICY.export_check_flag == "--check-zones"

    def test_policy_is_deterministic(self) -> None:
        """Policy attributes are deterministic across invocations."""
        # Multiple reads should return same values
        assert DEFAULT_ZONE_POLICY.policy_id == "kicad-cli-zones-v1"
        assert DEFAULT_ZONE_POLICY.policy_id == "kicad-cli-zones-v1"


class TestReqM1013ManifestCompliance:
    """Tests for REQ-M1-013 manifest compliance.

    REQ-M1-013: The manifest MUST include an explicit zone policy
    record (refill/check behavior and toolchain versioning).
    """

    def test_policy_record_for_manifest(self) -> None:
        """Policy record contains required manifest fields."""
        record = get_zone_policy_record(kicad_cli_version="9.0.7")

        # Required fields per REQ-M1-013
        assert "policy_id" in record
        assert "drc" in record
        assert "export" in record
        assert "kicad_cli_version" in record

        # DRC behavior documented
        assert "refill_zones" in record["drc"]
        assert "flag" in record["drc"]

        # Export behavior documented
        assert "check_zones" in record["export"]
        assert "flag" in record["export"]

    def test_policy_record_documents_actual_behavior(self) -> None:
        """Policy record accurately documents actual behavior."""
        record = get_zone_policy_record()

        # Record must match actual policy behavior
        assert record["drc"]["refill_zones"] == DEFAULT_ZONE_POLICY.drc_refill_zones
        assert record["drc"]["flag"] == DEFAULT_ZONE_POLICY.drc_refill_flag
        assert record["export"]["check_zones"] == DEFAULT_ZONE_POLICY.export_check_zones
        assert record["export"]["flag"] == DEFAULT_ZONE_POLICY.export_check_flag
