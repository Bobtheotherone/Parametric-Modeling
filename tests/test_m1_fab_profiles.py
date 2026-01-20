"""Tests for fab capability profiles and DFM constraint loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from formula_foundry.coupongen.fab_profiles import (
    FAB_PROFILES_DIR,
    clear_profile_cache,
    get_fab_limits,
    list_available_profiles,
    load_fab_profile,
    load_fab_profile_from_dict,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the profile cache before each test."""
    clear_profile_cache()


def test_fab_profiles_dir_exists() -> None:
    """Test that the fab_profiles directory exists."""
    assert FAB_PROFILES_DIR.exists()
    assert FAB_PROFILES_DIR.is_dir()


def test_list_available_profiles() -> None:
    """Test listing available fab profiles."""
    profiles = list_available_profiles()
    assert isinstance(profiles, list)
    assert "oshpark_4layer" in profiles
    assert "generic" in profiles


def test_load_oshpark_4layer_profile() -> None:
    """Test loading the OSH Park 4-layer profile."""
    profile = load_fab_profile("oshpark_4layer")

    assert profile.id == "oshpark_4layer"
    assert profile.vendor == "OSH Park"
    assert profile.schema_version == 1

    # Verify trace constraints
    assert profile.trace.min_width_nm == 127000  # 5 mil
    assert profile.trace.min_spacing_nm == 127000  # 5 mil

    # Verify drill constraints
    assert profile.drill.min_pth_diameter_nm == 254000  # 10 mil

    # Verify via constraints
    assert profile.via.min_annular_ring_nm == 101600  # 4 mil


def test_load_generic_profile() -> None:
    """Test loading the generic conservative profile."""
    profile = load_fab_profile("generic")

    assert profile.id == "generic"
    assert profile.vendor == "Generic"
    assert 2 in profile.layer_counts
    assert 4 in profile.layer_counts


def test_load_nonexistent_profile_raises() -> None:
    """Test that loading a nonexistent profile raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Fab profile not found"):
        load_fab_profile("nonexistent_profile")


def test_load_fab_profile_from_dict() -> None:
    """Test loading a fab profile from a dictionary."""
    data = {
        "schema_version": 1,
        "id": "test_profile",
        "name": "Test Profile",
        "vendor": "Test Vendor",
        "trace": {"min_width_nm": 100000, "min_spacing_nm": 100000},
        "drill": {"min_diameter_nm": 200000, "min_pth_diameter_nm": 200000},
        "via": {"min_annular_ring_nm": 100000, "min_diameter_nm": 400000},
        "soldermask": {"min_expansion_nm": 50000, "min_web_nm": 100000},
        "silkscreen": {"min_width_nm": 150000, "min_height_nm": 750000, "min_clearance_nm": 125000},
        "board": {"min_edge_clearance_nm": 250000},
    }

    profile = load_fab_profile_from_dict(data)
    assert profile.id == "test_profile"
    assert profile.trace.min_width_nm == 100000


def test_get_fab_limits() -> None:
    """Test extracting flat limits dictionary from a profile."""
    profile = load_fab_profile("oshpark_4layer")
    limits = get_fab_limits(profile)

    assert "min_trace_width_nm" in limits
    assert "min_gap_nm" in limits
    assert "min_drill_nm" in limits
    assert "min_annular_ring_nm" in limits
    assert "min_via_diameter_nm" in limits
    assert "min_edge_clearance_nm" in limits

    # Check that values are integers
    assert all(isinstance(v, int) for v in limits.values())

    # Check specific values from oshpark profile
    assert limits["min_trace_width_nm"] == 127000
    assert limits["min_gap_nm"] == 127000


def test_profile_schema_validation_rejects_invalid() -> None:
    """Test that invalid profiles are rejected by validation."""
    # Missing required field
    invalid_data = {
        "schema_version": 1,
        "id": "invalid",
        "name": "Invalid",
        # Missing vendor and other required fields
    }

    with pytest.raises(ValidationError):
        load_fab_profile_from_dict(invalid_data)


def test_profile_schema_rejects_extra_fields() -> None:
    """Test that extra fields are rejected by strict validation."""
    data = {
        "schema_version": 1,
        "id": "test_profile",
        "name": "Test Profile",
        "vendor": "Test Vendor",
        "extra_field": "should fail",  # Extra field
        "trace": {"min_width_nm": 100000, "min_spacing_nm": 100000},
        "drill": {"min_diameter_nm": 200000, "min_pth_diameter_nm": 200000},
        "via": {"min_annular_ring_nm": 100000, "min_diameter_nm": 400000},
        "soldermask": {"min_expansion_nm": 50000, "min_web_nm": 100000},
        "silkscreen": {"min_width_nm": 150000, "min_height_nm": 750000, "min_clearance_nm": 125000},
        "board": {"min_edge_clearance_nm": 250000},
    }

    with pytest.raises(ValidationError, match="extra_field"):
        load_fab_profile_from_dict(data)


def test_profile_id_pattern_validation() -> None:
    """Test that profile IDs must match the pattern [a-z0-9_]+."""
    data = {
        "schema_version": 1,
        "id": "Invalid-ID",  # Hyphens not allowed
        "name": "Test Profile",
        "vendor": "Test Vendor",
        "trace": {"min_width_nm": 100000, "min_spacing_nm": 100000},
        "drill": {"min_diameter_nm": 200000, "min_pth_diameter_nm": 200000},
        "via": {"min_annular_ring_nm": 100000, "min_diameter_nm": 400000},
        "soldermask": {"min_expansion_nm": 50000, "min_web_nm": 100000},
        "silkscreen": {"min_width_nm": 150000, "min_height_nm": 750000, "min_clearance_nm": 125000},
        "board": {"min_edge_clearance_nm": 250000},
    }

    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        load_fab_profile_from_dict(data)


def test_profile_caching() -> None:
    """Test that profiles are cached after first load."""
    clear_profile_cache()

    profile1 = load_fab_profile("oshpark_4layer")
    profile2 = load_fab_profile("oshpark_4layer")

    # Should be the same cached instance
    assert profile1 is profile2


def test_all_bundled_profiles_valid() -> None:
    """Test that all bundled fab profiles load and validate successfully."""
    profiles = list_available_profiles()
    assert len(profiles) >= 2  # At least oshpark_4layer and generic

    for profile_id in profiles:
        profile = load_fab_profile(profile_id)
        assert profile.id == profile_id
        assert profile.schema_version == 1
        # Verify essential constraints exist
        assert profile.trace.min_width_nm > 0
        assert profile.drill.min_diameter_nm > 0
        assert profile.via.min_annular_ring_nm > 0


def test_profile_length_unit_parsing() -> None:
    """Test that length values can be parsed from strings with units."""
    data = {
        "schema_version": 1,
        "id": "test_units",
        "name": "Test Units",
        "vendor": "Test",
        "trace": {"min_width_nm": "5mil", "min_spacing_nm": "127um"},  # String with units
        "drill": {"min_diameter_nm": "0.2mm", "min_pth_diameter_nm": 200000},
        "via": {"min_annular_ring_nm": "4mil", "min_diameter_nm": 400000},
        "soldermask": {"min_expansion_nm": 50000, "min_web_nm": 100000},
        "silkscreen": {"min_width_nm": 150000, "min_height_nm": 750000, "min_clearance_nm": 125000},
        "board": {"min_edge_clearance_nm": 250000},
    }

    profile = load_fab_profile_from_dict(data)
    assert profile.trace.min_width_nm == 127000  # 5 mil = 127000 nm
    assert profile.trace.min_spacing_nm == 127000  # 127 um = 127000 nm
    assert profile.drill.min_diameter_nm == 200000  # 0.2 mm = 200000 nm
