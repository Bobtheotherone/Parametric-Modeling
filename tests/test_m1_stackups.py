"""Tests for stackup profiles and layer structure."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from formula_foundry.coupongen.stackups import (
    STACKUPS_DIR,
    clear_stackup_cache,
    compute_total_thickness,
    get_copper_layer_names,
    get_dielectric_between_layers,
    get_effective_er,
    get_thickness_between_layers,
    list_available_stackups,
    load_stackup,
    load_stackup_from_dict,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the stackup cache before each test."""
    clear_stackup_cache()


def test_stackups_dir_exists() -> None:
    """Test that the stackups directory exists."""
    assert STACKUPS_DIR.exists()
    assert STACKUPS_DIR.is_dir()


def test_list_available_stackups() -> None:
    """Test listing available stackup profiles."""
    stackups = list_available_stackups()
    assert isinstance(stackups, list)
    assert "oshpark_4layer" in stackups
    assert "generic_4layer" in stackups


def test_load_oshpark_4layer_stackup() -> None:
    """Test loading the OSH Park 4-layer stackup."""
    stackup = load_stackup("oshpark_4layer")

    assert stackup.id == "oshpark_4layer"
    assert stackup.vendor == "OSH Park"
    assert stackup.copper_layers == 4
    assert stackup.schema_version == 1

    # Verify layer count matches copper_layers
    copper_count = sum(1 for layer in stackup.layers if layer.type == "copper")
    assert copper_count == 4


def test_load_generic_4layer_stackup() -> None:
    """Test loading the generic 4-layer stackup."""
    stackup = load_stackup("generic_4layer")

    assert stackup.id == "generic_4layer"
    assert stackup.copper_layers == 4


def test_load_nonexistent_stackup_raises() -> None:
    """Test that loading a nonexistent stackup raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Stackup profile not found"):
        load_stackup("nonexistent_stackup")


def test_load_stackup_from_dict() -> None:
    """Test loading a stackup profile from a dictionary."""
    data = {
        "schema_version": 1,
        "id": "test_stackup",
        "name": "Test Stackup",
        "copper_layers": 2,
        "layers": [
            {"name": "F.Cu", "type": "copper", "thickness_nm": 35000},
            {"name": "core", "type": "dielectric", "thickness_nm": 1600000, "dielectric": {"er": 4.5, "loss_tangent": 0.02}},
            {"name": "B.Cu", "type": "copper", "thickness_nm": 35000},
        ],
    }

    stackup = load_stackup_from_dict(data)
    assert stackup.id == "test_stackup"
    assert stackup.copper_layers == 2
    assert len(stackup.layers) == 3


def test_copper_layer_count_validation() -> None:
    """Test that copper_layers must match actual copper layers in list."""
    data = {
        "schema_version": 1,
        "id": "test_invalid",
        "name": "Invalid Stackup",
        "copper_layers": 4,  # Says 4 but only 2 copper layers defined
        "layers": [
            {"name": "F.Cu", "type": "copper", "thickness_nm": 35000},
            {"name": "core", "type": "dielectric", "thickness_nm": 1600000},
            {"name": "B.Cu", "type": "copper", "thickness_nm": 35000},
        ],
    }

    with pytest.raises(ValidationError, match="copper_layers=4 but found 2"):
        load_stackup_from_dict(data)


def test_compute_total_thickness() -> None:
    """Test computing total stackup thickness."""
    stackup = load_stackup("oshpark_4layer")
    total = compute_total_thickness(stackup)

    # Sum of all layer thicknesses
    expected = sum(int(layer.thickness_nm) for layer in stackup.layers)
    assert total == expected
    assert total > 0


def test_get_copper_layer_names() -> None:
    """Test getting ordered copper layer names."""
    stackup = load_stackup("oshpark_4layer")
    names = get_copper_layer_names(stackup)

    assert len(names) == 4
    assert names[0] == "F.Cu"
    assert names[-1] == "B.Cu"


def test_get_dielectric_between_layers() -> None:
    """Test getting dielectric layers between copper layers."""
    stackup = load_stackup("oshpark_4layer")

    # Between first two copper layers (F.Cu and In1.Cu)
    dielectrics = get_dielectric_between_layers(stackup, 0, 1)
    assert len(dielectrics) >= 1
    assert all(layer.type == "dielectric" for layer in dielectrics)


def test_get_thickness_between_layers() -> None:
    """Test getting total dielectric thickness between copper layers."""
    stackup = load_stackup("oshpark_4layer")

    # Thickness between outer copper layers
    thickness = get_thickness_between_layers(stackup, 0, 3)
    assert thickness > 0

    # Thickness should be less than or equal to total
    total = compute_total_thickness(stackup)
    assert thickness <= total


def test_get_effective_er() -> None:
    """Test getting effective relative permittivity between layers."""
    stackup = load_stackup("oshpark_4layer")

    er = get_effective_er(stackup, 0, 1)
    assert er > 1  # Er must be > 1
    assert er < 10  # Reasonable upper bound for FR4-type materials


def test_layer_index_validation() -> None:
    """Test that invalid layer indices raise ValueError."""
    stackup = load_stackup("oshpark_4layer")

    with pytest.raises(ValueError, match="Layer index out of range"):
        get_dielectric_between_layers(stackup, 0, 10)


def test_stackup_caching() -> None:
    """Test that stackups are cached after first load."""
    clear_stackup_cache()

    stackup1 = load_stackup("oshpark_4layer")
    stackup2 = load_stackup("oshpark_4layer")

    # Should be the same cached instance
    assert stackup1 is stackup2


def test_all_bundled_stackups_valid() -> None:
    """Test that all bundled stackup profiles load and validate successfully."""
    stackups = list_available_stackups()
    assert len(stackups) >= 2  # At least oshpark_4layer and generic_4layer

    for stackup_id in stackups:
        stackup = load_stackup(stackup_id)
        assert stackup.id == stackup_id
        assert stackup.schema_version == 1
        assert stackup.copper_layers >= 1

        # Verify copper count matches
        copper_count = sum(1 for layer in stackup.layers if layer.type == "copper")
        assert copper_count == stackup.copper_layers


def test_dielectric_properties() -> None:
    """Test that dielectric properties are properly loaded."""
    stackup = load_stackup("oshpark_4layer")

    dielectric_layers = [layer for layer in stackup.layers if layer.type == "dielectric"]
    assert len(dielectric_layers) > 0

    for layer in dielectric_layers:
        if layer.dielectric:
            assert layer.dielectric.er > 1
            assert 0 <= layer.dielectric.loss_tangent <= 1


def test_stackup_length_unit_parsing() -> None:
    """Test that thickness values can be parsed from strings with units."""
    data = {
        "schema_version": 1,
        "id": "test_units",
        "name": "Test Units",
        "copper_layers": 2,
        "layers": [
            {"name": "F.Cu", "type": "copper", "thickness_nm": "35um"},  # String with units
            {"name": "core", "type": "dielectric", "thickness_nm": "1.6mm"},
            {"name": "B.Cu", "type": "copper", "thickness_nm": "1.4mil"},
        ],
    }

    stackup = load_stackup_from_dict(data)
    assert stackup.layers[0].thickness_nm == 35000  # 35um = 35000nm
    assert stackup.layers[1].thickness_nm == 1600000  # 1.6mm = 1600000nm
    assert stackup.layers[2].thickness_nm == 35560  # 1.4mil ≈ 35560nm (rounded)


def test_default_dielectric_properties() -> None:
    """Test that default dielectric properties are available at stackup level."""
    stackup = load_stackup("oshpark_4layer")

    if stackup.dielectric_defaults:
        assert stackup.dielectric_defaults.er > 1
        assert stackup.dielectric_defaults.loss_tangent >= 0
