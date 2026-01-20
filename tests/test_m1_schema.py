from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from formula_foundry.coupongen.spec import (
    COUPONSPEC_SCHEMA,
    COUPONSPEC_SCHEMA_PATH,
    CouponSpec,
    get_json_schema,
    load_couponspec,
    load_couponspec_from_file,
)


def _example_spec_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "coupon_family": "F1_SINGLE_ENDED_VIA",
        "units": "nm",
        "toolchain": {
            "kicad": {
                "version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7@sha256:deadbeef",
            }
        },
        "fab_profile": {"id": "oshpark_4layer", "overrides": {}},
        "stackup": {
            "copper_layers": 4,
            "thicknesses_nm": {
                "L1_to_L2": 180000,
                "L2_to_L3": 800000,
                "L3_to_L4": 180000,
            },
            "materials": {"er": 4.1, "loss_tangent": 0.02},
        },
        "board": {
            "outline": {
                "width_nm": 20000000,
                "length_nm": 80000000,
                "corner_radius_nm": 2000000,
            },
            "origin": {"mode": "EDGE_L_CENTER"},
            "text": {"coupon_id": "${COUPON_ID}", "include_manifest_hash": True},
        },
        "connectors": {
            "left": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [5000000, 0],
                "rotation_deg": 180,
            },
            "right": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [75000000, 0],
                "rotation_deg": 0,
            },
        },
        "transmission_line": {
            "type": "CPWG",
            "layer": "F.Cu",
            "w_nm": 300000,
            "gap_nm": 180000,
            "length_left_nm": 25000000,
            "length_right_nm": 25000000,
            "ground_via_fence": {
                "enabled": True,
                "pitch_nm": 1500000,
                "offset_from_gap_nm": 800000,
                "via": {"drill_nm": 300000, "diameter_nm": 600000},
            },
        },
        "discontinuity": {
            "type": "VIA_TRANSITION",
            "signal_via": {
                "drill_nm": 300000,
                "diameter_nm": 650000,
                "pad_diameter_nm": 900000,
            },
            "antipads": {
                "L2": {
                    "shape": "ROUNDRECT",
                    "rx_nm": 1200000,
                    "ry_nm": 900000,
                    "corner_nm": 250000,
                },
                "L3": {"shape": "CIRCLE", "r_nm": 1100000},
            },
            "return_vias": {
                "pattern": "RING",
                "count": 4,
                "radius_nm": 1700000,
                "via": {"drill_nm": 300000, "diameter_nm": 650000},
            },
            "plane_cutouts": {
                "L2": {
                    "shape": "SLOT",
                    "length_nm": 3000000,
                    "width_nm": 1500000,
                    "rotation_deg": 0,
                }
            },
        },
        "constraints": {
            "mode": "REJECT",
            "drc": {"must_pass": True, "severity": "all"},
            "symmetry": {"enforce": True},
            "allow_unconnected_copper": False,
        },
        "export": {
            "gerbers": {"enabled": True, "format": "gerbers"},
            "drill": {"enabled": True, "format": "excellon"},
            "outputs_dir": "artifacts/",
        },
    }


def test_couponspec_schema_validation() -> None:
    schema = COUPONSPEC_SCHEMA
    assert schema["type"] == "object"
    assert schema.get("additionalProperties") is False

    data = _example_spec_data()
    spec = CouponSpec.model_validate(data)
    assert spec.schema_version == 1

    data["unexpected_field"] = "nope"
    with pytest.raises(ValidationError):
        CouponSpec.model_validate(data)


def test_json_schema_file_exists() -> None:
    """REQ-M1-001: JSON Schema file must exist at the canonical path."""
    assert COUPONSPEC_SCHEMA_PATH.exists(), f"Schema file not found: {COUPONSPEC_SCHEMA_PATH}"


def test_json_schema_file_is_valid() -> None:
    """REQ-M1-001: JSON Schema file must be valid JSON Schema."""
    schema = get_json_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "CouponSpec"
    assert schema["type"] == "object"
    assert "schema_version" in schema["required"]


def test_load_couponspec_from_dict() -> None:
    """Test loading CouponSpec from dict validates and normalizes units."""
    data = _example_spec_data()
    spec = load_couponspec(data)
    assert spec.schema_version == 1
    assert spec.coupon_family == "F1_SINGLE_ENDED_VIA"
    assert spec.units == "nm"


def test_load_couponspec_from_json_file() -> None:
    """REQ-M1-002: Must support loading CouponSpec from JSON files."""
    data = _example_spec_data()
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(data, f)
        f.flush()
        path = Path(f.name)

    try:
        spec = load_couponspec_from_file(path)
        assert spec.schema_version == 1
        assert spec.coupon_family == "F1_SINGLE_ENDED_VIA"
    finally:
        path.unlink()


def test_load_couponspec_from_yaml_file() -> None:
    """REQ-M1-002: Must support loading CouponSpec from YAML files."""
    yaml = pytest.importorskip("yaml")
    data = _example_spec_data()
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.safe_dump(data, f)
        f.flush()
        path = Path(f.name)

    try:
        spec = load_couponspec_from_file(path)
        assert spec.schema_version == 1
        assert spec.coupon_family == "F1_SINGLE_ENDED_VIA"
    finally:
        path.unlink()


def test_load_couponspec_from_yml_file() -> None:
    """REQ-M1-002: Must support .yml extension."""
    yaml = pytest.importorskip("yaml")
    data = _example_spec_data()
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        yaml.safe_dump(data, f)
        f.flush()
        path = Path(f.name)

    try:
        spec = load_couponspec_from_file(path)
        assert spec.schema_version == 1
    finally:
        path.unlink()


def test_load_couponspec_with_unit_strings() -> None:
    """REQ-M1-001: CouponSpec must support mm/mil/um unit strings that normalize to nm."""
    data = _example_spec_data()
    # Replace integer nm values with unit strings
    data["board"]["outline"]["width_nm"] = "20mm"  # type: ignore[index]
    data["board"]["outline"]["length_nm"] = "80mm"  # type: ignore[index]
    data["board"]["outline"]["corner_radius_nm"] = "2mm"  # type: ignore[index]
    data["transmission_line"]["w_nm"] = "0.3mm"  # type: ignore[index]
    data["transmission_line"]["gap_nm"] = "7.09mil"  # type: ignore[index]

    spec = load_couponspec(data)
    assert spec.board.outline.width_nm == 20_000_000
    assert spec.board.outline.length_nm == 80_000_000
    assert spec.board.outline.corner_radius_nm == 2_000_000
    assert spec.transmission_line.w_nm == 300_000
    # 7.09mil = 7.09 * 25400 = 180086 nm (approximately)
    assert spec.transmission_line.gap_nm == 180086


def test_load_couponspec_unsupported_extension_raises() -> None:
    """Unsupported file extensions must raise ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("{}")
        f.flush()
        path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_couponspec_from_file(path)
    finally:
        path.unlink()


def test_load_couponspec_file_not_found_raises() -> None:
    """Missing file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_couponspec_from_file("/nonexistent/path/to/spec.json")
