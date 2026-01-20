"""Tests for CouponSpec schema validation, unit parsing, and error handling.

This module provides comprehensive tests for the Geometry DSL schema validation
as defined in M1 of the design document. It covers:

- REQ-M1-001: JSON Schema validation and existence
- REQ-M1-002: Loading CouponSpec from JSON/YAML files
- Unit parsing (nm, um, mm, mil) with normalization to integer nanometers
- Error handling for invalid inputs (missing required fields, type errors, etc.)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

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
from formula_foundry.coupongen.units import (
    AngleMdeg,
    FrequencyHz,
    LengthNM,
    parse_angle_mdeg,
    parse_frequency_hz,
    parse_length_nm,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def valid_spec_data() -> dict[str, Any]:
    """Return a valid CouponSpec data dictionary."""
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


@pytest.fixture
def minimal_spec_data() -> dict[str, Any]:
    """Return a minimal valid CouponSpec without optional fields."""
    return {
        "schema_version": 1,
        "coupon_family": "F0_CALIBRATION_THRU",
        "units": "nm",
        "toolchain": {
            "kicad": {
                "version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7@sha256:abc123",
            }
        },
        "fab_profile": {"id": "generic_4layer"},
        "stackup": {
            "copper_layers": 4,
            "thicknesses_nm": {
                "L1_to_L2": 180000,
                "L2_to_L3": 800000,
                "L3_to_L4": 180000,
            },
            "materials": {"er": 4.5, "loss_tangent": 0.015},
        },
        "board": {
            "outline": {
                "width_nm": 10000000,
                "length_nm": 40000000,
                "corner_radius_nm": 1000000,
            },
            "origin": {"mode": "EDGE_L_CENTER"},
            "text": {"coupon_id": "THRU001", "include_manifest_hash": False},
        },
        "connectors": {
            "left": {
                "footprint": "SMA:SMA_Vertical",
                "position_nm": [2000000, 0],
                "rotation_deg": 0,
            },
            "right": {
                "footprint": "SMA:SMA_Vertical",
                "position_nm": [38000000, 0],
                "rotation_deg": 180,
            },
        },
        "transmission_line": {
            "type": "CPWG",
            "layer": "F.Cu",
            "w_nm": 200000,
            "gap_nm": 150000,
            "length_left_nm": 15000000,
            "length_right_nm": 15000000,
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
            "outputs_dir": "output/",
        },
    }


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestCouponSpecSchema:
    """Tests for CouponSpec JSON Schema structure and validation."""

    def test_schema_exists_at_canonical_path(self) -> None:
        """REQ-M1-001: JSON Schema file must exist at the canonical path."""
        assert COUPONSPEC_SCHEMA_PATH.exists(), f"Schema not found: {COUPONSPEC_SCHEMA_PATH}"

    def test_schema_is_valid_json(self) -> None:
        """REQ-M1-001: Schema file must be valid JSON."""
        schema = get_json_schema()
        assert isinstance(schema, dict)
        assert "$schema" in schema

    def test_schema_uses_2020_12_draft(self) -> None:
        """REQ-M1-001: Schema must use JSON Schema draft 2020-12."""
        schema = get_json_schema()
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"

    def test_schema_has_correct_title(self) -> None:
        """Schema must have correct title."""
        schema = get_json_schema()
        assert schema["title"] == "CouponSpec"

    def test_schema_is_object_type(self) -> None:
        """Schema root must be object type."""
        assert COUPONSPEC_SCHEMA["type"] == "object"

    def test_schema_forbids_additional_properties(self) -> None:
        """Schema must forbid additional properties for strict validation."""
        assert COUPONSPEC_SCHEMA.get("additionalProperties") is False

    def test_schema_has_required_fields(self) -> None:
        """Schema must define required top-level fields."""
        schema = get_json_schema()
        required = schema.get("required", [])
        assert "schema_version" in required


class TestCouponSpecValidation:
    """Tests for CouponSpec validation via Pydantic."""

    def test_valid_spec_loads_successfully(self, valid_spec_data: dict[str, Any]) -> None:
        """Valid specification must load without errors."""
        spec = CouponSpec.model_validate(valid_spec_data)
        assert spec.schema_version == 1
        assert spec.coupon_family == "F1_SINGLE_ENDED_VIA"

    def test_minimal_spec_loads_successfully(self, minimal_spec_data: dict[str, Any]) -> None:
        """Minimal valid specification (without optional fields) must load."""
        spec = CouponSpec.model_validate(minimal_spec_data)
        assert spec.schema_version == 1
        assert spec.discontinuity is None  # Optional field

    def test_extra_fields_rejected(self, valid_spec_data: dict[str, Any]) -> None:
        """Additional properties must be rejected (strict mode)."""
        valid_spec_data["unexpected_field"] = "should_fail"
        with pytest.raises(ValidationError):
            CouponSpec.model_validate(valid_spec_data)

    def test_missing_required_field_rejected(self, valid_spec_data: dict[str, Any]) -> None:
        """Missing required fields must raise ValidationError."""
        del valid_spec_data["schema_version"]
        with pytest.raises(ValidationError) as exc_info:
            CouponSpec.model_validate(valid_spec_data)
        assert "schema_version" in str(exc_info.value)

    def test_wrong_type_rejected(self, valid_spec_data: dict[str, Any]) -> None:
        """Wrong types must be rejected."""
        valid_spec_data["schema_version"] = "not_an_int"
        with pytest.raises(ValidationError):
            CouponSpec.model_validate(valid_spec_data)

    def test_invalid_constraint_mode_rejected(self, valid_spec_data: dict[str, Any]) -> None:
        """Invalid constraint mode (not REJECT/REPAIR) must be rejected."""
        valid_spec_data["constraints"]["mode"] = "INVALID_MODE"
        with pytest.raises(ValidationError) as exc_info:
            CouponSpec.model_validate(valid_spec_data)
        assert "mode" in str(exc_info.value).lower()

    def test_nested_missing_field_rejected(self, valid_spec_data: dict[str, Any]) -> None:
        """Missing nested required fields must be rejected."""
        del valid_spec_data["toolchain"]["kicad"]["version"]
        with pytest.raises(ValidationError):
            CouponSpec.model_validate(valid_spec_data)


# =============================================================================
# File Loading Tests
# =============================================================================


class TestCouponSpecFileLoading:
    """Tests for loading CouponSpec from files."""

    def test_load_from_json_file(self, valid_spec_data: dict[str, Any]) -> None:
        """REQ-M1-002: Must support loading from JSON files."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(valid_spec_data, f)
            path = Path(f.name)

        try:
            spec = load_couponspec_from_file(path)
            assert spec.schema_version == 1
        finally:
            path.unlink()

    def test_load_from_yaml_file(self, valid_spec_data: dict[str, Any]) -> None:
        """REQ-M1-002: Must support loading from YAML files."""
        yaml = pytest.importorskip("yaml")
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.safe_dump(valid_spec_data, f)
            path = Path(f.name)

        try:
            spec = load_couponspec_from_file(path)
            assert spec.schema_version == 1
        finally:
            path.unlink()

    def test_load_from_yml_file(self, valid_spec_data: dict[str, Any]) -> None:
        """REQ-M1-002: Must support .yml extension."""
        yaml = pytest.importorskip("yaml")
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            yaml.safe_dump(valid_spec_data, f)
            path = Path(f.name)

        try:
            spec = load_couponspec_from_file(path)
            assert spec.schema_version == 1
        finally:
            path.unlink()

    def test_unsupported_extension_raises(self) -> None:
        """Unsupported file extensions must raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("{}")
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unsupported file extension"):
                load_couponspec_from_file(path)
        finally:
            path.unlink()

    def test_nonexistent_file_raises(self) -> None:
        """Missing file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_couponspec_from_file("/nonexistent/path/spec.json")

    def test_invalid_json_raises(self) -> None:
        """Malformed JSON must raise JSONDecodeError."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{invalid json}")
            path = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                load_couponspec_from_file(path)
        finally:
            path.unlink()


# =============================================================================
# Unit Parsing Tests: LengthNM
# =============================================================================


class TestLengthNMParsing:
    """Tests for parsing lengths to integer nanometers."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("0.25mm", 250_000),
            ("1mm", 1_000_000),
            ("0.001mm", 1_000),
            ("10mil", 254_000),
            ("1mil", 25_400),
            ("250um", 250_000),
            ("1um", 1_000),
            ("1000nm", 1_000),
            ("1nm", 1),
        ],
    )
    def test_string_with_units(self, value: str, expected: int) -> None:
        """String values with units must parse correctly."""
        assert parse_length_nm(value) == expected

    @pytest.mark.parametrize("value,expected", [("1000", 1_000), ("0", 0), ("-500", -500)])
    def test_integer_strings_as_nm(self, value: str, expected: int) -> None:
        """Plain integer strings are treated as nanometers."""
        assert parse_length_nm(value) == expected

    def test_raw_integer_passthrough(self) -> None:
        """Raw integers pass through as nanometers."""
        assert parse_length_nm(1_000_000) == 1_000_000
        assert parse_length_nm(0) == 0
        assert parse_length_nm(-100) == -100

    def test_float_integer_valued(self) -> None:
        """Float values exactly equal to integers are accepted."""
        assert parse_length_nm(1000.0) == 1_000
        assert parse_length_nm(1e6) == 1_000_000

    def test_returns_int_type(self) -> None:
        """Parser must always return Python int."""
        assert isinstance(parse_length_nm("1mm"), int)
        assert isinstance(parse_length_nm(1000), int)
        assert isinstance(parse_length_nm(1000.0), int)

    def test_whitespace_tolerance(self) -> None:
        """Whitespace around value and unit is tolerated."""
        assert parse_length_nm("  1mm  ") == 1_000_000
        assert parse_length_nm(" 10 mil ") == 254_000

    def test_case_insensitive_units(self) -> None:
        """Units are case-insensitive."""
        assert parse_length_nm("1MM") == 1_000_000
        assert parse_length_nm("10MIL") == 254_000
        assert parse_length_nm("1UM") == 1_000
        assert parse_length_nm("1NM") == 1

    def test_negative_values(self) -> None:
        """Negative values are supported."""
        assert parse_length_nm("-1mm") == -1_000_000
        assert parse_length_nm("-10mil") == -254_000


class TestLengthNMErrors:
    """Tests for LengthNM error handling."""

    def test_unknown_unit_raises(self) -> None:
        """Unknown units must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown LengthNM unit"):
            parse_length_nm("12.34foo")

    def test_unitless_decimal_fails(self) -> None:
        """Decimal without unit is ambiguous and must fail."""
        with pytest.raises(ValueError):
            parse_length_nm("1.5")

    def test_non_integer_result_fails(self) -> None:
        """Conversions yielding non-integer nm must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_length_nm("0.5nm")

    def test_boolean_rejected(self) -> None:
        """Boolean values must be rejected."""
        with pytest.raises(ValueError, match="boolean"):
            parse_length_nm(True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="boolean"):
            parse_length_nm(False)  # type: ignore[arg-type]

    def test_empty_string_fails(self) -> None:
        """Empty string must fail."""
        with pytest.raises(ValueError, match="numeric value"):
            parse_length_nm("")

    def test_whitespace_only_fails(self) -> None:
        """Whitespace-only string must fail."""
        with pytest.raises(ValueError, match="numeric value"):
            parse_length_nm("   ")

    def test_non_integer_float_fails(self) -> None:
        """Float that is not exactly integer must fail."""
        with pytest.raises(ValueError, match="integer"):
            parse_length_nm(1000.5)


class TestLengthNMBounds:
    """Tests for LengthNM 64-bit integer bounds."""

    def test_max_i64_accepted(self) -> None:
        """Maximum signed 64-bit integer is accepted."""
        max_i64 = 2**63 - 1
        assert parse_length_nm(max_i64) == max_i64

    def test_min_i64_accepted(self) -> None:
        """Minimum signed 64-bit integer is accepted."""
        min_i64 = -(2**63)
        assert parse_length_nm(min_i64) == min_i64

    def test_overflow_fails(self) -> None:
        """Values exceeding 64-bit range must fail."""
        with pytest.raises(ValueError, match="64-bit"):
            parse_length_nm(2**63)

    def test_underflow_fails(self) -> None:
        """Values below 64-bit range must fail."""
        with pytest.raises(ValueError, match="64-bit"):
            parse_length_nm(-(2**63) - 1)


# =============================================================================
# Unit Parsing Tests: AngleMdeg
# =============================================================================


class TestAngleMdegParsing:
    """Tests for parsing angles to integer millidegrees."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("45deg", 45_000),
            ("90deg", 90_000),
            ("-180deg", -180_000),
            ("0.5deg", 500),
            ("1000mdeg", 1_000),
            ("1mdeg", 1),
        ],
    )
    def test_string_with_units(self, value: str, expected: int) -> None:
        """String values with angle units must parse correctly."""
        assert parse_angle_mdeg(value) == expected

    def test_raw_integer_passthrough(self) -> None:
        """Raw integers pass through as millidegrees."""
        assert parse_angle_mdeg(45_000) == 45_000
        assert parse_angle_mdeg(0) == 0
        assert parse_angle_mdeg(-90_000) == -90_000

    def test_returns_int_type(self) -> None:
        """Parser must always return Python int."""
        assert isinstance(parse_angle_mdeg("45deg"), int)
        assert isinstance(parse_angle_mdeg(1000), int)

    def test_case_insensitive_units(self) -> None:
        """Units are case-insensitive."""
        assert parse_angle_mdeg("45DEG") == 45_000
        assert parse_angle_mdeg("1000MDEG") == 1_000


class TestAngleMdegErrors:
    """Tests for AngleMdeg error handling."""

    def test_unknown_unit_raises(self) -> None:
        """Unknown angle units must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown AngleMdeg unit"):
            parse_angle_mdeg("45rad")

    def test_boolean_rejected(self) -> None:
        """Boolean values must be rejected."""
        with pytest.raises(ValueError, match="boolean"):
            parse_angle_mdeg(True)  # type: ignore[arg-type]

    def test_empty_string_fails(self) -> None:
        """Empty string must fail."""
        with pytest.raises(ValueError, match="numeric value"):
            parse_angle_mdeg("")


# =============================================================================
# Unit Parsing Tests: FrequencyHz
# =============================================================================


class TestFrequencyHzParsing:
    """Tests for parsing frequencies to integer Hz."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1000Hz", 1_000),
            ("1kHz", 1_000),
            ("50kHz", 50_000),
            ("100MHz", 100_000_000),
            ("1GHz", 1_000_000_000),
            ("2.4GHz", 2_400_000_000),
        ],
    )
    def test_string_with_units(self, value: str, expected: int) -> None:
        """String values with frequency units must parse correctly."""
        assert parse_frequency_hz(value) == expected

    def test_raw_integer_passthrough(self) -> None:
        """Raw integers pass through as Hz."""
        assert parse_frequency_hz(1_000_000_000) == 1_000_000_000
        assert parse_frequency_hz(0) == 0

    def test_case_insensitive_units(self) -> None:
        """Units are case-insensitive."""
        assert parse_frequency_hz("1ghz") == 1_000_000_000
        assert parse_frequency_hz("1GHZ") == 1_000_000_000
        assert parse_frequency_hz("100mhz") == 100_000_000


class TestFrequencyHzErrors:
    """Tests for FrequencyHz error handling."""

    def test_unknown_unit_raises(self) -> None:
        """Unknown frequency units must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown FrequencyHz unit"):
            parse_frequency_hz("100THz")

    def test_boolean_rejected(self) -> None:
        """Boolean values must be rejected."""
        with pytest.raises(ValueError, match="boolean"):
            parse_frequency_hz(True)  # type: ignore[arg-type]

    def test_empty_string_fails(self) -> None:
        """Empty string must fail."""
        with pytest.raises(ValueError, match="numeric value"):
            parse_frequency_hz("")


# =============================================================================
# Pydantic Type Integration Tests
# =============================================================================


class TestPydanticTypeIntegration:
    """Tests for Pydantic annotated types (LengthNM, AngleMdeg, FrequencyHz)."""

    def test_lengthnm_in_model(self) -> None:
        """LengthNM type works in Pydantic models."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            width_nm: LengthNM

        m = TestModel(width_nm="1mm")  # type: ignore[arg-type]
        assert m.width_nm == 1_000_000

        m2 = TestModel(width_nm=500_000)
        assert m2.width_nm == 500_000

    def test_lengthnm_validation_error(self) -> None:
        """LengthNM with invalid unit raises ValidationError."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            width_nm: LengthNM

        with pytest.raises(ValidationError):
            TestModel(width_nm="10feet")  # type: ignore[arg-type]

    def test_anglemdeg_in_model(self) -> None:
        """AngleMdeg type works in Pydantic models."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            angle: AngleMdeg

        m = TestModel(angle="45deg")  # type: ignore[arg-type]
        assert m.angle == 45_000

    def test_frequencyhz_in_model(self) -> None:
        """FrequencyHz type works in Pydantic models."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            freq: FrequencyHz

        m = TestModel(freq="1GHz")  # type: ignore[arg-type]
        assert m.freq == 1_000_000_000


# =============================================================================
# CouponSpec Unit String Normalization Tests
# =============================================================================


class TestCouponSpecUnitNormalization:
    """Tests for unit string normalization in CouponSpec."""

    def test_unit_strings_normalize_to_nm(self, valid_spec_data: dict[str, Any]) -> None:
        """REQ-M1-001: CouponSpec must support mm/mil/um unit strings that normalize to nm."""
        # Replace integer nm values with unit strings
        valid_spec_data["board"]["outline"]["width_nm"] = "20mm"
        valid_spec_data["board"]["outline"]["length_nm"] = "80mm"
        valid_spec_data["board"]["outline"]["corner_radius_nm"] = "2mm"
        valid_spec_data["transmission_line"]["w_nm"] = "0.3mm"
        valid_spec_data["transmission_line"]["gap_nm"] = "7.09mil"

        spec = load_couponspec(valid_spec_data)
        assert spec.board.outline.width_nm == 20_000_000
        assert spec.board.outline.length_nm == 80_000_000
        assert spec.board.outline.corner_radius_nm == 2_000_000
        assert spec.transmission_line.w_nm == 300_000
        # 7.09mil = 7.09 * 25400 = 180086 nm
        assert spec.transmission_line.gap_nm == 180086

    def test_mixed_integer_and_string_units(self, valid_spec_data: dict[str, Any]) -> None:
        """CouponSpec should accept mix of integer nm and string units."""
        valid_spec_data["board"]["outline"]["width_nm"] = 20_000_000  # integer nm
        valid_spec_data["board"]["outline"]["length_nm"] = "80mm"  # string mm

        spec = load_couponspec(valid_spec_data)
        assert spec.board.outline.width_nm == 20_000_000
        assert spec.board.outline.length_nm == 80_000_000
