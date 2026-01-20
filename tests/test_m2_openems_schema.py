"""Tests for M2 openEMS simulation config schema.

These tests validate:
- Schema validation (strict mode, no extra fields)
- Unit parsing for frequency/time values
- JSON schema generation
- Round-trip serialization
- Edge cases and error handling
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from formula_foundry.openems import (
    SIMULATIONSPEC_SCHEMA,
    BoundarySpec,
    ExcitationSpec,
    GeometryRefSpec,
    MeshSpec,
    OpenEMSToolchainSpec,
    PortSpec,
    SimulationSpec,
    load_simulationspec,
)
from formula_foundry.openems.units import parse_frequency_hz, parse_time_ps

# =============================================================================
# Unit Parsing Tests
# =============================================================================


class TestFrequencyParsing:
    """Tests for FrequencyHz parsing."""

    def test_parse_integer_hz(self) -> None:
        assert parse_frequency_hz(1_000_000_000) == 1_000_000_000

    def test_parse_string_integer(self) -> None:
        assert parse_frequency_hz("1000000000") == 1_000_000_000

    def test_parse_ghz_string(self) -> None:
        assert parse_frequency_hz("1GHz") == 1_000_000_000

    def test_parse_mhz_string(self) -> None:
        assert parse_frequency_hz("500MHz") == 500_000_000

    def test_parse_khz_string(self) -> None:
        assert parse_frequency_hz("100kHz") == 100_000

    def test_parse_hz_string(self) -> None:
        assert parse_frequency_hz("50Hz") == 50

    def test_parse_ghz_decimal(self) -> None:
        assert parse_frequency_hz("2.5GHz") == 2_500_000_000

    def test_parse_mhz_lowercase(self) -> None:
        assert parse_frequency_hz("100mhz") == 100_000_000

    def test_parse_float_value(self) -> None:
        assert parse_frequency_hz(1.5e9) == 1_500_000_000

    def test_reject_boolean(self) -> None:
        with pytest.raises(ValueError, match="does not accept boolean"):
            parse_frequency_hz(True)  # type: ignore[arg-type]

    def test_reject_empty_string(self) -> None:
        with pytest.raises(ValueError, match="requires a numeric value"):
            parse_frequency_hz("")

    def test_reject_invalid_unit(self) -> None:
        with pytest.raises(ValueError, match="Unknown FrequencyHz unit"):
            parse_frequency_hz("100THz")

    def test_reject_non_integer_result(self) -> None:
        # 1.5 Hz cannot resolve to an integer
        with pytest.raises(ValueError, match="must resolve to an integer"):
            parse_frequency_hz("1.5Hz")


class TestTimeParsing:
    """Tests for TimePS parsing."""

    def test_parse_integer_ps(self) -> None:
        assert parse_time_ps(1000) == 1000

    def test_parse_ns_string(self) -> None:
        assert parse_time_ps("1ns") == 1000

    def test_parse_us_string(self) -> None:
        assert parse_time_ps("1us") == 1_000_000

    def test_parse_ms_string(self) -> None:
        assert parse_time_ps("1ms") == 1_000_000_000

    def test_parse_s_string(self) -> None:
        assert parse_time_ps("1s") == 1_000_000_000_000

    def test_parse_ps_string(self) -> None:
        assert parse_time_ps("100ps") == 100

    def test_reject_invalid_unit(self) -> None:
        with pytest.raises(ValueError, match="Unknown TimePS unit"):
            parse_time_ps("100fs")  # femtoseconds not supported


# =============================================================================
# Spec Model Tests
# =============================================================================


def _minimal_simulation_spec_dict() -> dict:
    """Return a minimal valid SimulationSpec dictionary."""
    return {
        "schema_version": 1,
        "toolchain": {
            "openems": {
                "version": "0.0.35",
                "docker_image": "ghcr.io/thliebig/openems:0.0.35@sha256:abc123",
            }
        },
        "geometry_ref": {
            "design_hash": "abc123def456",
        },
        "excitation": {
            "f0_hz": "5GHz",
            "fc_hz": "10GHz",
        },
        "frequency": {
            "f_start_hz": "1MHz",
            "f_stop_hz": "20GHz",
            "n_points": 401,
        },
        "ports": [
            {
                "id": "port1",
                "position_nm": [0, 0, 0],
                "direction": "x",
                "excite": True,
            },
            {
                "id": "port2",
                "position_nm": [10_000_000, 0, 0],
                "direction": "-x",
            },
        ],
    }


class TestSimulationSpecValidation:
    """Tests for SimulationSpec validation."""

    def test_minimal_spec_valid(self) -> None:
        """A minimal spec with required fields should validate."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        assert spec.schema_version == 1
        assert spec.toolchain.openems.version == "0.0.35"
        assert len(spec.ports) == 2

    def test_frequency_unit_conversion(self) -> None:
        """Frequency strings should be converted to Hz integers."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        assert spec.excitation.f0_hz == 5_000_000_000
        assert spec.excitation.fc_hz == 10_000_000_000
        assert spec.frequency.f_start_hz == 1_000_000
        assert spec.frequency.f_stop_hz == 20_000_000_000

    def test_extra_fields_rejected(self) -> None:
        """Extra fields should be rejected (strict mode)."""
        data = _minimal_simulation_spec_dict()
        data["unknown_field"] = "should fail"
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            load_simulationspec(data)

    def test_nested_extra_fields_rejected(self) -> None:
        """Extra fields in nested objects should be rejected."""
        data = _minimal_simulation_spec_dict()
        data["excitation"]["extra"] = "bad"
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            load_simulationspec(data)

    def test_missing_required_field_rejected(self) -> None:
        """Missing required fields should be rejected."""
        data = _minimal_simulation_spec_dict()
        del data["ports"]
        with pytest.raises(ValidationError, match="Field required"):
            load_simulationspec(data)

    def test_empty_ports_rejected(self) -> None:
        """At least one port is required."""
        data = _minimal_simulation_spec_dict()
        data["ports"] = []
        with pytest.raises(ValidationError, match="at least 1"):
            load_simulationspec(data)

    def test_defaults_applied(self) -> None:
        """Default values should be applied for optional fields."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        assert spec.boundaries.x_min == "PML_8"
        assert spec.mesh.resolution.lambda_resolution == 20
        assert spec.control.termination.end_criteria_db == -50.0
        assert spec.output.s_params is True


class TestBoundarySpec:
    """Tests for BoundarySpec."""

    def test_all_boundary_types(self) -> None:
        """All boundary types should be valid."""
        for bc in ["PEC", "PMC", "MUR", "PML_8", "PML_16", "PML_32"]:
            spec = BoundarySpec(x_min=bc, x_max=bc, y_min=bc, y_max=bc, z_min=bc, z_max=bc)
            assert spec.x_min == bc

    def test_invalid_boundary_type_rejected(self) -> None:
        """Invalid boundary type should be rejected."""
        with pytest.raises(ValidationError):
            BoundarySpec(x_min="INVALID")  # type: ignore[arg-type]


class TestMeshSpec:
    """Tests for MeshSpec."""

    def test_mesh_with_fixed_lines(self) -> None:
        """Fixed mesh lines should accept length values."""
        data = {
            "fixed_lines_x_nm": [0, "1mm", "10mm"],
            "fixed_lines_y_nm": ["0.5mm"],
            "fixed_lines_z_nm": [],
        }
        spec = MeshSpec.model_validate(data)
        assert spec.fixed_lines_x_nm == [0, 1_000_000, 10_000_000]
        assert spec.fixed_lines_y_nm == [500_000]

    def test_resolution_bounds(self) -> None:
        """Lambda resolution should be within bounds."""
        with pytest.raises(ValidationError, match="greater than or equal to 10"):
            MeshSpec(resolution={"lambda_resolution": 5})


class TestPortSpec:
    """Tests for PortSpec."""

    def test_port_with_all_fields(self) -> None:
        """Port with all optional fields should be valid."""
        port = PortSpec(
            id="port1",
            type="msl",
            impedance_ohm=50.0,
            excite=True,
            position_nm=("1mm", "0.5mm", "0"),
            direction="x",
            width_nm="0.3mm",
            height_nm="0.2mm",
        )
        assert port.position_nm == (1_000_000, 500_000, 0)
        assert port.width_nm == 300_000
        assert port.height_nm == 200_000

    def test_port_direction_variants(self) -> None:
        """All direction variants should be valid."""
        for direction in ["x", "y", "z", "-x", "-y", "-z"]:
            port = PortSpec(
                id="test",
                position_nm=(0, 0, 0),
                direction=direction,
            )
            assert port.direction == direction


# =============================================================================
# JSON Schema Tests
# =============================================================================


class TestJsonSchema:
    """Tests for JSON schema generation."""

    def test_schema_is_dict(self) -> None:
        """Generated schema should be a valid dictionary."""
        assert isinstance(SIMULATIONSPEC_SCHEMA, dict)

    def test_schema_has_properties(self) -> None:
        """Schema should define properties."""
        assert "properties" in SIMULATIONSPEC_SCHEMA
        assert "schema_version" in SIMULATIONSPEC_SCHEMA["properties"]
        assert "toolchain" in SIMULATIONSPEC_SCHEMA["properties"]

    def test_schema_has_required_fields(self) -> None:
        """Schema should list required fields."""
        assert "required" in SIMULATIONSPEC_SCHEMA
        required = SIMULATIONSPEC_SCHEMA["required"]
        assert "toolchain" in required
        assert "geometry_ref" in required
        assert "ports" in required

    def test_schema_serializable(self) -> None:
        """Schema should be JSON-serializable."""
        json_str = json.dumps(SIMULATIONSPEC_SCHEMA)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert parsed == SIMULATIONSPEC_SCHEMA


# =============================================================================
# Round-trip Tests
# =============================================================================


class TestRoundTrip:
    """Tests for serialization round-trip."""

    def test_dict_to_spec_to_dict(self) -> None:
        """Spec should round-trip through dict serialization."""
        original = _minimal_simulation_spec_dict()
        spec = load_simulationspec(original)
        serialized = spec.model_dump()

        # Reload and compare
        reloaded = load_simulationspec(serialized)
        assert spec == reloaded

    def test_json_roundtrip(self) -> None:
        """Spec should round-trip through JSON serialization."""
        original = _minimal_simulation_spec_dict()
        spec = load_simulationspec(original)

        json_str = spec.model_dump_json()
        reloaded = SimulationSpec.model_validate_json(json_str)
        assert spec == reloaded


# =============================================================================
# Integration Tests
# =============================================================================


class TestSchemaVersioning:
    """Tests for schema versioning."""

    def test_schema_version_required(self) -> None:
        """Schema version is included with default value."""
        data = _minimal_simulation_spec_dict()
        del data["schema_version"]
        spec = load_simulationspec(data)
        assert spec.schema_version == 1

    def test_invalid_schema_version_rejected(self) -> None:
        """Schema version must be >= 1."""
        data = _minimal_simulation_spec_dict()
        data["schema_version"] = 0
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            load_simulationspec(data)


class TestToolchainValidation:
    """Tests for toolchain specification."""

    def test_toolchain_version_required(self) -> None:
        """Toolchain version must be non-empty."""
        with pytest.raises(ValidationError, match="at least 1"):
            OpenEMSToolchainSpec(version="", docker_image="image:tag")

    def test_docker_image_required(self) -> None:
        """Docker image must be non-empty."""
        with pytest.raises(ValidationError, match="at least 1"):
            OpenEMSToolchainSpec(version="0.0.35", docker_image="")


class TestExcitationSpec:
    """Tests for excitation specification."""

    def test_gaussian_default(self) -> None:
        """Default excitation type is gaussian."""
        exc = ExcitationSpec(f0_hz=1_000_000_000, fc_hz=5_000_000_000)
        assert exc.type == "gaussian"

    def test_sinusoidal_type(self) -> None:
        """Sinusoidal excitation type should be valid."""
        exc = ExcitationSpec(type="sinusoidal", f0_hz="1GHz", fc_hz="5GHz")
        assert exc.type == "sinusoidal"


class TestGeometryRef:
    """Tests for geometry reference."""

    def test_design_hash_required(self) -> None:
        """Design hash is required."""
        with pytest.raises(ValidationError, match="Field required"):
            GeometryRefSpec()  # type: ignore[call-arg]

    def test_coupon_id_optional(self) -> None:
        """Coupon ID is optional."""
        ref = GeometryRefSpec(design_hash="abc123")
        assert ref.coupon_id is None

        ref_with_id = GeometryRefSpec(design_hash="abc123", coupon_id="COUPON001")
        assert ref_with_id.coupon_id == "COUPON001"
