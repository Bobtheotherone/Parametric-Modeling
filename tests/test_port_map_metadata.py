"""Tests for port map metadata recording.

REQ-M2-013: Port map records mapping, orientation, reference plane, and backend.

This module tests that port map metadata:
- Records port mapping (port_id, port_index, position)
- Records orientation (cardinal direction)
- Records reference plane (enabled, distance, epsilon_r_eff, method)
- Records backend (openems, gerber2ems, etc.)
- Can be serialized into meta.json format
- Validates required fields and types

Test matrix coverage:
| REQ-M2-013 | test_port_map_records_mapping_orientation_reference_plane_and_backend |
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from formula_foundry.ports.metadata import (
    PortMapEntry,
    PortMapMetadata,
    PortOrientation,
    ReferencePlaneSpec,
    validate_port_map_metadata,
)


class TestPortOrientation:
    """Tests for PortOrientation enum."""

    def test_orientation_values(self) -> None:
        """Verify all cardinal orientations are available."""
        assert PortOrientation.X_POSITIVE.value == "+x"
        assert PortOrientation.X_NEGATIVE.value == "-x"
        assert PortOrientation.Y_POSITIVE.value == "+y"
        assert PortOrientation.Y_NEGATIVE.value == "-y"
        assert PortOrientation.Z_POSITIVE.value == "+z"
        assert PortOrientation.Z_NEGATIVE.value == "-z"

    def test_from_string_shorthand(self) -> None:
        """Test parsing shorthand orientation strings."""
        assert PortOrientation.from_string("x") == PortOrientation.X_POSITIVE
        assert PortOrientation.from_string("y") == PortOrientation.Y_POSITIVE
        assert PortOrientation.from_string("z") == PortOrientation.Z_POSITIVE

    def test_from_string_explicit(self) -> None:
        """Test parsing explicit signed orientation strings."""
        assert PortOrientation.from_string("+x") == PortOrientation.X_POSITIVE
        assert PortOrientation.from_string("-x") == PortOrientation.X_NEGATIVE
        assert PortOrientation.from_string("+y") == PortOrientation.Y_POSITIVE
        assert PortOrientation.from_string("-y") == PortOrientation.Y_NEGATIVE
        assert PortOrientation.from_string("+z") == PortOrientation.Z_POSITIVE
        assert PortOrientation.from_string("-z") == PortOrientation.Z_NEGATIVE

    def test_from_string_case_insensitive(self) -> None:
        """Test that orientation parsing is case insensitive."""
        assert PortOrientation.from_string("X") == PortOrientation.X_POSITIVE
        assert PortOrientation.from_string("+X") == PortOrientation.X_POSITIVE
        assert PortOrientation.from_string("-X") == PortOrientation.X_NEGATIVE

    def test_from_string_invalid(self) -> None:
        """Test that invalid orientation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid port orientation"):
            PortOrientation.from_string("w")
        with pytest.raises(ValueError, match="Invalid port orientation"):
            PortOrientation.from_string("diagonal")


class TestReferencePlaneSpec:
    """Tests for ReferencePlaneSpec dataclass."""

    def test_default_values(self) -> None:
        """Test default reference plane is disabled."""
        ref_plane = ReferencePlaneSpec()
        assert ref_plane.enabled is False
        assert ref_plane.distance_nm == 0
        assert ref_plane.epsilon_r_eff is None
        assert ref_plane.method == "none"

    def test_enabled_reference_plane(self) -> None:
        """Test enabled reference plane with all fields."""
        ref_plane = ReferencePlaneSpec(
            enabled=True,
            distance_nm=500_000,
            epsilon_r_eff=3.5,
            method="reference_plane",
        )
        assert ref_plane.enabled is True
        assert ref_plane.distance_nm == 500_000
        assert ref_plane.epsilon_r_eff == 3.5
        assert ref_plane.method == "reference_plane"

    def test_to_dict_disabled(self) -> None:
        """Test serialization of disabled reference plane."""
        ref_plane = ReferencePlaneSpec()
        result = ref_plane.to_dict()

        assert result["enabled"] is False
        assert result["method"] == "none"
        # distance_nm should not be present when disabled
        assert "distance_nm" not in result

    def test_to_dict_enabled(self) -> None:
        """Test serialization of enabled reference plane."""
        ref_plane = ReferencePlaneSpec(
            enabled=True,
            distance_nm=100_000,
            epsilon_r_eff=2.8,
            method="reference_plane",
        )
        result = ref_plane.to_dict()

        assert result["enabled"] is True
        assert result["method"] == "reference_plane"
        assert result["distance_nm"] == 100_000
        assert result["epsilon_r_eff"] == 2.8


class TestPortMapEntry:
    """Tests for PortMapEntry dataclass."""

    def test_minimal_port_entry(self) -> None:
        """Test creating port entry with minimal required fields."""
        port = PortMapEntry(
            port_id="P1",
            port_index=0,
            port_type="waveguide",
            position_nm=(0, 0, 100_000),
            orientation=PortOrientation.X_POSITIVE,
        )
        assert port.port_id == "P1"
        assert port.port_index == 0
        assert port.port_type == "waveguide"
        assert port.position_nm == (0, 0, 100_000)
        assert port.orientation == PortOrientation.X_POSITIVE
        assert port.impedance_ohm == 50.0
        assert port.excite is False

    def test_full_port_entry(self) -> None:
        """Test creating port entry with all fields."""
        ref_plane = ReferencePlaneSpec(
            enabled=True,
            distance_nm=500_000,
            epsilon_r_eff=3.2,
            method="reference_plane",
        )
        port = PortMapEntry(
            port_id="P1",
            port_index=0,
            port_type="cpwg",
            position_nm=(-1_000_000, 0, 150_000),
            orientation=PortOrientation.X_POSITIVE,
            impedance_ohm=50.0,
            excite=True,
            reference_plane=ref_plane,
            layer_id="L1",
            signal_width_nm=200_000,
            gap_nm=100_000,
        )
        assert port.port_id == "P1"
        assert port.layer_id == "L1"
        assert port.signal_width_nm == 200_000
        assert port.gap_nm == 100_000

    def test_to_dict_includes_required_fields(self) -> None:
        """Test that to_dict includes all required fields."""
        port = PortMapEntry(
            port_id="P1",
            port_index=0,
            port_type="waveguide",
            position_nm=(100, 200, 300),
            orientation=PortOrientation.X_NEGATIVE,
        )
        result = port.to_dict()

        # Required fields
        assert "port_id" in result
        assert "port_index" in result
        assert "port_type" in result
        assert "position_nm" in result
        assert "orientation" in result
        assert "impedance_ohm" in result
        assert "excite" in result
        assert "reference_plane" in result

        # Check types
        assert isinstance(result["port_id"], str)
        assert isinstance(result["port_index"], int)
        assert isinstance(result["port_type"], str)
        assert isinstance(result["position_nm"], list)
        assert isinstance(result["orientation"], str)
        assert isinstance(result["impedance_ohm"], float)
        assert isinstance(result["excite"], bool)
        assert isinstance(result["reference_plane"], dict)

    def test_to_dict_optional_fields(self) -> None:
        """Test that to_dict includes optional fields only when set."""
        port_without_optional = PortMapEntry(
            port_id="P1",
            port_index=0,
            port_type="waveguide",
            position_nm=(0, 0, 0),
            orientation=PortOrientation.X_POSITIVE,
        )
        result = port_without_optional.to_dict()
        assert "layer_id" not in result
        assert "signal_width_nm" not in result
        assert "gap_nm" not in result

        port_with_optional = PortMapEntry(
            port_id="P1",
            port_index=0,
            port_type="cpwg",
            position_nm=(0, 0, 0),
            orientation=PortOrientation.X_POSITIVE,
            layer_id="L1",
            signal_width_nm=200_000,
            gap_nm=100_000,
        )
        result = port_with_optional.to_dict()
        assert result["layer_id"] == "L1"
        assert result["signal_width_nm"] == 200_000
        assert result["gap_nm"] == 100_000


class TestPortMapMetadata:
    """Tests for PortMapMetadata dataclass."""

    def _create_two_port_metadata(self) -> PortMapMetadata:
        """Helper to create a standard two-port metadata."""
        port1 = PortMapEntry(
            port_id="P1",
            port_index=0,
            port_type="waveguide",
            position_nm=(-1_000_000, 0, 100_000),
            orientation=PortOrientation.X_POSITIVE,
            excite=True,
            reference_plane=ReferencePlaneSpec(
                enabled=True,
                distance_nm=500_000,
                method="reference_plane",
            ),
        )
        port2 = PortMapEntry(
            port_id="P2",
            port_index=1,
            port_type="waveguide",
            position_nm=(1_000_000, 0, 100_000),
            orientation=PortOrientation.X_NEGATIVE,
            excite=False,
            reference_plane=ReferencePlaneSpec(
                enabled=True,
                distance_nm=500_000,
                method="reference_plane",
            ),
        )
        return PortMapMetadata(
            ports=[port1, port2],
            backend="openems",
            design_hash="abc123",
            simulation_id="sim-001",
        )

    def test_port_count(self) -> None:
        """Test port_count property."""
        metadata = self._create_two_port_metadata()
        assert metadata.port_count == 2

    def test_get_port_by_id(self) -> None:
        """Test getting port by ID."""
        metadata = self._create_two_port_metadata()
        port = metadata.get_port("P1")
        assert port is not None
        assert port.port_id == "P1"
        assert port.excite is True

        # Non-existent port
        assert metadata.get_port("P99") is None

    def test_get_port_by_index(self) -> None:
        """Test getting port by index."""
        metadata = self._create_two_port_metadata()
        port = metadata.get_port_by_index(1)
        assert port is not None
        assert port.port_id == "P2"
        assert port.port_index == 1

        # Non-existent index
        assert metadata.get_port_by_index(99) is None

    def test_to_dict_structure(self) -> None:
        """Test to_dict produces correct structure."""
        metadata = self._create_two_port_metadata()
        result = metadata.to_dict()

        # Required fields
        assert "schema_version" in result
        assert "backend" in result
        assert "port_count" in result
        assert "created_utc" in result
        assert "ports" in result

        # Types
        assert isinstance(result["schema_version"], int)
        assert isinstance(result["backend"], str)
        assert isinstance(result["port_count"], int)
        assert isinstance(result["created_utc"], str)
        assert isinstance(result["ports"], list)

        # Values
        assert result["schema_version"] == 1
        assert result["backend"] == "openems"
        assert result["port_count"] == 2
        assert len(result["ports"]) == 2

    def test_to_dict_optional_fields(self) -> None:
        """Test that optional fields are included when set."""
        metadata = self._create_two_port_metadata()
        result = metadata.to_dict()

        assert result["design_hash"] == "abc123"
        assert result["simulation_id"] == "sim-001"

    def test_to_meta_json(self) -> None:
        """Test to_meta_json produces meta.json compatible output."""
        metadata = self._create_two_port_metadata()
        result = metadata.to_meta_json()

        assert "port_map" in result
        assert result["port_map"]["backend"] == "openems"
        assert result["port_map"]["port_count"] == 2

    def test_json_serialization(self) -> None:
        """Test that to_dict output is JSON serializable."""
        metadata = self._create_two_port_metadata()
        result = metadata.to_dict()

        # Should not raise
        json_str = json.dumps(result, indent=2, sort_keys=True)
        assert isinstance(json_str, str)

        # Round-trip should work
        parsed = json.loads(json_str)
        assert parsed["backend"] == "openems"
        assert len(parsed["ports"]) == 2

    def test_write_and_load_json(self) -> None:
        """Test writing and loading from JSON file."""
        metadata = self._create_two_port_metadata()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "port_map.json"
            metadata.write_json(path)

            # File should exist
            assert path.exists()

            # Load and compare
            loaded = PortMapMetadata.load_json(path)
            assert loaded.port_count == metadata.port_count
            assert loaded.backend == metadata.backend
            assert loaded.design_hash == metadata.design_hash

            # Check ports are preserved
            p1 = loaded.get_port("P1")
            assert p1 is not None
            assert p1.orientation == PortOrientation.X_POSITIVE
            assert p1.reference_plane.enabled is True

    def test_from_dict(self) -> None:
        """Test creating metadata from dictionary."""
        data = {
            "schema_version": 1,
            "backend": "gerber2ems",
            "port_count": 1,
            "created_utc": "2024-01-01T00:00:00Z",
            "ports": [
                {
                    "port_id": "P1",
                    "port_index": 0,
                    "port_type": "cpwg",
                    "position_nm": [0, 0, 100_000],
                    "orientation": "+x",
                    "impedance_ohm": 50.0,
                    "excite": True,
                    "reference_plane": {
                        "enabled": True,
                        "method": "reference_plane",
                        "distance_nm": 250_000,
                    },
                    "layer_id": "L1",
                }
            ],
        }

        metadata = PortMapMetadata.from_dict(data)
        assert metadata.backend == "gerber2ems"
        assert metadata.port_count == 1

        port = metadata.get_port("P1")
        assert port is not None
        assert port.layer_id == "L1"
        assert port.orientation == PortOrientation.X_POSITIVE
        assert port.reference_plane.enabled is True


class TestValidation:
    """Tests for port map validation."""

    def test_valid_metadata_passes(self) -> None:
        """Test that valid metadata has no errors."""
        port = PortMapEntry(
            port_id="P1",
            port_index=0,
            port_type="waveguide",
            position_nm=(0, 0, 0),
            orientation=PortOrientation.X_POSITIVE,
        )
        metadata = PortMapMetadata(ports=[port], backend="openems")

        errors = validate_port_map_metadata(metadata)
        assert errors == []

    def test_empty_ports_fails(self) -> None:
        """Test that empty port list fails validation."""
        metadata = PortMapMetadata(ports=[], backend="openems")

        errors = validate_port_map_metadata(metadata)
        assert any("at least one port" in e for e in errors)

    def test_missing_backend_fails(self) -> None:
        """Test that missing backend fails validation."""
        port = PortMapEntry(
            port_id="P1",
            port_index=0,
            port_type="waveguide",
            position_nm=(0, 0, 0),
            orientation=PortOrientation.X_POSITIVE,
        )
        metadata = PortMapMetadata(ports=[port], backend="")

        errors = validate_port_map_metadata(metadata)
        assert any("Backend must be specified" in e for e in errors)

    def test_duplicate_port_id_fails(self) -> None:
        """Test that duplicate port IDs fail validation."""
        port1 = PortMapEntry(
            port_id="P1",
            port_index=0,
            port_type="waveguide",
            position_nm=(0, 0, 0),
            orientation=PortOrientation.X_POSITIVE,
        )
        port2 = PortMapEntry(
            port_id="P1",  # Duplicate ID
            port_index=1,
            port_type="waveguide",
            position_nm=(100, 0, 0),
            orientation=PortOrientation.X_NEGATIVE,
        )
        metadata = PortMapMetadata(ports=[port1, port2], backend="openems")

        errors = validate_port_map_metadata(metadata)
        assert any("Duplicate port_id: P1" in e for e in errors)

    def test_duplicate_port_index_fails(self) -> None:
        """Test that duplicate port indices fail validation."""
        port1 = PortMapEntry(
            port_id="P1",
            port_index=0,
            port_type="waveguide",
            position_nm=(0, 0, 0),
            orientation=PortOrientation.X_POSITIVE,
        )
        port2 = PortMapEntry(
            port_id="P2",
            port_index=0,  # Duplicate index
            port_type="waveguide",
            position_nm=(100, 0, 0),
            orientation=PortOrientation.X_NEGATIVE,
        )
        metadata = PortMapMetadata(ports=[port1, port2], backend="openems")

        errors = validate_port_map_metadata(metadata)
        assert any("Duplicate port_index: 0" in e for e in errors)

    def test_non_sequential_indices_fails(self) -> None:
        """Test that non-sequential port indices fail validation."""
        port1 = PortMapEntry(
            port_id="P1",
            port_index=0,
            port_type="waveguide",
            position_nm=(0, 0, 0),
            orientation=PortOrientation.X_POSITIVE,
        )
        port2 = PortMapEntry(
            port_id="P2",
            port_index=2,  # Should be 1
            port_type="waveguide",
            position_nm=(100, 0, 0),
            orientation=PortOrientation.X_NEGATIVE,
        )
        metadata = PortMapMetadata(ports=[port1, port2], backend="openems")

        errors = validate_port_map_metadata(metadata)
        assert any("sequential" in e for e in errors)


class TestPortMapRecordsRequiredFields:
    """REQ-M2-013: Port map records mapping, orientation, reference plane, and backend.

    This test class verifies the main requirement that port maps record all
    required information for auditability and can be serialized into meta.json.
    """

    def test_port_map_records_mapping_orientation_reference_plane_and_backend(
        self,
    ) -> None:
        """REQ-M2-013: Verify port map records all required fields.

        This is the primary test for REQ-M2-013. It verifies that:
        1. Mapping: port_id, port_index, position are recorded
        2. Orientation: cardinal direction is recorded
        3. Reference plane: de-embedding config is recorded
        4. Backend: geometry backend is recorded
        5. Output: can be serialized into meta.json format
        """
        # Create port with all required metadata
        ref_plane = ReferencePlaneSpec(
            enabled=True,
            distance_nm=500_000,
            epsilon_r_eff=3.5,
            method="reference_plane",
        )
        port = PortMapEntry(
            port_id="P1",
            port_index=0,
            port_type="waveguide",
            position_nm=(-1_000_000, 0, 150_000),
            orientation=PortOrientation.X_POSITIVE,
            impedance_ohm=50.0,
            excite=True,
            reference_plane=ref_plane,
            layer_id="L1",
        )

        # Create port map with backend
        metadata = PortMapMetadata(
            ports=[port],
            backend="openems",
            design_hash="abc123def456",
            simulation_id="sim-test-001",
        )

        # Validate no errors
        errors = validate_port_map_metadata(metadata)
        assert errors == [], f"Validation errors: {errors}"

        # Serialize to dict
        data = metadata.to_dict()

        # 1. Verify MAPPING is recorded
        assert data["port_count"] == 1
        port_data = data["ports"][0]
        assert port_data["port_id"] == "P1"
        assert port_data["port_index"] == 0
        assert port_data["position_nm"] == [-1_000_000, 0, 150_000]

        # 2. Verify ORIENTATION is recorded
        assert port_data["orientation"] == "+x"
        assert port_data["orientation"] in ["+x", "-x", "+y", "-y", "+z", "-z"]

        # 3. Verify REFERENCE PLANE is recorded
        ref_data = port_data["reference_plane"]
        assert ref_data["enabled"] is True
        assert ref_data["distance_nm"] == 500_000
        assert ref_data["epsilon_r_eff"] == 3.5
        assert ref_data["method"] == "reference_plane"

        # 4. Verify BACKEND is recorded
        assert data["backend"] == "openems"

        # 5. Verify can be serialized into meta.json format
        meta_json = metadata.to_meta_json()
        assert "port_map" in meta_json

        # Verify JSON serializable
        json_str = json.dumps(meta_json, indent=2, sort_keys=True)
        assert isinstance(json_str, str)

        # Verify round-trip
        parsed = json.loads(json_str)
        assert parsed["port_map"]["backend"] == "openems"
        assert parsed["port_map"]["ports"][0]["orientation"] == "+x"
        assert parsed["port_map"]["ports"][0]["reference_plane"]["enabled"] is True

    def test_port_map_required_fields_types(self) -> None:
        """Verify all required fields have correct types."""
        port = PortMapEntry(
            port_id="P1",
            port_index=0,
            port_type="waveguide",
            position_nm=(0, 0, 0),
            orientation=PortOrientation.X_POSITIVE,
        )
        metadata = PortMapMetadata(ports=[port], backend="openems")
        data = metadata.to_dict()

        # Top-level field types
        assert isinstance(data["schema_version"], int)
        assert isinstance(data["backend"], str)
        assert isinstance(data["port_count"], int)
        assert isinstance(data["created_utc"], str)
        assert isinstance(data["ports"], list)

        # Port field types
        port_data = data["ports"][0]
        assert isinstance(port_data["port_id"], str)
        assert isinstance(port_data["port_index"], int)
        assert isinstance(port_data["port_type"], str)
        assert isinstance(port_data["position_nm"], list)
        assert len(port_data["position_nm"]) == 3
        assert all(isinstance(x, int) for x in port_data["position_nm"])
        assert isinstance(port_data["orientation"], str)
        assert isinstance(port_data["impedance_ohm"], float)
        assert isinstance(port_data["excite"], bool)
        assert isinstance(port_data["reference_plane"], dict)

        # Reference plane field types
        ref_data = port_data["reference_plane"]
        assert isinstance(ref_data["enabled"], bool)
        assert isinstance(ref_data["method"], str)
