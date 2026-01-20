"""Tests for M2 port configuration (REQ-M2-005).

These tests validate:
- Port type support (lumped, MSL, waveguide, CPWG)
- Port position calculation from connector locations
- Impedance matching configuration
- De-embedding support for reference plane shifting
- Port builder functionality
- Integration with SimulationSpec
"""
from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from formula_foundry.openems import (
    BoardOutlineSpec,
    DeembedConfigSpec,
    DeembedSpec,
    DeembedType,
    DiscontinuitySpec,
    GeometrySpec,
    ImpedanceSpec,
    LayerSpec,
    PortBuilder,
    PortGeometrySpec,
    PortPosition,
    PortSpec,
    PortType,
    StackupSpec,
    TransmissionLineSpec,
    WaveguidePortSpec,
    build_ports_from_resolved,
    load_simulationspec,
    waveguide_port_to_basic_port_spec,
)
from formula_foundry.openems.geometry import StackupMaterialsSpec


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_stackup_spec() -> StackupSpec:
    """4-layer stackup with typical CPWG dimensions."""
    return StackupSpec(
        copper_layers=4,
        thicknesses_nm={
            "L1_to_L2": 200_000,  # 200um prepreg
            "L2_to_L3": 1_000_000,  # 1mm core
            "L3_to_L4": 200_000,  # 200um prepreg
        },
        materials=StackupMaterialsSpec(er=4.2, loss_tangent=0.02),
    )


@pytest.fixture
def sample_geometry_spec(sample_stackup_spec: StackupSpec) -> GeometrySpec:
    """Sample GeometrySpec with CPWG transmission line."""
    return GeometrySpec(
        design_hash="test_port_config_12345",
        coupon_family="F1_SINGLE_ENDED_VIA",
        board=BoardOutlineSpec(
            width_nm=20_000_000,
            length_nm=80_000_000,
            corner_radius_nm=2_000_000,
        ),
        stackup=sample_stackup_spec,
        layers=[
            LayerSpec(id="L1", z_nm=0, role="signal"),
            LayerSpec(id="L2", z_nm=200_000, role="ground"),
            LayerSpec(id="L3", z_nm=1_200_000, role="ground"),
            LayerSpec(id="L4", z_nm=1_400_000, role="signal"),
        ],
        transmission_line=TransmissionLineSpec(
            type="CPWG",
            layer="F.Cu",
            w_nm=300_000,  # 300um signal width
            gap_nm=180_000,  # 180um gap
            length_left_nm=25_000_000,
            length_right_nm=25_000_000,
        ),
        discontinuity=DiscontinuitySpec(
            type="VIA_TRANSITION",
            parameters_nm={"signal_via.drill_nm": 300_000},
        ),
    )


# =============================================================================
# Port Type Tests
# =============================================================================


class TestPortType:
    """Tests for PortType enumeration."""

    def test_all_port_types_defined(self) -> None:
        """All required port types should be defined."""
        assert PortType.LUMPED.value == "lumped"
        assert PortType.MSL.value == "msl"
        assert PortType.WAVEGUIDE.value == "waveguide"
        assert PortType.CPWG.value == "cpwg"

    def test_port_type_string_conversion(self) -> None:
        """Port types should convert to strings correctly."""
        assert str(PortType.WAVEGUIDE.value) == "waveguide"
        assert str(PortType.CPWG.value) == "cpwg"


class TestDeembedType:
    """Tests for DeembedType enumeration."""

    def test_all_deembed_types_defined(self) -> None:
        """All de-embedding types should be defined."""
        assert DeembedType.NONE.value == "none"
        assert DeembedType.REFERENCE_PLANE.value == "reference_plane"
        assert DeembedType.OPEN_SHORT.value == "open_short"
        assert DeembedType.TRL.value == "trl"


# =============================================================================
# Port Geometry Spec Tests
# =============================================================================


class TestPortGeometrySpec:
    """Tests for PortGeometrySpec model."""

    def test_minimal_geometry(self) -> None:
        """Geometry with required fields should validate."""
        geom = PortGeometrySpec(width_nm=500_000, height_nm=400_000)
        assert geom.width_nm == 500_000
        assert geom.height_nm == 400_000
        assert geom.signal_width_nm is None
        assert geom.gap_nm is None

    def test_full_cpwg_geometry(self) -> None:
        """Full CPWG geometry with all fields should validate."""
        geom = PortGeometrySpec(
            width_nm=660_000,  # w + 2*gap
            height_nm=400_000,
            signal_width_nm=300_000,
            gap_nm=180_000,
        )
        assert geom.signal_width_nm == 300_000
        assert geom.gap_nm == 180_000

    def test_extra_fields_rejected(self) -> None:
        """Extra fields should be rejected."""
        with pytest.raises(ValidationError, match="Extra inputs"):
            PortGeometrySpec(
                width_nm=500_000,
                height_nm=400_000,
                unknown_field=123,  # type: ignore[call-arg]
            )


# =============================================================================
# Impedance Spec Tests
# =============================================================================


class TestImpedanceSpec:
    """Tests for ImpedanceSpec model."""

    def test_default_impedance(self) -> None:
        """Default impedance should be 50 Ohm."""
        imp = ImpedanceSpec()
        assert imp.z0_ohm == 50.0
        assert imp.match_to_line is False
        assert imp.calculated_z0_ohm is None

    def test_custom_impedance(self) -> None:
        """Custom impedance should be accepted."""
        imp = ImpedanceSpec(z0_ohm=75.0)
        assert imp.z0_ohm == 75.0

    def test_match_to_line(self) -> None:
        """Match to line flag should work with calculated impedance."""
        imp = ImpedanceSpec(
            z0_ohm=50.0,
            match_to_line=True,
            calculated_z0_ohm=48.5,
        )
        assert imp.match_to_line is True
        assert imp.calculated_z0_ohm == 48.5

    def test_invalid_impedance_rejected(self) -> None:
        """Zero or negative impedance should be rejected."""
        with pytest.raises(ValidationError, match="greater than 0"):
            ImpedanceSpec(z0_ohm=0)

        with pytest.raises(ValidationError, match="greater than 0"):
            ImpedanceSpec(z0_ohm=-50)


# =============================================================================
# De-embed Spec Tests
# =============================================================================


class TestDeembedSpec:
    """Tests for DeembedSpec model."""

    def test_default_no_deembedding(self) -> None:
        """Default should be no de-embedding."""
        deembed = DeembedSpec()
        assert deembed.method == DeembedType.NONE
        assert deembed.distance_nm is None
        assert deembed.epsilon_r_eff is None

    def test_reference_plane_deembedding(self) -> None:
        """Reference plane de-embedding with distance."""
        deembed = DeembedSpec(
            method=DeembedType.REFERENCE_PLANE,
            distance_nm=5_000_000,
            epsilon_r_eff=2.6,
        )
        assert deembed.method == DeembedType.REFERENCE_PLANE
        assert deembed.distance_nm == 5_000_000
        assert deembed.epsilon_r_eff == 2.6

    def test_invalid_epsilon_rejected(self) -> None:
        """Epsilon_r must be positive."""
        with pytest.raises(ValidationError, match="greater than 0"):
            DeembedSpec(
                method=DeembedType.REFERENCE_PLANE,
                epsilon_r_eff=0,
            )


# =============================================================================
# Waveguide Port Spec Tests
# =============================================================================


class TestWaveguidePortSpec:
    """Tests for WaveguidePortSpec model."""

    def test_minimal_waveguide_port(self) -> None:
        """Minimal waveguide port should validate."""
        port = WaveguidePortSpec(
            id="P1",
            position_nm=(0, 0, 0),
            direction="x",
        )
        assert port.id == "P1"
        assert port.port_type == PortType.WAVEGUIDE
        assert port.excite is False
        assert port.impedance.z0_ohm == 50.0

    def test_full_waveguide_port(self) -> None:
        """Full waveguide port with all fields."""
        port = WaveguidePortSpec(
            id="P1",
            port_type=PortType.CPWG,
            position_nm=(5_000_000, 0, 17_500),
            direction="x",
            excite=True,
            excite_weight=1.0,
            geometry=PortGeometrySpec(
                width_nm=660_000,
                height_nm=400_000,
                signal_width_nm=300_000,
                gap_nm=180_000,
            ),
            impedance=ImpedanceSpec(
                z0_ohm=50.0,
                match_to_line=True,
                calculated_z0_ohm=48.5,
            ),
            deembed=DeembedSpec(
                method=DeembedType.REFERENCE_PLANE,
                distance_nm=5_000_000,
                epsilon_r_eff=2.6,
            ),
            polarization="E_transverse",
        )
        assert port.port_type == PortType.CPWG
        assert port.excite is True
        assert port.geometry is not None
        assert port.geometry.signal_width_nm == 300_000
        assert port.polarization == "E_transverse"

    def test_all_directions_valid(self) -> None:
        """All direction variants should be valid."""
        for direction in ["x", "y", "z", "-x", "-y", "-z"]:
            port = WaveguidePortSpec(
                id="test",
                position_nm=(0, 0, 0),
                direction=direction,
            )
            assert port.direction == direction


# =============================================================================
# Port Builder Tests
# =============================================================================


class TestPortBuilder:
    """Tests for PortBuilder class."""

    def test_builder_creation(self, sample_geometry_spec: GeometrySpec) -> None:
        """Builder should initialize from geometry spec."""
        builder = PortBuilder(geometry=sample_geometry_spec)
        assert builder.signal_layer_id == "L1"
        assert builder.copper_thickness_nm == 35_000

    def test_build_connector_port(self, sample_geometry_spec: GeometrySpec) -> None:
        """Build a single port at connector location."""
        builder = PortBuilder(geometry=sample_geometry_spec)
        port = builder.build_connector_port(
            "P1",
            (5_000_000, 0),
            excite=True,
            direction="x",
        )
        assert port.id == "P1"
        assert port.excite is True
        assert port.position_nm[0] == 5_000_000
        assert port.position_nm[1] == 0
        # Z should be at signal layer center
        assert port.position_nm[2] > 0

    def test_build_port_with_deembedding(self, sample_geometry_spec: GeometrySpec) -> None:
        """Build port with de-embedding enabled."""
        builder = PortBuilder(geometry=sample_geometry_spec)
        port = builder.build_connector_port(
            "P1",
            (5_000_000, 0),
            excite=True,
            direction="x",
            deembed_distance_nm=3_000_000,
        )
        assert port.deembed.method == DeembedType.REFERENCE_PLANE
        assert port.deembed.distance_nm == 3_000_000
        assert port.deembed.epsilon_r_eff is not None
        assert port.deembed.epsilon_r_eff > 1.0

    def test_build_port_geometry(self, sample_geometry_spec: GeometrySpec) -> None:
        """Port should have correct geometry from transmission line."""
        builder = PortBuilder(geometry=sample_geometry_spec)
        port = builder.build_connector_port(
            "P1",
            (5_000_000, 0),
            excite=True,
            direction="x",
        )
        assert port.geometry is not None
        # Port width should be signal + 2*gap
        expected_width = 300_000 + 2 * 180_000  # 660_000
        assert port.geometry.width_nm == expected_width
        assert port.geometry.signal_width_nm == 300_000
        assert port.geometry.gap_nm == 180_000

    def test_build_transmission_line_ports(self, sample_geometry_spec: GeometrySpec) -> None:
        """Build a pair of ports for 2-port S-parameter extraction."""
        builder = PortBuilder(geometry=sample_geometry_spec)
        p1, p2 = builder.build_transmission_line_ports(
            (-25_000_000, 0),
            (25_000_000, 0),
        )

        # P1 should be excited, P2 should not
        assert p1.id == "P1"
        assert p1.excite is True
        assert p1.direction == "x"

        assert p2.id == "P2"
        assert p2.excite is False
        assert p2.direction == "-x"

        # Positions should match input
        assert p1.position_nm[0] == -25_000_000
        assert p2.position_nm[0] == 25_000_000

    def test_port_types(self, sample_geometry_spec: GeometrySpec) -> None:
        """Builder should support all port types."""
        builder = PortBuilder(geometry=sample_geometry_spec)

        for port_type in [PortType.LUMPED, PortType.MSL, PortType.WAVEGUIDE, PortType.CPWG]:
            port = builder.build_connector_port(
                "test",
                (0, 0),
                port_type=port_type,
            )
            assert port.port_type == port_type


# =============================================================================
# Impedance Calculation Tests
# =============================================================================


class TestImpedanceCalculation:
    """Tests for transmission line impedance calculation."""

    def test_calculated_impedance_reasonable(self, sample_geometry_spec: GeometrySpec) -> None:
        """Calculated impedance should be in reasonable range."""
        builder = PortBuilder(geometry=sample_geometry_spec)
        port = builder.build_connector_port("P1", (0, 0))

        # For CPWG with our dimensions, impedance should be roughly 40-60 Ohm
        assert port.impedance.calculated_z0_ohm is not None
        assert 20.0 < port.impedance.calculated_z0_ohm < 100.0

    def test_match_to_line_flag_set(self, sample_geometry_spec: GeometrySpec) -> None:
        """Match to line flag should be set."""
        builder = PortBuilder(geometry=sample_geometry_spec)
        port = builder.build_connector_port("P1", (0, 0))
        assert port.impedance.match_to_line is True


# =============================================================================
# Port Spec Integration Tests
# =============================================================================


class TestPortSpecIntegration:
    """Tests for PortSpec model with new fields."""

    def test_portspec_with_deembed(self) -> None:
        """PortSpec should accept de-embedding configuration."""
        port = PortSpec(
            id="P1",
            position_nm=(0, 0, 0),
            direction="x",
            deembed=DeembedConfigSpec(
                enabled=True,
                distance_nm=5_000_000,
                epsilon_r_eff=2.6,
            ),
        )
        assert port.deembed.enabled is True
        assert port.deembed.distance_nm == 5_000_000

    def test_portspec_with_cpwg_fields(self) -> None:
        """PortSpec should accept CPWG-specific fields."""
        port = PortSpec(
            id="P1",
            type="cpwg",
            position_nm=(0, 0, 0),
            direction="x",
            width_nm=660_000,
            height_nm=400_000,
            signal_width_nm=300_000,
            gap_nm=180_000,
        )
        assert port.type == "cpwg"
        assert port.signal_width_nm == 300_000
        assert port.gap_nm == 180_000

    def test_portspec_match_to_line(self) -> None:
        """PortSpec should support impedance matching fields."""
        port = PortSpec(
            id="P1",
            position_nm=(0, 0, 0),
            direction="x",
            match_to_line=True,
            calculated_z0_ohm=48.5,
        )
        assert port.match_to_line is True
        assert port.calculated_z0_ohm == 48.5

    def test_portspec_polarization(self) -> None:
        """PortSpec should support polarization selection."""
        port = PortSpec(
            id="P1",
            position_nm=(0, 0, 0),
            direction="x",
            polarization="E_transverse",
        )
        assert port.polarization == "E_transverse"


# =============================================================================
# Conversion Tests
# =============================================================================


class TestWaveguideToPortSpecConversion:
    """Tests for waveguide_port_to_basic_port_spec conversion."""

    def test_basic_conversion(self) -> None:
        """Basic conversion should preserve essential fields."""
        wp = WaveguidePortSpec(
            id="P1",
            port_type=PortType.WAVEGUIDE,
            position_nm=(1_000_000, 0, 35_000),
            direction="x",
            excite=True,
        )
        result = waveguide_port_to_basic_port_spec(wp)

        assert result["id"] == "P1"
        assert result["type"] == "waveguide"
        assert result["position_nm"] == (1_000_000, 0, 35_000)
        assert result["direction"] == "x"
        assert result["excite"] is True
        assert result["impedance_ohm"] == 50.0

    def test_conversion_with_geometry(self) -> None:
        """Conversion should include geometry dimensions."""
        wp = WaveguidePortSpec(
            id="P1",
            position_nm=(0, 0, 0),
            direction="x",
            geometry=PortGeometrySpec(
                width_nm=660_000,
                height_nm=400_000,
            ),
        )
        result = waveguide_port_to_basic_port_spec(wp)

        assert result["width_nm"] == 660_000
        assert result["height_nm"] == 400_000


# =============================================================================
# SimulationSpec Integration Tests
# =============================================================================


class TestSimulationSpecPortIntegration:
    """Tests for port configuration in SimulationSpec."""

    def _minimal_spec_dict(self) -> dict:
        """Return minimal valid SimulationSpec with new port fields."""
        return {
            "schema_version": 1,
            "toolchain": {
                "openems": {
                    "version": "0.0.35",
                    "docker_image": "ghcr.io/thliebig/openems:0.0.35",
                }
            },
            "geometry_ref": {"design_hash": "abc123"},
            "excitation": {"f0_hz": "5GHz", "fc_hz": "10GHz"},
            "frequency": {"f_start_hz": "1MHz", "f_stop_hz": "20GHz", "n_points": 401},
            "ports": [
                {
                    "id": "P1",
                    "type": "waveguide",
                    "position_nm": [0, 0, 0],
                    "direction": "x",
                    "excite": True,
                    "width_nm": 660_000,
                    "height_nm": 400_000,
                    "signal_width_nm": 300_000,
                    "gap_nm": 180_000,
                    "match_to_line": True,
                    "calculated_z0_ohm": 48.5,
                    "deembed": {
                        "enabled": True,
                        "distance_nm": 5_000_000,
                        "epsilon_r_eff": 2.6,
                    },
                    "polarization": "E_transverse",
                }
            ],
        }

    def test_full_port_config_validates(self) -> None:
        """Full port configuration should validate."""
        data = self._minimal_spec_dict()
        spec = load_simulationspec(data)

        port = spec.ports[0]
        assert port.type == "waveguide"
        assert port.signal_width_nm == 300_000
        assert port.gap_nm == 180_000
        assert port.match_to_line is True
        assert port.calculated_z0_ohm == 48.5
        assert port.deembed.enabled is True
        assert port.deembed.distance_nm == 5_000_000
        assert port.polarization == "E_transverse"

    def test_cpwg_port_type(self) -> None:
        """CPWG port type should be valid."""
        data = self._minimal_spec_dict()
        data["ports"][0]["type"] = "cpwg"
        spec = load_simulationspec(data)
        assert spec.ports[0].type == "cpwg"

    def test_default_deembed_disabled(self) -> None:
        """Default de-embedding should be disabled."""
        data = self._minimal_spec_dict()
        del data["ports"][0]["deembed"]
        spec = load_simulationspec(data)
        assert spec.ports[0].deembed.enabled is False


# =============================================================================
# Build Ports From Resolved Tests
# =============================================================================


class TestBuildPortsFromResolved:
    """Tests for build_ports_from_resolved function."""

    def test_basic_port_building(self, sample_geometry_spec: GeometrySpec) -> None:
        """Basic port building should work."""
        params = {
            "transmission_line.length_left_nm": 25_000_000,
            "transmission_line.length_right_nm": 25_000_000,
        }
        ports = build_ports_from_resolved(
            sample_geometry_spec,
            params,
            port_type=PortType.WAVEGUIDE,
        )

        assert len(ports) == 2
        assert ports[0].id == "P1"
        assert ports[1].id == "P2"

    def test_with_connector_positions(self, sample_geometry_spec: GeometrySpec) -> None:
        """Port building with explicit connector positions."""
        params = {
            "connectors.left.position_nm[0]": 5_000_000,
            "connectors.left.position_nm[1]": 0,
            "connectors.right.position_nm[0]": 75_000_000,
            "connectors.right.position_nm[1]": 0,
            "transmission_line.length_left_nm": 25_000_000,
            "transmission_line.length_right_nm": 25_000_000,
        }
        ports = build_ports_from_resolved(sample_geometry_spec, params)

        assert ports[0].position_nm[0] == 5_000_000
        assert ports[1].position_nm[0] == 75_000_000

    def test_with_deembedding(self, sample_geometry_spec: GeometrySpec) -> None:
        """Port building with de-embedding from launch length."""
        params = {
            "transmission_line.length_left_nm": 25_000_000,
            "transmission_line.length_right_nm": 25_000_000,
            "launch.length_nm": 3_000_000,
        }
        ports = build_ports_from_resolved(
            sample_geometry_spec,
            params,
            include_deembedding=True,
        )

        assert ports[0].deembed.method == DeembedType.REFERENCE_PLANE
        assert ports[0].deembed.distance_nm == 3_000_000


# =============================================================================
# Port Position Tests
# =============================================================================


class TestPortPosition:
    """Tests for PortPosition dataclass."""

    def test_creation(self) -> None:
        """PortPosition should store coordinates correctly."""
        pos = PortPosition(
            x_nm=1_000_000,
            y_nm=2_000_000,
            z_nm=35_000,
            direction="x",
            layer_id="L1",
        )
        assert pos.x_nm == 1_000_000
        assert pos.y_nm == 2_000_000
        assert pos.z_nm == 35_000
        assert pos.direction == "x"
        assert pos.layer_id == "L1"

    def test_as_tuple(self) -> None:
        """as_tuple should return (x, y, z) tuple."""
        pos = PortPosition(
            x_nm=1_000_000,
            y_nm=2_000_000,
            z_nm=35_000,
            direction="x",
            layer_id="L1",
        )
        assert pos.as_tuple() == (1_000_000, 2_000_000, 35_000)

    def test_frozen(self) -> None:
        """PortPosition should be immutable."""
        pos = PortPosition(
            x_nm=1_000_000,
            y_nm=0,
            z_nm=0,
            direction="x",
            layer_id="L1",
        )
        with pytest.raises(AttributeError):
            pos.x_nm = 2_000_000  # type: ignore[misc]
