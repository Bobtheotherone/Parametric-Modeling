"""Tests for port verification (REQ-M2-011, REQ-M2-012).

These tests validate:
- Cardinal rotation enforcement by default
- Port geometry overlap detection and rejection
- Anti-short checks for signal/ground separation
- Minimum port length vs mesh cell size validation
"""

from __future__ import annotations

import pytest
from formula_foundry.ports.verification import (
    CardinalDirection,
    CardinalRotationError,
    PortGeometry,
    PortGeometryOverlapError,
    PortLengthError,
    PortShortError,
    PortVerificationReport,
    PortVerificationResult,
    check_cardinal_rotation,
    check_port_length_vs_mesh,
    check_port_overlap,
    check_port_short,
    validate_port_geometry,
    verify_port_definitions,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def port_p1() -> PortGeometry:
    """Standard port P1 at left side with cardinal rotation."""
    return PortGeometry(
        id="P1",
        position_nm=(0, 0, 35_000),
        width_nm=660_000,
        height_nm=400_000,
        length_nm=100_000,
        direction="x",
        rotation_deg=0.0,
        signal_width_nm=300_000,
        gap_nm=180_000,
    )


@pytest.fixture
def port_p2() -> PortGeometry:
    """Standard port P2 at right side with cardinal rotation."""
    return PortGeometry(
        id="P2",
        position_nm=(50_000_000, 0, 35_000),
        width_nm=660_000,
        height_nm=400_000,
        length_nm=100_000,
        direction="-x",
        rotation_deg=0.0,
        signal_width_nm=300_000,
        gap_nm=180_000,
    )


# =============================================================================
# Cardinal Direction Tests
# =============================================================================


class TestCardinalDirection:
    """Tests for CardinalDirection enumeration."""

    def test_all_directions_defined(self) -> None:
        """All cardinal directions should be defined."""
        assert CardinalDirection.X_POSITIVE.value == "x"
        assert CardinalDirection.X_NEGATIVE.value == "-x"
        assert CardinalDirection.Y_POSITIVE.value == "y"
        assert CardinalDirection.Y_NEGATIVE.value == "-y"
        assert CardinalDirection.Z_POSITIVE.value == "z"
        assert CardinalDirection.Z_NEGATIVE.value == "-z"

    def test_cardinal_direction_count(self) -> None:
        """Should have exactly 6 cardinal directions."""
        assert len(CardinalDirection) == 6


# =============================================================================
# Port Geometry Tests
# =============================================================================


class TestPortGeometry:
    """Tests for PortGeometry dataclass."""

    def test_creation(self, port_p1: PortGeometry) -> None:
        """PortGeometry should store all attributes correctly."""
        assert port_p1.id == "P1"
        assert port_p1.position_nm == (0, 0, 35_000)
        assert port_p1.width_nm == 660_000
        assert port_p1.height_nm == 400_000
        assert port_p1.length_nm == 100_000
        assert port_p1.direction == "x"
        assert port_p1.rotation_deg == 0.0
        assert port_p1.signal_width_nm == 300_000
        assert port_p1.gap_nm == 180_000

    def test_frozen(self, port_p1: PortGeometry) -> None:
        """PortGeometry should be immutable."""
        with pytest.raises(AttributeError):
            port_p1.id = "P2"  # type: ignore[misc]

    def test_bounds_x_direction(self, port_p1: PortGeometry) -> None:
        """Bounds should be correct for x-direction port."""
        min_c, max_c = port_p1.bounds()
        # Center at (0, 0, 35000), length=100000, width=660000, height=400000
        # For x direction: dx=length/2, dy=width/2, dz=height/2
        assert min_c == (-50_000, -330_000, -165_000)
        assert max_c == (50_000, 330_000, 235_000)

    def test_bounds_y_direction(self) -> None:
        """Bounds should be correct for y-direction port."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=200_000,
            length_nm=50_000,
            direction="y",
        )
        min_c, max_c = port.bounds()
        # For y direction: dx=width/2, dy=length/2, dz=height/2
        assert min_c == (-50_000, -25_000, -100_000)
        assert max_c == (50_000, 25_000, 100_000)

    def test_bounds_z_direction(self) -> None:
        """Bounds should be correct for z-direction port."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=200_000,
            length_nm=50_000,
            direction="z",
        )
        min_c, max_c = port.bounds()
        # For z direction: dx=width/2, dy=height/2, dz=length/2
        assert min_c == (-50_000, -100_000, -25_000)
        assert max_c == (50_000, 100_000, 25_000)


# =============================================================================
# Cardinal Rotation Tests (REQ-M2-011)
# =============================================================================


class TestCardinalRotation:
    """Tests for cardinal rotation validation (REQ-M2-011)."""

    def test_port_rotation_must_be_cardinal_by_default(self, port_p1: PortGeometry) -> None:
        """Port with 0 degree rotation should pass cardinal check (REQ-M2-011)."""
        result = check_cardinal_rotation(port_p1, strict=True)
        assert result.passed is True
        assert result.check_name == "cardinal_rotation"
        assert "cardinal" in result.message.lower()

    @pytest.mark.parametrize("rotation", [0.0, 90.0, 180.0, 270.0])
    def test_all_cardinal_rotations_valid(self, rotation: float) -> None:
        """All four cardinal rotations should be valid."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=100_000,
            length_nm=50_000,
            direction="x",
            rotation_deg=rotation,
        )
        result = check_cardinal_rotation(port, strict=True)
        assert result.passed is True

    @pytest.mark.parametrize("rotation", [45.0, 30.0, 135.0, 225.0, 315.0])
    def test_non_cardinal_rotation_fails_strict(self, rotation: float) -> None:
        """Non-cardinal rotations should fail in strict mode."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=100_000,
            length_nm=50_000,
            direction="x",
            rotation_deg=rotation,
        )
        with pytest.raises(CardinalRotationError) as exc_info:
            check_cardinal_rotation(port, strict=True)
        assert exc_info.value.port_id == "P"
        assert exc_info.value.rotation_deg == rotation

    def test_non_cardinal_rotation_non_strict(self) -> None:
        """Non-cardinal rotation in non-strict mode should return failed result."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=100_000,
            length_nm=50_000,
            direction="x",
            rotation_deg=45.0,
        )
        result = check_cardinal_rotation(port, strict=False)
        assert result.passed is False
        assert "non-cardinal" in result.message.lower()

    def test_rotation_360_equals_zero(self) -> None:
        """360 degrees should be equivalent to 0 (cardinal)."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=100_000,
            length_nm=50_000,
            direction="x",
            rotation_deg=360.0,
        )
        result = check_cardinal_rotation(port, strict=True)
        assert result.passed is True

    def test_rotation_tolerance(self) -> None:
        """Small floating-point deviations should pass within tolerance."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=100_000,
            length_nm=50_000,
            direction="x",
            rotation_deg=90.0001,  # Very close to 90
        )
        # Should pass because within tolerance
        result = check_cardinal_rotation(port, strict=True)
        assert result.passed is True


# =============================================================================
# Port Overlap Tests (REQ-M2-012)
# =============================================================================


class TestPortOverlap:
    """Tests for port overlap detection (REQ-M2-012)."""

    def test_non_overlapping_ports_pass(self, port_p1: PortGeometry, port_p2: PortGeometry) -> None:
        """Non-overlapping ports should pass."""
        result = check_port_overlap(port_p1, port_p2, strict=True)
        assert result.passed is True
        assert result.check_name == "geometry_overlap"

    def test_overlapping_ports_fail_strict(self) -> None:
        """Overlapping ports should raise error in strict mode."""
        # Two ports at same position
        port1 = PortGeometry(
            id="P1",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=100_000,
            length_nm=50_000,
            direction="x",
        )
        port2 = PortGeometry(
            id="P2",
            position_nm=(0, 0, 0),  # Same position -> overlap
            width_nm=100_000,
            height_nm=100_000,
            length_nm=50_000,
            direction="-x",
        )
        with pytest.raises(PortGeometryOverlapError) as exc_info:
            check_port_overlap(port1, port2, strict=True)
        assert exc_info.value.port_id_1 == "P1"
        assert exc_info.value.port_id_2 == "P2"

    def test_overlapping_ports_non_strict(self) -> None:
        """Overlapping ports in non-strict mode should return failed result."""
        port1 = PortGeometry(
            id="P1",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=100_000,
            length_nm=50_000,
            direction="x",
        )
        port2 = PortGeometry(
            id="P2",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=100_000,
            length_nm=50_000,
            direction="-x",
        )
        result = check_port_overlap(port1, port2, strict=False)
        assert result.passed is False
        assert "overlap" in result.message.lower()

    def test_partial_overlap_detected(self) -> None:
        """Partially overlapping ports should be detected."""
        port1 = PortGeometry(
            id="P1",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=100_000,
            length_nm=100_000,
            direction="x",
        )
        port2 = PortGeometry(
            id="P2",
            position_nm=(50_000, 0, 0),  # Partial overlap on X
            width_nm=100_000,
            height_nm=100_000,
            length_nm=100_000,
            direction="-x",
        )
        result = check_port_overlap(port1, port2, strict=False)
        assert result.passed is False

    def test_edge_touching_no_overlap(self) -> None:
        """Edge-touching boxes should not be considered overlapping."""
        port1 = PortGeometry(
            id="P1",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=100_000,
            length_nm=100_000,
            direction="x",
        )
        # Port2 exactly adjacent on X axis
        port2 = PortGeometry(
            id="P2",
            position_nm=(100_000, 0, 0),  # Exactly touching
            width_nm=100_000,
            height_nm=100_000,
            length_nm=100_000,
            direction="-x",
        )
        result = check_port_overlap(port1, port2, strict=True)
        assert result.passed is True


# =============================================================================
# Anti-Short Tests (REQ-M2-012)
# =============================================================================


class TestPortShort:
    """Tests for anti-short validation (REQ-M2-012)."""

    def test_valid_port_passes(self, port_p1: PortGeometry) -> None:
        """Port with proper signal/gap geometry should pass."""
        result = check_port_short(port_p1, strict=True)
        assert result.passed is True
        assert result.check_name == "anti_short"

    def test_zero_gap_fails(self) -> None:
        """Zero gap should be detected as potential short."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=660_000,
            height_nm=400_000,
            length_nm=100_000,
            direction="x",
            signal_width_nm=300_000,
            gap_nm=0,  # Zero gap -> short
        )
        with pytest.raises(PortShortError) as exc_info:
            check_port_short(port, strict=True)
        assert exc_info.value.port_id == "P"
        assert "positive" in exc_info.value.reason

    def test_negative_gap_fails(self) -> None:
        """Negative gap should be detected as short."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=660_000,
            height_nm=400_000,
            length_nm=100_000,
            direction="x",
            signal_width_nm=300_000,
            gap_nm=-10_000,  # Negative gap
        )
        with pytest.raises(PortShortError):
            check_port_short(port, strict=True)

    def test_signal_wider_than_port_fails(self) -> None:
        """Signal wider than port aperture should fail."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=300_000,  # Port width
            height_nm=400_000,
            length_nm=100_000,
            direction="x",
            signal_width_nm=400_000,  # Signal wider than port
            gap_nm=50_000,
        )
        with pytest.raises(PortShortError) as exc_info:
            check_port_short(port, strict=True)
        assert "signal width" in exc_info.value.reason.lower()

    def test_signal_plus_gap_exceeds_port_fails(self) -> None:
        """Signal + 2*gap exceeding port width should fail."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=500_000,  # Port width
            height_nm=400_000,
            length_nm=100_000,
            direction="x",
            signal_width_nm=300_000,  # Signal
            gap_nm=150_000,  # 300 + 2*150 = 600 > 500
        )
        with pytest.raises(PortShortError) as exc_info:
            check_port_short(port, strict=True)
        assert "exceeds port width" in exc_info.value.reason

    def test_no_signal_info_skipped(self) -> None:
        """Port without signal/gap info should skip check."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=660_000,
            height_nm=400_000,
            length_nm=100_000,
            direction="x",
            # No signal_width_nm or gap_nm
        )
        result = check_port_short(port, strict=True)
        assert result.passed is True
        assert "skipped" in result.message.lower()

    def test_short_non_strict(self) -> None:
        """Short in non-strict mode should return failed result."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=660_000,
            height_nm=400_000,
            length_nm=100_000,
            direction="x",
            signal_width_nm=300_000,
            gap_nm=0,
        )
        result = check_port_short(port, strict=False)
        assert result.passed is False
        assert "short" in result.message.lower()


# =============================================================================
# Port Length vs Mesh Tests (REQ-M2-012)
# =============================================================================


class TestPortLengthVsMesh:
    """Tests for port length vs mesh cell size validation (REQ-M2-012)."""

    def test_port_geometry_overlap_no_short_and_min_length_vs_mesh(self, port_p1: PortGeometry) -> None:
        """Combined test for overlap, short, and min length (REQ-M2-012)."""
        # port_p1 has length_nm=100_000
        # With max_cell_size=40_000, min_required = 80_000, so should pass
        result = check_port_length_vs_mesh(port_p1, max_cell_size_nm=40_000, strict=True)
        assert result.passed is True
        assert result.check_name == "min_length_vs_mesh"

    def test_sufficient_length_passes(self, port_p1: PortGeometry) -> None:
        """Port with sufficient length should pass."""
        # port_p1 has length_nm=100_000, with max_cell=40_000, min=80_000
        result = check_port_length_vs_mesh(port_p1, max_cell_size_nm=40_000, strict=True)
        assert result.passed is True

    def test_insufficient_length_fails_strict(self) -> None:
        """Port with insufficient length should fail in strict mode."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=660_000,
            height_nm=400_000,
            length_nm=50_000,  # Too short
            direction="x",
        )
        # With max_cell=50_000, min_required = 100_000
        with pytest.raises(PortLengthError) as exc_info:
            check_port_length_vs_mesh(port, max_cell_size_nm=50_000, strict=True)
        assert exc_info.value.port_id == "P"
        assert exc_info.value.port_length_nm == 50_000
        assert exc_info.value.min_required_nm == 100_000
        assert exc_info.value.max_cell_size_nm == 50_000

    def test_insufficient_length_non_strict(self) -> None:
        """Insufficient length in non-strict mode should return failed result."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=660_000,
            height_nm=400_000,
            length_nm=50_000,
            direction="x",
        )
        result = check_port_length_vs_mesh(port, max_cell_size_nm=50_000, strict=False)
        assert result.passed is False
        assert "below minimum" in result.message

    def test_custom_length_ratio(self) -> None:
        """Custom min_length_ratio should be respected."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=660_000,
            height_nm=400_000,
            length_nm=100_000,
            direction="x",
        )
        # With max_cell=50_000 and ratio=3, min=150_000 -> should fail
        with pytest.raises(PortLengthError):
            check_port_length_vs_mesh(port, max_cell_size_nm=50_000, min_length_ratio=3.0, strict=True)

    def test_exact_minimum_passes(self) -> None:
        """Port with exactly minimum length should pass."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=660_000,
            height_nm=400_000,
            length_nm=100_000,  # Exactly 2x max_cell
            direction="x",
        )
        result = check_port_length_vs_mesh(port, max_cell_size_nm=50_000, strict=True)
        assert result.passed is True


# =============================================================================
# Validate Port Geometry Tests
# =============================================================================


class TestValidatePortGeometry:
    """Tests for validate_port_geometry function."""

    def test_valid_port_all_checks_pass(self, port_p1: PortGeometry) -> None:
        """Valid port should pass all checks."""
        results = validate_port_geometry(port_p1, max_cell_size_nm=40_000, strict=True)
        assert len(results) >= 3  # rotation, anti_short, length
        assert all(r.passed for r in results)

    def test_without_mesh_size_skips_length_check(self, port_p1: PortGeometry) -> None:
        """Without max_cell_size, length check should be skipped."""
        results = validate_port_geometry(port_p1, max_cell_size_nm=None, strict=True)
        assert len(results) == 2  # Only rotation and anti_short
        check_names = [r.check_name for r in results]
        assert "min_length_vs_mesh" not in check_names

    def test_multiple_failures_reported_non_strict(self) -> None:
        """Multiple failures should all be reported in non-strict mode."""
        port = PortGeometry(
            id="P",
            position_nm=(0, 0, 0),
            width_nm=500_000,
            height_nm=400_000,
            length_nm=30_000,  # Too short
            direction="x",
            rotation_deg=45.0,  # Non-cardinal
            signal_width_nm=300_000,
            gap_nm=150_000,  # Signal + 2*gap > port width
        )
        results = validate_port_geometry(port, max_cell_size_nm=50_000, strict=False)
        failed = [r for r in results if not r.passed]
        assert len(failed) == 3  # rotation, anti_short, length


# =============================================================================
# Verify Port Definitions Tests
# =============================================================================


class TestVerifyPortDefinitions:
    """Tests for verify_port_definitions function."""

    def test_valid_ports_all_pass(self, port_p1: PortGeometry, port_p2: PortGeometry) -> None:
        """Valid non-overlapping ports should pass all checks."""
        report = verify_port_definitions([port_p1, port_p2], max_cell_size_nm=40_000, strict=True)
        assert report.all_passed is True
        assert len(report.failed_checks) == 0

    def test_report_structure(self, port_p1: PortGeometry, port_p2: PortGeometry) -> None:
        """Report should have correct structure."""
        report = verify_port_definitions([port_p1, port_p2], max_cell_size_nm=40_000)
        assert isinstance(report, PortVerificationReport)
        assert isinstance(report.results, tuple)
        assert all(isinstance(r, PortVerificationResult) for r in report.results)

    def test_overlap_detected_in_report(self) -> None:
        """Overlapping ports should be detected in full verification."""
        port1 = PortGeometry(
            id="P1",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=100_000,
            length_nm=100_000,
            direction="x",
        )
        port2 = PortGeometry(
            id="P2",
            position_nm=(0, 0, 0),  # Overlaps with port1
            width_nm=100_000,
            height_nm=100_000,
            length_nm=100_000,
            direction="-x",
        )
        report = verify_port_definitions([port1, port2], strict=False)
        assert report.all_passed is False
        assert "geometry_overlap" in report.failed_checks

    def test_multiple_failures_collected(self) -> None:
        """Multiple failures across ports should be collected."""
        port1 = PortGeometry(
            id="P1",
            position_nm=(0, 0, 0),
            width_nm=100_000,
            height_nm=100_000,
            length_nm=30_000,  # Too short
            direction="x",
            rotation_deg=45.0,  # Non-cardinal
        )
        port2 = PortGeometry(
            id="P2",
            position_nm=(0, 0, 0),  # Overlaps
            width_nm=100_000,
            height_nm=100_000,
            length_nm=100_000,
            direction="-x",
        )
        report = verify_port_definitions([port1, port2], max_cell_size_nm=50_000, strict=False)
        assert report.all_passed is False
        # Should have: cardinal_rotation fail, min_length fail, geometry_overlap fail
        assert len(report.failed_checks) >= 3

    def test_empty_ports_list(self) -> None:
        """Empty ports list should pass trivially."""
        report = verify_port_definitions([], strict=True)
        assert report.all_passed is True
        assert len(report.results) == 0

    def test_single_port_no_overlap_check(self, port_p1: PortGeometry) -> None:
        """Single port should not have overlap checks."""
        report = verify_port_definitions([port_p1], max_cell_size_nm=40_000, strict=True)
        assert report.all_passed is True
        check_names = [r.check_name for r in report.results]
        assert "geometry_overlap" not in check_names

    def test_three_ports_pairwise_overlap_checks(self) -> None:
        """Three ports should have 3 pairwise overlap checks."""
        ports = [
            PortGeometry(
                id=f"P{i}",
                position_nm=(i * 1_000_000, 0, 0),
                width_nm=100_000,
                height_nm=100_000,
                length_nm=100_000,
                direction="x",
            )
            for i in range(3)
        ]
        report = verify_port_definitions(ports, strict=True)
        overlap_checks = [r for r in report.results if r.check_name == "geometry_overlap"]
        # C(3,2) = 3 pairwise checks
        assert len(overlap_checks) == 3


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for exception classes."""

    def test_cardinal_rotation_error_message(self) -> None:
        """CardinalRotationError should have informative message."""
        error = CardinalRotationError("P1", 45.0)
        assert error.port_id == "P1"
        assert error.rotation_deg == 45.0
        assert "non-cardinal" in str(error).lower()
        assert "0, 90, 180, 270" in str(error)

    def test_port_geometry_overlap_error_message(self) -> None:
        """PortGeometryOverlapError should have informative message."""
        error = PortGeometryOverlapError("P1", "P2")
        assert error.port_id_1 == "P1"
        assert error.port_id_2 == "P2"
        assert "overlap" in str(error).lower()

    def test_port_short_error_message(self) -> None:
        """PortShortError should include reason."""
        error = PortShortError("P1", "gap is zero")
        assert error.port_id == "P1"
        assert error.reason == "gap is zero"
        assert "short" in str(error).lower()
        assert "gap is zero" in str(error)

    def test_port_length_error_message(self) -> None:
        """PortLengthError should have informative message."""
        error = PortLengthError("P1", 50_000, 100_000, 50_000)
        assert error.port_id == "P1"
        assert error.port_length_nm == 50_000
        assert error.min_required_nm == 100_000
        assert error.max_cell_size_nm == 50_000
        assert "50000" in str(error)
        assert "100000" in str(error)
