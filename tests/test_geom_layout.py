# SPDX-License-Identifier: MIT
"""Unit tests for coupongen geometry layout module.

Tests the LayoutPlan data structures and factory functions that serve as
the Single Source of Truth for coupon geometry:
- PortPlan: Connector port placement
- SegmentPlan: Trace segment definition
- LayoutPlan: Full coupon layout
- derive_right_length_nm: F1 continuity formula

Per CP-2.1, LayoutPlan enforces the F1 continuity invariant:
    x_discontinuity_center_nm == x_left_connector_ref_nm + length_left_nm
    x_discontinuity_center_nm == x_right_connector_ref_nm - length_right_nm
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

# Direct import to avoid broken import chain in formula_foundry.__init__
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _load_module(name: str, path: Path) -> ModuleType:
    """Load a module from file with proper sys.modules registration."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {name} at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Register before exec to handle dataclasses
    spec.loader.exec_module(module)
    return module


# Load primitives first (layout depends on it)
_primitives = _load_module(
    "formula_foundry.coupongen.geom.primitives",
    _SRC_DIR / "formula_foundry" / "coupongen" / "geom" / "primitives.py",
)

# Load layout module
_layout = _load_module(
    "formula_foundry.coupongen.geom.layout",
    _SRC_DIR / "formula_foundry" / "coupongen" / "geom" / "layout.py",
)

# Import from loaded modules
OriginMode = _primitives.OriginMode
PortPlan = _layout.PortPlan
SegmentPlan = _layout.SegmentPlan
LayoutPlan = _layout.LayoutPlan
derive_right_length_nm = _layout.derive_right_length_nm


# =============================================================================
# PortPlan tests
# =============================================================================


class TestPortPlan:
    """Tests for PortPlan dataclass."""

    def test_create_left_port(self) -> None:
        """Create a left port plan."""
        port = PortPlan(
            x_ref_nm=5_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=8_000_000,
            signal_pad_y_nm=0,
            footprint="Coupongen_Connectors:SMA_EndLaunch_Generic",
            rotation_mdeg=0,
            side="left",
        )
        assert port.side == "left"
        assert port.x_ref_nm == 5_000_000
        assert port.signal_pad_x_nm == 8_000_000

    def test_create_right_port(self) -> None:
        """Create a right port plan."""
        port = PortPlan(
            x_ref_nm=75_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=72_000_000,
            signal_pad_y_nm=0,
            footprint="Coupongen_Connectors:SMA_EndLaunch_Generic",
            rotation_mdeg=180000,
            side="right",
        )
        assert port.side == "right"
        assert port.rotation_mdeg == 180000

    def test_invalid_side_raises(self) -> None:
        """Invalid port side raises ValueError."""
        with pytest.raises(ValueError, match="left.*right"):
            PortPlan(
                x_ref_nm=0,
                y_ref_nm=0,
                signal_pad_x_nm=0,
                signal_pad_y_nm=0,
                footprint="Test:FP",
                rotation_mdeg=0,
                side="middle",
            )

    def test_invalid_rotation_raises(self) -> None:
        """Invalid rotation value raises ValueError."""
        with pytest.raises(ValueError, match="90000"):
            PortPlan(
                x_ref_nm=0,
                y_ref_nm=0,
                signal_pad_x_nm=0,
                signal_pad_y_nm=0,
                footprint="Test:FP",
                rotation_mdeg=45000,  # Invalid: not 0, 90000, 180000, 270000
                side="left",
            )

    def test_valid_rotations(self) -> None:
        """All valid rotation values should work."""
        valid_rotations = [0, 90000, 180000, 270000]
        for rot in valid_rotations:
            port = PortPlan(
                x_ref_nm=0,
                y_ref_nm=0,
                signal_pad_x_nm=0,
                signal_pad_y_nm=0,
                footprint="Test:FP",
                rotation_mdeg=rot,
                side="left",
            )
            assert port.rotation_mdeg == rot


# =============================================================================
# SegmentPlan tests
# =============================================================================


class TestSegmentPlan:
    """Tests for SegmentPlan dataclass."""

    def test_create_segment(self) -> None:
        """Create a trace segment."""
        segment = SegmentPlan(
            x_start_nm=8_000_000,
            x_end_nm=40_000_000,
            y_nm=0,
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="left",
        )
        assert segment.x_start_nm == 8_000_000
        assert segment.x_end_nm == 40_000_000
        assert segment.layer == "F.Cu"

    def test_segment_length_property(self) -> None:
        """Segment length_nm is computed correctly."""
        segment = SegmentPlan(
            x_start_nm=10_000_000,
            x_end_nm=50_000_000,
            y_nm=0,
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="test",
        )
        assert segment.length_nm == 40_000_000

    def test_zero_length_segment_allowed(self) -> None:
        """Zero-length segment (start == end) is allowed."""
        segment = SegmentPlan(
            x_start_nm=10_000_000,
            x_end_nm=10_000_000,
            y_nm=0,
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="point",
        )
        assert segment.length_nm == 0

    def test_negative_length_raises(self) -> None:
        """Segment with end < start raises ValueError."""
        with pytest.raises(ValueError, match="end.*start"):
            SegmentPlan(
                x_start_nm=50_000_000,
                x_end_nm=10_000_000,  # Invalid: end < start
                y_nm=0,
                width_nm=300_000,
                layer="F.Cu",
                net_name="SIG",
                label="invalid",
            )

    def test_zero_width_raises(self) -> None:
        """Segment with zero width raises ValueError."""
        with pytest.raises(ValueError, match="width.*positive"):
            SegmentPlan(
                x_start_nm=0,
                x_end_nm=10_000_000,
                y_nm=0,
                width_nm=0,
                layer="F.Cu",
                net_name="SIG",
                label="invalid",
            )

    def test_negative_width_raises(self) -> None:
        """Segment with negative width raises ValueError."""
        with pytest.raises(ValueError, match="width.*positive"):
            SegmentPlan(
                x_start_nm=0,
                x_end_nm=10_000_000,
                y_nm=0,
                width_nm=-100,
                layer="F.Cu",
                net_name="SIG",
                label="invalid",
            )


# =============================================================================
# LayoutPlan tests
# =============================================================================


class TestLayoutPlan:
    """Tests for LayoutPlan dataclass."""

    @pytest.fixture
    def left_port(self):
        """Create a left port for testing."""
        return PortPlan(
            x_ref_nm=5_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=8_000_000,
            signal_pad_y_nm=0,
            footprint="Coupongen_Connectors:SMA_EndLaunch_Generic",
            rotation_mdeg=0,
            side="left",
        )

    @pytest.fixture
    def right_port(self):
        """Create a right port for testing."""
        return PortPlan(
            x_ref_nm=75_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=72_000_000,
            signal_pad_y_nm=0,
            footprint="Coupongen_Connectors:SMA_EndLaunch_Generic",
            rotation_mdeg=180000,
            side="right",
        )

    @pytest.fixture
    def through_segment(self, left_port, right_port):
        """Create a through segment for F0 layout."""
        return SegmentPlan(
            x_start_nm=left_port.signal_pad_x_nm,
            x_end_nm=right_port.signal_pad_x_nm,
            y_nm=0,
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="through",
        )

    def test_create_f0_layout(self, left_port, right_port, through_segment) -> None:
        """Create F0 (through-line) layout."""
        layout = LayoutPlan(
            origin_mode=OriginMode.EDGE_L_CENTER,
            board_length_nm=80_000_000,
            board_width_nm=20_000_000,
            board_corner_radius_nm=2_000_000,
            left_port=left_port,
            right_port=right_port,
            segments=(through_segment,),
            x_disc_nm=None,
            y_centerline_nm=0,
            coupon_family="F0_THROUGH_LINE",
        )
        assert layout.has_discontinuity is False
        assert layout.coupon_family == "F0_THROUGH_LINE"

    def test_board_edge_properties(self, left_port, right_port, through_segment) -> None:
        """Board edge properties are computed correctly."""
        layout = LayoutPlan(
            origin_mode=OriginMode.EDGE_L_CENTER,
            board_length_nm=80_000_000,
            board_width_nm=20_000_000,
            board_corner_radius_nm=2_000_000,
            left_port=left_port,
            right_port=right_port,
            segments=(through_segment,),
            x_disc_nm=None,
            y_centerline_nm=0,
            coupon_family="F0",
        )
        assert layout.x_board_left_edge_nm == 0
        assert layout.x_board_right_edge_nm == 80_000_000
        assert layout.y_board_top_edge_nm == 10_000_000
        assert layout.y_board_bottom_edge_nm == -10_000_000

    def test_total_trace_length(self, left_port, right_port, through_segment) -> None:
        """Total trace length is sum of all segments."""
        layout = LayoutPlan(
            origin_mode=OriginMode.EDGE_L_CENTER,
            board_length_nm=80_000_000,
            board_width_nm=20_000_000,
            board_corner_radius_nm=0,
            left_port=left_port,
            right_port=right_port,
            segments=(through_segment,),
            x_disc_nm=None,
            y_centerline_nm=0,
            coupon_family="F0",
        )
        # 72_000_000 - 8_000_000 = 64_000_000
        assert layout.total_trace_length_nm == 64_000_000

    def test_get_segment_by_label(self, left_port, right_port, through_segment) -> None:
        """Get segment by label works correctly."""
        layout = LayoutPlan(
            origin_mode=OriginMode.EDGE_L_CENTER,
            board_length_nm=80_000_000,
            board_width_nm=20_000_000,
            board_corner_radius_nm=0,
            left_port=left_port,
            right_port=right_port,
            segments=(through_segment,),
            x_disc_nm=None,
            y_centerline_nm=0,
            coupon_family="F0",
        )
        assert layout.get_segment_by_label("through") == through_segment
        assert layout.get_segment_by_label("nonexistent") is None

    def test_zero_board_length_raises(self, left_port, right_port, through_segment) -> None:
        """Zero board length raises ValueError."""
        with pytest.raises(ValueError, match="Board length.*positive"):
            LayoutPlan(
                origin_mode=OriginMode.EDGE_L_CENTER,
                board_length_nm=0,
                board_width_nm=20_000_000,
                board_corner_radius_nm=0,
                left_port=left_port,
                right_port=right_port,
                segments=(through_segment,),
                x_disc_nm=None,
                y_centerline_nm=0,
                coupon_family="F0",
            )

    def test_negative_corner_radius_raises(self, left_port, right_port, through_segment) -> None:
        """Negative corner radius raises ValueError."""
        with pytest.raises(ValueError, match="corner radius.*non-negative"):
            LayoutPlan(
                origin_mode=OriginMode.EDGE_L_CENTER,
                board_length_nm=80_000_000,
                board_width_nm=20_000_000,
                board_corner_radius_nm=-100,
                left_port=left_port,
                right_port=right_port,
                segments=(through_segment,),
                x_disc_nm=None,
                y_centerline_nm=0,
                coupon_family="F0",
            )

    def test_empty_segments_raises(self, left_port, right_port) -> None:
        """Empty segments tuple raises ValueError."""
        with pytest.raises(ValueError, match="At least one.*segment"):
            LayoutPlan(
                origin_mode=OriginMode.EDGE_L_CENTER,
                board_length_nm=80_000_000,
                board_width_nm=20_000_000,
                board_corner_radius_nm=0,
                left_port=left_port,
                right_port=right_port,
                segments=(),
                x_disc_nm=None,
                y_centerline_nm=0,
                coupon_family="F0",
            )


# =============================================================================
# derive_right_length_nm tests
# =============================================================================


class TestDeriveRightLengthNm:
    """Tests for derive_right_length_nm function."""

    def test_basic_derivation(self) -> None:
        """Basic right length derivation."""
        # left_port_x = 10, right_port_x = 70, left_length = 30
        # x_disc = 10 + 30 = 40
        # right_length = 70 - 40 = 30
        result = derive_right_length_nm(
            left_port_x_nm=10_000_000,
            right_port_x_nm=70_000_000,
            left_length_nm=30_000_000,
        )
        assert result == 30_000_000

    def test_asymmetric_lengths(self) -> None:
        """Asymmetric trace lengths work correctly."""
        # left_port_x = 10, right_port_x = 70, left_length = 40
        # x_disc = 10 + 40 = 50
        # right_length = 70 - 50 = 20
        result = derive_right_length_nm(
            left_port_x_nm=10_000_000,
            right_port_x_nm=70_000_000,
            left_length_nm=40_000_000,
        )
        assert result == 20_000_000

    def test_zero_right_length(self) -> None:
        """Zero right length (discontinuity at right port) is valid."""
        result = derive_right_length_nm(
            left_port_x_nm=10_000_000,
            right_port_x_nm=70_000_000,
            left_length_nm=60_000_000,
        )
        assert result == 0

    def test_negative_result_raises(self) -> None:
        """Left length too long raises ValueError."""
        with pytest.raises(ValueError, match="negative"):
            derive_right_length_nm(
                left_port_x_nm=10_000_000,
                right_port_x_nm=70_000_000,
                left_length_nm=70_000_000,  # Too long
            )


# =============================================================================
# F1 layout invariant tests
# =============================================================================


class TestF1LayoutInvariant:
    """Tests for F1 layout continuity invariant enforcement."""

    def test_f1_left_segment_must_end_at_discontinuity(self) -> None:
        """F1 left segment end must equal discontinuity X."""
        left_port = PortPlan(
            x_ref_nm=5_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=8_000_000,
            signal_pad_y_nm=0,
            footprint="Test:FP",
            rotation_mdeg=0,
            side="left",
        )
        right_port = PortPlan(
            x_ref_nm=75_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=72_000_000,
            signal_pad_y_nm=0,
            footprint="Test:FP",
            rotation_mdeg=180000,
            side="right",
        )
        # Left segment ends at wrong position
        left_seg = SegmentPlan(
            x_start_nm=8_000_000,
            x_end_nm=30_000_000,  # Should be 40_000_000
            y_nm=0,
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="left",
        )
        right_seg = SegmentPlan(
            x_start_nm=40_000_000,
            x_end_nm=72_000_000,
            y_nm=0,
            width_nm=300_000,
            layer="B.Cu",
            net_name="SIG",
            label="right",
        )

        with pytest.raises(ValueError, match="Left segment end"):
            LayoutPlan(
                origin_mode=OriginMode.EDGE_L_CENTER,
                board_length_nm=80_000_000,
                board_width_nm=20_000_000,
                board_corner_radius_nm=0,
                left_port=left_port,
                right_port=right_port,
                segments=(left_seg, right_seg),
                x_disc_nm=40_000_000,
                y_centerline_nm=0,
                coupon_family="F1",
            )

    def test_f1_right_segment_must_start_at_discontinuity(self) -> None:
        """F1 right segment start must equal discontinuity X."""
        left_port = PortPlan(
            x_ref_nm=5_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=8_000_000,
            signal_pad_y_nm=0,
            footprint="Test:FP",
            rotation_mdeg=0,
            side="left",
        )
        right_port = PortPlan(
            x_ref_nm=75_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=72_000_000,
            signal_pad_y_nm=0,
            footprint="Test:FP",
            rotation_mdeg=180000,
            side="right",
        )
        left_seg = SegmentPlan(
            x_start_nm=8_000_000,
            x_end_nm=40_000_000,
            y_nm=0,
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="left",
        )
        # Right segment starts at wrong position
        right_seg = SegmentPlan(
            x_start_nm=45_000_000,
            x_end_nm=72_000_000,  # Should be 40_000_000
            y_nm=0,
            width_nm=300_000,
            layer="B.Cu",
            net_name="SIG",
            label="right",
        )

        with pytest.raises(ValueError, match="Right segment start"):
            LayoutPlan(
                origin_mode=OriginMode.EDGE_L_CENTER,
                board_length_nm=80_000_000,
                board_width_nm=20_000_000,
                board_corner_radius_nm=0,
                left_port=left_port,
                right_port=right_port,
                segments=(left_seg, right_seg),
                x_disc_nm=40_000_000,
                y_centerline_nm=0,
                coupon_family="F1",
            )

    def test_f1_valid_layout_passes_invariant(self) -> None:
        """Valid F1 layout passes continuity invariant check."""
        left_port = PortPlan(
            x_ref_nm=5_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=8_000_000,
            signal_pad_y_nm=0,
            footprint="Test:FP",
            rotation_mdeg=0,
            side="left",
        )
        right_port = PortPlan(
            x_ref_nm=75_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=72_000_000,
            signal_pad_y_nm=0,
            footprint="Test:FP",
            rotation_mdeg=180000,
            side="right",
        )
        left_seg = SegmentPlan(
            x_start_nm=8_000_000,
            x_end_nm=40_000_000,
            y_nm=0,
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="left",
        )
        right_seg = SegmentPlan(
            x_start_nm=40_000_000,
            x_end_nm=72_000_000,
            y_nm=0,
            width_nm=300_000,
            layer="B.Cu",
            net_name="SIG",
            label="right",
        )

        # Should not raise
        layout = LayoutPlan(
            origin_mode=OriginMode.EDGE_L_CENTER,
            board_length_nm=80_000_000,
            board_width_nm=20_000_000,
            board_corner_radius_nm=2_000_000,
            left_port=left_port,
            right_port=right_port,
            segments=(left_seg, right_seg),
            x_disc_nm=40_000_000,
            y_centerline_nm=0,
            coupon_family="F1_SINGLE_ENDED_VIA",
        )
        assert layout.has_discontinuity is True
        assert layout.x_disc_nm == 40_000_000


# =============================================================================
# validate_connectivity tests
# =============================================================================


class TestValidateConnectivity:
    """Tests for LayoutPlan.validate_connectivity method."""

    def test_valid_f0_layout_no_errors(self) -> None:
        """Valid F0 layout passes connectivity validation."""
        left_port = PortPlan(
            x_ref_nm=5_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=8_000_000,
            signal_pad_y_nm=0,
            footprint="Test:FP",
            rotation_mdeg=0,
            side="left",
        )
        right_port = PortPlan(
            x_ref_nm=75_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=72_000_000,
            signal_pad_y_nm=0,
            footprint="Test:FP",
            rotation_mdeg=180000,
            side="right",
        )
        through_seg = SegmentPlan(
            x_start_nm=8_000_000,
            x_end_nm=72_000_000,
            y_nm=0,
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="through",
        )

        layout = LayoutPlan(
            origin_mode=OriginMode.EDGE_L_CENTER,
            board_length_nm=80_000_000,
            board_width_nm=20_000_000,
            board_corner_radius_nm=0,
            left_port=left_port,
            right_port=right_port,
            segments=(through_seg,),
            x_disc_nm=None,
            y_centerline_nm=0,
            coupon_family="F0",
        )
        errors = layout.validate_connectivity()
        assert errors == []

    def test_segment_not_connected_to_left_port(self) -> None:
        """Segment not starting at left port signal pad is flagged."""
        left_port = PortPlan(
            x_ref_nm=5_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=8_000_000,
            signal_pad_y_nm=0,
            footprint="Test:FP",
            rotation_mdeg=0,
            side="left",
        )
        right_port = PortPlan(
            x_ref_nm=75_000_000,
            y_ref_nm=0,
            signal_pad_x_nm=72_000_000,
            signal_pad_y_nm=0,
            footprint="Test:FP",
            rotation_mdeg=180000,
            side="right",
        )
        # Segment starts at wrong position
        through_seg = SegmentPlan(
            x_start_nm=10_000_000,
            x_end_nm=72_000_000,  # Should start at 8_000_000
            y_nm=0,
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="through",
        )

        layout = LayoutPlan(
            origin_mode=OriginMode.EDGE_L_CENTER,
            board_length_nm=80_000_000,
            board_width_nm=20_000_000,
            board_corner_radius_nm=0,
            left_port=left_port,
            right_port=right_port,
            segments=(through_seg,),
            x_disc_nm=None,
            y_centerline_nm=0,
            coupon_family="F0",
        )
        errors = layout.validate_connectivity()
        assert len(errors) == 1
        assert "left port" in errors[0].lower()
