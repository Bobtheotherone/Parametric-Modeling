# SPDX-License-Identifier: MIT
"""Unit tests for launch geometry generators.

Tests the LaunchSegment, LaunchPlan, and build_launch_plan functions
for connector-to-CPWG transition geometry generation.
"""

from __future__ import annotations

import pytest

from formula_foundry.coupongen.geom.launch import (
    LaunchPlan,
    LaunchSegment,
    build_launch_plan,
)
from formula_foundry.coupongen.geom.primitives import PositionNM


class TestLaunchSegment:
    """Tests for LaunchSegment data class."""

    def test_launch_segment_creation(self) -> None:
        """Basic LaunchSegment creation."""
        start = PositionNM(0, 0)
        end = PositionNM(10_000_000, 0)  # 10mm horizontal
        segment = LaunchSegment(
            start=start,
            end=end,
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="left_launch_0",
        )

        assert segment.start == start
        assert segment.end == end
        assert segment.width_nm == 300_000
        assert segment.layer == "F.Cu"
        assert segment.net_name == "SIG"
        assert segment.label == "left_launch_0"

    def test_launch_segment_horizontal_length(self) -> None:
        """Horizontal segment length calculation."""
        segment = LaunchSegment(
            start=PositionNM(0, 0),
            end=PositionNM(5_000_000, 0),
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="test",
        )

        assert segment.length_nm == 5_000_000

    def test_launch_segment_negative_direction_length(self) -> None:
        """Segment length is positive regardless of direction."""
        segment = LaunchSegment(
            start=PositionNM(10_000_000, 0),
            end=PositionNM(0, 0),  # Negative x direction
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="test",
        )

        assert segment.length_nm == 10_000_000

    def test_launch_segment_zero_length(self) -> None:
        """Zero-length segment when start equals end."""
        pos = PositionNM(5_000_000, 0)
        segment = LaunchSegment(
            start=pos,
            end=pos,
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="test",
        )

        assert segment.length_nm == 0

    def test_launch_segment_is_frozen(self) -> None:
        """LaunchSegment is immutable (frozen dataclass)."""
        segment = LaunchSegment(
            start=PositionNM(0, 0),
            end=PositionNM(1_000_000, 0),
            width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            label="test",
        )

        with pytest.raises(AttributeError):
            segment.width_nm = 500_000  # type: ignore[misc]


class TestLaunchPlan:
    """Tests for LaunchPlan data class."""

    def test_launch_plan_creation_left(self) -> None:
        """Basic LaunchPlan creation for left side."""
        pad_center = PositionNM(5_000_000, 0)
        launch_point = PositionNM(10_000_000, 0)
        segments = (
            LaunchSegment(
                start=pad_center,
                end=launch_point,
                width_nm=300_000,
                layer="F.Cu",
                net_name="SIG",
                label="left_launch_0",
            ),
        )

        plan = LaunchPlan(
            side="left",
            pad_center=pad_center,
            launch_point=launch_point,
            direction_deg=0.0,
            pad_width_nm=500_000,
            trace_width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            transition_length_nm=5_000_000,
            segments=segments,
            stitch_vias=(),
        )

        assert plan.side == "left"
        assert plan.pad_center == pad_center
        assert plan.launch_point == launch_point
        assert plan.pad_width_nm == 500_000
        assert plan.trace_width_nm == 300_000
        assert len(plan.segments) == 1
        assert len(plan.stitch_vias) == 0

    def test_launch_plan_creation_right(self) -> None:
        """Basic LaunchPlan creation for right side."""
        plan = LaunchPlan(
            side="right",
            pad_center=PositionNM(95_000_000, 0),
            launch_point=PositionNM(90_000_000, 0),
            direction_deg=180.0,
            pad_width_nm=500_000,
            trace_width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            transition_length_nm=5_000_000,
            segments=(),
            stitch_vias=(),
        )

        assert plan.side == "right"
        assert plan.direction_deg == 180.0

    def test_launch_plan_invalid_side_raises(self) -> None:
        """Invalid side value raises ValueError."""
        with pytest.raises(ValueError, match="must be 'left' or 'right'"):
            LaunchPlan(
                side="center",
                pad_center=PositionNM(0, 0),
                launch_point=PositionNM(1_000_000, 0),
                direction_deg=0.0,
                pad_width_nm=500_000,
                trace_width_nm=300_000,
                layer="F.Cu",
                net_name="SIG",
                transition_length_nm=1_000_000,
                segments=(),
                stitch_vias=(),
            )

    def test_launch_plan_zero_pad_width_raises(self) -> None:
        """Zero pad width raises ValueError."""
        with pytest.raises(ValueError, match="pad_width_nm must be positive"):
            LaunchPlan(
                side="left",
                pad_center=PositionNM(0, 0),
                launch_point=PositionNM(1_000_000, 0),
                direction_deg=0.0,
                pad_width_nm=0,
                trace_width_nm=300_000,
                layer="F.Cu",
                net_name="SIG",
                transition_length_nm=1_000_000,
                segments=(),
                stitch_vias=(),
            )

    def test_launch_plan_negative_pad_width_raises(self) -> None:
        """Negative pad width raises ValueError."""
        with pytest.raises(ValueError, match="pad_width_nm must be positive"):
            LaunchPlan(
                side="left",
                pad_center=PositionNM(0, 0),
                launch_point=PositionNM(1_000_000, 0),
                direction_deg=0.0,
                pad_width_nm=-100_000,
                trace_width_nm=300_000,
                layer="F.Cu",
                net_name="SIG",
                transition_length_nm=1_000_000,
                segments=(),
                stitch_vias=(),
            )

    def test_launch_plan_zero_trace_width_raises(self) -> None:
        """Zero trace width raises ValueError."""
        with pytest.raises(ValueError, match="trace_width_nm must be positive"):
            LaunchPlan(
                side="left",
                pad_center=PositionNM(0, 0),
                launch_point=PositionNM(1_000_000, 0),
                direction_deg=0.0,
                pad_width_nm=500_000,
                trace_width_nm=0,
                layer="F.Cu",
                net_name="SIG",
                transition_length_nm=1_000_000,
                segments=(),
                stitch_vias=(),
            )

    def test_launch_plan_negative_transition_length_raises(self) -> None:
        """Negative transition length raises ValueError."""
        with pytest.raises(ValueError, match="transition_length_nm must be non-negative"):
            LaunchPlan(
                side="left",
                pad_center=PositionNM(0, 0),
                launch_point=PositionNM(1_000_000, 0),
                direction_deg=0.0,
                pad_width_nm=500_000,
                trace_width_nm=300_000,
                layer="F.Cu",
                net_name="SIG",
                transition_length_nm=-100,
                segments=(),
                stitch_vias=(),
            )

    def test_launch_plan_zero_transition_length_allowed(self) -> None:
        """Zero transition length is valid."""
        plan = LaunchPlan(
            side="left",
            pad_center=PositionNM(5_000_000, 0),
            launch_point=PositionNM(5_000_000, 0),  # Same as pad center
            direction_deg=0.0,
            pad_width_nm=500_000,
            trace_width_nm=300_000,
            layer="F.Cu",
            net_name="SIG",
            transition_length_nm=0,
            segments=(),
            stitch_vias=(),
        )

        assert plan.transition_length_nm == 0


class TestBuildLaunchPlan:
    """Tests for build_launch_plan function."""

    def test_build_launch_plan_basic_left(self) -> None:
        """Build a basic left-side launch plan."""
        plan = build_launch_plan(
            side="left",
            pad_center=PositionNM(5_000_000, 0),
            launch_point=PositionNM(10_000_000, 0),
            launch_direction_deg=0.0,
            rotation_deg=0,
            pad_size_x_nm=800_000,
            pad_size_y_nm=500_000,
            trace_width_nm=300_000,
            trace_layer="F.Cu",
            gap_nm=180_000,
            min_trace_width_nm=150_000,
            min_gap_nm=127_000,
        )

        assert plan.side == "left"
        assert plan.pad_center == PositionNM(5_000_000, 0)
        assert plan.launch_point == PositionNM(10_000_000, 0)
        assert plan.transition_length_nm == 5_000_000
        assert len(plan.segments) > 0
        assert plan.stitch_vias == ()

    def test_build_launch_plan_basic_right(self) -> None:
        """Build a basic right-side launch plan."""
        plan = build_launch_plan(
            side="right",
            pad_center=PositionNM(95_000_000, 0),
            launch_point=PositionNM(90_000_000, 0),
            launch_direction_deg=180.0,
            rotation_deg=0,
            pad_size_x_nm=800_000,
            pad_size_y_nm=500_000,
            trace_width_nm=300_000,
            trace_layer="F.Cu",
            gap_nm=180_000,
            min_trace_width_nm=150_000,
            min_gap_nm=127_000,
        )

        assert plan.side == "right"
        assert plan.transition_length_nm == 5_000_000
        assert len(plan.segments) > 0

    def test_build_launch_plan_zero_transition(self) -> None:
        """Build launch plan where pad center equals launch point."""
        pad_pos = PositionNM(40_000_000, 0)
        plan = build_launch_plan(
            side="left",
            pad_center=pad_pos,
            launch_point=pad_pos,  # Same position
            launch_direction_deg=0.0,
            rotation_deg=0,
            pad_size_x_nm=800_000,
            pad_size_y_nm=500_000,
            trace_width_nm=300_000,
            trace_layer="F.Cu",
            gap_nm=180_000,
            min_trace_width_nm=150_000,
            min_gap_nm=127_000,
        )

        assert plan.transition_length_nm == 0
        assert len(plan.segments) == 0

    def test_build_launch_plan_with_rotation(self) -> None:
        """Build launch plan with connector rotation."""
        plan = build_launch_plan(
            side="left",
            pad_center=PositionNM(5_000_000, 0),
            launch_point=PositionNM(10_000_000, 0),
            launch_direction_deg=0.0,
            rotation_deg=180,  # Rotated connector
            pad_size_x_nm=800_000,
            pad_size_y_nm=500_000,
            trace_width_nm=300_000,
            trace_layer="F.Cu",
            gap_nm=180_000,
            min_trace_width_nm=150_000,
            min_gap_nm=127_000,
        )

        assert plan.direction_deg == 180.0

    def test_build_launch_plan_non_horizontal_raises(self) -> None:
        """Non-horizontal launch transition raises ValueError."""
        with pytest.raises(ValueError, match="horizontal pad-to-launch alignment"):
            build_launch_plan(
                side="left",
                pad_center=PositionNM(5_000_000, 0),
                launch_point=PositionNM(10_000_000, 5_000_000),  # Different y
                launch_direction_deg=0.0,
                rotation_deg=0,
                pad_size_x_nm=800_000,
                pad_size_y_nm=500_000,
                trace_width_nm=300_000,
                trace_layer="F.Cu",
                gap_nm=180_000,
                min_trace_width_nm=150_000,
                min_gap_nm=127_000,
            )

    def test_build_launch_plan_segment_widths_taper(self) -> None:
        """Launch segments should taper between pad width and trace width."""
        plan = build_launch_plan(
            side="left",
            pad_center=PositionNM(0, 0),
            launch_point=PositionNM(10_000_000, 0),  # Long enough for multiple segments
            launch_direction_deg=0.0,
            rotation_deg=0,
            pad_size_x_nm=800_000,  # Larger than trace
            pad_size_y_nm=500_000,
            trace_width_nm=300_000,
            trace_layer="F.Cu",
            gap_nm=180_000,
            min_trace_width_nm=150_000,
            min_gap_nm=127_000,
        )

        # With enough length and width difference, should get multiple segments
        if len(plan.segments) > 1:
            # First segment should be wider (closer to pad)
            # Last segment should be narrower (closer to trace)
            first_width = plan.segments[0].width_nm
            last_width = plan.segments[-1].width_nm
            assert first_width >= last_width

    def test_build_launch_plan_all_segments_meet_min_width(self) -> None:
        """All segment widths should meet minimum trace width."""
        plan = build_launch_plan(
            side="left",
            pad_center=PositionNM(0, 0),
            launch_point=PositionNM(5_000_000, 0),
            launch_direction_deg=0.0,
            rotation_deg=0,
            pad_size_x_nm=800_000,
            pad_size_y_nm=500_000,
            trace_width_nm=300_000,
            trace_layer="F.Cu",
            gap_nm=180_000,
            min_trace_width_nm=150_000,
            min_gap_nm=127_000,
        )

        for segment in plan.segments:
            assert segment.width_nm >= 150_000

    def test_build_launch_plan_deterministic(self) -> None:
        """build_launch_plan produces identical results on repeated calls."""
        kwargs = {
            "side": "left",
            "pad_center": PositionNM(5_000_000, 0),
            "launch_point": PositionNM(15_000_000, 0),
            "launch_direction_deg": 0.0,
            "rotation_deg": 0,
            "pad_size_x_nm": 800_000,
            "pad_size_y_nm": 500_000,
            "trace_width_nm": 300_000,
            "trace_layer": "F.Cu",
            "gap_nm": 180_000,
            "min_trace_width_nm": 150_000,
            "min_gap_nm": 127_000,
        }

        plan1 = build_launch_plan(**kwargs)
        plan2 = build_launch_plan(**kwargs)

        assert plan1.segments == plan2.segments
        assert plan1.stitch_vias == plan2.stitch_vias
        assert plan1.pad_width_nm == plan2.pad_width_nm
        assert plan1.trace_width_nm == plan2.trace_width_nm

    def test_build_launch_plan_segments_are_connected(self) -> None:
        """Adjacent launch segments should be connected (end of one = start of next)."""
        plan = build_launch_plan(
            side="left",
            pad_center=PositionNM(0, 0),
            launch_point=PositionNM(10_000_000, 0),
            launch_direction_deg=0.0,
            rotation_deg=0,
            pad_size_x_nm=800_000,
            pad_size_y_nm=500_000,
            trace_width_nm=300_000,
            trace_layer="F.Cu",
            gap_nm=180_000,
            min_trace_width_nm=150_000,
            min_gap_nm=127_000,
        )

        segments = plan.segments
        for i in range(len(segments) - 1):
            assert segments[i].end == segments[i + 1].start

    def test_build_launch_plan_uses_pad_x_for_horizontal(self) -> None:
        """For horizontal launch, pad_size_x_nm is used as pad width."""
        plan = build_launch_plan(
            side="left",
            pad_center=PositionNM(0, 0),
            launch_point=PositionNM(5_000_000, 0),
            launch_direction_deg=0.0,
            rotation_deg=0,
            pad_size_x_nm=900_000,  # X dimension (used for horizontal)
            pad_size_y_nm=500_000,  # Y dimension
            trace_width_nm=300_000,
            trace_layer="F.Cu",
            gap_nm=180_000,
            min_trace_width_nm=150_000,
            min_gap_nm=127_000,
        )

        assert plan.pad_width_nm == 900_000
