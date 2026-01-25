"""Launch geometry generators for connector-to-CPWG transitions.

Defines deterministic launch plans that connect connector pads to the CPWG
transmission line via a stepped (or tapered) transition. The launch plan is
computed from footprint metadata, connector placement, and fab DFM limits.

All coordinates use integer nanometers (nm) for determinism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .cpwg import CPWGSpec, GroundViaFenceSpec, generate_ground_via_fence
from .primitives import PositionNM, Via


@dataclass(frozen=True, slots=True)
class LaunchSegment:
    """Single launch transition segment.

    Attributes:
        start: Segment start position in nm.
        end: Segment end position in nm.
        width_nm: Segment width in nm.
        layer: Copper layer name.
        net_name: Net name (typically "SIG").
        label: Human-readable label for deterministic tracking.
    """

    start: PositionNM
    end: PositionNM
    width_nm: int
    layer: str
    net_name: str
    label: str

    @property
    def length_nm(self) -> int:
        """Return segment length in nm (Euclidean)."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        if dy == 0:
            return abs(dx)
        return int(round(math.hypot(dx, dy)))


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    """Launch plan for a connector.

    Captures the stepped transition between pad width and CPWG trace width,
    along with optional ground stitching vias for the launch region.
    """

    side: str
    pad_center: PositionNM
    launch_point: PositionNM
    direction_deg: float
    pad_width_nm: int
    trace_width_nm: int
    layer: str
    net_name: str
    transition_length_nm: int
    segments: tuple[LaunchSegment, ...]
    stitch_vias: tuple[Via, ...]

    def __post_init__(self) -> None:
        """Validate launch plan invariants."""
        if self.side not in ("left", "right"):
            raise ValueError(f"LaunchPlan side must be 'left' or 'right', got {self.side!r}")
        if self.pad_width_nm <= 0:
            raise ValueError(f"pad_width_nm must be positive, got {self.pad_width_nm}")
        if self.trace_width_nm <= 0:
            raise ValueError(f"trace_width_nm must be positive, got {self.trace_width_nm}")
        if self.transition_length_nm < 0:
            raise ValueError(
                f"transition_length_nm must be non-negative, got {self.transition_length_nm}"
            )


def build_launch_plan(
    *,
    side: str,
    pad_center: PositionNM,
    launch_point: PositionNM,
    launch_direction_deg: float,
    rotation_deg: int,
    pad_size_x_nm: int,
    pad_size_y_nm: int,
    trace_width_nm: int,
    trace_layer: str,
    gap_nm: int,
    min_trace_width_nm: int,
    min_gap_nm: int,
    ground_via_fence: object | None = None,
) -> LaunchPlan:
    """Build a deterministic launch plan from pad center to launch reference.

    The launch transition is represented as one or more stepped segments
    between the pad width and the CPWG trace width. Optional ground stitching
    vias are placed along the launch segment if a fence spec is provided.
    """
    if min_trace_width_nm <= 0:
        min_trace_width_nm = 1

    direction_deg = (launch_direction_deg + rotation_deg) % 360
    rad = math.radians(direction_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    # Pad dimension along launch direction (assume axis-aligned pads).
    pad_width_nm = pad_size_x_nm if abs(cos_r) >= abs(sin_r) else pad_size_y_nm
    min_step_length_nm = max(min_trace_width_nm, min_gap_nm, 1)

    dx_total = launch_point.x - pad_center.x
    dy_total = launch_point.y - pad_center.y
    if dy_total != 0:
        raise ValueError(
            "Launch transitions currently require horizontal pad-to-launch alignment."
        )

    transition_length_nm = abs(dx_total)
    segments: list[LaunchSegment] = []

    if transition_length_nm > 0:
        width_delta = abs(pad_width_nm - trace_width_nm)
        if width_delta == 0:
            step_count = 1
        elif transition_length_nm >= 3 * min_step_length_nm and width_delta >= trace_width_nm:
            step_count = 3
        elif transition_length_nm >= 2 * min_step_length_nm:
            step_count = 2
        else:
            step_count = 1

        direction_sign = 1 if dx_total >= 0 else -1
        base_len = transition_length_nm // step_count
        remainder = transition_length_nm % step_count

        current_x = pad_center.x
        for i in range(step_count):
            seg_len = base_len + (1 if i < remainder else 0)
            next_x = current_x + direction_sign * seg_len
            if step_count == 1:
                width_nm = trace_width_nm if pad_width_nm >= trace_width_nm else pad_width_nm
            else:
                frac = i / (step_count - 1)
                width_nm = int(round(pad_width_nm + (trace_width_nm - pad_width_nm) * frac))
            width_nm = max(width_nm, min_trace_width_nm)

            segments.append(
                LaunchSegment(
                    start=PositionNM(current_x, pad_center.y),
                    end=PositionNM(next_x, pad_center.y),
                    width_nm=width_nm,
                    layer=trace_layer,
                    net_name="SIG",
                    label=f"{side}_launch_{i}",
                )
            )
            current_x = next_x

    stitch_vias: tuple[Via, ...] = ()
    if ground_via_fence is not None:
        fence_enabled = getattr(ground_via_fence, "enabled", False)
        if fence_enabled and transition_length_nm > 0:
            via_spec = ground_via_fence.via
            fence_spec = GroundViaFenceSpec(
                pitch_nm=int(ground_via_fence.pitch_nm),
                offset_from_gap_nm=int(ground_via_fence.offset_from_gap_nm),
                drill_nm=int(via_spec.drill_nm),
                diameter_nm=int(via_spec.diameter_nm),
                layers=("F.Cu", "B.Cu"),
                net_id=2,
            )
            cpwg_spec = CPWGSpec(
                w_nm=int(trace_width_nm),
                gap_nm=int(gap_nm),
                length_nm=int(transition_length_nm),
                layer=trace_layer,
                net_id=1,
            )
            pos_vias, neg_vias = generate_ground_via_fence(
                pad_center, launch_point, cpwg_spec, fence_spec
            )
            stitch_vias = tuple(pos_vias + neg_vias)

    return LaunchPlan(
        side=side,
        pad_center=pad_center,
        launch_point=launch_point,
        direction_deg=direction_deg,
        pad_width_nm=pad_width_nm,
        trace_width_nm=trace_width_nm,
        layer=trace_layer,
        net_name="SIG",
        transition_length_nm=transition_length_nm,
        segments=tuple(segments),
        stitch_vias=stitch_vias,
    )
