"""Parameter mapping from normalized u vectors to CouponSpec for F1 family.

Per CP-4.3, this module provides the mapping between normalized design vectors
u in [0,1]^d and the physical parameters in CouponSpec.

The mapping is defined per coupon family (currently only F1 is supported) and
must be consistent with the GPU filter's parameter space definition.

This module enables the build-batch command to:
1. Take a spec template and normalized u vectors
2. Map each u vector to physical parameters
3. Generate a CouponSpec for each u vector
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .constraints.gpu_filter import FamilyF1ParameterSpace
from .spec import CouponSpec


def get_f1_parameter_space() -> FamilyF1ParameterSpace:
    """Get the F1 family parameter space definition.

    Returns:
        FamilyF1ParameterSpace with all parameter mappings
    """
    return FamilyF1ParameterSpace()


def u_to_spec_params_f1(
    u: NDArray[np.floating[Any]],
    param_space: FamilyF1ParameterSpace | None = None,
) -> dict[str, int]:
    """Convert a single normalized u vector to physical parameters for F1 family.

    This maps a normalized design vector u in [0,1]^d to the integer nm
    parameter values used in CouponSpec.

    Args:
        u: Single normalized vector of shape (d,) with values in [0, 1]
        param_space: Parameter space definition (uses default F1 space if None)

    Returns:
        Dictionary mapping parameter names to integer nm values
    """
    param_space = param_space or FamilyF1ParameterSpace()

    # Convert single vector to batch of 1 for consistency with batch API
    u_batch = u.reshape(1, -1)
    params = param_space.to_physical_batch(u_batch, np)

    # Extract single values and convert to int
    return {name: int(arr[0]) for name, arr in params.items()}


def apply_params_to_spec(
    spec_template: CouponSpec,
    params: dict[str, int],
) -> CouponSpec:
    """Apply physical parameters to a spec template to create a new CouponSpec.

    This updates the relevant fields of a CouponSpec based on the parameter values.
    The mapping from parameter names to spec fields is:

    - trace_width_nm -> transmission_line.w_nm
    - trace_gap_nm -> transmission_line.gap_nm
    - board_width_nm -> board.outline.width_nm
    - board_length_nm -> board.outline.length_nm
    - corner_radius_nm -> board.outline.corner_radius_nm
    - signal_drill_nm -> discontinuity.signal_via.drill_nm
    - signal_via_diameter_nm -> discontinuity.signal_via.diameter_nm
    - signal_pad_diameter_nm -> discontinuity.signal_via.pad_diameter_nm
    - return_via_drill_nm -> discontinuity.return_vias.via.drill_nm
    - return_via_diameter_nm -> discontinuity.return_vias.via.diameter_nm
    - fence_via_drill_nm -> transmission_line.ground_via_fence.via.drill_nm
    - fence_via_diameter_nm -> transmission_line.ground_via_fence.via.diameter_nm
    - left_connector_x_nm -> connectors.left.position_nm[0]
    - right_connector_x_nm -> connectors.right.position_nm[0]
    - trace_length_left_nm -> transmission_line.length_left_nm
    - trace_length_right_nm -> transmission_line.length_right_nm
    - return_via_ring_radius_nm -> discontinuity.return_vias.radius_nm
    - fence_pitch_nm -> transmission_line.ground_via_fence.pitch_nm
    - fence_offset_nm -> transmission_line.ground_via_fence.offset_from_gap_nm

    Args:
        spec_template: Template CouponSpec to clone and modify
        params: Dictionary of parameter names to integer nm values

    Returns:
        New CouponSpec with updated parameters
    """
    # Serialize template to dict for modification
    spec_dict = spec_template.model_dump(mode="json")

    # Board outline
    if "board_width_nm" in params:
        spec_dict["board"]["outline"]["width_nm"] = params["board_width_nm"]
    if "board_length_nm" in params:
        spec_dict["board"]["outline"]["length_nm"] = params["board_length_nm"]
    if "corner_radius_nm" in params:
        spec_dict["board"]["outline"]["corner_radius_nm"] = params["corner_radius_nm"]

    # Transmission line
    if "trace_width_nm" in params:
        spec_dict["transmission_line"]["w_nm"] = params["trace_width_nm"]
    if "trace_gap_nm" in params:
        spec_dict["transmission_line"]["gap_nm"] = params["trace_gap_nm"]
    if "trace_length_left_nm" in params:
        spec_dict["transmission_line"]["length_left_nm"] = params["trace_length_left_nm"]
    if "trace_length_right_nm" in params:
        spec_dict["transmission_line"]["length_right_nm"] = params["trace_length_right_nm"]

    # Ground via fence (if present in template)
    if spec_dict["transmission_line"].get("ground_via_fence") is not None:
        fence = spec_dict["transmission_line"]["ground_via_fence"]
        if "fence_pitch_nm" in params:
            fence["pitch_nm"] = params["fence_pitch_nm"]
        if "fence_offset_nm" in params:
            fence["offset_from_gap_nm"] = params["fence_offset_nm"]
        if "fence_via_drill_nm" in params:
            fence["via"]["drill_nm"] = params["fence_via_drill_nm"]
        if "fence_via_diameter_nm" in params:
            fence["via"]["diameter_nm"] = params["fence_via_diameter_nm"]

    # Connectors
    if "left_connector_x_nm" in params:
        left_pos = list(spec_dict["connectors"]["left"]["position_nm"])
        left_pos[0] = params["left_connector_x_nm"]
        spec_dict["connectors"]["left"]["position_nm"] = tuple(left_pos)

    if "right_connector_x_nm" in params:
        right_pos = list(spec_dict["connectors"]["right"]["position_nm"])
        right_pos[0] = params["right_connector_x_nm"]
        spec_dict["connectors"]["right"]["position_nm"] = tuple(right_pos)

    # Discontinuity (if present in template - required for F1)
    if spec_dict.get("discontinuity") is not None:
        disc = spec_dict["discontinuity"]
        if "signal_drill_nm" in params:
            disc["signal_via"]["drill_nm"] = params["signal_drill_nm"]
        if "signal_via_diameter_nm" in params:
            disc["signal_via"]["diameter_nm"] = params["signal_via_diameter_nm"]
        if "signal_pad_diameter_nm" in params:
            disc["signal_via"]["pad_diameter_nm"] = params["signal_pad_diameter_nm"]

        # Return vias (if present)
        if disc.get("return_vias") is not None:
            ret_vias = disc["return_vias"]
            if "return_via_ring_radius_nm" in params:
                ret_vias["radius_nm"] = params["return_via_ring_radius_nm"]
            if "return_via_drill_nm" in params:
                ret_vias["via"]["drill_nm"] = params["return_via_drill_nm"]
            if "return_via_diameter_nm" in params:
                ret_vias["via"]["diameter_nm"] = params["return_via_diameter_nm"]

    # Reconstruct and validate the spec
    return CouponSpec.model_validate(spec_dict)


def u_to_spec_f1(
    u: NDArray[np.floating[Any]],
    spec_template: CouponSpec,
    param_space: FamilyF1ParameterSpace | None = None,
) -> CouponSpec:
    """Convert a normalized u vector to a CouponSpec for F1 family.

    This is the main entry point for u-to-spec conversion. It:
    1. Converts the normalized u vector to physical parameters
    2. Applies those parameters to the spec template
    3. Returns a new, validated CouponSpec

    Args:
        u: Single normalized vector of shape (d,) with values in [0, 1]
        spec_template: Template CouponSpec to use as base
        param_space: Parameter space definition (uses default F1 space if None)

    Returns:
        New CouponSpec with parameters from u vector
    """
    params = u_to_spec_params_f1(u, param_space)
    return apply_params_to_spec(spec_template, params)


def batch_u_to_specs_f1(
    u_batch: NDArray[np.floating[Any]],
    spec_template: CouponSpec,
    param_space: FamilyF1ParameterSpace | None = None,
) -> list[CouponSpec]:
    """Convert a batch of normalized u vectors to CouponSpecs for F1 family.

    This processes multiple u vectors efficiently by reusing the parameter
    space definition across all conversions.

    Args:
        u_batch: Array of shape (N, d) with normalized values in [0, 1]
        spec_template: Template CouponSpec to use as base
        param_space: Parameter space definition (uses default F1 space if None)

    Returns:
        List of N CouponSpec instances, one for each u vector
    """
    param_space = param_space or FamilyF1ParameterSpace()
    specs = []

    for i in range(len(u_batch)):
        spec = u_to_spec_f1(u_batch[i], spec_template, param_space)
        specs.append(spec)

    return specs
