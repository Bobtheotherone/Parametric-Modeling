"""GPU-accelerated batch constraint prefilter using CuPy.

This module implements vectorized constraint checking for Tier 0-2 constraints
on GPU, enabling filtering of millions of candidates before invoking KiCad
or expensive geometry operations.

The filter operates on normalized design vectors u in [0,1]^d and returns:
- feasible_mask: Boolean mask indicating which candidates pass all constraints
- repaired_candidates: Candidates repaired to satisfy constraints
- repair_metadata: Information about repairs applied

REQ-M1-GPU-FILTER: GPU vectorized constraint prefilter for batch candidate filtering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Try to import CuPy, fall back to NumPy if unavailable
try:
    import cupy as cp

    # Verify CUDA device is actually available and NVRTC works
    try:
        cp.cuda.Device(0).compute_capability
        # Try a tiny kernel operation to verify NVRTC is available
        # This will fail if libnvrtc.so is missing
        _test_arr = cp.array([1.0, 2.0])
        _ = _test_arr * 2.0  # This triggers NVRTC compilation
        del _test_arr
        _HAS_CUPY = True
    except (cp.cuda.runtime.CUDARuntimeError, RuntimeError, OSError):
        cp = None  # type: ignore[assignment]
        _HAS_CUPY = False
except ImportError:
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False


# Type alias for array module (either cupy or numpy)
ArrayModule = Any


def get_array_module(use_gpu: bool = True) -> ArrayModule:
    """Get the appropriate array module (CuPy or NumPy).

    Args:
        use_gpu: Whether to use GPU acceleration if available

    Returns:
        cupy if available and use_gpu=True, else numpy
    """
    if use_gpu and _HAS_CUPY:
        return cp
    return np


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return _HAS_CUPY


@dataclass(frozen=True, slots=True)
class ParameterMapping:
    """Mapping from normalized [0,1] to physical parameter space.

    Supports linear, logarithmic, and discrete mappings.

    Attributes:
        name: Parameter name (e.g., "trace_width_nm")
        index: Index in the normalized vector
        scale: Mapping type ("linear", "log", "discrete")
        min_val: Minimum physical value (for linear/log)
        max_val: Maximum physical value (for linear/log)
        values: Discrete values (for discrete mapping)
    """

    name: str
    index: int
    scale: Literal["linear", "log", "discrete"] = "linear"
    min_val: float = 0.0
    max_val: float = 1.0
    values: tuple[float, ...] | None = None

    def to_physical(self, u: Any, xp: ArrayModule) -> Any:
        """Convert normalized value(s) to physical value(s).

        Args:
            u: Normalized value(s) in [0, 1]
            xp: Array module (cupy or numpy)

        Returns:
            Physical value(s)
        """
        if self.scale == "linear":
            return self.min_val + u * (self.max_val - self.min_val)
        elif self.scale == "log":
            log_min = xp.log(max(self.min_val, 1e-10))
            log_max = xp.log(max(self.max_val, 1e-10))
            return xp.exp(log_min + u * (log_max - log_min))
        elif self.scale == "discrete" and self.values is not None:
            # For discrete, map u to index and select from values
            n_values = len(self.values)
            indices = xp.clip(xp.floor(u * n_values).astype(int), 0, n_values - 1)
            values_arr = xp.asarray(self.values)
            return values_arr[indices]
        else:
            return u * (self.max_val - self.min_val) + self.min_val

    def to_normalized(self, x: Any, xp: ArrayModule) -> Any:
        """Convert physical value(s) to normalized value(s).

        Args:
            x: Physical value(s)
            xp: Array module (cupy or numpy)

        Returns:
            Normalized value(s) in [0, 1]
        """
        if self.scale == "linear":
            return (x - self.min_val) / max(self.max_val - self.min_val, 1e-10)
        elif self.scale == "log":
            log_min = xp.log(max(self.min_val, 1e-10))
            log_max = xp.log(max(self.max_val, 1e-10))
            return (xp.log(xp.maximum(x, 1e-10)) - log_min) / max(log_max - log_min, 1e-10)
        elif self.scale == "discrete" and self.values is not None:
            # Find closest discrete value and return its normalized position
            values_arr = xp.asarray(self.values)
            n_values = len(self.values)
            # This is approximate - find closest value
            diffs = xp.abs(x[..., None] - values_arr)
            indices = xp.argmin(diffs, axis=-1)
            return (indices + 0.5) / n_values
        else:
            return (x - self.min_val) / max(self.max_val - self.min_val, 1e-10)


@dataclass
class FamilyF1ParameterSpace:
    """Parameter space definition for F1 (single-ended via) coupon family.

    This defines the mapping between normalized vectors and physical parameters
    for the F1 coupon family. Parameters are ordered to group related constraints.

    The normalized vector has dimension d where each component maps to a
    physical parameter used in constraint checking.
    """

    # Default parameter mappings for F1 family
    # Indices are assigned to group related constraints together
    mappings: tuple[ParameterMapping, ...] = field(
        default_factory=lambda: (
            # Tier 0: Direct parameter bounds
            ParameterMapping("trace_width_nm", 0, "linear", 100_000, 500_000),
            ParameterMapping("trace_gap_nm", 1, "linear", 100_000, 300_000),
            ParameterMapping("board_width_nm", 2, "linear", 10_000_000, 50_000_000),
            ParameterMapping("board_length_nm", 3, "linear", 30_000_000, 150_000_000),
            ParameterMapping("corner_radius_nm", 4, "linear", 0, 5_000_000),
            ParameterMapping("signal_drill_nm", 5, "linear", 200_000, 500_000),
            ParameterMapping("signal_via_diameter_nm", 6, "linear", 300_000, 800_000),
            ParameterMapping("signal_pad_diameter_nm", 7, "linear", 400_000, 1_200_000),
            ParameterMapping("return_via_drill_nm", 8, "linear", 200_000, 500_000),
            ParameterMapping("return_via_diameter_nm", 9, "linear", 300_000, 800_000),
            ParameterMapping("fence_via_drill_nm", 10, "linear", 200_000, 400_000),
            ParameterMapping("fence_via_diameter_nm", 11, "linear", 300_000, 700_000),
            # Tier 1: Derived parameters (annular ring computed from above)
            # Tier 2: Spatial parameters
            ParameterMapping("left_connector_x_nm", 12, "linear", 2_000_000, 10_000_000),
            ParameterMapping("right_connector_x_nm", 13, "linear", 70_000_000, 145_000_000),
            ParameterMapping("trace_length_left_nm", 14, "linear", 5_000_000, 50_000_000),
            ParameterMapping("trace_length_right_nm", 15, "linear", 5_000_000, 50_000_000),
            ParameterMapping("return_via_ring_radius_nm", 16, "linear", 800_000, 3_000_000),
            ParameterMapping("fence_pitch_nm", 17, "linear", 500_000, 3_000_000),
            ParameterMapping("fence_offset_nm", 18, "linear", 200_000, 1_500_000),
        )
    )

    @property
    def dimension(self) -> int:
        """Return the dimension of the normalized parameter space."""
        return len(self.mappings)

    def get_mapping(self, name: str) -> ParameterMapping | None:
        """Get mapping by parameter name."""
        for m in self.mappings:
            if m.name == name:
                return m
        return None

    def to_physical_batch(self, u_batch: Any, xp: ArrayModule) -> dict[str, Any]:
        """Convert batch of normalized vectors to physical parameters.

        Args:
            u_batch: Array of shape (N, d) with normalized values in [0, 1]
            xp: Array module (cupy or numpy)

        Returns:
            Dictionary mapping parameter names to arrays of shape (N,)
        """
        result = {}
        for mapping in self.mappings:
            u_col = u_batch[:, mapping.index]
            result[mapping.name] = mapping.to_physical(u_col, xp)
        return result


@dataclass(frozen=True)
class BatchFilterResult:
    """Result of batch constraint filtering.

    Attributes:
        feasible_mask: Boolean array of shape (N,) - True for feasible candidates
        repaired_u: Array of shape (N, d) with repaired normalized vectors
        repair_counts: Array of shape (N,) counting repairs per candidate
        repair_distances: Array of shape (N,) with repair distances per candidate
        tier_violations: Dict mapping tier to violation counts per candidate
        constraint_margins: Dict mapping constraint ID to margin arrays
    """

    feasible_mask: NDArray[np.bool_]
    repaired_u: NDArray[np.floating[Any]]
    repair_counts: NDArray[np.intp]
    repair_distances: NDArray[np.floating[Any]]
    tier_violations: dict[str, NDArray[np.intp]]
    constraint_margins: dict[str, NDArray[np.floating[Any]]]

    @property
    def n_candidates(self) -> int:
        """Return number of candidates."""
        return len(self.feasible_mask)

    @property
    def n_feasible(self) -> int:
        """Return number of feasible candidates."""
        return int(self.feasible_mask.sum())

    @property
    def feasibility_rate(self) -> float:
        """Return fraction of feasible candidates."""
        return self.n_feasible / max(self.n_candidates, 1)


class GPUConstraintFilter:
    """GPU-accelerated constraint filter for batch candidate evaluation.

    This filter checks Tier 0-2 constraints on batches of candidates using
    vectorized GPU operations. It can also repair candidates to satisfy
    constraints when possible.

    Tier 0: Parameter bounds (direct value checks)
    Tier 1: Derived scalar constraints (annular ring, aspect ratio)
    Tier 2: Analytic spatial constraints (clearances, fitting)

    Tier 3 (exact geometry collision) is NOT included as it requires
    non-vectorizable geometry operations.
    """

    def __init__(
        self,
        fab_limits: dict[str, int],
        param_space: FamilyF1ParameterSpace | None = None,
        use_gpu: bool = True,
    ) -> None:
        """Initialize the GPU constraint filter.

        Args:
            fab_limits: Dictionary of fab capability limits in nm
            param_space: Parameter space definition (defaults to F1 family)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.fab_limits = fab_limits
        self.param_space = param_space or FamilyF1ParameterSpace()
        self.use_gpu = use_gpu and _HAS_CUPY
        self.xp = get_array_module(self.use_gpu)

        # Cache commonly used limits
        self._min_trace_width = fab_limits.get("min_trace_width_nm", 100_000)
        self._min_gap = fab_limits.get("min_gap_nm", 100_000)
        self._min_drill = fab_limits.get("min_drill_nm", 200_000)
        self._min_via_diameter = fab_limits.get("min_via_diameter_nm", 300_000)
        self._min_annular_ring = fab_limits.get("min_annular_ring_nm", 100_000)
        self._min_edge_clearance = fab_limits.get("min_edge_clearance_nm", 200_000)
        self._min_via_to_via = fab_limits.get("min_via_to_via_nm", 200_000)
        self._min_board_width = fab_limits.get("min_board_width_nm", 5_000_000)

    def _to_device(self, arr: Any) -> Any:
        """Transfer array to GPU if using CUDA."""
        if self.use_gpu:
            return self.xp.asarray(arr)
        return np.asarray(arr)

    def _to_host(self, arr: Any) -> NDArray[Any]:
        """Transfer array back to CPU."""
        if self.use_gpu and hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    def check_tier0(self, params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Check Tier 0 parameter bounds constraints.

        Args:
            params: Dictionary of physical parameter arrays

        Returns:
            Tuple of (passed_mask, margins_dict)
        """
        xp = self.xp
        n = len(params["trace_width_nm"])
        passed = xp.ones(n, dtype=bool)
        margins: dict[str, Any] = {}

        # T0_TRACE_WIDTH_MIN
        margin = params["trace_width_nm"] - self._min_trace_width
        margins["T0_TRACE_WIDTH_MIN"] = margin
        passed &= margin >= 0

        # T0_TRACE_GAP_MIN
        margin = params["trace_gap_nm"] - self._min_gap
        margins["T0_TRACE_GAP_MIN"] = margin
        passed &= margin >= 0

        # T0_BOARD_WIDTH_MIN
        margin = params["board_width_nm"] - self._min_board_width
        margins["T0_BOARD_WIDTH_MIN"] = margin
        passed &= margin >= 0

        # T0_BOARD_LENGTH_MIN
        margin = params["board_length_nm"] - self._min_board_width
        margins["T0_BOARD_LENGTH_MIN"] = margin
        passed &= margin >= 0

        # T0_CORNER_RADIUS_MIN (must be non-negative)
        margin = params["corner_radius_nm"]
        margins["T0_CORNER_RADIUS_MIN"] = margin
        passed &= margin >= 0

        # T0_CORNER_RADIUS_MAX (must not exceed half min dimension)
        max_corner = xp.minimum(params["board_width_nm"], params["board_length_nm"]) / 2
        margin = max_corner - params["corner_radius_nm"]
        margins["T0_CORNER_RADIUS_MAX"] = margin
        passed &= margin >= 0

        # T0_SIGNAL_DRILL_MIN
        margin = params["signal_drill_nm"] - self._min_drill
        margins["T0_SIGNAL_DRILL_MIN"] = margin
        passed &= margin >= 0

        # T0_SIGNAL_VIA_DIAMETER_MIN
        margin = params["signal_via_diameter_nm"] - self._min_via_diameter
        margins["T0_SIGNAL_VIA_DIAMETER_MIN"] = margin
        passed &= margin >= 0

        # T0_SIGNAL_PAD_DIAMETER_MIN
        margin = params["signal_pad_diameter_nm"] - self._min_via_diameter
        margins["T0_SIGNAL_PAD_DIAMETER_MIN"] = margin
        passed &= margin >= 0

        # T0_RETURN_VIA_DRILL_MIN
        margin = params["return_via_drill_nm"] - self._min_drill
        margins["T0_RETURN_VIA_DRILL_MIN"] = margin
        passed &= margin >= 0

        # T0_RETURN_VIA_DIAMETER_MIN
        margin = params["return_via_diameter_nm"] - self._min_via_diameter
        margins["T0_RETURN_VIA_DIAMETER_MIN"] = margin
        passed &= margin >= 0

        # T0_FENCE_VIA_DRILL_MIN
        margin = params["fence_via_drill_nm"] - self._min_drill
        margins["T0_FENCE_VIA_DRILL_MIN"] = margin
        passed &= margin >= 0

        # T0_FENCE_VIA_DIAMETER_MIN
        margin = params["fence_via_diameter_nm"] - self._min_via_diameter
        margins["T0_FENCE_VIA_DIAMETER_MIN"] = margin
        passed &= margin >= 0

        return passed, margins

    def check_tier1(self, params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Check Tier 1 derived scalar constraints.

        Args:
            params: Dictionary of physical parameter arrays

        Returns:
            Tuple of (passed_mask, margins_dict)
        """
        xp = self.xp
        n = len(params["trace_width_nm"])
        passed = xp.ones(n, dtype=bool)
        margins: dict[str, Any] = {}

        # T1_SIGNAL_VIA_DIAMETER_GT_DRILL
        margin = params["signal_via_diameter_nm"] - params["signal_drill_nm"]
        margins["T1_SIGNAL_VIA_DIAMETER_GT_DRILL"] = margin
        passed &= margin > 0

        # T1_SIGNAL_PAD_GT_VIA
        margin = params["signal_pad_diameter_nm"] - params["signal_via_diameter_nm"]
        margins["T1_SIGNAL_PAD_GT_VIA"] = margin
        passed &= margin >= 0

        # T1_SIGNAL_ANNULAR_MIN (annular ring = (pad - drill) / 2)
        annular_ring = (params["signal_pad_diameter_nm"] - params["signal_drill_nm"]) / 2
        margin = annular_ring - self._min_annular_ring
        margins["T1_SIGNAL_ANNULAR_MIN"] = margin
        passed &= margin >= 0

        # T1_RETURN_ANNULAR_MIN
        return_annular = (params["return_via_diameter_nm"] - params["return_via_drill_nm"]) / 2
        margin = return_annular - self._min_annular_ring
        margins["T1_RETURN_ANNULAR_MIN"] = margin
        passed &= margin >= 0

        # T1_FENCE_ANNULAR_MIN
        fence_annular = (params["fence_via_diameter_nm"] - params["fence_via_drill_nm"]) / 2
        margin = fence_annular - self._min_annular_ring
        margins["T1_FENCE_ANNULAR_MIN"] = margin
        passed &= margin >= 0

        # T1_TRACE_LEFT_POSITIVE
        margin = params["trace_length_left_nm"] - 1
        margins["T1_TRACE_LEFT_POSITIVE"] = margin
        passed &= margin >= 0

        # T1_TRACE_RIGHT_POSITIVE
        margin = params["trace_length_right_nm"] - 1
        margins["T1_TRACE_RIGHT_POSITIVE"] = margin
        passed &= margin >= 0

        # T1_BOARD_ASPECT_RATIO_MAX (max 20:1)
        max_dim = xp.maximum(params["board_length_nm"], params["board_width_nm"])
        min_dim = xp.maximum(xp.minimum(params["board_length_nm"], params["board_width_nm"]), 1)
        aspect_ratio = max_dim / min_dim
        margin = 20.0 - aspect_ratio
        margins["T1_BOARD_ASPECT_RATIO_MAX"] = margin
        passed &= margin >= 0

        return passed, margins

    def check_tier2(self, params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Check Tier 2 analytic spatial constraints.

        Args:
            params: Dictionary of physical parameter arrays

        Returns:
            Tuple of (passed_mask, margins_dict)
        """
        xp = self.xp
        n = len(params["trace_width_nm"])
        passed = xp.ones(n, dtype=bool)
        margins: dict[str, Any] = {}

        board_length = params["board_length_nm"]

        # T2_LEFT_CONNECTOR_X_MIN
        margin = params["left_connector_x_nm"] - self._min_edge_clearance
        margins["T2_LEFT_CONNECTOR_X_MIN"] = margin
        passed &= margin >= 0

        # T2_LEFT_CONNECTOR_X_MAX
        margin = board_length - self._min_edge_clearance - params["left_connector_x_nm"]
        margins["T2_LEFT_CONNECTOR_X_MAX"] = margin
        passed &= margin >= 0

        # T2_RIGHT_CONNECTOR_X_MIN
        margin = params["right_connector_x_nm"] - self._min_edge_clearance
        margins["T2_RIGHT_CONNECTOR_X_MIN"] = margin
        passed &= margin >= 0

        # T2_RIGHT_CONNECTOR_X_MAX
        margin = board_length - self._min_edge_clearance - params["right_connector_x_nm"]
        margins["T2_RIGHT_CONNECTOR_X_MAX"] = margin
        passed &= margin >= 0

        # T2_TRACE_FITS_IN_BOARD
        available_length = params["right_connector_x_nm"] - params["left_connector_x_nm"]
        total_trace = params["trace_length_left_nm"] + params["trace_length_right_nm"]
        margin = available_length - total_trace
        margins["T2_TRACE_FITS_IN_BOARD"] = margin
        passed &= margin >= 0

        # T2_RETURN_VIA_RING_RADIUS (clearance from signal via)
        signal_pad_radius = params["signal_pad_diameter_nm"] / 2
        return_via_radius = params["return_via_diameter_nm"] / 2
        required_ring_radius = signal_pad_radius + return_via_radius + self._min_via_to_via
        margin = params["return_via_ring_radius_nm"] - required_ring_radius
        margins["T2_RETURN_VIA_RING_RADIUS"] = margin
        passed &= margin >= 0

        # T2_FENCE_VIA_GAP_CLEARANCE
        fence_via_edge = params["fence_offset_nm"] - params["fence_via_diameter_nm"] / 2
        margin = fence_via_edge
        margins["T2_FENCE_VIA_GAP_CLEARANCE"] = margin
        passed &= margin >= 0

        # T2_FENCE_PITCH_MIN
        min_pitch = params["fence_via_diameter_nm"] + self._min_via_to_via
        margin = params["fence_pitch_nm"] - min_pitch
        margins["T2_FENCE_PITCH_MIN"] = margin
        passed &= margin >= 0

        return passed, margins

    def repair_tier0(self, u_batch: Any, params: dict[str, Any], margins: dict[str, Any]) -> tuple[Any, Any]:
        """Repair Tier 0 violations by clamping to bounds.

        Args:
            u_batch: Normalized parameter vectors (N, d)
            params: Physical parameter values
            margins: Constraint margins from check_tier0

        Returns:
            Tuple of (repaired_u, repair_counts)
        """
        xp = self.xp
        repaired = u_batch.copy()
        repair_counts = xp.zeros(len(u_batch), dtype=int)

        # Repair trace width
        idx = self.param_space.get_mapping("trace_width_nm")
        if idx is not None:
            violating = margins["T0_TRACE_WIDTH_MIN"] < 0
            if xp.any(violating):
                u_min = idx.to_normalized(xp.asarray(self._min_trace_width), xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_min)
                repair_counts[violating] += 1

        # Repair trace gap
        idx = self.param_space.get_mapping("trace_gap_nm")
        if idx is not None:
            violating = margins["T0_TRACE_GAP_MIN"] < 0
            if xp.any(violating):
                u_min = idx.to_normalized(xp.asarray(self._min_gap), xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_min)
                repair_counts[violating] += 1

        # Repair board width
        idx = self.param_space.get_mapping("board_width_nm")
        if idx is not None:
            violating = margins["T0_BOARD_WIDTH_MIN"] < 0
            if xp.any(violating):
                u_min = idx.to_normalized(xp.asarray(self._min_board_width), xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_min)
                repair_counts[violating] += 1

        # Repair board length
        idx = self.param_space.get_mapping("board_length_nm")
        if idx is not None:
            violating = margins["T0_BOARD_LENGTH_MIN"] < 0
            if xp.any(violating):
                u_min = idx.to_normalized(xp.asarray(self._min_board_width), xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_min)
                repair_counts[violating] += 1

        # Repair corner radius min (clamp to 0)
        idx = self.param_space.get_mapping("corner_radius_nm")
        if idx is not None:
            violating = margins["T0_CORNER_RADIUS_MIN"] < 0
            if xp.any(violating):
                repaired[violating, idx.index] = 0.0
                repair_counts[violating] += 1

        # Repair signal drill
        idx = self.param_space.get_mapping("signal_drill_nm")
        if idx is not None:
            violating = margins["T0_SIGNAL_DRILL_MIN"] < 0
            if xp.any(violating):
                u_min = idx.to_normalized(xp.asarray(self._min_drill), xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_min)
                repair_counts[violating] += 1

        # Repair signal via diameter
        idx = self.param_space.get_mapping("signal_via_diameter_nm")
        if idx is not None:
            violating = margins["T0_SIGNAL_VIA_DIAMETER_MIN"] < 0
            if xp.any(violating):
                u_min = idx.to_normalized(xp.asarray(self._min_via_diameter), xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_min)
                repair_counts[violating] += 1

        # Repair signal pad diameter
        idx = self.param_space.get_mapping("signal_pad_diameter_nm")
        if idx is not None:
            violating = margins["T0_SIGNAL_PAD_DIAMETER_MIN"] < 0
            if xp.any(violating):
                u_min = idx.to_normalized(xp.asarray(self._min_via_diameter), xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_min)
                repair_counts[violating] += 1

        # Repair return via drill
        idx = self.param_space.get_mapping("return_via_drill_nm")
        if idx is not None:
            violating = margins["T0_RETURN_VIA_DRILL_MIN"] < 0
            if xp.any(violating):
                u_min = idx.to_normalized(xp.asarray(self._min_drill), xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_min)
                repair_counts[violating] += 1

        # Repair return via diameter
        idx = self.param_space.get_mapping("return_via_diameter_nm")
        if idx is not None:
            violating = margins["T0_RETURN_VIA_DIAMETER_MIN"] < 0
            if xp.any(violating):
                u_min = idx.to_normalized(xp.asarray(self._min_via_diameter), xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_min)
                repair_counts[violating] += 1

        # Repair fence via drill
        idx = self.param_space.get_mapping("fence_via_drill_nm")
        if idx is not None:
            violating = margins["T0_FENCE_VIA_DRILL_MIN"] < 0
            if xp.any(violating):
                u_min = idx.to_normalized(xp.asarray(self._min_drill), xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_min)
                repair_counts[violating] += 1

        # Repair fence via diameter
        idx = self.param_space.get_mapping("fence_via_diameter_nm")
        if idx is not None:
            violating = margins["T0_FENCE_VIA_DIAMETER_MIN"] < 0
            if xp.any(violating):
                u_min = idx.to_normalized(xp.asarray(self._min_via_diameter), xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_min)
                repair_counts[violating] += 1

        return repaired, repair_counts

    def repair_tier1(self, u_batch: Any, params: dict[str, Any], margins: dict[str, Any]) -> tuple[Any, Any]:
        """Repair Tier 1 violations (annular rings, etc.).

        Args:
            u_batch: Normalized parameter vectors (N, d)
            params: Physical parameter values
            margins: Constraint margins from check_tier1

        Returns:
            Tuple of (repaired_u, repair_counts)
        """
        xp = self.xp
        repaired = u_batch.copy()
        repair_counts = xp.zeros(len(u_batch), dtype=int)

        # Repair signal annular ring by increasing pad diameter
        pad_idx = self.param_space.get_mapping("signal_pad_diameter_nm")
        drill_idx = self.param_space.get_mapping("signal_drill_nm")
        if pad_idx is not None and drill_idx is not None:
            violating = margins["T1_SIGNAL_ANNULAR_MIN"] < 0
            if xp.any(violating):
                # Required pad = drill + 2 * min_annular
                drill_vals = params["signal_drill_nm"][violating]
                required_pad = drill_vals + 2 * self._min_annular_ring
                u_required = pad_idx.to_normalized(required_pad, xp)
                repaired[violating, pad_idx.index] = xp.maximum(repaired[violating, pad_idx.index], u_required)
                repair_counts[violating] += 1

        # Repair return via annular ring
        dia_idx = self.param_space.get_mapping("return_via_diameter_nm")
        drill_idx = self.param_space.get_mapping("return_via_drill_nm")
        if dia_idx is not None and drill_idx is not None:
            violating = margins["T1_RETURN_ANNULAR_MIN"] < 0
            if xp.any(violating):
                drill_vals = params["return_via_drill_nm"][violating]
                required_dia = drill_vals + 2 * self._min_annular_ring
                u_required = dia_idx.to_normalized(required_dia, xp)
                repaired[violating, dia_idx.index] = xp.maximum(repaired[violating, dia_idx.index], u_required)
                repair_counts[violating] += 1

        # Repair fence via annular ring
        dia_idx = self.param_space.get_mapping("fence_via_diameter_nm")
        drill_idx = self.param_space.get_mapping("fence_via_drill_nm")
        if dia_idx is not None and drill_idx is not None:
            violating = margins["T1_FENCE_ANNULAR_MIN"] < 0
            if xp.any(violating):
                drill_vals = params["fence_via_drill_nm"][violating]
                required_dia = drill_vals + 2 * self._min_annular_ring
                u_required = dia_idx.to_normalized(required_dia, xp)
                repaired[violating, dia_idx.index] = xp.maximum(repaired[violating, dia_idx.index], u_required)
                repair_counts[violating] += 1

        return repaired, repair_counts

    def repair_tier2(self, u_batch: Any, params: dict[str, Any], margins: dict[str, Any]) -> tuple[Any, Any]:
        """Repair Tier 2 violations (spatial constraints).

        Args:
            u_batch: Normalized parameter vectors (N, d)
            params: Physical parameter values
            margins: Constraint margins from check_tier2

        Returns:
            Tuple of (repaired_u, repair_counts)
        """
        xp = self.xp
        repaired = u_batch.copy()
        repair_counts = xp.zeros(len(u_batch), dtype=int)

        # Repair left connector X position
        idx = self.param_space.get_mapping("left_connector_x_nm")
        if idx is not None:
            violating = margins["T2_LEFT_CONNECTOR_X_MIN"] < 0
            if xp.any(violating):
                u_min = idx.to_normalized(xp.asarray(self._min_edge_clearance), xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_min)
                repair_counts[violating] += 1

        # Repair right connector X position
        idx = self.param_space.get_mapping("right_connector_x_nm")
        if idx is not None:
            violating = margins["T2_RIGHT_CONNECTOR_X_MIN"] < 0
            if xp.any(violating):
                u_min = idx.to_normalized(xp.asarray(self._min_edge_clearance), xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_min)
                repair_counts[violating] += 1

        # Repair return via ring radius
        idx = self.param_space.get_mapping("return_via_ring_radius_nm")
        if idx is not None:
            violating = margins["T2_RETURN_VIA_RING_RADIUS"] < 0
            if xp.any(violating):
                # Required radius = signal_pad/2 + return_via/2 + min_via_to_via
                signal_pad_radius = params["signal_pad_diameter_nm"][violating] / 2
                return_via_radius = params["return_via_diameter_nm"][violating] / 2
                required_radius = signal_pad_radius + return_via_radius + self._min_via_to_via
                u_required = idx.to_normalized(required_radius, xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_required)
                repair_counts[violating] += 1

        # Repair fence offset
        idx = self.param_space.get_mapping("fence_offset_nm")
        if idx is not None:
            violating = margins["T2_FENCE_VIA_GAP_CLEARANCE"] < 0
            if xp.any(violating):
                # Required offset = fence_via_diameter / 2
                fence_dia = params["fence_via_diameter_nm"][violating]
                required_offset = fence_dia / 2
                u_required = idx.to_normalized(required_offset, xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_required)
                repair_counts[violating] += 1

        # Repair fence pitch
        idx = self.param_space.get_mapping("fence_pitch_nm")
        if idx is not None:
            violating = margins["T2_FENCE_PITCH_MIN"] < 0
            if xp.any(violating):
                # Required pitch = fence_via_diameter + min_via_to_via
                fence_dia = params["fence_via_diameter_nm"][violating]
                required_pitch = fence_dia + self._min_via_to_via
                u_required = idx.to_normalized(required_pitch, xp)
                repaired[violating, idx.index] = xp.maximum(repaired[violating, idx.index], u_required)
                repair_counts[violating] += 1

        return repaired, repair_counts

    def batch_filter(
        self,
        u_batch: NDArray[np.floating[Any]],
        repair: bool = True,
        max_repair_iterations: int = 3,
    ) -> BatchFilterResult:
        """Filter a batch of candidate design vectors.

        This is the main entry point for batch constraint filtering.
        It checks Tier 0-2 constraints and optionally repairs violations.

        Args:
            u_batch: Array of shape (N, d) with normalized values in [0, 1]
            repair: Whether to attempt repair of constraint violations
            max_repair_iterations: Maximum repair iterations

        Returns:
            BatchFilterResult with feasibility mask, repaired vectors, and metadata
        """
        xp = self.xp

        # Transfer to device
        u = self._to_device(u_batch.astype(np.float64))
        n_candidates = len(u)

        # Initialize tracking
        all_margins: dict[str, Any] = {}
        tier_violations: dict[str, Any] = {
            "T0": xp.zeros(n_candidates, dtype=int),
            "T1": xp.zeros(n_candidates, dtype=int),
            "T2": xp.zeros(n_candidates, dtype=int),
        }
        total_repair_counts = xp.zeros(n_candidates, dtype=int)

        # Convert to physical parameters
        params = self.param_space.to_physical_batch(u, xp)

        # Check Tier 0
        t0_passed, t0_margins = self.check_tier0(params)
        all_margins.update(t0_margins)
        for _key, margin in t0_margins.items():
            tier_violations["T0"] += (margin < 0).astype(int)

        # Check Tier 1
        t1_passed, t1_margins = self.check_tier1(params)
        all_margins.update(t1_margins)
        for _key, margin in t1_margins.items():
            tier_violations["T1"] += (margin < 0).astype(int)

        # Check Tier 2
        t2_passed, t2_margins = self.check_tier2(params)
        all_margins.update(t2_margins)
        for _key, margin in t2_margins.items():
            tier_violations["T2"] += (margin < 0).astype(int)

        # Initial feasibility
        feasible = t0_passed & t1_passed & t2_passed

        # Repair loop
        repaired_u = u.copy()
        if repair:
            for _iteration in range(max_repair_iterations):
                if xp.all(feasible):
                    break

                # Repair Tier 0
                repaired_u, repair_counts = self.repair_tier0(repaired_u, params, t0_margins)
                total_repair_counts += repair_counts

                # Recompute parameters and check
                params = self.param_space.to_physical_batch(repaired_u, xp)
                t0_passed, t0_margins = self.check_tier0(params)

                # Repair Tier 1
                repaired_u, repair_counts = self.repair_tier1(repaired_u, params, t1_margins)
                total_repair_counts += repair_counts

                # Recompute and check
                params = self.param_space.to_physical_batch(repaired_u, xp)
                t1_passed, t1_margins = self.check_tier1(params)

                # Repair Tier 2
                repaired_u, repair_counts = self.repair_tier2(repaired_u, params, t2_margins)
                total_repair_counts += repair_counts

                # Recompute and final check
                params = self.param_space.to_physical_batch(repaired_u, xp)
                t0_passed, _ = self.check_tier0(params)
                t1_passed, _ = self.check_tier1(params)
                t2_passed, _ = self.check_tier2(params)

                feasible = t0_passed & t1_passed & t2_passed

        # Compute repair distances (L2 norm in normalized space)
        repair_distances = xp.linalg.norm(repaired_u - u, axis=1)

        # Transfer results back to CPU
        result_feasible = self._to_host(feasible)
        result_repaired = self._to_host(repaired_u)
        result_repair_counts = self._to_host(total_repair_counts)
        result_repair_distances = self._to_host(repair_distances)
        result_tier_violations = {tier: self._to_host(counts) for tier, counts in tier_violations.items()}
        result_margins = {key: self._to_host(margin) for key, margin in all_margins.items()}

        return BatchFilterResult(
            feasible_mask=result_feasible,
            repaired_u=result_repaired,
            repair_counts=result_repair_counts,
            repair_distances=result_repair_distances,
            tier_violations=result_tier_violations,
            constraint_margins=result_margins,
        )


def batch_filter(
    u_batch: NDArray[np.floating[Any]],
    fab_limits: dict[str, int] | None = None,
    repair: bool = True,
    use_gpu: bool = True,
) -> BatchFilterResult:
    """Convenience function for batch constraint filtering.

    This is the primary API for GPU-accelerated constraint prefiltering.
    It checks Tier 0-2 constraints on a batch of normalized design vectors
    and returns a feasibility mask along with repaired candidates.

    Args:
        u_batch: Array of shape (N, d) with normalized values in [0, 1]
                 where d is the dimension of the parameter space (19 for F1 family)
        fab_limits: Dictionary of fab capability limits in nm. If None, uses defaults.
        repair: Whether to attempt repair of constraint violations
        use_gpu: Whether to use GPU acceleration if available

    Returns:
        BatchFilterResult containing:
        - feasible_mask: Boolean mask of feasible candidates
        - repaired_u: Repaired normalized vectors
        - repair_counts: Number of repairs per candidate
        - repair_distances: Distance moved by repair in normalized space
        - tier_violations: Violation counts by tier
        - constraint_margins: Margin values for each constraint

    Example:
        >>> import numpy as np
        >>> from formula_foundry.coupongen.constraints.gpu_filter import batch_filter
        >>>
        >>> # Generate 1M random candidate vectors
        >>> u_batch = np.random.rand(1_000_000, 19)
        >>>
        >>> # Filter candidates
        >>> result = batch_filter(u_batch)
        >>>
        >>> print(f"Feasibility rate: {result.feasibility_rate:.2%}")
        >>> print(f"Feasible candidates: {result.n_feasible}")
    """
    if fab_limits is None:
        fab_limits = {
            "min_trace_width_nm": 100_000,
            "min_gap_nm": 100_000,
            "min_drill_nm": 200_000,
            "min_via_diameter_nm": 300_000,
            "min_annular_ring_nm": 100_000,
            "min_edge_clearance_nm": 200_000,
            "min_via_to_via_nm": 200_000,
            "min_board_width_nm": 5_000_000,
        }

    filter_instance = GPUConstraintFilter(fab_limits, use_gpu=use_gpu)
    return filter_instance.batch_filter(u_batch, repair=repair)
