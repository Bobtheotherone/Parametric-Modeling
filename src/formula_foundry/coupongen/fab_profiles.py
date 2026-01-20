"""Fabrication capability profiles for DFM constraints.

This module provides:
- Pydantic models for fab profile schema validation
- Loading and caching of fab profile JSON files
- Default fab profiles for common PCB vendors

All dimensions are in integer nanometers (LengthNM).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .units import LengthNM

# Path to fab_profiles directory
FAB_PROFILES_DIR = Path(__file__).resolve().parents[3] / "coupongen" / "fab_profiles"


class _FabProfileBase(BaseModel):
    """Base model for fab profile components with strict validation."""

    model_config = ConfigDict(extra="forbid")


class TraceConstraints(_FabProfileBase):
    """Trace (copper track) DFM constraints."""

    min_width_nm: LengthNM = Field(..., description="Minimum trace width in nm")
    min_spacing_nm: LengthNM = Field(..., description="Minimum trace-to-trace spacing in nm")
    min_width_outer_nm: LengthNM | None = Field(default=None, description="Minimum trace width on outer layers (optional)")
    min_width_inner_nm: LengthNM | None = Field(default=None, description="Minimum trace width on inner layers (optional)")


class DrillConstraints(_FabProfileBase):
    """Drill and hole DFM constraints."""

    min_diameter_nm: LengthNM = Field(..., description="Minimum drill diameter for any hole")
    min_pth_diameter_nm: LengthNM = Field(..., description="Minimum plated through-hole drill")
    min_npth_diameter_nm: LengthNM | None = Field(default=None, description="Minimum non-plated through-hole drill (optional)")
    min_via_drill_nm: LengthNM | None = Field(default=None, description="Minimum via drill diameter (optional)")
    min_hole_to_hole_nm: LengthNM | None = Field(default=None, description="Minimum hole-to-hole spacing (optional)")
    aspect_ratio_max: float | None = Field(default=None, gt=0, description="Maximum aspect ratio for PTH holes")


class ViaConstraints(_FabProfileBase):
    """Via DFM constraints."""

    min_annular_ring_nm: LengthNM = Field(..., description="Minimum annular ring width")
    min_diameter_nm: LengthNM = Field(..., description="Minimum via pad diameter")
    min_via_to_via_nm: LengthNM | None = Field(default=None, description="Minimum via-to-via pad spacing (optional)")
    min_via_to_trace_nm: LengthNM | None = Field(default=None, description="Minimum via-to-trace spacing (optional)")


class SoldermaskConstraints(_FabProfileBase):
    """Soldermask DFM constraints."""

    min_expansion_nm: LengthNM = Field(..., ge=0, description="Minimum soldermask expansion")
    max_expansion_nm: LengthNM | None = Field(default=None, ge=0, description="Maximum soldermask expansion (optional)")
    min_web_nm: LengthNM = Field(..., description="Minimum soldermask web/dam width")
    min_opening_nm: LengthNM | None = Field(default=None, description="Minimum soldermask opening size (optional)")


class SilkscreenConstraints(_FabProfileBase):
    """Silkscreen DFM constraints."""

    min_width_nm: LengthNM = Field(..., description="Minimum silkscreen line width")
    min_height_nm: LengthNM = Field(..., description="Minimum silkscreen text height")
    min_clearance_nm: LengthNM = Field(..., ge=0, description="Minimum silkscreen-to-pad clearance")


class BoardConstraints(_FabProfileBase):
    """Board outline and edge DFM constraints."""

    min_edge_clearance_nm: LengthNM = Field(..., ge=0, description="Minimum copper-to-board-edge clearance")
    min_board_width_nm: LengthNM | None = Field(default=None, description="Minimum board width (optional)")
    min_board_length_nm: LengthNM | None = Field(default=None, description="Minimum board length (optional)")
    max_board_width_nm: LengthNM | None = Field(default=None, description="Maximum board width (optional)")
    max_board_length_nm: LengthNM | None = Field(default=None, description="Maximum board length (optional)")


class FabCapabilityProfile(_FabProfileBase):
    """Full fabrication capability profile with all DFM constraints.

    This model represents a complete set of DFM (Design for Manufacturability)
    constraints for a specific PCB fabrication vendor/process. All dimensions
    are in integer nanometers.

    Example usage:
        profile = load_fab_profile("oshpark_4layer")
        min_trace = profile.trace.min_width_nm
    """

    schema_version: Literal[1] = Field(1, description="Schema version for fab profile format")
    id: str = Field(..., min_length=1, pattern=r"^[a-z0-9_]+$")
    name: str = Field(..., min_length=1)
    vendor: str = Field(..., min_length=1)
    description: str | None = Field(default=None)
    layer_counts: list[int] | None = Field(default=None, min_length=1)
    trace: TraceConstraints
    drill: DrillConstraints
    via: ViaConstraints
    soldermask: SoldermaskConstraints
    silkscreen: SilkscreenConstraints
    board: BoardConstraints


# Generate JSON schema for external validation
FAB_PROFILE_SCHEMA = FabCapabilityProfile.model_json_schema()


def load_fab_profile(profile_id: str) -> FabCapabilityProfile:
    """Load a fab profile by ID from the fab_profiles directory.

    Args:
        profile_id: The profile ID (e.g., "oshpark_4layer", "generic")

    Returns:
        Validated FabCapabilityProfile instance

    Raises:
        FileNotFoundError: If profile JSON file doesn't exist
        ValidationError: If profile JSON is invalid
    """
    return _load_fab_profile_cached(profile_id)


@lru_cache(maxsize=32)
def _load_fab_profile_cached(profile_id: str) -> FabCapabilityProfile:
    """Cached implementation of load_fab_profile."""
    profile_path = FAB_PROFILES_DIR / f"{profile_id}.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"Fab profile not found: {profile_path}")
    with open(profile_path, encoding="utf-8") as f:
        data = json.load(f)
    return FabCapabilityProfile.model_validate(data)


def load_fab_profile_from_dict(data: dict[str, Any]) -> FabCapabilityProfile:
    """Load a fab profile from a dictionary (for inline/override specs).

    Args:
        data: Dictionary containing fab profile data

    Returns:
        Validated FabCapabilityProfile instance
    """
    return FabCapabilityProfile.model_validate(data)


def list_available_profiles() -> list[str]:
    """List all available fab profile IDs.

    Returns:
        List of profile IDs (filenames without .json extension)
    """
    if not FAB_PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in FAB_PROFILES_DIR.glob("*.json") if not p.name.endswith(".schema.json"))


def get_fab_limits(profile: FabCapabilityProfile) -> dict[str, int]:
    """Extract constraint limits from a fab profile in a flat dict format.

    This provides backwards compatibility with the existing constraint system
    that expects a flat dictionary of limits.

    Args:
        profile: A loaded FabCapabilityProfile

    Returns:
        Dictionary with constraint limits used by evaluate_constraints and gpu_filter
    """
    via_drill = profile.drill.min_via_drill_nm
    if via_drill is None:
        via_drill = profile.drill.min_pth_diameter_nm

    # Get via-to-via spacing (used by gpu_filter Tier2 constraints)
    min_via_to_via = profile.via.min_via_to_via_nm
    if min_via_to_via is None:
        # Default to hole-to-hole spacing if available, else use via diameter
        min_via_to_via = profile.drill.min_hole_to_hole_nm or 200_000

    # Get minimum board dimension (used by gpu_filter Tier0 constraints)
    min_board_width = profile.board.min_board_width_nm or 5_000_000

    return {
        "min_trace_width_nm": int(profile.trace.min_width_nm),
        "min_gap_nm": int(profile.trace.min_spacing_nm),
        "min_drill_nm": int(via_drill),
        "min_annular_ring_nm": int(profile.via.min_annular_ring_nm),
        "min_via_diameter_nm": int(profile.via.min_diameter_nm),
        "min_edge_clearance_nm": int(profile.board.min_edge_clearance_nm),
        "min_soldermask_expansion_nm": int(profile.soldermask.min_expansion_nm),
        "min_soldermask_web_nm": int(profile.soldermask.min_web_nm),
        "min_silkscreen_width_nm": int(profile.silkscreen.min_width_nm),
        "min_silkscreen_clearance_nm": int(profile.silkscreen.min_clearance_nm),
        # Additional limits for GPU batch filter (CP-4.1)
        "min_via_to_via_nm": int(min_via_to_via),
        "min_board_width_nm": int(min_board_width),
    }


def clear_profile_cache() -> None:
    """Clear the fab profile cache. Useful for testing."""
    _load_fab_profile_cached.cache_clear()
