"""PCB stackup profiles for layer structure and material properties.

This module provides:
- Pydantic models for stackup schema validation
- Loading and caching of stackup profile JSON files
- Utility functions for stackup calculations

All dimensions are in integer nanometers (LengthNM).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .units import LengthNM

# Path to stackups directory
STACKUPS_DIR = Path(__file__).resolve().parents[3] / "coupongen" / "stackups"


class _StackupBase(BaseModel):
    """Base model for stackup components with strict validation."""

    model_config = ConfigDict(extra="forbid")


class DielectricProperties(_StackupBase):
    """Dielectric material properties."""

    er: float = Field(..., gt=1, description="Relative permittivity (dielectric constant)")
    loss_tangent: float = Field(
        ..., ge=0, le=1, description="Loss tangent (tan delta)"
    )
    material: str | None = Field(default=None, description="Material name (e.g., 'FR4')")


class StackupLayer(_StackupBase):
    """Individual layer in a stackup."""

    name: str | None = Field(default=None, min_length=1, description="Layer name")
    type: Literal["copper", "dielectric", "soldermask", "silkscreen", "paste"] = Field(
        ..., description="Type of layer"
    )
    thickness_nm: LengthNM = Field(..., ge=0, description="Layer thickness in nm")
    copper_weight_oz: float | None = Field(
        default=None, gt=0, description="Copper weight in oz/ft² (copper layers only)"
    )
    dielectric: DielectricProperties | None = Field(
        default=None, description="Dielectric properties (dielectric layers only)"
    )


class StackupProfile(_StackupBase):
    """Full PCB stackup profile.

    Defines the complete layer structure from top to bottom, including
    copper layers, dielectric substrates, and surface finishes.

    Example usage:
        stackup = load_stackup("oshpark_4layer")
        for layer in stackup.layers:
            if layer.type == "dielectric":
                print(f"Er: {layer.dielectric.er}")
    """

    schema_version: Literal[1] = Field(1, description="Schema version for stackup format")
    id: str = Field(..., min_length=1, pattern=r"^[a-z0-9_]+$")
    name: str = Field(..., min_length=1)
    description: str | None = Field(default=None)
    vendor: str | None = Field(default=None)
    copper_layers: int = Field(..., ge=1, le=32)
    total_thickness_nm: LengthNM | None = Field(
        default=None, description="Total board thickness (computed if not provided)"
    )
    layers: list[StackupLayer] = Field(..., min_length=1)
    dielectric_defaults: DielectricProperties | None = Field(
        default=None, description="Default dielectric properties"
    )

    @model_validator(mode="after")
    def validate_copper_count(self) -> "StackupProfile":
        """Validate that copper layer count matches layers list."""
        copper_count = sum(1 for layer in self.layers if layer.type == "copper")
        if copper_count != self.copper_layers:
            raise ValueError(
                f"copper_layers={self.copper_layers} but found {copper_count} copper layers in layers list"
            )
        return self


# Generate JSON schema for external validation
STACKUP_SCHEMA = StackupProfile.model_json_schema()


def load_stackup(stackup_id: str) -> StackupProfile:
    """Load a stackup profile by ID from the stackups directory.

    Args:
        stackup_id: The stackup ID (e.g., "oshpark_4layer", "generic_4layer")

    Returns:
        Validated StackupProfile instance

    Raises:
        FileNotFoundError: If stackup JSON file doesn't exist
        ValidationError: If stackup JSON is invalid
    """
    return _load_stackup_cached(stackup_id)


@lru_cache(maxsize=32)
def _load_stackup_cached(stackup_id: str) -> StackupProfile:
    """Cached implementation of load_stackup."""
    stackup_path = STACKUPS_DIR / f"{stackup_id}.json"
    if not stackup_path.exists():
        raise FileNotFoundError(f"Stackup profile not found: {stackup_path}")
    with open(stackup_path, encoding="utf-8") as f:
        data = json.load(f)
    return StackupProfile.model_validate(data)


def load_stackup_from_dict(data: dict[str, Any]) -> StackupProfile:
    """Load a stackup profile from a dictionary.

    Args:
        data: Dictionary containing stackup profile data

    Returns:
        Validated StackupProfile instance
    """
    return StackupProfile.model_validate(data)


def list_available_stackups() -> list[str]:
    """List all available stackup profile IDs.

    Returns:
        List of stackup IDs (filenames without .json extension)
    """
    if not STACKUPS_DIR.exists():
        return []
    return sorted(
        p.stem for p in STACKUPS_DIR.glob("*.json") if not p.name.endswith(".schema.json")
    )


def compute_total_thickness(stackup: StackupProfile) -> int:
    """Compute total stackup thickness from layer thicknesses.

    Args:
        stackup: A loaded StackupProfile

    Returns:
        Total thickness in nanometers
    """
    return sum(int(layer.thickness_nm) for layer in stackup.layers)


def get_copper_layer_names(stackup: StackupProfile) -> list[str]:
    """Get ordered list of copper layer names.

    Args:
        stackup: A loaded StackupProfile

    Returns:
        List of copper layer names from top to bottom
    """
    names = []
    for i, layer in enumerate(stackup.layers):
        if layer.type == "copper":
            if layer.name:
                names.append(layer.name)
            else:
                names.append(f"L{len(names) + 1}")
    return names


def get_dielectric_between_layers(
    stackup: StackupProfile, layer1_idx: int, layer2_idx: int
) -> list[StackupLayer]:
    """Get dielectric layers between two copper layers.

    Args:
        stackup: A loaded StackupProfile
        layer1_idx: Index of first copper layer (0-based from top)
        layer2_idx: Index of second copper layer (0-based from top)

    Returns:
        List of dielectric layers between the specified copper layers
    """
    copper_indices = [
        i for i, layer in enumerate(stackup.layers) if layer.type == "copper"
    ]
    if layer1_idx >= len(copper_indices) or layer2_idx >= len(copper_indices):
        raise ValueError(f"Layer index out of range (max: {len(copper_indices) - 1})")

    start_idx = copper_indices[min(layer1_idx, layer2_idx)]
    end_idx = copper_indices[max(layer1_idx, layer2_idx)]

    return [
        layer
        for layer in stackup.layers[start_idx + 1 : end_idx]
        if layer.type == "dielectric"
    ]


def get_thickness_between_layers(
    stackup: StackupProfile, layer1_idx: int, layer2_idx: int
) -> int:
    """Get total dielectric thickness between two copper layers.

    Args:
        stackup: A loaded StackupProfile
        layer1_idx: Index of first copper layer (0-based from top)
        layer2_idx: Index of second copper layer (0-based from top)

    Returns:
        Total dielectric thickness in nanometers
    """
    dielectric_layers = get_dielectric_between_layers(stackup, layer1_idx, layer2_idx)
    return sum(int(layer.thickness_nm) for layer in dielectric_layers)


def get_effective_er(stackup: StackupProfile, layer1_idx: int, layer2_idx: int) -> float:
    """Get weighted average relative permittivity between two copper layers.

    Uses thickness-weighted average of Er values for the dielectric layers
    between the specified copper layers.

    Args:
        stackup: A loaded StackupProfile
        layer1_idx: Index of first copper layer (0-based from top)
        layer2_idx: Index of second copper layer (0-based from top)

    Returns:
        Effective relative permittivity
    """
    dielectric_layers = get_dielectric_between_layers(stackup, layer1_idx, layer2_idx)
    if not dielectric_layers:
        return 1.0

    total_thickness = 0
    weighted_er = 0.0

    default_er = 4.5  # FR4 default
    if stackup.dielectric_defaults:
        default_er = stackup.dielectric_defaults.er

    for layer in dielectric_layers:
        thickness = int(layer.thickness_nm)
        total_thickness += thickness
        er = layer.dielectric.er if layer.dielectric else default_er
        weighted_er += thickness * er

    return weighted_er / total_thickness if total_thickness > 0 else default_er


def clear_stackup_cache() -> None:
    """Clear the stackup profile cache. Useful for testing."""
    _load_stackup_cached.cache_clear()
