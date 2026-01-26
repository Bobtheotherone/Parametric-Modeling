"""Layer set validation for coupon generation exports.

This module defines expected layer sets per family and validates that all
required layers are present in export outputs.

Per Section 13.5.3 of the design doc:
    Define and enforce a locked set for fabrication exports:
    - F.Cu, In1.Cu, In2.Cu, B.Cu (for 4-layer boards)
    - F.Mask, B.Mask
    - F.SilkS, B.SilkS (optional for fab but commonly included)
    - Edge.Cuts

    Enforce in tests that every exported fab directory contains all expected layers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Path to layer sets configuration
LAYER_SETS_PATH = Path(__file__).resolve().parents[3] / "coupongen" / "layer_sets.json"


@dataclass(frozen=True)
class LayerSetConfig:
    """Configuration for a layer set based on copper layer count."""

    copper: tuple[str, ...]
    mask: tuple[str, ...]
    silkscreen: tuple[str, ...]
    edge: tuple[str, ...]
    required: tuple[str, ...]
    optional: tuple[str, ...]

    @property
    def all_layers(self) -> tuple[str, ...]:
        """Return all layers (required + optional)."""
        return self.required + self.optional


@dataclass(frozen=True)
class FamilyOverride:
    """Family-specific layer requirements."""

    description: str
    signal_layers_min: int
    requires_via_layers: bool


@dataclass(frozen=True)
class LayerValidationResult:
    """Result of layer set validation."""

    passed: bool
    missing_layers: tuple[str, ...]
    extra_layers: tuple[str, ...]
    expected_layers: tuple[str, ...]
    actual_layers: tuple[str, ...]
    copper_layer_count: int
    family: str


class LayerSetValidationError(Exception):
    """Raised when layer set validation fails."""

    def __init__(self, result: LayerValidationResult):
        self.result = result
        missing = ", ".join(result.missing_layers) if result.missing_layers else "none"
        super().__init__(
            f"Layer set validation failed for {result.family} ({result.copper_layer_count}-layer): missing layers: {missing}"
        )


@lru_cache(maxsize=1)
def _load_layer_sets_config() -> dict:
    """Load and cache the layer sets configuration."""
    if not LAYER_SETS_PATH.exists():
        raise FileNotFoundError(f"Layer sets configuration not found: {LAYER_SETS_PATH}")
    with open(LAYER_SETS_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_layer_set_for_copper_count(copper_layers: int) -> LayerSetConfig:
    """Get the layer set configuration for a given copper layer count.

    Args:
        copper_layers: Number of copper layers (2, 4, 6, etc.)

    Returns:
        LayerSetConfig for the specified copper count

    Raises:
        ValueError: If copper layer count is not supported
    """
    config = _load_layer_sets_config()
    key = f"{copper_layers}_layer"

    if key not in config["layer_sets"]:
        supported = sorted(int(k.replace("_layer", "")) for k in config["layer_sets"])
        raise ValueError(f"Unsupported copper layer count: {copper_layers}. Supported counts: {supported}")

    layer_set = config["layer_sets"][key]
    return LayerSetConfig(
        copper=tuple(layer_set["copper"]),
        mask=tuple(layer_set["mask"]),
        silkscreen=tuple(layer_set["silkscreen"]),
        edge=tuple(layer_set["edge"]),
        required=tuple(layer_set["required"]),
        optional=tuple(layer_set["optional"]),
    )


def get_family_override(family: str) -> FamilyOverride | None:
    """Get family-specific layer requirements.

    Args:
        family: Coupon family identifier (e.g., "F0_CAL_THRU_LINE")

    Returns:
        FamilyOverride if family has specific requirements, None otherwise
    """
    config = _load_layer_sets_config()
    overrides = config.get("family_overrides", {})

    if family not in overrides:
        return None

    override = overrides[family]
    return FamilyOverride(
        description=override["description"],
        signal_layers_min=override["signal_layers_min"],
        requires_via_layers=override["requires_via_layers"],
    )


def get_gerber_extension_map() -> dict[str, str]:
    """Get mapping of layer names to gerber file extensions.

    Returns:
        Dictionary mapping layer names (e.g., "F.Cu") to extensions (e.g., "-F_Cu.gbr")
    """
    config = _load_layer_sets_config()
    return dict(config.get("gerber_extension_map", {}))


def extract_layers_from_exports(
    export_paths: list[str],
    gerber_dir: str = "gerbers/",
) -> list[str]:
    """Extract layer names from export file paths.

    Args:
        export_paths: List of relative paths to exported files
        gerber_dir: Prefix for gerber directory (default "gerbers/")

    Returns:
        List of layer names found in exports
    """
    extension_map = get_gerber_extension_map()
    # Invert map: extension -> layer name
    ext_to_layer = {ext: layer for layer, ext in extension_map.items()}

    layers = []
    for path in export_paths:
        # Only look at files in the gerber directory
        if not path.startswith(gerber_dir):
            continue

        # Check each extension mapping
        for ext, layer in ext_to_layer.items():
            if path.endswith(ext):
                layers.append(layer)
                break

    return layers


def validate_layer_set(
    export_paths: list[str],
    copper_layers: int,
    family: str,
    *,
    gerber_dir: str = "gerbers/",
    strict: bool = True,
) -> LayerValidationResult:
    """Validate that all required layers are present in exports.

    Args:
        export_paths: List of relative paths to exported files
        copper_layers: Number of copper layers in the design
        family: Coupon family identifier
        gerber_dir: Prefix for gerber directory (default "gerbers/")
        strict: If True, all required layers must be present; if False, log warnings only

    Returns:
        LayerValidationResult with validation details

    Raises:
        LayerSetValidationError: If validation fails and strict=True
    """
    layer_set = get_layer_set_for_copper_count(copper_layers)
    actual_layers = extract_layers_from_exports(export_paths, gerber_dir)

    # Check for missing required layers
    missing = [layer for layer in layer_set.required if layer not in actual_layers]

    # Check for unexpected layers (not in required or optional)
    all_expected = set(layer_set.required) | set(layer_set.optional)
    extra = [layer for layer in actual_layers if layer not in all_expected]

    passed = len(missing) == 0

    result = LayerValidationResult(
        passed=passed,
        missing_layers=tuple(missing),
        extra_layers=tuple(extra),
        expected_layers=layer_set.required,
        actual_layers=tuple(sorted(set(actual_layers))),
        copper_layer_count=copper_layers,
        family=family,
    )

    if not passed and strict:
        raise LayerSetValidationError(result)

    return result


def validate_family_layer_requirements(
    copper_layers: int,
    family: str,
) -> None:
    """Validate that the copper layer count is sufficient for the family.

    Args:
        copper_layers: Number of copper layers in the design
        family: Coupon family identifier

    Raises:
        ValueError: If the family requires more layers than available
    """
    override = get_family_override(family)
    if override is None:
        return

    if copper_layers < override.signal_layers_min:
        raise ValueError(
            f"Family {family} requires at least {override.signal_layers_min} "
            f"copper layers, but design has only {copper_layers}. "
            f"({override.description})"
        )


def layer_validation_payload(result: LayerValidationResult) -> dict:
    """Convert LayerValidationResult to a dictionary for manifest inclusion.

    Args:
        result: The validation result

    Returns:
        Dictionary suitable for JSON serialization
    """
    return {
        "passed": result.passed,
        "copper_layer_count": result.copper_layer_count,
        "family": result.family,
        "expected_layers": list(result.expected_layers),
        "actual_layers": list(result.actual_layers),
        "missing_layers": list(result.missing_layers),
        "extra_layers": list(result.extra_layers),
    }


def clear_layer_sets_cache() -> None:
    """Clear the layer sets configuration cache. Useful for testing."""
    _load_layer_sets_config.cache_clear()
