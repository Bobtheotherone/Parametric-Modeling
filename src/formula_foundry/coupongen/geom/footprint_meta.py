"""Footprint metadata loader for connector footprints.

This module provides dataclasses and a loader function for footprint metadata
JSON files. Each metadata file defines anchor point, signal_pad, ground_pads,
and launch_reference for a connector footprint, enabling the geometry pipeline
to correctly position and connect transmission lines to connectors.

All coordinates are in integer nanometers (nm) for determinism.

Satisfies task cp2-2-footprint-metadata per ECO-M1-ALIGN-0001.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from formula_foundry.substrate import canonical_json_dumps, sha256_bytes

from ..paths import FOOTPRINT_META_DIR, get_footprint_module_path

if TYPE_CHECKING:
    pass

__all__ = [
    "FootprintMeta",
    "PadMeta",
    "PointMeta",
    "LaunchRefMeta",
    "CourtyardMeta",
    "load_footprint_meta",
    "get_footprint_meta_path",
    "list_available_footprint_meta",
]

_SIG_NET_NAME = "SIG"
_GND_NET_NAME = "GND"


def _extract_pad_numbers_from_kicad_mod(footprint_path: Path) -> set[str]:
    """Extract all pad numbers from a .kicad_mod file.

    Parses the footprint S-expression to find all pad definitions and returns
    the set of unique pad numbers. This is used for validating that all pads
    in the footprint are accounted for in the metadata.

    Args:
        footprint_path: Path to the .kicad_mod file.

    Returns:
        Set of normalized pad number strings found in the footprint.

    Raises:
        ValueError: If the file cannot be parsed or contains no pads.
    """
    import re

    text = footprint_path.read_text(encoding="utf-8", errors="replace")

    # Parse pad numbers using regex pattern matching
    # Matches (pad "number" ...) or (pad number ...) patterns
    # KiCad pad numbers can be quoted strings or bare identifiers
    pad_pattern = re.compile(r'\(\s*pad\s+(?:"([^"]+)"|(\S+))\s+')
    matches = pad_pattern.findall(text)

    pad_numbers: set[str] = set()
    for quoted, bare in matches:
        pad_num = quoted if quoted else bare
        if pad_num:
            normalized = _normalize_pad_number(pad_num)
            pad_numbers.add(normalized)

    if not pad_numbers:
        raise ValueError(
            f"No pads found in footprint file: {footprint_path}. A valid KiCad footprint must have at least one pad definition."
        )

    return pad_numbers


def _normalize_pad_number(pad_number: str) -> str:
    """Normalize pad numbers for deterministic comparisons."""
    normalized = str(pad_number).strip()
    if not normalized:
        raise ValueError("Pad number must be non-empty")
    return normalized


@dataclass(frozen=True, slots=True)
class PointMeta:
    """2D point in integer nanometers.

    Attributes:
        x_nm: X coordinate in nanometers.
        y_nm: Y coordinate in nanometers.
        description: Optional description of the point.
    """

    x_nm: int
    y_nm: int
    description: str = ""

    def to_tuple(self) -> tuple[int, int]:
        """Return point as (x, y) tuple."""
        return (self.x_nm, self.y_nm)


@dataclass(frozen=True, slots=True)
class PadMeta:
    """Pad metadata for signal or ground pads.

    Attributes:
        pad_number: Pad number/name in the footprint (e.g., "1", "2").
        center_x_nm: X coordinate of pad center relative to anchor in nm.
        center_y_nm: Y coordinate of pad center relative to anchor in nm.
        size_x_nm: Pad width (X dimension) in nanometers.
        size_y_nm: Pad height (Y dimension) in nanometers.
        shape: Pad shape ("rect", "roundrect", "oval", "circle", "custom").
        layers: Copper layers the pad exists on.
        net_name: Default net name for the pad.
    """

    pad_number: str
    center_x_nm: int
    center_y_nm: int
    size_x_nm: int
    size_y_nm: int
    shape: str = "rect"
    layers: tuple[str, ...] = field(default_factory=lambda: ("F.Cu",))
    net_name: str = ""

    def __post_init__(self) -> None:
        """Validate pad metadata."""
        if self.size_x_nm <= 0:
            raise ValueError(f"Pad size_x_nm must be positive, got {self.size_x_nm}")
        if self.size_y_nm <= 0:
            raise ValueError(f"Pad size_y_nm must be positive, got {self.size_y_nm}")

    @property
    def center(self) -> tuple[int, int]:
        """Return pad center as (x, y) tuple."""
        return (self.center_x_nm, self.center_y_nm)


@dataclass(frozen=True, slots=True)
class LaunchRefMeta:
    """Launch reference point metadata.

    The launch reference defines where the transmission line connects
    to the signal pad and the direction of the launch.

    Attributes:
        x_nm: X coordinate of the launch reference point in nm.
        y_nm: Y coordinate of the launch reference point in nm.
        direction_deg: Direction of transmission line launch in degrees
                       (0 = +X, 90 = +Y).
        description: Optional description of the launch reference.
    """

    x_nm: int
    y_nm: int
    direction_deg: float
    description: str = ""

    def __post_init__(self) -> None:
        """Validate launch reference."""
        if not (0 <= self.direction_deg < 360):
            raise ValueError(f"direction_deg must be in [0, 360), got {self.direction_deg}")

    def to_tuple(self) -> tuple[int, int]:
        """Return launch point as (x, y) tuple."""
        return (self.x_nm, self.y_nm)


@dataclass(frozen=True, slots=True)
class CourtyardMeta:
    """Courtyard bounding box metadata.

    Attributes:
        min_x_nm: Minimum X coordinate of courtyard bounding box.
        max_x_nm: Maximum X coordinate of courtyard bounding box.
        min_y_nm: Minimum Y coordinate of courtyard bounding box.
        max_y_nm: Maximum Y coordinate of courtyard bounding box.
    """

    min_x_nm: int
    max_x_nm: int
    min_y_nm: int
    max_y_nm: int

    @property
    def width_nm(self) -> int:
        """Courtyard width in nanometers."""
        return self.max_x_nm - self.min_x_nm

    @property
    def height_nm(self) -> int:
        """Courtyard height in nanometers."""
        return self.max_y_nm - self.min_y_nm


@dataclass(frozen=True, slots=True)
class FootprintMeta:
    """Complete footprint metadata.

    Attributes:
        schema_version: Schema version for footprint metadata format.
        id: Unique identifier for the footprint metadata.
        name: Human-readable name for the footprint.
        description: Optional description.
        footprint_lib: KiCad footprint library name.
        footprint_name: KiCad footprint name within the library.
        footprint_file: Path to the vendored .kicad_mod file.
        footprint_hash: SHA256 hash of the normalized footprint file contents.
        metadata_hash: SHA256 hash of canonicalized metadata JSON.
        connector_type: RF connector type/series (e.g., "SMA").
        anchor: Anchor point (footprint placement origin).
        signal_pad: Signal pad metadata.
        ground_pads: Tuple of ground pad metadata.
        launch_reference: Launch reference point.
        courtyard: Optional courtyard bounding box.
        impedance_ohms: Characteristic impedance in ohms.
        max_frequency_ghz: Maximum operating frequency in GHz.
    """

    schema_version: int
    id: str
    name: str
    footprint_lib: str
    footprint_name: str
    footprint_file: Path
    footprint_hash: str
    metadata_hash: str
    anchor: PointMeta
    signal_pad: PadMeta
    ground_pads: tuple[PadMeta, ...]
    launch_reference: LaunchRefMeta
    description: str = ""
    connector_type: str = ""
    courtyard: CourtyardMeta | None = None
    impedance_ohms: float | None = None
    max_frequency_ghz: float | None = None

    def __post_init__(self) -> None:
        """Validate footprint metadata."""
        if self.schema_version != 1:
            raise ValueError(f"Unsupported schema_version {self.schema_version}, expected 1")
        if len(self.ground_pads) == 0:
            raise ValueError("At least one ground pad is required")
        self._validate_pad_map()

    @property
    def footprint_path(self) -> str:
        """Full footprint path as 'lib:name'."""
        return f"{self.footprint_lib}:{self.footprint_name}"

    @property
    def signal_pad_center_nm(self) -> tuple[int, int]:
        """Signal pad center in nanometers."""
        return self.signal_pad.center

    def pad_net_map(self) -> dict[str, str]:
        """Return deterministic pad->net mapping for this footprint."""
        return self._build_pad_map()

    def _validate_pad_map(self) -> None:
        """Validate pad map conventions against metadata and footprint file.

        This method validates:
        1. Signal pad has correct net_name (SIG)
        2. Ground pads have correct net_name (GND)
        3. No pad maps to both signal and ground nets
        4. All pads in the actual footprint file are accounted for (REQ-M1-004)

        Raises:
            ValueError: If any validation fails.
        """
        pad_map = self._build_pad_map()
        self._validate_footprint_pads_coverage(pad_map)

    def _build_pad_map(self) -> dict[str, str]:
        pad_map: dict[str, str] = {}
        signal_pad_number = _normalize_pad_number(self.signal_pad.pad_number)
        signal_net = self.signal_pad.net_name or _SIG_NET_NAME
        if signal_net != _SIG_NET_NAME:
            raise ValueError(f"Signal pad net_name must be '{_SIG_NET_NAME}', got {signal_net!r}")
        pad_map[signal_pad_number] = _SIG_NET_NAME

        for ground_pad in self.ground_pads:
            ground_pad_number = _normalize_pad_number(ground_pad.pad_number)
            ground_net = ground_pad.net_name or _GND_NET_NAME
            if ground_net != _GND_NET_NAME:
                raise ValueError(f"Ground pad net_name must be '{_GND_NET_NAME}', got {ground_net!r}")
            existing = pad_map.get(ground_pad_number)
            if existing is not None and existing != _GND_NET_NAME:
                raise ValueError(f"Pad number cannot map to both signal and ground nets: {ground_pad_number!r}")
            pad_map[ground_pad_number] = _GND_NET_NAME
        return pad_map

    def _validate_footprint_pads_coverage(self, pad_map: dict[str, str]) -> None:
        """Validate that all pads in the footprint file are mapped.

        REQ-M1-004: Footprint-to-net and anchor-pad mapping MUST be deterministic
        and explicit (via `pad_map` or documented conventions) so the launch
        connects to the true signal pad and GND pads/nets are correctly assigned.

        This validation ensures no pads in the actual .kicad_mod file are missing
        from the metadata, which would leave them with undefined net assignments.

        Args:
            pad_map: The pad-to-net mapping built from metadata.

        Raises:
            ValueError: If any footprint pads are missing from the metadata.
        """
        # Extract actual pad numbers from the footprint file
        actual_pads = _extract_pad_numbers_from_kicad_mod(self.footprint_file)
        mapped_pads = set(pad_map.keys())

        # Check for missing pads (in footprint but not in metadata)
        missing_pads = actual_pads - mapped_pads
        if missing_pads:
            raise ValueError(
                f"Missing pad mapping for footprint {self.id!r}: "
                f"pads {sorted(missing_pads)} exist in {self.footprint_file.name} "
                f"but are not defined in metadata (signal_pad or ground_pads). "
                f"REQ-M1-004 requires explicit mapping for all pads."
            )

        # Check for extra pads (in metadata but not in footprint)
        extra_pads = mapped_pads - actual_pads
        if extra_pads:
            raise ValueError(
                f"Extraneous pad mapping for footprint {self.id!r}: "
                f"pads {sorted(extra_pads)} are defined in metadata "
                f"but do not exist in {self.footprint_file.name}. "
                f"Metadata must match actual footprint pads exactly."
            )


def get_footprint_meta_path(footprint_id: str) -> Path:
    """Get the path to a footprint metadata file.

    Args:
        footprint_id: The footprint identifier (e.g., "SMA_EndLaunch_Generic").

    Returns:
        Path to the metadata JSON file.
    """
    return FOOTPRINT_META_DIR / f"{footprint_id}.json"


def list_available_footprint_meta() -> list[str]:
    """List all available footprint metadata IDs.

    Returns:
        List of footprint metadata IDs (filenames without .json extension).
    """
    if not FOOTPRINT_META_DIR.exists():
        return []
    return sorted(p.stem for p in FOOTPRINT_META_DIR.glob("*.json") if not p.name.endswith(".schema.json"))


def _normalize_line_endings(text: str) -> str:
    """Normalize line endings to LF for deterministic hashing."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalized and not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def _hash_metadata(data: dict) -> str:
    """Compute deterministic hash for footprint metadata JSON."""
    canonical = canonical_json_dumps(data)
    return sha256_bytes(canonical.encode("utf-8"))


def _hash_footprint_file(path: Path) -> str:
    """Compute deterministic hash for a footprint module file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    normalized = _normalize_line_endings(text)
    return sha256_bytes(normalized.encode("utf-8"))


def _parse_pad_meta(data: dict, default_net: str = "") -> PadMeta:
    """Parse pad metadata from JSON dict."""
    layers = data.get("layers", ["F.Cu"])
    return PadMeta(
        pad_number=data["pad_number"],
        center_x_nm=data["center_x_nm"],
        center_y_nm=data["center_y_nm"],
        size_x_nm=data["size_x_nm"],
        size_y_nm=data["size_y_nm"],
        shape=data.get("shape", "rect"),
        layers=tuple(layers) if isinstance(layers, list) else (layers,),
        net_name=data.get("net_name", default_net),
    )


@lru_cache(maxsize=32)
def load_footprint_meta(footprint_id: str) -> FootprintMeta:
    """Load footprint metadata from JSON file.

    Args:
        footprint_id: The footprint identifier (e.g., "SMA_EndLaunch_Generic").
                      Can also be a full footprint path like
                      "Coupongen_Connectors:SMA_EndLaunch_Generic".

    Returns:
        FootprintMeta dataclass with all metadata.

    Raises:
        FileNotFoundError: If the metadata file doesn't exist.
        ValueError: If the JSON is invalid or doesn't match schema.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    # Extract footprint name if full path provided
    if ":" in footprint_id:
        footprint_id = footprint_id.split(":", 1)[1]

    meta_path = get_footprint_meta_path(footprint_id)

    if not meta_path.exists():
        raise FileNotFoundError(f"Footprint metadata not found: {meta_path}\nAvailable: {list_available_footprint_meta()}")

    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metadata_hash = _hash_metadata(data)

    footprint_file = get_footprint_module_path(
        data["footprint_lib"],
        data["footprint_name"],
    )
    if not footprint_file.exists():
        raise FileNotFoundError(f"Footprint module not found: {footprint_file}")
    footprint_hash = _hash_footprint_file(footprint_file)

    # Parse anchor
    anchor_data = data["anchor"]
    anchor = PointMeta(
        x_nm=anchor_data["x_nm"],
        y_nm=anchor_data["y_nm"],
        description=anchor_data.get("description", ""),
    )

    # Parse signal pad
    signal_pad = _parse_pad_meta(data["signal_pad"], default_net=_SIG_NET_NAME)

    # Parse ground pads
    ground_pads = tuple(_parse_pad_meta(gp, default_net=_GND_NET_NAME) for gp in data["ground_pads"])

    # Parse launch reference
    launch_data = data["launch_reference"]
    launch_reference = LaunchRefMeta(
        x_nm=launch_data["x_nm"],
        y_nm=launch_data["y_nm"],
        direction_deg=launch_data["direction_deg"],
        description=launch_data.get("description", ""),
    )

    # Parse optional courtyard
    courtyard = None
    if "courtyard" in data and data["courtyard"]:
        cy = data["courtyard"]
        courtyard = CourtyardMeta(
            min_x_nm=cy["min_x_nm"],
            max_x_nm=cy["max_x_nm"],
            min_y_nm=cy["min_y_nm"],
            max_y_nm=cy["max_y_nm"],
        )

    return FootprintMeta(
        schema_version=data["schema_version"],
        id=data["id"],
        name=data["name"],
        description=data.get("description", ""),
        footprint_lib=data["footprint_lib"],
        footprint_name=data["footprint_name"],
        footprint_file=footprint_file,
        footprint_hash=footprint_hash,
        metadata_hash=metadata_hash,
        connector_type=data.get("connector_type", ""),
        anchor=anchor,
        signal_pad=signal_pad,
        ground_pads=ground_pads,
        launch_reference=launch_reference,
        courtyard=courtyard,
        impedance_ohms=data.get("impedance_ohms"),
        max_frequency_ghz=data.get("max_frequency_ghz"),
    )
