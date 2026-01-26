from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FOOTPRINT_LIB_DIR = REPO_ROOT / "coupongen" / "libs" / "footprints"
FOOTPRINT_META_DIR = REPO_ROOT / "coupongen" / "libs" / "footprints_meta"


def parse_footprint_path(footprint_path: str) -> tuple[str, str]:
    """Parse a footprint path of the form 'Lib:Name' into components."""
    if ":" not in footprint_path:
        raise ValueError(f"Footprint path must be 'Lib:Name', got {footprint_path!r}")
    footprint_lib, footprint_name = footprint_path.split(":", 1)
    if not footprint_lib or not footprint_name:
        raise ValueError(f"Footprint path must be 'Lib:Name', got {footprint_path!r}")
    return footprint_lib, footprint_name


def get_footprint_lib_dir(footprint_lib: str) -> Path:
    """Return the .pretty directory for a footprint library."""
    return FOOTPRINT_LIB_DIR / f"{footprint_lib}.pretty"


def get_footprint_module_path(footprint_lib: str, footprint_name: str) -> Path:
    """Return the path to a .kicad_mod file for the given footprint."""
    return get_footprint_lib_dir(footprint_lib) / f"{footprint_name}.kicad_mod"
