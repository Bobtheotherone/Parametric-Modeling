"""Golden export fixtures for G1-G5 determinism gate verification.

This module provides expected layer sets and export hashes for F0 and F1 coupon families.
Used by integration tests to verify export completeness (G4) and hash stability (G5).
"""

import json
from pathlib import Path

_HERE = Path(__file__).parent


def load_layer_sets() -> dict:
    """Load expected layer sets per coupon family.

    Returns:
        Dict with keys 'F0_CAL_THRU_LINE' and 'F1_SINGLE_ENDED_VIA',
        each containing 'gerbers' and 'drills' lists.
    """
    with open(_HERE / "layer_sets.json") as f:
        return json.load(f)


def load_export_hashes() -> dict:
    """Load expected canonical export hashes per golden spec.

    Returns:
        Dict mapping spec name (e.g., 'f0_cal_001') to expected hashes.
        Hash values are 'PLACEHOLDER_PENDING_KICAD_BUILD' until first
        authoritative KiCad build populates them.
    """
    with open(_HERE / "export_hashes.json") as f:
        return json.load(f)


# Convenience exports
layer_sets = load_layer_sets()
export_hashes = load_export_hashes()

# Expected layer filenames for 4-layer boards
EXPECTED_GERBER_LAYERS_4L = [
    "F.Cu.gbr",
    "In1.Cu.gbr",
    "In2.Cu.gbr",
    "B.Cu.gbr",
    "F.Mask.gbr",
    "B.Mask.gbr",
    "F.SilkS.gbr",
    "B.SilkS.gbr",
    "Edge.Cuts.gbr",
]

EXPECTED_DRILL_FILES = [
    "drill.drl",
    "drill-NPTH.drl",
]

__all__ = [
    "load_layer_sets",
    "load_export_hashes",
    "layer_sets",
    "export_hashes",
    "EXPECTED_GERBER_LAYERS_4L",
    "EXPECTED_DRILL_FILES",
]
