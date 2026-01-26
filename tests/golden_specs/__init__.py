"""Golden spec YAML fixtures for G1-G5 determinism gates.

This package contains 10 F0 (calibration thru line) and 10 F1 (single-ended via)
spec YAML files that cover the parameter space for testing.

F0 specs (F0_CAL_THRU_LINE):
  - f0_cal_001: Baseline (150um trace, 100um gap, 10mm lengths)
  - f0_cal_002: Narrow trace (127um, 5mil minimum)
  - f0_cal_003: Wide trace (200um)
  - f0_cal_004: Tight gap (80um), generic profile
  - f0_cal_005: Wide gap (200um), long traces (20mm)
  - f0_cal_006: Short thru line (5mm)
  - f0_cal_007: JLCPCB fab profile
  - f0_cal_008: Asymmetric lengths (7mm/14mm)
  - f0_cal_009: Wide board (18mm width)
  - f0_cal_010: PCBWay fab profile

F1 specs (F1_SINGLE_ENDED_VIA):
  - f1_via_001: Baseline (150um trace, 300um via, QUAD return vias)
  - f1_via_002: Small via (254um minimum drill)
  - f1_via_003: Large via (400um drill, 8 RING return vias)
  - f1_via_004: Roundrect antipads
  - f1_via_005: No return vias
  - f1_via_006: JLCPCB fab profile
  - f1_via_007: Long traces (20mm per side, 6 return vias)
  - f1_via_008: Generic profile with plane cutouts
  - f1_via_009: Asymmetric lengths (8mm/16mm)
  - f1_via_010: PCBWay fab profile (6 RING return vias)

Usage:
    from pathlib import Path
    GOLDEN_SPECS_DIR = Path(__file__).parent

    # List all F0 specs
    f0_specs = list(GOLDEN_SPECS_DIR.glob("f0_cal_*.yaml"))

    # List all F1 specs
    f1_specs = list(GOLDEN_SPECS_DIR.glob("f1_via_*.yaml"))
"""

from pathlib import Path

GOLDEN_SPECS_DIR = Path(__file__).parent


def get_f0_spec_paths() -> list[Path]:
    """Return paths to all F0 golden spec YAML files."""
    return sorted(GOLDEN_SPECS_DIR.glob("f0_cal_*.yaml"))


def get_f1_spec_paths() -> list[Path]:
    """Return paths to all F1 golden spec YAML files."""
    return sorted(GOLDEN_SPECS_DIR.glob("f1_via_*.yaml"))


def get_all_spec_paths() -> list[Path]:
    """Return paths to all golden spec YAML files."""
    return get_f0_spec_paths() + get_f1_spec_paths()
