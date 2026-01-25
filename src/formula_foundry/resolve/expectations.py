"""Expected paths and optional patterns for spec consumption tracking.

This module defines which paths are expected for each coupon family and
which paths are considered optional (not required to be consumed).

Satisfies REQ-M1-001:
    - Defines expected paths per coupon family
    - Defines optional path patterns
"""

from __future__ import annotations

import re

# Common expected paths for all coupon families
COMMON_EXPECTED_PATHS: frozenset[str] = frozenset({
    "schema_version",
    "coupon_family",
    "units",
    "toolchain",
    "toolchain.kicad",
    "toolchain.kicad.version",
    "fab_profile",
    "fab_profile.id",
    "stackup",
    "stackup.copper_layers",
    "stackup.thicknesses_nm",
    "stackup.materials",
    "board",
    "board.outline",
    "board.origin",
    "connectors",
    "connectors.left",
    "connectors.right",
    "transmission_line",
    "transmission_line.type",
    "transmission_line.layer",
    "transmission_line.w_nm",
    "transmission_line.gap_nm",
    "transmission_line.length_left_nm",
    "constraints",
    "export",
})

# F0-specific expected paths (calibration thru-line)
F0_EXPECTED_PATHS: frozenset[str] = COMMON_EXPECTED_PATHS | frozenset({
    "transmission_line.length_right_nm",
})

# F1-specific expected paths (single-ended via)
F1_EXPECTED_PATHS: frozenset[str] = COMMON_EXPECTED_PATHS | frozenset({
    "discontinuity",
    "discontinuity.type",
    "discontinuity.signal_via",
})

# Family to expected paths mapping
FAMILY_EXPECTED_PATHS: dict[str, frozenset[str]] = {
    "F0": F0_EXPECTED_PATHS,
    "F0_CAL_THRU_LINE": F0_EXPECTED_PATHS,
    "F1": F1_EXPECTED_PATHS,
    "F1_SINGLE_ENDED_VIA": F1_EXPECTED_PATHS,
}

# Patterns for optional paths (regex patterns)
OPTIONAL_PATH_PATTERNS: list[re.Pattern[str]] = [
    # Ground via fence is optional
    re.compile(r"^transmission_line\.ground_via_fence"),
    # Return vias are optional
    re.compile(r"^discontinuity\.return_vias"),
    # Antipads are optional
    re.compile(r"^discontinuity\.antipads"),
    # Plane cutouts are optional
    re.compile(r"^discontinuity\.plane_cutouts"),
    # Text fields are optional
    re.compile(r"^board\.text"),
    # Docker image hash is optional
    re.compile(r"^toolchain\.kicad\.docker_image"),
    # Constraint details are optional
    re.compile(r"^constraints\."),
    # Export details are optional
    re.compile(r"^export\."),
    # Material details are optional
    re.compile(r"^stackup\.materials\."),
    # Individual thickness values are optional (dynamic keys)
    re.compile(r"^stackup\.thicknesses_nm\."),
    # Connector position/rotation details are optional
    re.compile(r"^connectors\.(left|right)\.(position_nm|rotation_deg)"),
]


def expected_paths_for_family(coupon_family: str) -> frozenset[str]:
    """Get the expected paths for a given coupon family.

    Args:
        coupon_family: The coupon family identifier (e.g., "F0", "F1",
            "F0_CAL_THRU_LINE", "F1_SINGLE_ENDED_VIA").

    Returns:
        Frozenset of expected dot-delimited paths for the family.
        Returns empty frozenset for unknown families.
    """
    return FAMILY_EXPECTED_PATHS.get(coupon_family, frozenset())


def is_optional_path(path: str) -> bool:
    """Check if a path matches any optional pattern.

    Optional paths are not required to be consumed during resolution.
    This includes optional features like ground via fences, return vias,
    and other non-essential configuration.

    Args:
        path: The dot-delimited path to check.

    Returns:
        True if the path matches an optional pattern, False otherwise.
    """
    return any(pattern.match(path) for pattern in OPTIONAL_PATH_PATTERNS)
