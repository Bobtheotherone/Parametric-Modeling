# SPDX-License-Identifier: MIT
"""Unit tests for resolve/expectations module.

Tests the expected paths and optional pattern matching for spec consumption tracking.

Satisfies REQ-M1-001:
    - The generator MUST track and emit spec consumption (consumed paths,
      expected paths, unused provided paths)
"""

from __future__ import annotations

import pytest

from formula_foundry.resolve.expectations import (
    COMMON_EXPECTED_PATHS,
    F0_EXPECTED_PATHS,
    F1_EXPECTED_PATHS,
    FAMILY_EXPECTED_PATHS,
    OPTIONAL_PATH_PATTERNS,
    expected_paths_for_family,
    is_optional_path,
)


class TestExpectedPathConstants:
    """Tests for the expected path constant definitions."""

    def test_common_paths_is_frozenset(self) -> None:
        """COMMON_EXPECTED_PATHS is a frozenset."""
        assert isinstance(COMMON_EXPECTED_PATHS, frozenset)

    def test_common_paths_contains_essentials(self) -> None:
        """COMMON_EXPECTED_PATHS contains essential spec paths."""
        assert "schema_version" in COMMON_EXPECTED_PATHS
        assert "coupon_family" in COMMON_EXPECTED_PATHS
        assert "units" in COMMON_EXPECTED_PATHS
        assert "toolchain" in COMMON_EXPECTED_PATHS
        assert "stackup" in COMMON_EXPECTED_PATHS
        assert "board" in COMMON_EXPECTED_PATHS
        assert "connectors" in COMMON_EXPECTED_PATHS
        assert "transmission_line" in COMMON_EXPECTED_PATHS
        assert "constraints" in COMMON_EXPECTED_PATHS
        assert "export" in COMMON_EXPECTED_PATHS

    def test_f0_paths_extends_common(self) -> None:
        """F0_EXPECTED_PATHS contains all common paths plus F0-specific."""
        assert COMMON_EXPECTED_PATHS.issubset(F0_EXPECTED_PATHS)
        assert "transmission_line.length_right_nm" in F0_EXPECTED_PATHS

    def test_f1_paths_extends_common(self) -> None:
        """F1_EXPECTED_PATHS contains all common paths plus F1-specific."""
        assert COMMON_EXPECTED_PATHS.issubset(F1_EXPECTED_PATHS)
        assert "discontinuity" in F1_EXPECTED_PATHS
        assert "discontinuity.type" in F1_EXPECTED_PATHS
        assert "discontinuity.signal_via" in F1_EXPECTED_PATHS

    def test_family_mapping_has_all_variants(self) -> None:
        """FAMILY_EXPECTED_PATHS has all family variant mappings."""
        assert "F0" in FAMILY_EXPECTED_PATHS
        assert "F0_CAL_THRU_LINE" in FAMILY_EXPECTED_PATHS
        assert "F1" in FAMILY_EXPECTED_PATHS
        assert "F1_SINGLE_ENDED_VIA" in FAMILY_EXPECTED_PATHS

    def test_family_short_and_long_names_match(self) -> None:
        """Short and long family names map to the same path sets."""
        assert FAMILY_EXPECTED_PATHS["F0"] == FAMILY_EXPECTED_PATHS["F0_CAL_THRU_LINE"]
        assert FAMILY_EXPECTED_PATHS["F1"] == FAMILY_EXPECTED_PATHS["F1_SINGLE_ENDED_VIA"]


class TestExpectedPathsForFamily:
    """Tests for expected_paths_for_family function."""

    def test_f0_returns_f0_paths(self) -> None:
        """expected_paths_for_family('F0') returns F0 paths."""
        paths = expected_paths_for_family("F0")
        assert paths == F0_EXPECTED_PATHS

    def test_f0_long_name_returns_f0_paths(self) -> None:
        """expected_paths_for_family('F0_CAL_THRU_LINE') returns F0 paths."""
        paths = expected_paths_for_family("F0_CAL_THRU_LINE")
        assert paths == F0_EXPECTED_PATHS

    def test_f1_returns_f1_paths(self) -> None:
        """expected_paths_for_family('F1') returns F1 paths."""
        paths = expected_paths_for_family("F1")
        assert paths == F1_EXPECTED_PATHS

    def test_f1_long_name_returns_f1_paths(self) -> None:
        """expected_paths_for_family('F1_SINGLE_ENDED_VIA') returns F1 paths."""
        paths = expected_paths_for_family("F1_SINGLE_ENDED_VIA")
        assert paths == F1_EXPECTED_PATHS

    def test_unknown_family_returns_empty_frozenset(self) -> None:
        """expected_paths_for_family returns empty frozenset for unknown families."""
        paths = expected_paths_for_family("F99_UNKNOWN")
        assert paths == frozenset()
        assert isinstance(paths, frozenset)

    def test_case_sensitivity(self) -> None:
        """expected_paths_for_family is case-sensitive."""
        paths = expected_paths_for_family("f0")
        assert paths == frozenset()

        paths = expected_paths_for_family("F0")
        assert paths == F0_EXPECTED_PATHS


class TestOptionalPathPatterns:
    """Tests for OPTIONAL_PATH_PATTERNS and is_optional_path function."""

    def test_optional_patterns_is_list(self) -> None:
        """OPTIONAL_PATH_PATTERNS is a list of compiled patterns."""
        assert isinstance(OPTIONAL_PATH_PATTERNS, list)
        assert len(OPTIONAL_PATH_PATTERNS) > 0

    def test_ground_via_fence_is_optional(self) -> None:
        """Ground via fence paths are optional."""
        assert is_optional_path("transmission_line.ground_via_fence")
        assert is_optional_path("transmission_line.ground_via_fence.enabled")
        assert is_optional_path("transmission_line.ground_via_fence.pitch_nm")

    def test_return_vias_is_optional(self) -> None:
        """Return vias paths are optional."""
        assert is_optional_path("discontinuity.return_vias")
        assert is_optional_path("discontinuity.return_vias.count")
        assert is_optional_path("discontinuity.return_vias.radius_nm")

    def test_antipads_is_optional(self) -> None:
        """Antipad paths are optional."""
        assert is_optional_path("discontinuity.antipads")
        assert is_optional_path("discontinuity.antipads.L2")
        assert is_optional_path("discontinuity.antipads.L2.shape")

    def test_plane_cutouts_is_optional(self) -> None:
        """Plane cutout paths are optional."""
        assert is_optional_path("discontinuity.plane_cutouts")
        assert is_optional_path("discontinuity.plane_cutouts.gnd")

    def test_board_text_is_optional(self) -> None:
        """Board text paths are optional."""
        assert is_optional_path("board.text")
        assert is_optional_path("board.text.coupon_id")
        assert is_optional_path("board.text.include_manifest_hash")

    def test_docker_image_is_optional(self) -> None:
        """Docker image path is optional."""
        assert is_optional_path("toolchain.kicad.docker_image")

    def test_constraint_details_are_optional(self) -> None:
        """Constraint detail paths are optional."""
        assert is_optional_path("constraints.drc")
        assert is_optional_path("constraints.drc.must_pass")
        assert is_optional_path("constraints.symmetry")

    def test_export_details_are_optional(self) -> None:
        """Export detail paths are optional."""
        assert is_optional_path("export.gerbers")
        assert is_optional_path("export.gerbers.enabled")
        assert is_optional_path("export.drill")

    def test_material_details_are_optional(self) -> None:
        """Material detail paths are optional."""
        assert is_optional_path("stackup.materials.er")
        assert is_optional_path("stackup.materials.loss_tangent")

    def test_thickness_values_are_optional(self) -> None:
        """Individual thickness values are optional."""
        assert is_optional_path("stackup.thicknesses_nm.core")
        assert is_optional_path("stackup.thicknesses_nm.prepreg")
        assert is_optional_path("stackup.thicknesses_nm.L1_to_L2")

    def test_connector_position_is_optional(self) -> None:
        """Connector position/rotation paths are optional."""
        assert is_optional_path("connectors.left.position_nm")
        assert is_optional_path("connectors.left.rotation_deg")
        assert is_optional_path("connectors.right.position_nm")
        assert is_optional_path("connectors.right.rotation_deg")

    def test_required_paths_are_not_optional(self) -> None:
        """Required paths are NOT optional."""
        assert not is_optional_path("schema_version")
        assert not is_optional_path("coupon_family")
        assert not is_optional_path("units")
        assert not is_optional_path("toolchain")
        assert not is_optional_path("toolchain.kicad")
        assert not is_optional_path("toolchain.kicad.version")
        assert not is_optional_path("stackup")
        assert not is_optional_path("stackup.copper_layers")
        assert not is_optional_path("board")
        assert not is_optional_path("board.outline")
        assert not is_optional_path("connectors")
        assert not is_optional_path("transmission_line")
        assert not is_optional_path("transmission_line.type")
        assert not is_optional_path("transmission_line.w_nm")
        assert not is_optional_path("transmission_line.gap_nm")

    def test_discontinuity_core_is_not_optional(self) -> None:
        """Discontinuity core paths are NOT optional (they're expected for F1)."""
        assert not is_optional_path("discontinuity")
        assert not is_optional_path("discontinuity.type")
        assert not is_optional_path("discontinuity.signal_via")


class TestIsOptionalPathEdgeCases:
    """Edge case tests for is_optional_path function."""

    def test_empty_path(self) -> None:
        """Empty path returns False."""
        assert not is_optional_path("")

    def test_partial_prefix_no_match(self) -> None:
        """Partial prefix that doesn't match returns False."""
        assert not is_optional_path("transmissi")  # incomplete
        assert not is_optional_path("discontinui")  # incomplete

    def test_nested_optional_paths(self) -> None:
        """Deeply nested optional paths still match."""
        assert is_optional_path("transmission_line.ground_via_fence.via.drill_nm")
        assert is_optional_path("discontinuity.return_vias.via.diameter_nm")

    def test_path_with_array_index(self) -> None:
        """Paths with array indices that match optional patterns."""
        # Array indices shouldn't affect optional matching for these patterns
        assert is_optional_path("connectors.left.position_nm[0]")
        assert is_optional_path("connectors.right.rotation_deg")
