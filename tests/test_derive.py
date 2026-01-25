# SPDX-License-Identifier: MIT
"""Unit tests for derived features and dimensionless groups computation.

Tests the derive module which computes physics-relevant dimensionless groups
and derived features for RF/microwave transmission line analysis.

Satisfies:
    - REQ-M1-014: Derived features and dimensionless groups MUST be expanded to
                  include CPWG/via/fence/launch-relevant groups and MUST be emitted
                  deterministically in manifest.json.
    - REQ-M1-015: Key dimensional groups drive design hash sensitivity.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest


# Fixture for valid F1 spec data
@pytest.fixture
def valid_f1_spec_data() -> dict[str, Any]:
    """Return a valid F1 CouponSpec data dictionary with discontinuity."""
    return {
        "schema_version": 1,
        "coupon_family": "F1_SINGLE_ENDED_VIA",
        "units": "nm",
        "toolchain": {
            "kicad": {
                "version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7@sha256:deadbeef",
            }
        },
        "fab_profile": {"id": "oshpark_4layer", "overrides": {}},
        "stackup": {
            "copper_layers": 4,
            "thicknesses_nm": {
                "core": 800_000,
                "prepreg": 180_000,
            },
            "materials": {"er": 4.1, "loss_tangent": 0.02},
        },
        "board": {
            "outline": {
                "width_nm": 20_000_000,
                "length_nm": 80_000_000,
                "corner_radius_nm": 2_000_000,
            },
            "origin": {"mode": "EDGE_L_CENTER"},
            "text": {"coupon_id": "${COUPON_ID}", "include_manifest_hash": True},
        },
        "connectors": {
            "left": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [5_000_000, 0],
                "rotation_deg": 180,
            },
            "right": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [75_000_000, 0],
                "rotation_deg": 0,
            },
        },
        "transmission_line": {
            "type": "CPWG",
            "layer": "F.Cu",
            "w_nm": 300_000,
            "gap_nm": 180_000,
            "length_left_nm": 25_000_000,
            "length_right_nm": 25_000_000,
            "ground_via_fence": {
                "enabled": True,
                "pitch_nm": 1_500_000,
                "offset_from_gap_nm": 800_000,
                "via": {"drill_nm": 300_000, "diameter_nm": 600_000},
            },
        },
        "discontinuity": {
            "type": "VIA_TRANSITION",
            "signal_via": {
                "drill_nm": 300_000,
                "diameter_nm": 650_000,
                "pad_diameter_nm": 900_000,
            },
            "antipads": {
                "L2": {
                    "shape": "ROUNDRECT",
                    "rx_nm": 1_200_000,
                    "ry_nm": 900_000,
                    "corner_nm": 250_000,
                },
            },
            "return_vias": {
                "pattern": "RING",
                "count": 4,
                "radius_nm": 1_700_000,
                "via": {"drill_nm": 300_000, "diameter_nm": 650_000},
            },
            "plane_cutouts": {},
        },
        "constraints": {
            "mode": "REJECT",
            "drc": {"must_pass": True, "severity": "all"},
            "symmetry": {"enforce": True},
            "allow_unconnected_copper": False,
        },
        "export": {
            "gerbers": {"enabled": True, "format": "gerbers"},
            "drill": {"enabled": True, "format": "excellon"},
            "outputs_dir": "artifacts/",
        },
    }


@pytest.fixture
def minimal_f0_spec_data() -> dict[str, Any]:
    """Return a minimal F0 spec (no discontinuity, no fence)."""
    return {
        "schema_version": 1,
        "coupon_family": "F0_CALIBRATION_THRU",
        "units": "nm",
        "toolchain": {
            "kicad": {
                "version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7@sha256:abc123",
            }
        },
        "fab_profile": {"id": "generic_4layer"},
        "stackup": {
            "copper_layers": 4,
            "thicknesses_nm": {
                "core": 800_000,
            },
            "materials": {"er": 4.5, "loss_tangent": 0.015},
        },
        "board": {
            "outline": {
                "width_nm": 10_000_000,
                "length_nm": 40_000_000,
                "corner_radius_nm": 1_000_000,
            },
            "origin": {"mode": "EDGE_L_CENTER"},
            "text": {"coupon_id": "THRU001", "include_manifest_hash": False},
        },
        "connectors": {
            "left": {
                "footprint": "SMA:SMA_Vertical",
                "position_nm": [2_000_000, 5_000_000],
                "rotation_deg": 0,
            },
            "right": {
                "footprint": "SMA:SMA_Vertical",
                "position_nm": [38_000_000, 5_000_000],
                "rotation_deg": 180,
            },
        },
        "transmission_line": {
            "type": "CPWG",
            "layer": "F.Cu",
            "w_nm": 200_000,
            "gap_nm": 150_000,
            "length_left_nm": 15_000_000,
            "length_right_nm": 15_000_000,
        },
        "constraints": {
            "mode": "REJECT",
            "drc": {"must_pass": True, "severity": "all"},
            "symmetry": {"enforce": True},
            "allow_unconnected_copper": False,
        },
        "export": {
            "gerbers": {"enabled": True, "format": "gerbers"},
            "drill": {"enabled": True, "format": "excellon"},
            "outputs_dir": "output/",
        },
    }


# =============================================================================
# safe_ratio tests
# =============================================================================


class TestSafeRatio:
    """Tests for safe_ratio function."""

    def test_normal_division(self) -> None:
        """Normal division returns correct float result."""
        from formula_foundry.derive import safe_ratio

        assert safe_ratio(10, 5) == 2.0
        assert safe_ratio(3, 4) == 0.75
        assert safe_ratio(1, 3) == pytest.approx(0.3333333, rel=1e-5)

    def test_zero_denominator_returns_zero(self) -> None:
        """Division by zero returns 0.0 instead of raising."""
        from formula_foundry.derive import safe_ratio

        assert safe_ratio(10, 0) == 0.0
        assert safe_ratio(0, 0) == 0.0
        assert safe_ratio(-5, 0) == 0.0

    def test_integer_inputs(self) -> None:
        """Integer inputs return float result."""
        from formula_foundry.derive import safe_ratio

        result = safe_ratio(100, 25)
        assert result == 4.0
        assert isinstance(result, float)

    def test_float_inputs(self) -> None:
        """Float inputs are handled correctly."""
        from formula_foundry.derive import safe_ratio

        assert safe_ratio(1.5, 0.5) == 3.0
        assert safe_ratio(2.5, 2.5) == 1.0

    def test_negative_values(self) -> None:
        """Negative values are handled correctly."""
        from formula_foundry.derive import safe_ratio

        assert safe_ratio(-10, 5) == -2.0
        assert safe_ratio(10, -5) == -2.0
        assert safe_ratio(-10, -5) == 2.0


# =============================================================================
# compute_derived_features tests
# =============================================================================


class TestComputeDerivedFeatures:
    """Tests for compute_derived_features function."""

    def test_returns_dict_with_sorted_keys(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Derived features must return dict with alphabetically sorted keys."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(valid_f1_spec_data)
        features = compute_derived_features(spec)

        assert isinstance(features, dict)
        assert list(features.keys()) == sorted(features.keys())

    def test_board_derived_features(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Board area and perimeter are correctly computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(valid_f1_spec_data)
        features = compute_derived_features(spec)

        # width=20mm, length=80mm
        expected_area = 20_000_000 * 80_000_000
        expected_perimeter = 2 * (20_000_000 + 80_000_000)

        assert features["board_area_nm2"] == expected_area
        assert features["board_perimeter_nm"] == expected_perimeter

    def test_cpwg_derived_features(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """CPWG ground opening is correctly computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(valid_f1_spec_data)
        features = compute_derived_features(spec)

        # w=300000, gap=180000
        # ground_opening = w + 2*gap = 300000 + 360000 = 660000
        assert features["cpwg_ground_opening_nm"] == 660_000
        assert features["cpwg_footprint_width_nm"] == 660_000

    def test_trace_total_length(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Trace total length is computed correctly."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(valid_f1_spec_data)
        features = compute_derived_features(spec)

        # length_left=25mm, length_right=25mm
        assert features["trace_total_length_nm"] == 50_000_000

    def test_trace_total_with_derived_right(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Trace total length uses passed length_right_nm when provided."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(valid_f1_spec_data)
        features = compute_derived_features(spec, length_right_nm=30_000_000)

        # length_left=25mm, derived length_right=30mm
        assert features["trace_total_length_nm"] == 55_000_000
        assert features["length_right_nm"] == 30_000_000

    def test_fence_derived_features(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Fence via derived features are computed when fence is enabled."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(valid_f1_spec_data)
        features = compute_derived_features(spec)

        # fence offset = w/2 + gap + offset = 150000 + 180000 + 800000 = 1130000
        assert features["fence_via_offset_from_center_nm"] == 1_130_000

        # annular ring = diameter - drill = 600000 - 300000 = 300000
        assert features["fence_via_annular_ring_nm"] == 300_000

    def test_discontinuity_derived_features(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Discontinuity/via derived features are computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(valid_f1_spec_data)
        features = compute_derived_features(spec)

        # signal_via: pad=900000, drill=300000, diameter=650000
        assert features["signal_via_annular_ring_nm"] == 600_000  # pad - drill
        assert features["signal_via_barrel_annular_nm"] == 350_000  # diameter - drill

    def test_return_vias_derived_features(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Return vias derived features are computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(valid_f1_spec_data)
        features = compute_derived_features(spec)

        assert features["return_via_count"] == 4
        assert features["return_via_radius_nm"] == 1_700_000
        # return via: diameter=650000, drill=300000 -> annular=350000
        assert features["return_via_annular_ring_nm"] == 350_000

    def test_connector_derived_features(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Connector span and launch offsets are correctly computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(valid_f1_spec_data)
        features = compute_derived_features(spec)

        # left_x=5mm, right_x=75mm -> span=70mm
        assert features["connector_span_nm"] == 70_000_000
        assert features["launch_span_nm"] == 70_000_000
        # board width=20mm, center_y=10mm, left/right y=0 -> offset=10mm
        assert features["launch_left_y_offset_nm"] == 10_000_000
        assert features["launch_right_y_offset_nm"] == 10_000_000
        # left/right edge clearance should match for symmetric placement
        assert features["launch_left_edge_clearance_nm"] == 5_000_000
        assert features["launch_right_edge_clearance_nm"] == 5_000_000
        assert features["launch_edge_symmetry_error_nm"] == 0

    def test_stackup_derived_features(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Stackup thickness derived features are computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(valid_f1_spec_data)
        features = compute_derived_features(spec)

        assert features["stackup_core_thickness_nm"] == 800_000
        assert features["stackup_prepreg_thickness_nm"] == 180_000

    def test_f0_spec_without_discontinuity(
        self, minimal_f0_spec_data: dict[str, Any]
    ) -> None:
        """F0 spec without discontinuity computes basic features."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(minimal_f0_spec_data)
        features = compute_derived_features(spec)

        # Should have board features
        assert "board_area_nm2" in features
        assert "board_perimeter_nm" in features

        # Should have CPWG features
        assert "cpwg_ground_opening_nm" in features

        # Should NOT have discontinuity features
        assert "signal_via_annular_ring_nm" not in features
        assert "return_via_count" not in features

    def test_f0_spec_without_fence(
        self, minimal_f0_spec_data: dict[str, Any]
    ) -> None:
        """F0 spec without fence computes basic features without fence features."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(minimal_f0_spec_data)
        features = compute_derived_features(spec)

        # Should NOT have fence features
        assert "fence_via_offset_from_center_nm" not in features
        assert "fence_via_annular_ring_nm" not in features
        assert "fence_via_count_per_side" not in features

    def test_all_values_are_integers(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """All derived feature values must be integers."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(valid_f1_spec_data)
        features = compute_derived_features(spec)

        for key, value in features.items():
            assert isinstance(value, int), f"{key} should be int, got {type(value)}"


# =============================================================================
# compute_dimensionless_groups tests
# =============================================================================


class TestComputeDimensionlessGroups:
    """Tests for compute_dimensionless_groups function."""

    def test_returns_dict_with_sorted_keys(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Dimensionless groups must return dict with alphabetically sorted keys."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_dimensionless_groups

        spec = load_couponspec(valid_f1_spec_data)
        groups = compute_dimensionless_groups(spec)

        assert isinstance(groups, dict)
        assert list(groups.keys()) == sorted(groups.keys())

    def test_all_values_are_floats(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """All dimensionless group values must be floats."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_dimensionless_groups

        spec = load_couponspec(valid_f1_spec_data)
        groups = compute_dimensionless_groups(spec)

        for key, value in groups.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"

    def test_cpwg_groups_present(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """CPWG geometry groups are computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_dimensionless_groups

        spec = load_couponspec(valid_f1_spec_data)
        groups = compute_dimensionless_groups(spec)

        assert "cpwg_gap_over_w" in groups
        assert "cpwg_w_over_gap" in groups
        assert "cpwg_k_ratio" in groups

    def test_via_groups_present(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Via geometry groups are computed for F1 spec."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_dimensionless_groups

        spec = load_couponspec(valid_f1_spec_data)
        groups = compute_dimensionless_groups(spec)

        assert "via_drill_over_pad" in groups
        assert "via_pad_over_trace_w" in groups

    def test_fence_groups_present(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Fence geometry groups are computed when fence is enabled."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_dimensionless_groups

        spec = load_couponspec(valid_f1_spec_data)
        groups = compute_dimensionless_groups(spec)

        assert "fence_pitch_over_gap" in groups
        assert "fence_offset_over_gap" in groups

    def test_launch_groups_present(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Launch geometry groups are computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_dimensionless_groups

        spec = load_couponspec(valid_f1_spec_data)
        groups = compute_dimensionless_groups(spec)

        assert "launch_span_over_board_length" in groups

    def test_stackup_groups_present(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """Stackup material groups are computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_dimensionless_groups

        spec = load_couponspec(valid_f1_spec_data)
        groups = compute_dimensionless_groups(spec)

        assert "stackup_er" in groups
        assert "stackup_loss_tangent" in groups
        assert "stackup_sqrt_er" in groups

    def test_f0_spec_has_basic_groups(
        self, minimal_f0_spec_data: dict[str, Any]
    ) -> None:
        """F0 spec has basic CPWG and launch groups."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_dimensionless_groups

        spec = load_couponspec(minimal_f0_spec_data)
        groups = compute_dimensionless_groups(spec)

        # Basic CPWG groups should be present
        assert "cpwg_gap_over_w" in groups
        assert "board_aspect_ratio" in groups
        assert "stackup_er" in groups

    def test_f0_spec_no_via_groups(
        self, minimal_f0_spec_data: dict[str, Any]
    ) -> None:
        """F0 spec without discontinuity has no via groups."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_dimensionless_groups

        spec = load_couponspec(minimal_f0_spec_data)
        groups = compute_dimensionless_groups(spec)

        # Via groups should not be present (no discontinuity)
        assert "via_drill_over_pad" not in groups

    def test_f0_spec_no_fence_groups(
        self, minimal_f0_spec_data: dict[str, Any]
    ) -> None:
        """F0 spec without fence has no fence groups."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_dimensionless_groups

        spec = load_couponspec(minimal_f0_spec_data)
        groups = compute_dimensionless_groups(spec)

        # Fence groups should not be present (no fence)
        assert "fence_pitch_over_gap" not in groups


# =============================================================================
# Determinism tests (REQ-M1-014)
# =============================================================================


class TestDeterminism:
    """Tests for deterministic output per REQ-M1-014."""

    def test_derived_features_deterministic(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """REQ-M1-014: Derived features must be emitted deterministically."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_derived_features

        spec = load_couponspec(valid_f1_spec_data)

        # Compute twice
        features_1 = compute_derived_features(spec)
        features_2 = compute_derived_features(spec)

        # Must be identical
        assert features_1 == features_2

        # Keys must be sorted (for deterministic JSON output)
        assert list(features_1.keys()) == sorted(features_1.keys())

    def test_dimensionless_groups_deterministic(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """REQ-M1-014: Dimensionless groups must be emitted deterministically."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import compute_dimensionless_groups

        spec = load_couponspec(valid_f1_spec_data)

        # Compute twice
        groups_1 = compute_dimensionless_groups(spec)
        groups_2 = compute_dimensionless_groups(spec)

        # Must be identical
        assert groups_1 == groups_2

        # Keys must be sorted (for deterministic JSON output)
        assert list(groups_1.keys()) == sorted(groups_1.keys())

    def test_json_serialization_deterministic(
        self, valid_f1_spec_data: dict[str, Any]
    ) -> None:
        """JSON serialization of derived features/groups is deterministic."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.derive import (
            compute_derived_features,
            compute_dimensionless_groups,
        )

        spec = load_couponspec(valid_f1_spec_data)

        features = compute_derived_features(spec)
        groups = compute_dimensionless_groups(spec)

        # Serialize to JSON multiple times
        json_1 = json.dumps({"features": features, "groups": groups}, sort_keys=True)
        json_2 = json.dumps({"features": features, "groups": groups}, sort_keys=True)

        assert json_1 == json_2
