# SPDX-License-Identifier: MIT
"""Unit tests for derived dimensionless groups computation.

Tests the resolve/derived_groups module which computes physics-relevant
dimensionless groups for CPWG/via/fence/launch structures.

Satisfies:
    - REQ-M1-014: Derived features and dimensionless groups MUST be expanded to
                  include CPWG/via/fence/launch-relevant groups and MUST be emitted
                  deterministically in manifest.json.
    - REQ-M1-015: Key dimensional groups drive design hash sensitivity.
"""

from __future__ import annotations

import math
from typing import Any

import pytest


# Fixture for valid F1 spec data with all features
@pytest.fixture
def full_f1_spec_data() -> dict[str, Any]:
    """Return a valid F1 CouponSpec with fence and discontinuity."""
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
                "copper": 35_000,
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
                "position_nm": [5_000_000, 10_000_000],
                "rotation_deg": 180,
            },
            "right": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [75_000_000, 10_000_000],
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
            "antipads": {},
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
def spec_with_l1_to_l2_stackup() -> dict[str, Any]:
    """Return a spec using L1_to_L2 style stackup naming."""
    return {
        "schema_version": 1,
        "coupon_family": "F0_CALIBRATION_THRU",
        "units": "nm",
        "toolchain": {
            "kicad": {
                "version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7@sha256:abc",
            }
        },
        "fab_profile": {"id": "generic_4layer"},
        "stackup": {
            "copper_layers": 4,
            "thicknesses_nm": {
                "L1_to_L2": 180_000,
                "L2_to_L3": 800_000,
                "L3_to_L4": 180_000,
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
            "text": {"coupon_id": "TEST", "include_manifest_hash": False},
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
# _safe_ratio tests
# =============================================================================


class TestInternalSafeRatio:
    """Tests for internal _safe_ratio function in derived_groups module."""

    def test_normal_ratio(self) -> None:
        """Normal division returns correct float result."""
        from formula_foundry.resolve.derived_groups import _safe_ratio

        assert _safe_ratio(10, 5) == 2.0
        assert _safe_ratio(1, 4) == 0.25

    def test_zero_denominator(self) -> None:
        """Zero denominator returns 0.0."""
        from formula_foundry.resolve.derived_groups import _safe_ratio

        assert _safe_ratio(10, 0) == 0.0
        assert _safe_ratio(0, 0) == 0.0

    def test_returns_float(self) -> None:
        """Result is always a float."""
        from formula_foundry.resolve.derived_groups import _safe_ratio

        result = _safe_ratio(100, 50)
        assert isinstance(result, float)


# =============================================================================
# _get_substrate_height tests
# =============================================================================


class TestGetSubstrateHeight:
    """Tests for _get_substrate_height helper function."""

    def test_with_core_thickness(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Returns core thickness when present in legacy naming."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import _get_substrate_height

        spec = load_couponspec(full_f1_spec_data)
        height = _get_substrate_height(spec)

        assert height == 800_000  # core thickness

    def test_with_l1_to_l2_naming(
        self, spec_with_l1_to_l2_stackup: dict[str, Any]
    ) -> None:
        """Returns L1_to_L2 thickness when using new naming convention."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import _get_substrate_height

        spec = load_couponspec(spec_with_l1_to_l2_stackup)
        height = _get_substrate_height(spec)

        assert height == 180_000  # L1_to_L2 thickness


# =============================================================================
# compute_cpwg_groups tests
# =============================================================================


class TestComputeCpwgGroups:
    """Tests for compute_cpwg_groups function."""

    def test_basic_cpwg_ratios(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Basic CPWG gap/width ratios are computed correctly."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_cpwg_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_cpwg_groups(spec)

        # w=300000, gap=180000
        expected_gap_over_w = 180_000 / 300_000  # 0.6
        expected_w_over_gap = 300_000 / 180_000  # 1.6667

        assert groups["cpwg_gap_over_w"] == pytest.approx(expected_gap_over_w)
        assert groups["cpwg_w_over_gap"] == pytest.approx(expected_w_over_gap)

    def test_k_ratio_computation(self, full_f1_spec_data: dict[str, Any]) -> None:
        """CPWG k-ratio (w / (w + 2*gap)) is computed correctly."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_cpwg_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_cpwg_groups(spec)

        # w=300000, gap=180000
        # ground_opening = 300000 + 360000 = 660000
        # k = 300000 / 660000 ≈ 0.4545
        expected_k = 300_000 / 660_000

        assert groups["cpwg_k_ratio"] == pytest.approx(expected_k)

    def test_k_prime_computation(self, full_f1_spec_data: dict[str, Any]) -> None:
        """CPWG k' = sqrt(1 - k^2) is computed correctly."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_cpwg_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_cpwg_groups(spec)

        k = 300_000 / 660_000
        expected_k_prime = math.sqrt(1 - k * k)

        assert groups["cpwg_k_prime_ratio"] == pytest.approx(expected_k_prime)

    def test_board_aspect_ratio(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Board aspect ratio is computed correctly."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_cpwg_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_cpwg_groups(spec)

        # length=80mm, width=20mm
        expected_aspect = 80_000_000 / 20_000_000  # 4.0

        assert groups["board_aspect_ratio"] == pytest.approx(expected_aspect)

    def test_substrate_height_ratios(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Ratios to substrate height are computed when available."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_cpwg_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_cpwg_groups(spec)

        # core=800000 (substrate height)
        # w=300000, gap=180000
        h = 800_000

        assert groups["cpwg_w_over_h"] == pytest.approx(300_000 / h)
        assert groups["cpwg_gap_over_h"] == pytest.approx(180_000 / h)


# =============================================================================
# compute_via_groups tests
# =============================================================================


class TestComputeViaGroups:
    """Tests for compute_via_groups function."""

    def test_returns_empty_without_discontinuity(
        self, spec_with_l1_to_l2_stackup: dict[str, Any]
    ) -> None:
        """Returns empty dict when no discontinuity is present."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_via_groups

        spec = load_couponspec(spec_with_l1_to_l2_stackup)
        groups = compute_via_groups(spec)

        assert groups == {}

    def test_signal_via_ratios(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Signal via geometry ratios are computed correctly."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_via_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_via_groups(spec)

        # drill=300000, diameter=650000, pad=900000
        assert groups["via_drill_over_pad"] == pytest.approx(300_000 / 900_000)
        assert groups["via_pad_over_drill"] == pytest.approx(900_000 / 300_000)
        assert groups["via_diameter_over_drill"] == pytest.approx(650_000 / 300_000)

    def test_trace_matching_ratios(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Via to trace width ratios are computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_via_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_via_groups(spec)

        # pad=900000, w=300000
        assert groups["via_pad_over_trace_w"] == pytest.approx(900_000 / 300_000)

    def test_annular_ring_ratio(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Via annular ring ratio is computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_via_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_via_groups(spec)

        # pad=900000, drill=300000 -> annular=600000
        # ratio = 600000 / 900000
        assert groups["via_annular_ring_ratio"] == pytest.approx(600_000 / 900_000)

    def test_return_via_ratios(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Return via geometry ratios are computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_via_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_via_groups(spec)

        # return_via: radius=1700000, count=4, via: drill=300000, diameter=650000
        # pad=900000 (signal via pad)
        assert groups["via_return_radius_over_pad"] == pytest.approx(1_700_000 / 900_000)
        assert groups["via_return_drill_over_diam"] == pytest.approx(300_000 / 650_000)


# =============================================================================
# compute_fence_groups tests
# =============================================================================


class TestComputeFenceGroups:
    """Tests for compute_fence_groups function."""

    def test_returns_empty_without_fence(
        self, spec_with_l1_to_l2_stackup: dict[str, Any]
    ) -> None:
        """Returns empty dict when no fence is present."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_fence_groups

        spec = load_couponspec(spec_with_l1_to_l2_stackup)
        groups = compute_fence_groups(spec)

        assert groups == {}

    def test_pitch_ratios(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Fence pitch ratios are computed correctly."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_fence_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_fence_groups(spec)

        # pitch=1500000, gap=180000, w=300000
        assert groups["fence_pitch_over_gap"] == pytest.approx(1_500_000 / 180_000)
        assert groups["fence_pitch_over_w"] == pytest.approx(1_500_000 / 300_000)

    def test_offset_ratios(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Fence offset ratios are computed correctly."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_fence_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_fence_groups(spec)

        # offset=800000, gap=180000, w=300000
        assert groups["fence_offset_over_gap"] == pytest.approx(800_000 / 180_000)
        assert groups["fence_offset_over_w"] == pytest.approx(800_000 / 300_000)

    def test_total_offset_ratio(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Total fence offset from trace center is computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_fence_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_fence_groups(spec)

        # total_offset = w/2 + gap + offset = 150000 + 180000 + 800000 = 1130000
        # w=300000
        expected = 1_130_000 / 300_000

        assert groups["fence_total_offset_over_w"] == pytest.approx(expected)

    def test_fence_via_ratios(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Fence via geometry ratios are computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_fence_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_fence_groups(spec)

        # via: drill=300000, diameter=600000, pitch=1500000
        assert groups["fence_via_drill_over_pitch"] == pytest.approx(300_000 / 1_500_000)
        assert groups["fence_via_diam_over_pitch"] == pytest.approx(600_000 / 1_500_000)
        assert groups["fence_via_drill_over_diam"] == pytest.approx(300_000 / 600_000)

    def test_disabled_fence_returns_empty(
        self, full_f1_spec_data: dict[str, Any]
    ) -> None:
        """Returns empty dict when fence is disabled."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_fence_groups

        # Disable the fence
        full_f1_spec_data["transmission_line"]["ground_via_fence"]["enabled"] = False

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_fence_groups(spec)

        assert groups == {}


# =============================================================================
# compute_launch_groups tests
# =============================================================================


class TestComputeLaunchGroups:
    """Tests for compute_launch_groups function."""

    def test_connector_span_ratios(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Connector span ratios are computed correctly."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_launch_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_launch_groups(spec)

        # left_x=5mm, right_x=75mm -> span=70mm
        # board_length=80mm
        assert groups["launch_span_over_board_length"] == pytest.approx(70_000_000 / 80_000_000)

    def test_y_offset_ratios(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Y-axis offset ratios are computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_launch_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_launch_groups(spec)

        # Both connectors at y=10mm, board_width=20mm, center=10mm
        # offset = |10mm - 10mm| = 0
        assert groups["launch_left_y_offset_ratio"] == pytest.approx(0.0)
        assert groups["launch_right_y_offset_ratio"] == pytest.approx(0.0)

    def test_edge_symmetry(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Edge symmetry is computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_launch_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_launch_groups(spec)

        # left_x=5mm, right_x=75mm, board_length=80mm
        # left_edge_dist = 5mm, right_edge_dist = 80-75=5mm
        # edge_diff = |5 - 5| = 0
        assert groups["launch_edge_symmetry"] == pytest.approx(0.0)

    def test_trace_over_span(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Trace length to span ratio is computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_launch_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_launch_groups(spec)

        # total_trace = 25mm + 25mm = 50mm
        # span = 70mm
        assert groups["launch_trace_over_span"] == pytest.approx(50_000_000 / 70_000_000)

    def test_via_offset_ratio_for_f1(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Via offset ratio is computed for F1 specs with discontinuity."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_launch_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_launch_groups(spec)

        # Via is at length_left from left connector
        # total_trace = 50mm, length_left = 25mm
        # via_offset_ratio = 25/50 = 0.5 (center)
        assert groups["launch_via_offset_ratio"] == pytest.approx(0.5)


# =============================================================================
# compute_stackup_groups tests
# =============================================================================


class TestComputeStackupGroups:
    """Tests for compute_stackup_groups function."""

    def test_material_properties(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Material properties are included."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_stackup_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_stackup_groups(spec)

        assert groups["stackup_er"] == pytest.approx(4.1)
        assert groups["stackup_loss_tangent"] == pytest.approx(0.02)
        assert groups["stackup_sqrt_er"] == pytest.approx(math.sqrt(4.1))

    def test_height_ratios(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Substrate height ratios are computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_stackup_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_stackup_groups(spec)

        # h (core) = 800000, w = 300000
        assert groups["stackup_h_over_w"] == pytest.approx(800_000 / 300_000)

    def test_total_thickness(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Total stackup thickness is recorded."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_stackup_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_stackup_groups(spec)

        # core=800000, prepreg=180000, copper=35000
        total = 800_000 + 180_000 + 35_000

        assert groups["stackup_total_thickness_nm"] == pytest.approx(total)

    def test_copper_layers_count(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Copper layers count is included."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_stackup_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_stackup_groups(spec)

        assert groups["stackup_copper_layers"] == 4.0

    def test_legacy_copper_ratios(self, full_f1_spec_data: dict[str, Any]) -> None:
        """Legacy copper/core ratios are computed when available."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_stackup_groups

        spec = load_couponspec(full_f1_spec_data)
        groups = compute_stackup_groups(spec)

        # copper=35000, core=800000
        assert groups["stackup_copper_over_core"] == pytest.approx(35_000 / 800_000)

    def test_l_style_layer_ratios(
        self, spec_with_l1_to_l2_stackup: dict[str, Any]
    ) -> None:
        """L1_to_L2 style layer ratios are computed."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_stackup_groups

        spec = load_couponspec(spec_with_l1_to_l2_stackup)
        groups = compute_stackup_groups(spec)

        # L1_to_L2=180000, L2_to_L3=800000, L3_to_L4=180000
        # Reference = max = 800000
        assert groups["stackup_l1_to_l2_ratio"] == pytest.approx(180_000 / 800_000)
        assert groups["stackup_l2_to_l3_ratio"] == pytest.approx(800_000 / 800_000)
        assert groups["stackup_l3_to_l4_ratio"] == pytest.approx(180_000 / 800_000)


# =============================================================================
# Determinism tests (REQ-M1-014)
# =============================================================================


class TestDerivedGroupsDeterminism:
    """Tests for deterministic output of derived groups."""

    def test_cpwg_groups_deterministic(
        self, full_f1_spec_data: dict[str, Any]
    ) -> None:
        """CPWG groups are deterministic across multiple calls."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_cpwg_groups

        spec = load_couponspec(full_f1_spec_data)

        groups_1 = compute_cpwg_groups(spec)
        groups_2 = compute_cpwg_groups(spec)

        assert groups_1 == groups_2

    def test_via_groups_deterministic(
        self, full_f1_spec_data: dict[str, Any]
    ) -> None:
        """Via groups are deterministic across multiple calls."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_via_groups

        spec = load_couponspec(full_f1_spec_data)

        groups_1 = compute_via_groups(spec)
        groups_2 = compute_via_groups(spec)

        assert groups_1 == groups_2

    def test_fence_groups_deterministic(
        self, full_f1_spec_data: dict[str, Any]
    ) -> None:
        """Fence groups are deterministic across multiple calls."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_fence_groups

        spec = load_couponspec(full_f1_spec_data)

        groups_1 = compute_fence_groups(spec)
        groups_2 = compute_fence_groups(spec)

        assert groups_1 == groups_2

    def test_launch_groups_deterministic(
        self, full_f1_spec_data: dict[str, Any]
    ) -> None:
        """Launch groups are deterministic across multiple calls."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_launch_groups

        spec = load_couponspec(full_f1_spec_data)

        groups_1 = compute_launch_groups(spec)
        groups_2 = compute_launch_groups(spec)

        assert groups_1 == groups_2

    def test_stackup_groups_deterministic(
        self, full_f1_spec_data: dict[str, Any]
    ) -> None:
        """Stackup groups are deterministic across multiple calls."""
        from formula_foundry.coupongen.spec import load_couponspec
        from formula_foundry.resolve.derived_groups import compute_stackup_groups

        spec = load_couponspec(full_f1_spec_data)

        groups_1 = compute_stackup_groups(spec)
        groups_2 = compute_stackup_groups(spec)

        assert groups_1 == groups_2
