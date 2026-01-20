"""Tests for formula_foundry.openems.mesh_generator module.

Tests cover:
- RefinementZone dataclass
- MeshLineGenerator functionality
- Refinement zone detection (vias, antipads, traces)
- Z mesh line generation with layer alignment
- Complete generate_adaptive_mesh_lines function
- Mesh line summary statistics
"""
from __future__ import annotations

import pytest

from formula_foundry.em.mesh import (
    AdaptiveMeshDensity,
    FrequencyRange,
    MeshConfig,
    create_default_mesh_config,
)
from formula_foundry.openems.geometry import (
    BoardOutlineSpec,
    DiscontinuitySpec,
    GeometrySpec,
    LayerSpec,
    StackupMaterialsSpec,
    StackupSpec,
    TransmissionLineSpec,
)
from formula_foundry.openems.mesh_generator import (
    MeshLineGenerator,
    RefinementZone,
    detect_antipad_refinement_zones,
    detect_trace_refinement_zones,
    detect_via_refinement_zones,
    generate_adaptive_mesh_lines,
    generate_z_mesh_lines,
    mesh_line_summary,
)


def _make_test_geometry(
    *,
    trace_w_nm: int = 300_000,
    trace_gap_nm: int = 180_000,
    epsilon_r: float = 4.1,
    with_discontinuity: bool = True,
) -> GeometrySpec:
    """Create a test GeometrySpec for mesh generator tests."""
    board = BoardOutlineSpec(
        width_nm=20_000_000,
        length_nm=80_000_000,
        corner_radius_nm=2_000_000,
    )
    stackup = StackupSpec(
        copper_layers=4,
        thicknesses_nm={
            "L1_to_L2": 180_000,
            "L2_to_L3": 800_000,
            "L3_to_L4": 180_000,
        },
        materials=StackupMaterialsSpec(er=epsilon_r, loss_tangent=0.02),
    )
    layers = [
        LayerSpec(id="L1", z_nm=0, role="signal"),
        LayerSpec(id="L2", z_nm=180_000, role="ground"),
        LayerSpec(id="L3", z_nm=980_000, role="ground"),
        LayerSpec(id="L4", z_nm=1_160_000, role="ground"),
    ]
    transmission_line = TransmissionLineSpec(
        type="CPWG",
        layer="F.Cu",
        w_nm=trace_w_nm,
        gap_nm=trace_gap_nm,
        length_left_nm=25_000_000,
        length_right_nm=25_000_000,
    )
    discontinuity = None
    if with_discontinuity:
        discontinuity = DiscontinuitySpec(
            type="VIA_TRANSITION",
            parameters_nm={
                "signal_via.drill_nm": 300_000,
                "signal_via.diameter_nm": 650_000,
                "signal_via.pad_diameter_nm": 900_000,
                "antipad.L2.r_nm": 1_200_000,
                "antipad.L3.r_nm": 1_100_000,
            },
        )
    return GeometrySpec(
        design_hash="test_hash_mesh_gen",
        coupon_family="F1_SINGLE_ENDED_VIA",
        board=board,
        stackup=stackup,
        layers=layers,
        transmission_line=transmission_line,
        discontinuity=discontinuity,
        parameters_nm={},
        derived_features={},
        dimensionless_groups={},
    )


def _make_test_adaptive_density() -> AdaptiveMeshDensity:
    """Create a test AdaptiveMeshDensity."""
    return AdaptiveMeshDensity(
        base_cell_nm=500_000,
        trace_cell_nm=50_000,
        via_cell_nm=75_000,
        antipad_cell_nm=100_000,
        substrate_cell_nm=100_000,
        pml_cell_nm=500_000,
        min_feature_size_nm=50_000,
    )


class TestRefinementZone:
    """Tests for RefinementZone dataclass."""

    def test_create_refinement_zone(self) -> None:
        """RefinementZone can be created with valid parameters."""
        zone = RefinementZone(
            center_nm=1_000_000,
            radius_nm=500_000,
            cell_size_nm=50_000,
            axis="x",
        )
        assert zone.center_nm == 1_000_000
        assert zone.radius_nm == 500_000
        assert zone.cell_size_nm == 50_000
        assert zone.axis == "x"

    def test_refinement_zone_is_frozen(self) -> None:
        """RefinementZone is immutable."""
        zone = RefinementZone(
            center_nm=1_000_000,
            radius_nm=500_000,
            cell_size_nm=50_000,
            axis="x",
        )
        with pytest.raises(AttributeError):
            zone.center_nm = 2_000_000  # type: ignore[misc]


class TestMeshLineGenerator:
    """Tests for MeshLineGenerator class."""

    def test_generates_domain_boundaries(self) -> None:
        """Generator always includes domain boundaries."""
        generator = MeshLineGenerator(
            domain_min_nm=0,
            domain_max_nm=10_000_000,
            base_cell_nm=1_000_000,
            min_cell_nm=10_000,
            max_ratio=1.5,
        )
        lines = generator.generate_lines()
        assert lines[0] == 0
        assert lines[-1] == 10_000_000

    def test_generates_lines_within_bounds(self) -> None:
        """All generated lines are within domain bounds."""
        generator = MeshLineGenerator(
            domain_min_nm=0,
            domain_max_nm=10_000_000,
            base_cell_nm=1_000_000,
            min_cell_nm=10_000,
            max_ratio=1.5,
        )
        lines = generator.generate_lines()
        assert all(0 <= ln <= 10_000_000 for ln in lines)

    def test_lines_are_sorted(self) -> None:
        """Generated lines are sorted in ascending order."""
        generator = MeshLineGenerator(
            domain_min_nm=0,
            domain_max_nm=10_000_000,
            base_cell_nm=1_000_000,
            min_cell_nm=10_000,
            max_ratio=1.5,
        )
        lines = generator.generate_lines()
        assert lines == sorted(lines)

    def test_lines_are_unique(self) -> None:
        """Generated lines have no duplicates."""
        generator = MeshLineGenerator(
            domain_min_nm=0,
            domain_max_nm=10_000_000,
            base_cell_nm=1_000_000,
            min_cell_nm=10_000,
            max_ratio=1.5,
        )
        lines = generator.generate_lines()
        assert len(lines) == len(set(lines))

    def test_refinement_zone_adds_lines(self) -> None:
        """Adding a refinement zone increases the number of lines."""
        # Without refinement zone
        gen_no_zone = MeshLineGenerator(
            domain_min_nm=0,
            domain_max_nm=10_000_000,
            base_cell_nm=1_000_000,
            min_cell_nm=10_000,
            max_ratio=1.5,
        )
        lines_no_zone = gen_no_zone.generate_lines()

        # With refinement zone
        gen_with_zone = MeshLineGenerator(
            domain_min_nm=0,
            domain_max_nm=10_000_000,
            base_cell_nm=1_000_000,
            min_cell_nm=10_000,
            max_ratio=1.5,
            refinement_zones=[
                RefinementZone(
                    center_nm=5_000_000,
                    radius_nm=500_000,
                    cell_size_nm=50_000,
                    axis="x",
                )
            ],
        )
        lines_with_zone = gen_with_zone.generate_lines()

        assert len(lines_with_zone) > len(lines_no_zone)

    def test_refinement_zone_center_included(self) -> None:
        """Refinement zone center is included in generated lines."""
        zone_center = 5_000_000
        generator = MeshLineGenerator(
            domain_min_nm=0,
            domain_max_nm=10_000_000,
            base_cell_nm=1_000_000,
            min_cell_nm=10_000,
            max_ratio=1.5,
            refinement_zones=[
                RefinementZone(
                    center_nm=zone_center,
                    radius_nm=500_000,
                    cell_size_nm=50_000,
                    axis="x",
                )
            ],
        )
        lines = generator.generate_lines()
        assert zone_center in lines

    def test_finer_cells_near_refinement_zone(self) -> None:
        """Cells are finer near the refinement zone center."""
        zone_center = 5_000_000
        zone_cell = 50_000
        generator = MeshLineGenerator(
            domain_min_nm=0,
            domain_max_nm=10_000_000,
            base_cell_nm=1_000_000,
            min_cell_nm=10_000,
            max_ratio=1.5,
            refinement_zones=[
                RefinementZone(
                    center_nm=zone_center,
                    radius_nm=500_000,
                    cell_size_nm=zone_cell,
                    axis="x",
                )
            ],
        )
        lines = generator.generate_lines()

        # Find cells near the zone center
        center_idx = lines.index(zone_center)
        if center_idx > 0:
            cell_before = zone_center - lines[center_idx - 1]
            assert cell_before <= zone_cell * 2  # Allow some tolerance

        if center_idx < len(lines) - 1:
            cell_after = lines[center_idx + 1] - zone_center
            assert cell_after <= zone_cell * 2


class TestDetectViaRefinementZones:
    """Tests for detect_via_refinement_zones function."""

    def test_no_zones_without_discontinuity(self) -> None:
        """No via zones detected when geometry has no discontinuity."""
        geometry = _make_test_geometry(with_discontinuity=False)
        density = _make_test_adaptive_density()
        zones = detect_via_refinement_zones(geometry, density)
        assert zones == []

    def test_zones_detected_with_discontinuity(self) -> None:
        """Via zones are detected when discontinuity is present."""
        geometry = _make_test_geometry(with_discontinuity=True)
        density = _make_test_adaptive_density()
        zones = detect_via_refinement_zones(geometry, density)

        assert len(zones) >= 2  # At least X and Y zones
        axes = {z.axis for z in zones}
        assert "x" in axes
        assert "y" in axes

    def test_via_zone_cell_size_from_density(self) -> None:
        """Via zone cell size matches adaptive density."""
        geometry = _make_test_geometry(with_discontinuity=True)
        density = _make_test_adaptive_density()
        zones = detect_via_refinement_zones(geometry, density)

        for zone in zones:
            assert zone.cell_size_nm == density.via_cell_nm

    def test_via_zone_position_at_transmission_line_junction(self) -> None:
        """Via X position is at the junction of left/right transmission lines."""
        geometry = _make_test_geometry(with_discontinuity=True)
        density = _make_test_adaptive_density()
        zones = detect_via_refinement_zones(geometry, density)

        # Via should be at length_left_nm from origin
        expected_x = geometry.transmission_line.length_left_nm
        x_zones = [z for z in zones if z.axis == "x"]
        assert any(z.center_nm == expected_x for z in x_zones)


class TestDetectAntipadRefinementZones:
    """Tests for detect_antipad_refinement_zones function."""

    def test_no_zones_without_discontinuity(self) -> None:
        """No antipad zones detected when geometry has no discontinuity."""
        geometry = _make_test_geometry(with_discontinuity=False)
        density = _make_test_adaptive_density()
        zones = detect_antipad_refinement_zones(geometry, density)
        assert zones == []

    def test_zones_detected_for_each_antipad(self) -> None:
        """Antipad zones are detected for each antipad in discontinuity."""
        geometry = _make_test_geometry(with_discontinuity=True)
        density = _make_test_adaptive_density()
        zones = detect_antipad_refinement_zones(geometry, density)

        # We have antipad.L2.r_nm and antipad.L3.r_nm in our test geometry
        # Each should create X and Y zones
        assert len(zones) >= 4  # 2 layers * 2 axes

    def test_antipad_zone_cell_size_from_density(self) -> None:
        """Antipad zone cell size matches adaptive density."""
        geometry = _make_test_geometry(with_discontinuity=True)
        density = _make_test_adaptive_density()
        zones = detect_antipad_refinement_zones(geometry, density)

        for zone in zones:
            assert zone.cell_size_nm == density.antipad_cell_nm


class TestDetectTraceRefinementZones:
    """Tests for detect_trace_refinement_zones function."""

    def test_zones_created_for_trace_edges(self) -> None:
        """Trace refinement zones are created at trace edges."""
        geometry = _make_test_geometry()
        density = _make_test_adaptive_density()
        zones = detect_trace_refinement_zones(geometry, density)

        # Should have zones at positive and negative trace edges
        assert len(zones) >= 4  # ±trace_edge, ±gap_outer
        assert all(z.axis == "y" for z in zones)

    def test_trace_zone_positions_symmetric(self) -> None:
        """Trace zones are symmetric around y=0."""
        geometry = _make_test_geometry()
        density = _make_test_adaptive_density()
        zones = detect_trace_refinement_zones(geometry, density)

        # Collect zone centers
        centers = {z.center_nm for z in zones}

        # For each positive center, there should be a corresponding negative
        for center in centers:
            if center > 0:
                assert -center in centers

    def test_trace_zone_cell_size_from_density(self) -> None:
        """Trace zone cell size matches adaptive density."""
        geometry = _make_test_geometry()
        density = _make_test_adaptive_density()
        zones = detect_trace_refinement_zones(geometry, density)

        for zone in zones:
            assert zone.cell_size_nm == density.trace_cell_nm


class TestGenerateZMeshLines:
    """Tests for generate_z_mesh_lines function."""

    def test_lines_at_layer_boundaries(self) -> None:
        """Z mesh lines are placed at layer boundaries."""
        geometry = _make_test_geometry()
        density = _make_test_adaptive_density()
        z_lines = generate_z_mesh_lines(geometry, density)

        # Layer boundaries should be in the mesh
        # With 4 copper layers, we expect 8 boundaries (top and bottom of each)
        assert len(z_lines) >= 8

    def test_lines_are_sorted(self) -> None:
        """Z mesh lines are sorted."""
        geometry = _make_test_geometry()
        density = _make_test_adaptive_density()
        z_lines = generate_z_mesh_lines(geometry, density)
        assert z_lines == sorted(z_lines)

    def test_lines_include_air_padding(self) -> None:
        """Z mesh includes air padding above and below the stackup."""
        geometry = _make_test_geometry()
        density = _make_test_adaptive_density()
        z_lines = generate_z_mesh_lines(geometry, density)

        # Should include lines below 0 (air below stackup)
        assert any(z < 0 for z in z_lines)
        # Should include lines above total stackup thickness
        total_thickness = sum(geometry.stackup.thicknesses_nm.values())
        assert any(z > total_thickness for z in z_lines)

    def test_lines_fill_substrate_regions(self) -> None:
        """Z mesh fills substrate regions with intermediate lines."""
        geometry = _make_test_geometry()
        density = _make_test_adaptive_density()
        z_lines = generate_z_mesh_lines(geometry, density)

        # There should be more lines than just layer boundaries
        # Our stackup has 3 dielectric regions (L1-L2, L2-L3, L3-L4)
        min_lines = 8 + 3 * 2  # boundaries + some interior lines
        assert len(z_lines) >= min_lines


class TestGenerateAdaptiveMeshLines:
    """Tests for generate_adaptive_mesh_lines function."""

    def test_returns_mesh_spec(self) -> None:
        """Function returns a MeshSpec instance."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()
        mesh_spec = generate_adaptive_mesh_lines(config, geometry)

        from formula_foundry.openems.spec import MeshSpec

        assert isinstance(mesh_spec, MeshSpec)

    def test_mesh_spec_has_lines_in_all_axes(self) -> None:
        """Generated MeshSpec has lines in X, Y, and Z."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()
        mesh_spec = generate_adaptive_mesh_lines(config, geometry)

        assert len(mesh_spec.fixed_lines_x_nm) > 2
        assert len(mesh_spec.fixed_lines_y_nm) > 2
        assert len(mesh_spec.fixed_lines_z_nm) > 2

    def test_lines_are_sorted_in_all_axes(self) -> None:
        """All mesh lines are sorted."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()
        mesh_spec = generate_adaptive_mesh_lines(config, geometry)

        assert mesh_spec.fixed_lines_x_nm == sorted(mesh_spec.fixed_lines_x_nm)
        assert mesh_spec.fixed_lines_y_nm == sorted(mesh_spec.fixed_lines_y_nm)
        assert mesh_spec.fixed_lines_z_nm == sorted(mesh_spec.fixed_lines_z_nm)

    def test_more_lines_with_discontinuity(self) -> None:
        """More mesh lines are generated when discontinuity is present."""
        config = create_default_mesh_config()

        # Without discontinuity
        geom_no_disc = _make_test_geometry(with_discontinuity=False)
        mesh_no_disc = generate_adaptive_mesh_lines(config, geom_no_disc)

        # With discontinuity
        geom_with_disc = _make_test_geometry(with_discontinuity=True)
        mesh_with_disc = generate_adaptive_mesh_lines(config, geom_with_disc)

        # More lines expected with discontinuity due to refinement
        assert len(mesh_with_disc.fixed_lines_x_nm) >= len(mesh_no_disc.fixed_lines_x_nm)

    def test_resolution_spec_populated(self) -> None:
        """MeshSpec resolution is populated from config."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()
        mesh_spec = generate_adaptive_mesh_lines(config, geometry)

        assert mesh_spec.resolution.via_resolution_nm == config.via_refinement_nm
        assert mesh_spec.resolution.metal_edge_resolution_nm == config.edge_refinement_nm

    def test_smoothing_spec_populated(self) -> None:
        """MeshSpec smoothing is populated from config."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()
        mesh_spec = generate_adaptive_mesh_lines(config, geometry)

        assert mesh_spec.smoothing.max_ratio == config.smoothmesh_ratio
        assert mesh_spec.smoothing.smooth_mesh_lines is True

    def test_accepts_precomputed_density(self) -> None:
        """Function accepts precomputed adaptive density."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()
        density = _make_test_adaptive_density()

        # Should not raise
        mesh_spec = generate_adaptive_mesh_lines(config, geometry, density)
        assert mesh_spec is not None


class TestMeshLineSummary:
    """Tests for mesh_line_summary function."""

    def test_returns_statistics(self) -> None:
        """Summary returns expected statistics."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()
        mesh_spec = generate_adaptive_mesh_lines(config, geometry)

        summary = mesh_line_summary(mesh_spec)

        assert "total_cells" in summary
        assert "n_lines_x" in summary
        assert "n_lines_y" in summary
        assert "n_lines_z" in summary
        assert "x_cell_min_nm" in summary
        assert "x_cell_max_nm" in summary
        assert "x_cell_mean_nm" in summary

    def test_total_cells_calculation(self) -> None:
        """Total cells is computed correctly."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()
        mesh_spec = generate_adaptive_mesh_lines(config, geometry)

        summary = mesh_line_summary(mesh_spec)

        expected_total = (
            (len(mesh_spec.fixed_lines_x_nm) - 1)
            * (len(mesh_spec.fixed_lines_y_nm) - 1)
            * (len(mesh_spec.fixed_lines_z_nm) - 1)
        )
        assert summary["total_cells"] == expected_total

    def test_line_counts_match(self) -> None:
        """Line counts in summary match actual line counts."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()
        mesh_spec = generate_adaptive_mesh_lines(config, geometry)

        summary = mesh_line_summary(mesh_spec)

        assert summary["n_lines_x"] == len(mesh_spec.fixed_lines_x_nm)
        assert summary["n_lines_y"] == len(mesh_spec.fixed_lines_y_nm)
        assert summary["n_lines_z"] == len(mesh_spec.fixed_lines_z_nm)

    def test_cell_size_statistics_positive(self) -> None:
        """Cell size statistics are positive."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()
        mesh_spec = generate_adaptive_mesh_lines(config, geometry)

        summary = mesh_line_summary(mesh_spec)

        assert summary["x_cell_min_nm"] > 0
        assert summary["y_cell_min_nm"] > 0
        assert summary["z_cell_min_nm"] > 0
        assert summary["x_cell_max_nm"] >= summary["x_cell_min_nm"]
        assert summary["y_cell_max_nm"] >= summary["y_cell_min_nm"]
        assert summary["z_cell_max_nm"] >= summary["z_cell_min_nm"]


class TestMeshQuality:
    """Integration tests for mesh quality properties."""

    def test_domain_encompasses_board(self) -> None:
        """Mesh domain encompasses the entire board."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()
        mesh_spec = generate_adaptive_mesh_lines(config, geometry)

        x_lines = mesh_spec.fixed_lines_x_nm
        y_lines = mesh_spec.fixed_lines_y_nm

        # X domain should include 0 to board length
        assert x_lines[0] <= 0
        assert x_lines[-1] >= geometry.board.length_nm

        # Y domain should be symmetric around 0
        board_half_width = geometry.board.width_nm // 2
        assert y_lines[0] <= -board_half_width
        assert y_lines[-1] >= board_half_width

    def test_pml_padding_included(self) -> None:
        """Mesh extends beyond board for PML."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()
        mesh_spec = generate_adaptive_mesh_lines(config, geometry)

        x_lines = mesh_spec.fixed_lines_x_nm
        y_lines = mesh_spec.fixed_lines_y_nm

        # There should be lines outside the board boundaries
        assert x_lines[0] < 0
        assert x_lines[-1] > geometry.board.length_nm
        assert y_lines[0] < -geometry.board.width_nm // 2
        assert y_lines[-1] > geometry.board.width_nm // 2

    def test_reasonable_cell_count(self) -> None:
        """Total cell count is within reasonable bounds."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()
        mesh_spec = generate_adaptive_mesh_lines(config, geometry)

        summary = mesh_line_summary(mesh_spec)
        total_cells = summary["total_cells"]

        # Should be more than trivial (at least 1000 cells)
        assert total_cells > 1000

        # But not excessively large (less than 100 million for this geometry)
        assert total_cells < 100_000_000
