"""Tests for M2 geometry adapter (M1 -> openEMS CSX primitives).

These tests validate:
- CSX primitive creation and conversion
- TrackSegment to CSXBox mapping
- Via to CSXCylinder/CSXViaPad mapping
- Polygon to CSXPolygon mapping (for cutouts and copper pours)
- Material assignment
- Stackup Z-coordinate mapping
- Full geometry building
"""

from __future__ import annotations

import pytest

from formula_foundry.coupongen.geom.primitives import (
    Polygon,
    PolygonType,
    PositionNM,
    TrackSegment,
    Via,
)
from formula_foundry.openems import (
    BoardOutlineSpec,
    DiscontinuitySpec,
    GeometrySpec,
    LayerSpec,
    StackupSpec,
    TransmissionLineSpec,
)
from formula_foundry.openems.csx_primitives import (
    NM_TO_M,
    BoundingBox3D,
    CSXBox,
    CSXCylinder,
    CSXGeometry,
    CSXMaterialType,
    CSXPolygon,
    CSXViaPad,
    Point3D,
    air_material,
    copper_material,
    substrate_material,
)
from formula_foundry.openems.geometry import StackupMaterialsSpec
from formula_foundry.openems.geometry_adapter import (
    GeometryAdapter,
    StackupZMap,
    build_csx_geometry,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_stackup_spec() -> StackupSpec:
    """4-layer stackup with typical dimensions."""
    return StackupSpec(
        copper_layers=4,
        thicknesses_nm={
            "L1_to_L2": 200_000,  # 200um prepreg
            "L2_to_L3": 1_000_000,  # 1mm core
            "L3_to_L4": 200_000,  # 200um prepreg
        },
        materials=StackupMaterialsSpec(er=4.2, loss_tangent=0.02),
    )


@pytest.fixture
def sample_geometry_spec(sample_stackup_spec: StackupSpec) -> GeometrySpec:
    """Sample GeometrySpec for testing."""
    return GeometrySpec(
        design_hash="test_hash_12345",
        coupon_family="F1_SINGLE_ENDED_VIA",
        board=BoardOutlineSpec(
            width_nm=20_000_000,
            length_nm=80_000_000,
            corner_radius_nm=2_000_000,
        ),
        stackup=sample_stackup_spec,
        layers=[
            LayerSpec(id="L1", z_nm=0, role="signal"),
            LayerSpec(id="L2", z_nm=200_000, role="ground"),
            LayerSpec(id="L3", z_nm=1_200_000, role="ground"),
            LayerSpec(id="L4", z_nm=1_400_000, role="signal"),
        ],
        transmission_line=TransmissionLineSpec(
            type="CPWG",
            layer="F.Cu",
            w_nm=300_000,
            gap_nm=180_000,
            length_left_nm=25_000_000,
            length_right_nm=25_000_000,
        ),
        discontinuity=DiscontinuitySpec(
            type="VIA_TRANSITION",
            parameters_nm={"signal_via.drill_nm": 300_000},
        ),
    )


# =============================================================================
# CSX Primitive Tests
# =============================================================================


class TestPoint3D:
    """Tests for Point3D primitive."""

    def test_creation(self) -> None:
        pt = Point3D(1_000_000, 2_000_000, 3_000_000)
        assert pt.x == 1_000_000
        assert pt.y == 2_000_000
        assert pt.z == 3_000_000

    def test_to_meters(self) -> None:
        pt = Point3D(1_000_000, 2_000_000, 3_000_000)
        m = pt.to_meters()
        assert m == pytest.approx((0.001, 0.002, 0.003))

    def test_add(self) -> None:
        p1 = Point3D(100, 200, 300)
        p2 = Point3D(10, 20, 30)
        result = p1 + p2
        assert result == Point3D(110, 220, 330)

    def test_sub(self) -> None:
        p1 = Point3D(100, 200, 300)
        p2 = Point3D(10, 20, 30)
        result = p1 - p2
        assert result == Point3D(90, 180, 270)


class TestBoundingBox3D:
    """Tests for BoundingBox3D primitive."""

    def test_from_corners(self) -> None:
        bbox = BoundingBox3D.from_corners(0, 0, 0, 1000, 2000, 3000)
        assert bbox.min_pt == Point3D(0, 0, 0)
        assert bbox.max_pt == Point3D(1000, 2000, 3000)

    def test_from_corners_reorders(self) -> None:
        """Corners should be normalized regardless of input order."""
        bbox = BoundingBox3D.from_corners(1000, 2000, 3000, 0, 0, 0)
        assert bbox.min_pt == Point3D(0, 0, 0)
        assert bbox.max_pt == Point3D(1000, 2000, 3000)

    def test_size_nm(self) -> None:
        bbox = BoundingBox3D.from_corners(100, 200, 300, 500, 700, 900)
        assert bbox.size_nm == (400, 500, 600)

    def test_to_meters(self) -> None:
        bbox = BoundingBox3D.from_corners(0, 0, 0, 1_000_000, 2_000_000, 35_000)
        min_m, max_m = bbox.to_meters()
        assert min_m == pytest.approx((0.0, 0.0, 0.0))
        assert max_m == pytest.approx((0.001, 0.002, 0.000035))


class TestCSXMaterial:
    """Tests for CSXMaterial definitions."""

    def test_copper_material(self) -> None:
        mat = copper_material()
        assert mat.name == "copper"
        assert mat.material_type == CSXMaterialType.METAL
        assert mat.conductivity == 5.8e7

    def test_copper_material_custom_name(self) -> None:
        mat = copper_material(name="signal_copper", priority=150)
        assert mat.name == "signal_copper"
        assert mat.priority == 150

    def test_substrate_material(self) -> None:
        mat = substrate_material(epsilon_r=4.2, loss_tangent=0.02)
        assert mat.name == "substrate"
        assert mat.material_type == CSXMaterialType.DIELECTRIC
        assert mat.epsilon_r == 4.2
        assert mat.loss_tangent == 0.02

    def test_air_material(self) -> None:
        mat = air_material()
        assert mat.name == "air"
        assert mat.epsilon_r == 1.0
        assert mat.loss_tangent == 0.0


class TestCSXBox:
    """Tests for CSXBox primitive."""

    def test_creation(self) -> None:
        bbox = BoundingBox3D.from_corners(0, 0, 0, 1000, 500, 35_000)
        mat = copper_material()
        box = CSXBox(bbox=bbox, material=mat, name="trace_1")
        assert box.name == "trace_1"
        assert box.material == mat
        assert box.bbox == bbox

    def test_from_trace_horizontal(self) -> None:
        """Horizontal trace expands in Y direction."""
        mat = copper_material()
        box = CSXBox.from_trace(
            x_start=0,
            y_start=0,
            x_end=10_000_000,
            y_end=0,
            z_bottom=0,
            z_top=35_000,
            width_nm=300_000,
            material=mat,
        )
        # Y should expand by half width on each side
        assert box.bbox.min_pt.y == -150_000
        assert box.bbox.max_pt.y == 150_000
        # X should span the full trace length
        assert box.bbox.min_pt.x == 0
        assert box.bbox.max_pt.x == 10_000_000

    def test_from_trace_vertical(self) -> None:
        """Vertical trace expands in X direction."""
        mat = copper_material()
        box = CSXBox.from_trace(
            x_start=0,
            y_start=0,
            x_end=0,
            y_end=10_000_000,
            z_bottom=0,
            z_top=35_000,
            width_nm=300_000,
            material=mat,
        )
        # X should expand by half width on each side
        assert box.bbox.min_pt.x == -150_000
        assert box.bbox.max_pt.x == 150_000
        # Y should span the full trace length
        assert box.bbox.min_pt.y == 0
        assert box.bbox.max_pt.y == 10_000_000


class TestCSXCylinder:
    """Tests for CSXCylinder primitive."""

    def test_from_via(self) -> None:
        mat = copper_material()
        cyl = CSXCylinder.from_via(
            x=5_000_000,
            y=10_000_000,
            z_bottom=0,
            z_top=1_400_000,
            drill_nm=300_000,
            material=mat,
        )
        assert cyl.center_bottom == Point3D(5_000_000, 10_000_000, 0)
        assert cyl.center_top == Point3D(5_000_000, 10_000_000, 1_400_000)
        assert cyl.radius_nm == 150_000


class TestCSXPolygon:
    """Tests for CSXPolygon primitive."""

    def test_creation(self) -> None:
        vertices = ((0, 0), (1000, 0), (1000, 1000), (0, 1000))
        mat = air_material()
        poly = CSXPolygon(
            vertices_xy=vertices,
            z_bottom=0,
            z_top=35_000,
            material=mat,
        )
        assert len(poly.vertices_xy) == 4
        assert poly.z_bottom == 0
        assert poly.z_top == 35_000

    def test_vertices_to_meters(self) -> None:
        vertices = ((1_000_000, 2_000_000), (3_000_000, 4_000_000))
        poly = CSXPolygon(
            vertices_xy=vertices,
            z_bottom=0,
            z_top=35_000,
            material=air_material(),
        )
        m_verts = poly.vertices_to_meters()
        assert m_verts[0] == pytest.approx((0.001, 0.002))
        assert m_verts[1] == pytest.approx((0.003, 0.004))


class TestCSXGeometry:
    """Tests for CSXGeometry collection."""

    def test_add_material(self) -> None:
        geom = CSXGeometry()
        mat = copper_material()
        geom.add_material(mat)
        assert "copper" in geom.materials
        assert geom.materials["copper"] == mat

    def test_add_primitive_updates_bbox(self) -> None:
        geom = CSXGeometry()
        bbox = BoundingBox3D.from_corners(100, 200, 0, 1000, 2000, 35_000)
        box = CSXBox(bbox=bbox, material=copper_material())
        geom.add_primitive(box)

        assert geom.bbox is not None
        assert geom.bbox.min_pt.x == 100
        assert geom.bbox.max_pt.x == 1000

    def test_bbox_expands_with_primitives(self) -> None:
        geom = CSXGeometry()
        mat = copper_material()

        # Add first box
        bbox1 = BoundingBox3D.from_corners(0, 0, 0, 1000, 1000, 35_000)
        geom.add_primitive(CSXBox(bbox=bbox1, material=mat))

        # Add second box outside first
        bbox2 = BoundingBox3D.from_corners(2000, 2000, 0, 3000, 3000, 35_000)
        geom.add_primitive(CSXBox(bbox=bbox2, material=mat))

        # Overall bbox should encompass both
        assert geom.bbox is not None
        assert geom.bbox.min_pt == Point3D(0, 0, 0)
        assert geom.bbox.max_pt == Point3D(3000, 3000, 35_000)


# =============================================================================
# StackupZMap Tests
# =============================================================================


class TestStackupZMap:
    """Tests for StackupZMap Z-coordinate mapping."""

    def test_from_geometry_spec(self, sample_geometry_spec: GeometrySpec) -> None:
        z_map = StackupZMap.from_geometry_spec(sample_geometry_spec)

        # Should have 4 layers
        assert len(z_map.layers) == 4
        assert "L1" in z_map.layers
        assert "L4" in z_map.layers

        # L1 should be at top
        assert z_map.layers["L1"].is_top
        assert not z_map.layers["L1"].is_bottom

        # L4 should be at bottom
        assert z_map.layers["L4"].is_bottom
        assert not z_map.layers["L4"].is_top

    def test_layer_z_ordering(self, sample_geometry_spec: GeometrySpec) -> None:
        """Layers should have increasing Z from L1 (top) to L4 (bottom)."""
        z_map = StackupZMap.from_geometry_spec(sample_geometry_spec)

        # In our bottom-up coordinate system, L1 is higher Z than L4
        z1 = z_map.layers["L1"].z_bottom_nm
        z4 = z_map.layers["L4"].z_bottom_nm

        # L1 should have higher Z than L4 (top is higher)
        assert z1 > z4

    def test_copper_thickness(self, sample_geometry_spec: GeometrySpec) -> None:
        """Each layer should have correct copper thickness."""
        z_map = StackupZMap.from_geometry_spec(sample_geometry_spec, copper_thickness_nm=35_000)

        for _layer_id, layer_info in z_map.layers.items():
            thickness = layer_info.z_top_nm - layer_info.z_bottom_nm
            assert thickness == 35_000

    def test_get_layer_z(self, sample_geometry_spec: GeometrySpec) -> None:
        z_map = StackupZMap.from_geometry_spec(sample_geometry_spec)

        layer = z_map.get_layer_z("L2")
        assert layer.layer_id == "L2"
        assert not layer.is_top
        assert not layer.is_bottom

    def test_get_unknown_layer_raises(self, sample_geometry_spec: GeometrySpec) -> None:
        z_map = StackupZMap.from_geometry_spec(sample_geometry_spec)

        with pytest.raises(KeyError, match="Unknown layer"):
            z_map.get_layer_z("L5")


# =============================================================================
# GeometryAdapter Tests
# =============================================================================


class TestGeometryAdapter:
    """Tests for GeometryAdapter conversion methods."""

    def test_from_geometry_spec(self, sample_geometry_spec: GeometrySpec) -> None:
        adapter = GeometryAdapter.from_geometry_spec(sample_geometry_spec)

        assert adapter.copper_mat is not None
        assert adapter.substrate_mat is not None
        assert adapter.stackup_z is not None

    def test_track_to_csx_box(self, sample_geometry_spec: GeometrySpec) -> None:
        adapter = GeometryAdapter.from_geometry_spec(sample_geometry_spec)

        track = TrackSegment(
            start=PositionNM(0, 0),
            end=PositionNM(10_000_000, 0),
            width_nm=300_000,
            layer="F.Cu",
            net_id=1,
        )

        box = adapter.track_to_csx_box(track, layer_id="L1")

        # Check dimensions
        assert box.bbox.size_nm[0] == 10_000_000  # X length
        assert box.bbox.size_nm[1] == 300_000  # Y width
        assert box.material == adapter.copper_mat

    def test_via_to_csx_cylinder(self, sample_geometry_spec: GeometrySpec) -> None:
        adapter = GeometryAdapter.from_geometry_spec(sample_geometry_spec)

        via = Via(
            position=PositionNM(5_000_000, 0),
            diameter_nm=600_000,
            drill_nm=300_000,
        )

        cylinder = adapter.via_to_csx_cylinder(via, layer_start="L1", layer_end="L4")

        assert cylinder.radius_nm == 150_000  # drill/2
        assert cylinder.center_bottom.x == 5_000_000
        assert cylinder.center_top.x == 5_000_000

    def test_via_to_csx_pads(self, sample_geometry_spec: GeometrySpec) -> None:
        adapter = GeometryAdapter.from_geometry_spec(sample_geometry_spec)

        via = Via(
            position=PositionNM(5_000_000, 0),
            diameter_nm=600_000,
            drill_nm=300_000,
        )

        pads = adapter.via_to_csx_pads(via, layers=("L1", "L4"))

        assert len(pads) == 2
        assert pads[0].radius_nm == 300_000  # diameter/2

    def test_polygon_to_csx_cutout(self, sample_geometry_spec: GeometrySpec) -> None:
        adapter = GeometryAdapter.from_geometry_spec(sample_geometry_spec)

        # Create a circular antipad approximation as polygon
        vertices = (
            PositionNM(1_000_000, 0),
            PositionNM(0, 1_000_000),
            PositionNM(-1_000_000, 0),
            PositionNM(0, -1_000_000),
        )
        polygon = Polygon(
            vertices=vertices,
            layer="In1.Cu",
            polygon_type=PolygonType.CUTOUT,
        )

        csx_poly = adapter.polygon_to_csx(polygon, layer_id="L2")

        # Cutout should use air material
        assert csx_poly.material == adapter.air_mat
        assert len(csx_poly.vertices_xy) == 4

    def test_polygon_to_csx_copper_pour(self, sample_geometry_spec: GeometrySpec) -> None:
        adapter = GeometryAdapter.from_geometry_spec(sample_geometry_spec)

        vertices = (
            PositionNM(0, 0),
            PositionNM(10_000_000, 0),
            PositionNM(10_000_000, 10_000_000),
            PositionNM(0, 10_000_000),
        )
        polygon = Polygon(
            vertices=vertices,
            layer="F.Cu",
            polygon_type=PolygonType.COPPER_POUR,
        )

        csx_poly = adapter.polygon_to_csx(polygon, layer_id="L1")

        # Copper pour should use copper material
        assert csx_poly.material == adapter.copper_mat

    def test_create_ground_plane(self, sample_geometry_spec: GeometrySpec) -> None:
        adapter = GeometryAdapter.from_geometry_spec(sample_geometry_spec)

        plane = adapter.create_ground_plane(
            x_min=0,
            y_min=-10_000_000,
            x_max=80_000_000,
            y_max=10_000_000,
            layer_id="L2",
            name="gnd_L2",
        )

        assert plane.name == "gnd_L2"
        assert plane.material == adapter.copper_mat
        assert plane.bbox.size_nm[0] == 80_000_000
        assert plane.bbox.size_nm[1] == 20_000_000


# =============================================================================
# Full Geometry Building Tests
# =============================================================================


class TestBuildCSXGeometry:
    """Tests for build_csx_geometry function."""

    def test_build_with_tracks(self, sample_geometry_spec: GeometrySpec) -> None:
        tracks = [
            TrackSegment(
                start=PositionNM(0, 0),
                end=PositionNM(25_000_000, 0),
                width_nm=300_000,
                layer="F.Cu",
            ),
            TrackSegment(
                start=PositionNM(25_000_000, 0),
                end=PositionNM(50_000_000, 0),
                width_nm=300_000,
                layer="F.Cu",
            ),
        ]

        geom = build_csx_geometry(
            sample_geometry_spec,
            tracks=tracks,
            vias=[],
            polygons=[],
            include_substrate=False,
        )

        # Should have 2 track boxes + materials
        assert len(geom.primitives) == 2
        assert "copper" in geom.materials

    def test_build_with_vias(self, sample_geometry_spec: GeometrySpec) -> None:
        vias = [
            Via(
                position=PositionNM(40_000_000, 0),
                diameter_nm=600_000,
                drill_nm=300_000,
            ),
        ]

        geom = build_csx_geometry(
            sample_geometry_spec,
            tracks=[],
            vias=vias,
            polygons=[],
            include_substrate=False,
            include_via_pads=True,
        )

        # Should have 1 cylinder + 4 pads (one per layer)
        assert len(geom.primitives) == 5
        cylinders = [p for p in geom.primitives if isinstance(p, CSXCylinder)]
        pads = [p for p in geom.primitives if isinstance(p, CSXViaPad)]
        assert len(cylinders) == 1
        assert len(pads) == 4

    def test_build_with_polygons(self, sample_geometry_spec: GeometrySpec) -> None:
        vertices = (
            PositionNM(1_000_000, 0),
            PositionNM(0, 1_000_000),
            PositionNM(-1_000_000, 0),
            PositionNM(0, -1_000_000),
        )
        polygons = [
            Polygon(
                vertices=vertices,
                layer="In1.Cu",
                polygon_type=PolygonType.CUTOUT,
            ),
        ]

        geom = build_csx_geometry(
            sample_geometry_spec,
            tracks=[],
            vias=[],
            polygons=polygons,
            include_substrate=False,
        )

        # Should have 1 polygon
        assert len(geom.primitives) == 1
        assert isinstance(geom.primitives[0], CSXPolygon)

    def test_build_complete_coupon(self, sample_geometry_spec: GeometrySpec) -> None:
        """Test building a complete coupon geometry."""
        # Signal traces
        tracks = [
            TrackSegment(
                start=PositionNM(0, 0),
                end=PositionNM(25_000_000, 0),
                width_nm=300_000,
                layer="F.Cu",
            ),
            TrackSegment(
                start=PositionNM(25_000_000, 0),
                end=PositionNM(50_000_000, 0),
                width_nm=300_000,
                layer="F.Cu",
            ),
        ]

        # Via transition
        vias = [
            Via(
                position=PositionNM(25_000_000, 0),
                diameter_nm=600_000,
                drill_nm=300_000,
            ),
        ]

        # Antipad on L2
        antipad_verts = (
            PositionNM(25_000_000 + 1_000_000, 0),
            PositionNM(25_000_000, 1_000_000),
            PositionNM(25_000_000 - 1_000_000, 0),
            PositionNM(25_000_000, -1_000_000),
        )
        polygons = [
            Polygon(
                vertices=antipad_verts,
                layer="In1.Cu",
                polygon_type=PolygonType.CUTOUT,
            ),
        ]

        geom = build_csx_geometry(
            sample_geometry_spec,
            tracks=tracks,
            vias=vias,
            polygons=polygons,
            include_substrate=False,
            include_via_pads=True,
        )

        # Count primitives by type
        boxes = [p for p in geom.primitives if isinstance(p, CSXBox)]
        cylinders = [p for p in geom.primitives if isinstance(p, CSXCylinder)]
        polys = [p for p in geom.primitives if isinstance(p, CSXPolygon)]
        pads = [p for p in geom.primitives if isinstance(p, CSXViaPad)]

        assert len(boxes) == 2  # 2 tracks
        assert len(cylinders) == 1  # 1 via barrel
        assert len(polys) == 1  # 1 antipad
        assert len(pads) == 4  # 4 via pads

    def test_materials_registered(self, sample_geometry_spec: GeometrySpec) -> None:
        geom = build_csx_geometry(
            sample_geometry_spec,
            tracks=[],
            vias=[],
            polygons=[],
            include_substrate=False,
        )

        # Should have copper and air materials registered
        assert "copper" in geom.materials
        assert "air" in geom.materials


# =============================================================================
# NM to Meters Conversion Tests
# =============================================================================


class TestNMToMetersConversion:
    """Tests for nanometer to meter conversion."""

    def test_nm_to_m_constant(self) -> None:
        assert NM_TO_M == 1e-9

    def test_point_conversion_accuracy(self) -> None:
        # 1mm = 1,000,000 nm
        pt = Point3D(1_000_000, 0, 0)
        m = pt.to_meters()
        assert m[0] == pytest.approx(0.001, rel=1e-9)

    def test_typical_pcb_dimensions(self) -> None:
        # 300um trace width = 300,000 nm = 0.0003 m
        pt = Point3D(300_000, 0, 0)
        m = pt.to_meters()
        assert m[0] == pytest.approx(0.0003, rel=1e-9)

        # 35um copper = 35,000 nm = 0.000035 m
        pt2 = Point3D(0, 0, 35_000)
        m2 = pt2.to_meters()
        assert m2[2] == pytest.approx(0.000035, rel=1e-9)
