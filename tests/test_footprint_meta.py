"""Tests for footprint metadata loader.

Tests the loading, parsing, and validation of footprint metadata JSON files
that define anchor point, signal_pad, ground_pads, and launch_reference
for connector footprints.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from formula_foundry.coupongen.geom.footprint_meta import (
    CourtyardMeta,
    FootprintMeta,
    LaunchRefMeta,
    PadMeta,
    PointMeta,
    get_footprint_meta_path,
    list_available_footprint_meta,
    load_footprint_meta,
)


class TestPointMeta:
    """Tests for PointMeta dataclass."""

    def test_basic_point(self) -> None:
        """Test creating a basic point."""
        point = PointMeta(x_nm=1000000, y_nm=2000000)
        assert point.x_nm == 1000000
        assert point.y_nm == 2000000
        assert point.description == ""

    def test_point_with_description(self) -> None:
        """Test point with description."""
        point = PointMeta(x_nm=0, y_nm=0, description="Origin")
        assert point.description == "Origin"

    def test_to_tuple(self) -> None:
        """Test tuple conversion."""
        point = PointMeta(x_nm=100, y_nm=200)
        assert point.to_tuple() == (100, 200)


class TestPadMeta:
    """Tests for PadMeta dataclass."""

    def test_basic_pad(self) -> None:
        """Test creating a basic pad."""
        pad = PadMeta(
            pad_number="1",
            center_x_nm=0,
            center_y_nm=0,
            size_x_nm=500000,
            size_y_nm=1500000,
        )
        assert pad.pad_number == "1"
        assert pad.center_x_nm == 0
        assert pad.center_y_nm == 0
        assert pad.size_x_nm == 500000
        assert pad.size_y_nm == 1500000
        assert pad.shape == "rect"
        assert pad.net_name == ""

    def test_pad_with_all_fields(self) -> None:
        """Test pad with all fields specified."""
        pad = PadMeta(
            pad_number="2",
            center_x_nm=-1500000,
            center_y_nm=0,
            size_x_nm=1500000,
            size_y_nm=2500000,
            shape="roundrect",
            layers=("F.Cu", "F.Paste", "F.Mask"),
            net_name="GND",
        )
        assert pad.shape == "roundrect"
        assert pad.layers == ("F.Cu", "F.Paste", "F.Mask")
        assert pad.net_name == "GND"

    def test_pad_center_property(self) -> None:
        """Test center property."""
        pad = PadMeta(
            pad_number="1",
            center_x_nm=100,
            center_y_nm=200,
            size_x_nm=50,
            size_y_nm=50,
        )
        assert pad.center == (100, 200)

    def test_invalid_size_x(self) -> None:
        """Test that invalid size_x raises ValueError."""
        with pytest.raises(ValueError, match="size_x_nm must be positive"):
            PadMeta(
                pad_number="1",
                center_x_nm=0,
                center_y_nm=0,
                size_x_nm=0,
                size_y_nm=100,
            )

    def test_invalid_size_y(self) -> None:
        """Test that invalid size_y raises ValueError."""
        with pytest.raises(ValueError, match="size_y_nm must be positive"):
            PadMeta(
                pad_number="1",
                center_x_nm=0,
                center_y_nm=0,
                size_x_nm=100,
                size_y_nm=-50,
            )


class TestLaunchRefMeta:
    """Tests for LaunchRefMeta dataclass."""

    def test_basic_launch_ref(self) -> None:
        """Test creating a basic launch reference."""
        launch = LaunchRefMeta(x_nm=0, y_nm=0, direction_deg=0)
        assert launch.x_nm == 0
        assert launch.y_nm == 0
        assert launch.direction_deg == 0
        assert launch.description == ""

    def test_launch_ref_with_description(self) -> None:
        """Test launch reference with description."""
        launch = LaunchRefMeta(
            x_nm=250000,
            y_nm=0,
            direction_deg=0,
            description="Trace launches toward board center",
        )
        assert launch.description == "Trace launches toward board center"

    def test_to_tuple(self) -> None:
        """Test tuple conversion."""
        launch = LaunchRefMeta(x_nm=100, y_nm=200, direction_deg=90)
        assert launch.to_tuple() == (100, 200)

    def test_invalid_direction_negative(self) -> None:
        """Test that negative direction raises ValueError."""
        with pytest.raises(ValueError, match="direction_deg must be in"):
            LaunchRefMeta(x_nm=0, y_nm=0, direction_deg=-10)

    def test_invalid_direction_too_high(self) -> None:
        """Test that direction >= 360 raises ValueError."""
        with pytest.raises(ValueError, match="direction_deg must be in"):
            LaunchRefMeta(x_nm=0, y_nm=0, direction_deg=360)


class TestCourtyardMeta:
    """Tests for CourtyardMeta dataclass."""

    def test_basic_courtyard(self) -> None:
        """Test creating a basic courtyard."""
        cy = CourtyardMeta(
            min_x_nm=-3500000,
            max_x_nm=3500000,
            min_y_nm=-2800000,
            max_y_nm=2800000,
        )
        assert cy.min_x_nm == -3500000
        assert cy.max_x_nm == 3500000

    def test_courtyard_dimensions(self) -> None:
        """Test courtyard dimension properties."""
        cy = CourtyardMeta(
            min_x_nm=-1000000,
            max_x_nm=1000000,
            min_y_nm=-500000,
            max_y_nm=500000,
        )
        assert cy.width_nm == 2000000
        assert cy.height_nm == 1000000


class TestFootprintMeta:
    """Tests for FootprintMeta dataclass."""

    def test_footprint_path_property(self) -> None:
        """Test footprint_path property."""
        meta = FootprintMeta(
            schema_version=1,
            id="TestFootprint",
            name="Test Footprint",
            footprint_lib="TestLib",
            footprint_name="TestFP",
            anchor=PointMeta(x_nm=0, y_nm=0),
            signal_pad=PadMeta(
                pad_number="1",
                center_x_nm=0,
                center_y_nm=0,
                size_x_nm=100,
                size_y_nm=100,
            ),
            ground_pads=(
                PadMeta(
                    pad_number="2",
                    center_x_nm=100,
                    center_y_nm=0,
                    size_x_nm=100,
                    size_y_nm=100,
                ),
            ),
            launch_reference=LaunchRefMeta(x_nm=0, y_nm=0, direction_deg=0),
        )
        assert meta.footprint_path == "TestLib:TestFP"

    def test_signal_pad_center_property(self) -> None:
        """Test signal_pad_center_nm property."""
        meta = FootprintMeta(
            schema_version=1,
            id="TestFootprint",
            name="Test Footprint",
            footprint_lib="TestLib",
            footprint_name="TestFP",
            anchor=PointMeta(x_nm=0, y_nm=0),
            signal_pad=PadMeta(
                pad_number="1",
                center_x_nm=100,
                center_y_nm=200,
                size_x_nm=50,
                size_y_nm=50,
            ),
            ground_pads=(
                PadMeta(
                    pad_number="2",
                    center_x_nm=300,
                    center_y_nm=0,
                    size_x_nm=100,
                    size_y_nm=100,
                ),
            ),
            launch_reference=LaunchRefMeta(x_nm=0, y_nm=0, direction_deg=0),
        )
        assert meta.signal_pad_center_nm == (100, 200)

    def test_invalid_schema_version(self) -> None:
        """Test that invalid schema version raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            FootprintMeta(
                schema_version=2,
                id="Test",
                name="Test",
                footprint_lib="Lib",
                footprint_name="FP",
                anchor=PointMeta(x_nm=0, y_nm=0),
                signal_pad=PadMeta(
                    pad_number="1",
                    center_x_nm=0,
                    center_y_nm=0,
                    size_x_nm=100,
                    size_y_nm=100,
                ),
                ground_pads=(
                    PadMeta(
                        pad_number="2",
                        center_x_nm=100,
                        center_y_nm=0,
                        size_x_nm=100,
                        size_y_nm=100,
                    ),
                ),
                launch_reference=LaunchRefMeta(x_nm=0, y_nm=0, direction_deg=0),
            )

    def test_empty_ground_pads(self) -> None:
        """Test that empty ground_pads raises ValueError."""
        with pytest.raises(ValueError, match="At least one ground pad"):
            FootprintMeta(
                schema_version=1,
                id="Test",
                name="Test",
                footprint_lib="Lib",
                footprint_name="FP",
                anchor=PointMeta(x_nm=0, y_nm=0),
                signal_pad=PadMeta(
                    pad_number="1",
                    center_x_nm=0,
                    center_y_nm=0,
                    size_x_nm=100,
                    size_y_nm=100,
                ),
                ground_pads=(),
                launch_reference=LaunchRefMeta(x_nm=0, y_nm=0, direction_deg=0),
            )


class TestLoadFootprintMeta:
    """Tests for load_footprint_meta function."""

    def test_load_sma_endlaunch_generic(self) -> None:
        """Test loading the SMA_EndLaunch_Generic metadata."""
        meta = load_footprint_meta("SMA_EndLaunch_Generic")

        assert meta.id == "SMA_EndLaunch_Generic"
        assert meta.name == "Generic SMA End-Launch Connector"
        assert meta.footprint_lib == "Coupongen_Connectors"
        assert meta.footprint_name == "SMA_EndLaunch_Generic"
        assert meta.connector_type == "SMA"

        # Verify anchor
        assert meta.anchor.x_nm == 0
        assert meta.anchor.y_nm == 0

        # Verify signal pad
        assert meta.signal_pad.pad_number == "1"
        assert meta.signal_pad.center_x_nm == 0
        assert meta.signal_pad.center_y_nm == 0
        assert meta.signal_pad.size_x_nm == 500000  # 0.5mm
        assert meta.signal_pad.size_y_nm == 1500000  # 1.5mm

        # Verify ground pads (2 ground pads)
        assert len(meta.ground_pads) == 2
        left_gnd = meta.ground_pads[0]
        assert left_gnd.center_x_nm == -1500000  # -1.5mm
        right_gnd = meta.ground_pads[1]
        assert right_gnd.center_x_nm == 1500000  # 1.5mm

        # Verify launch reference
        assert meta.launch_reference.x_nm == 0
        assert meta.launch_reference.y_nm == 0
        assert meta.launch_reference.direction_deg == 0

        # Verify optional fields
        assert meta.impedance_ohms == 50
        assert meta.max_frequency_ghz == 18
        assert meta.courtyard is not None
        assert meta.courtyard.min_x_nm == -3500000

    def test_load_with_full_path(self) -> None:
        """Test loading with full 'lib:name' path."""
        meta = load_footprint_meta("Coupongen_Connectors:SMA_EndLaunch_Generic")
        assert meta.id == "SMA_EndLaunch_Generic"

    def test_load_nonexistent(self) -> None:
        """Test loading nonexistent metadata raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Footprint metadata not found"):
            load_footprint_meta("NonExistent_Footprint")

    def test_load_is_cached(self) -> None:
        """Test that repeated loads return same cached object."""
        meta1 = load_footprint_meta("SMA_EndLaunch_Generic")
        meta2 = load_footprint_meta("SMA_EndLaunch_Generic")
        assert meta1 is meta2


class TestListAvailableFootprintMeta:
    """Tests for list_available_footprint_meta function."""

    def test_list_contains_sma(self) -> None:
        """Test that list includes SMA_EndLaunch_Generic."""
        available = list_available_footprint_meta()
        assert "SMA_EndLaunch_Generic" in available

    def test_list_excludes_schema(self) -> None:
        """Test that schema file is excluded from list."""
        available = list_available_footprint_meta()
        assert "footprint_meta.schema" not in available
        assert all(not name.endswith(".schema") for name in available)


class TestGetFootprintMetaPath:
    """Tests for get_footprint_meta_path function."""

    def test_basic_path(self) -> None:
        """Test basic path generation."""
        path = get_footprint_meta_path("SMA_EndLaunch_Generic")
        assert path.name == "SMA_EndLaunch_Generic.json"
        assert path.parent.name == "footprints_meta"


class TestSchemaCompliance:
    """Tests that ensure metadata files comply with schema."""

    def test_sma_endlaunch_has_required_fields(self) -> None:
        """Test SMA_EndLaunch_Generic has all required schema fields."""
        meta = load_footprint_meta("SMA_EndLaunch_Generic")

        # Required top-level fields
        assert meta.schema_version == 1
        assert meta.id
        assert meta.name
        assert meta.footprint_lib
        assert meta.footprint_name

        # Required nested fields
        assert meta.anchor is not None
        assert meta.signal_pad is not None
        assert len(meta.ground_pads) >= 1
        assert meta.launch_reference is not None

    def test_pad_sizes_are_positive(self) -> None:
        """Test that all pad sizes are positive."""
        meta = load_footprint_meta("SMA_EndLaunch_Generic")

        assert meta.signal_pad.size_x_nm > 0
        assert meta.signal_pad.size_y_nm > 0

        for gp in meta.ground_pads:
            assert gp.size_x_nm > 0
            assert gp.size_y_nm > 0

    def test_launch_direction_valid(self) -> None:
        """Test launch direction is in valid range."""
        meta = load_footprint_meta("SMA_EndLaunch_Generic")
        assert 0 <= meta.launch_reference.direction_deg < 360
