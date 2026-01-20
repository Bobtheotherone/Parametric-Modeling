"""Tests for formula_foundry.em.mesh module.

Tests cover:
- MeshConfig dataclass validation and properties
- FrequencyRange validation
- AdaptiveMeshDensity computation
- Wavelength calculations
- Integration with GeometrySpec
"""

from __future__ import annotations

import pytest

from formula_foundry.em.mesh import (
    AdaptiveMeshDensity,
    FrequencyRange,
    MeshConfig,
    compute_adaptive_mesh_density,
    compute_min_wavelength_nm,
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


class TestFrequencyRange:
    """Tests for FrequencyRange dataclass."""

    def test_valid_range(self) -> None:
        """Valid frequency range is accepted."""
        freq = FrequencyRange(f_min_hz=100_000_000, f_max_hz=20_000_000_000)
        assert freq.f_min_hz == 100_000_000
        assert freq.f_max_hz == 20_000_000_000

    def test_equal_min_max(self) -> None:
        """Equal min and max is valid (single frequency)."""
        freq = FrequencyRange(f_min_hz=1_000_000_000, f_max_hz=1_000_000_000)
        assert freq.f_min_hz == freq.f_max_hz

    def test_negative_min_rejected(self) -> None:
        """Negative f_min_hz is rejected."""
        with pytest.raises(ValueError, match="f_min_hz must be positive"):
            FrequencyRange(f_min_hz=-100, f_max_hz=1_000_000_000)

    def test_zero_min_rejected(self) -> None:
        """Zero f_min_hz is rejected."""
        with pytest.raises(ValueError, match="f_min_hz must be positive"):
            FrequencyRange(f_min_hz=0, f_max_hz=1_000_000_000)

    def test_negative_max_rejected(self) -> None:
        """Negative f_max_hz is rejected."""
        with pytest.raises(ValueError, match="f_max_hz must be positive"):
            FrequencyRange(f_min_hz=100_000_000, f_max_hz=-100)

    def test_zero_max_rejected(self) -> None:
        """Zero f_max_hz is rejected."""
        with pytest.raises(ValueError, match="f_max_hz must be positive"):
            FrequencyRange(f_min_hz=100_000_000, f_max_hz=0)

    def test_min_greater_than_max_rejected(self) -> None:
        """f_min_hz > f_max_hz is rejected."""
        with pytest.raises(ValueError, match="f_min_hz must be <= f_max_hz"):
            FrequencyRange(f_min_hz=20_000_000_000, f_max_hz=100_000_000)


class TestMeshConfig:
    """Tests for MeshConfig dataclass."""

    def test_valid_config(self) -> None:
        """Valid mesh configuration is accepted."""
        config = MeshConfig(
            smoothmesh_ratio=1.4,
            edge_refinement_nm=50_000,
            pml_cells=8,
            frequency_range=FrequencyRange(f_min_hz=100_000_000, f_max_hz=20_000_000_000),
            min_wavelength_fraction=0.05,
        )
        assert config.smoothmesh_ratio == 1.4
        assert config.edge_refinement_nm == 50_000
        assert config.pml_cells == 8
        assert config.min_wavelength_fraction == 0.05

    def test_default_values(self) -> None:
        """Default values are applied correctly."""
        config = MeshConfig(
            smoothmesh_ratio=1.3,
            edge_refinement_nm=40_000,
            pml_cells=8,
            frequency_range=FrequencyRange(f_min_hz=100_000_000, f_max_hz=20_000_000_000),
            min_wavelength_fraction=0.05,
        )
        assert config.via_refinement_nm == 25_000
        assert config.substrate_refinement_nm == 100_000
        assert config.min_cell_size_nm == 1_000
        assert config.max_cell_size_nm == 1_000_000

    def test_smoothmesh_ratio_bounds(self) -> None:
        """Smoothmesh ratio bounds are enforced."""
        # Lower bound
        with pytest.raises(ValueError, match="smoothmesh_ratio must be >= 1.0"):
            MeshConfig(
                smoothmesh_ratio=0.9,
                edge_refinement_nm=50_000,
                pml_cells=8,
                frequency_range=FrequencyRange(f_min_hz=100_000_000, f_max_hz=20_000_000_000),
                min_wavelength_fraction=0.05,
            )
        # Upper bound
        with pytest.raises(ValueError, match="smoothmesh_ratio must be <= 3.0"):
            MeshConfig(
                smoothmesh_ratio=3.5,
                edge_refinement_nm=50_000,
                pml_cells=8,
                frequency_range=FrequencyRange(f_min_hz=100_000_000, f_max_hz=20_000_000_000),
                min_wavelength_fraction=0.05,
            )

    def test_pml_cells_bounds(self) -> None:
        """PML cells bounds are enforced."""
        # Lower bound
        with pytest.raises(ValueError, match="pml_cells must be >= 1"):
            MeshConfig(
                smoothmesh_ratio=1.4,
                edge_refinement_nm=50_000,
                pml_cells=0,
                frequency_range=FrequencyRange(f_min_hz=100_000_000, f_max_hz=20_000_000_000),
                min_wavelength_fraction=0.05,
            )
        # Upper bound
        with pytest.raises(ValueError, match="pml_cells must be <= 64"):
            MeshConfig(
                smoothmesh_ratio=1.4,
                edge_refinement_nm=50_000,
                pml_cells=100,
                frequency_range=FrequencyRange(f_min_hz=100_000_000, f_max_hz=20_000_000_000),
                min_wavelength_fraction=0.05,
            )

    def test_min_wavelength_fraction_bounds(self) -> None:
        """Min wavelength fraction bounds are enforced."""
        # Lower bound
        with pytest.raises(ValueError, match="min_wavelength_fraction must be positive"):
            MeshConfig(
                smoothmesh_ratio=1.4,
                edge_refinement_nm=50_000,
                pml_cells=8,
                frequency_range=FrequencyRange(f_min_hz=100_000_000, f_max_hz=20_000_000_000),
                min_wavelength_fraction=0.0,
            )
        # Upper bound
        with pytest.raises(ValueError, match="min_wavelength_fraction must be <= 1.0"):
            MeshConfig(
                smoothmesh_ratio=1.4,
                edge_refinement_nm=50_000,
                pml_cells=8,
                frequency_range=FrequencyRange(f_min_hz=100_000_000, f_max_hz=20_000_000_000),
                min_wavelength_fraction=1.5,
            )

    def test_cell_size_bounds_consistency(self) -> None:
        """min_cell_size_nm must be <= max_cell_size_nm."""
        with pytest.raises(ValueError, match="min_cell_size_nm must be <= max_cell_size_nm"):
            MeshConfig(
                smoothmesh_ratio=1.4,
                edge_refinement_nm=50_000,
                pml_cells=8,
                frequency_range=FrequencyRange(f_min_hz=100_000_000, f_max_hz=20_000_000_000),
                min_wavelength_fraction=0.05,
                min_cell_size_nm=100_000,
                max_cell_size_nm=50_000,
            )

    def test_min_wavelength_property(self) -> None:
        """min_wavelength_nm property computes correctly."""
        config = MeshConfig(
            smoothmesh_ratio=1.4,
            edge_refinement_nm=50_000,
            pml_cells=8,
            frequency_range=FrequencyRange(f_min_hz=100_000_000, f_max_hz=10_000_000_000),
            min_wavelength_fraction=0.05,
        )
        # At 10 GHz in vacuum: lambda = c/f = 3e8/1e10 = 0.03m = 30mm = 30_000_000 nm
        expected_wavelength = 299_792_458 * 1_000_000_000 // 10_000_000_000
        assert config.min_wavelength_nm == expected_wavelength

    def test_base_cell_size_property(self) -> None:
        """base_cell_size_nm property computes correctly with clamping."""
        # Use higher frequency so the computed base cell is within bounds
        config = MeshConfig(
            smoothmesh_ratio=1.4,
            edge_refinement_nm=50_000,
            pml_cells=8,
            frequency_range=FrequencyRange(f_min_hz=1_000_000_000, f_max_hz=100_000_000_000),
            min_wavelength_fraction=0.05,
            min_cell_size_nm=1_000,
            max_cell_size_nm=500_000,
        )
        # At 100 GHz: wavelength ~3mm = 3_000_000 nm; base = 3_000_000 * 0.05 = 150_000 nm
        raw_base = int(config.min_wavelength_nm * 0.05)
        expected_base = max(config.min_cell_size_nm, min(raw_base, config.max_cell_size_nm))
        assert config.base_cell_size_nm == expected_base
        # Verify it's within bounds
        assert config.base_cell_size_nm >= config.min_cell_size_nm
        assert config.base_cell_size_nm <= config.max_cell_size_nm


class TestComputeMinWavelength:
    """Tests for compute_min_wavelength_nm function."""

    def test_vacuum_wavelength(self) -> None:
        """Wavelength in vacuum is computed correctly."""
        # At 1 GHz: lambda = c/f = 3e8/1e9 = 0.3m = 300mm = 300_000_000 nm
        wavelength = compute_min_wavelength_nm(1_000_000_000)
        assert wavelength == 299_792_458 * 1_000_000_000 // 1_000_000_000

    def test_dielectric_wavelength(self) -> None:
        """Wavelength in dielectric is reduced by sqrt(epsilon_r)."""
        import math

        f_hz = 10_000_000_000  # 10 GHz
        epsilon_r = 4.0

        vacuum_wavelength = compute_min_wavelength_nm(f_hz, epsilon_r=1.0)
        dielectric_wavelength = compute_min_wavelength_nm(f_hz, epsilon_r=epsilon_r)

        # Dielectric wavelength should be ~half of vacuum (sqrt(4) = 2)
        assert dielectric_wavelength == int(vacuum_wavelength / math.sqrt(epsilon_r))

    def test_invalid_frequency_rejected(self) -> None:
        """Non-positive frequency is rejected."""
        with pytest.raises(ValueError, match="f_max_hz must be positive"):
            compute_min_wavelength_nm(0)
        with pytest.raises(ValueError, match="f_max_hz must be positive"):
            compute_min_wavelength_nm(-100)

    def test_invalid_epsilon_rejected(self) -> None:
        """Non-positive epsilon_r is rejected."""
        with pytest.raises(ValueError, match="epsilon_r must be positive"):
            compute_min_wavelength_nm(1_000_000_000, epsilon_r=0)
        with pytest.raises(ValueError, match="epsilon_r must be positive"):
            compute_min_wavelength_nm(1_000_000_000, epsilon_r=-1.0)


class TestCreateDefaultMeshConfig:
    """Tests for create_default_mesh_config function."""

    def test_default_config(self) -> None:
        """Default config has sensible values."""
        config = create_default_mesh_config()
        assert config.smoothmesh_ratio == 1.4
        assert config.edge_refinement_nm == 50_000
        assert config.pml_cells == 8
        assert config.min_wavelength_fraction == 0.05
        assert config.frequency_range.f_min_hz == 100_000_000
        assert config.frequency_range.f_max_hz == 20_000_000_000

    def test_custom_frequency_range(self) -> None:
        """Custom frequency range is accepted."""
        config = create_default_mesh_config(f_min_hz=1_000_000_000, f_max_hz=50_000_000_000)
        assert config.frequency_range.f_min_hz == 1_000_000_000
        assert config.frequency_range.f_max_hz == 50_000_000_000


def _make_test_geometry(
    *,
    trace_w_nm: int = 300_000,
    trace_gap_nm: int = 180_000,
    epsilon_r: float = 4.1,
    with_discontinuity: bool = False,
) -> GeometrySpec:
    """Create a test GeometrySpec for mesh density tests."""
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
            },
        )
    return GeometrySpec(
        design_hash="test_hash_12345",
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


class TestComputeAdaptiveMeshDensity:
    """Tests for compute_adaptive_mesh_density function."""

    def test_basic_density_computation(self) -> None:
        """Basic mesh density is computed without errors."""
        config = create_default_mesh_config()
        geometry = _make_test_geometry()

        density = compute_adaptive_mesh_density(config, geometry)

        assert isinstance(density, AdaptiveMeshDensity)
        assert density.base_cell_nm > 0
        assert density.trace_cell_nm > 0
        assert density.via_cell_nm > 0
        assert density.antipad_cell_nm > 0
        assert density.substrate_cell_nm > 0
        assert density.pml_cell_nm > 0
        assert density.min_feature_size_nm > 0

    def test_trace_refinement(self) -> None:
        """Trace cell size is appropriately refined based on trace dimensions."""
        config = create_default_mesh_config()

        # Narrow trace should have finer mesh
        narrow_geom = _make_test_geometry(trace_w_nm=100_000, trace_gap_nm=60_000)
        narrow_density = compute_adaptive_mesh_density(config, narrow_geom)

        # Wide trace can have coarser mesh
        wide_geom = _make_test_geometry(trace_w_nm=500_000, trace_gap_nm=300_000)
        wide_density = compute_adaptive_mesh_density(config, wide_geom)

        # Narrow trace should have smaller (or equal) trace cell
        assert narrow_density.trace_cell_nm <= wide_density.trace_cell_nm

    def test_via_refinement_with_discontinuity(self) -> None:
        """Via cell size is refined when discontinuity parameters are present."""
        config = create_default_mesh_config()

        # Without discontinuity
        no_disc_geom = _make_test_geometry(with_discontinuity=False)
        no_disc_density = compute_adaptive_mesh_density(config, no_disc_geom)

        # With discontinuity (has via parameters)
        with_disc_geom = _make_test_geometry(with_discontinuity=True)
        with_disc_density = compute_adaptive_mesh_density(config, with_disc_geom)

        # With discontinuity should have finer via mesh
        assert with_disc_density.via_cell_nm <= no_disc_density.via_cell_nm

    def test_dielectric_affects_base_cell(self) -> None:
        """Higher epsilon_r results in smaller base cell (shorter wavelength)."""
        config = create_default_mesh_config()

        # Low epsilon
        low_eps_geom = _make_test_geometry(epsilon_r=2.0)
        low_eps_density = compute_adaptive_mesh_density(config, low_eps_geom)

        # High epsilon
        high_eps_geom = _make_test_geometry(epsilon_r=10.0)
        high_eps_density = compute_adaptive_mesh_density(config, high_eps_geom)

        # Higher epsilon should have smaller base cell
        assert high_eps_density.base_cell_nm < low_eps_density.base_cell_nm

    def test_min_feature_size_detection(self) -> None:
        """Minimum feature size is correctly detected from geometry."""
        config = create_default_mesh_config()

        # Geometry with small gap
        small_gap_geom = _make_test_geometry(trace_w_nm=300_000, trace_gap_nm=50_000)
        density = compute_adaptive_mesh_density(config, small_gap_geom)

        # Min feature should be the gap (50_000 nm)
        assert density.min_feature_size_nm == 50_000

    def test_cell_sizes_respect_bounds(self) -> None:
        """All cell sizes respect min/max bounds from config."""
        config = MeshConfig(
            smoothmesh_ratio=1.4,
            edge_refinement_nm=50_000,
            pml_cells=8,
            frequency_range=FrequencyRange(f_min_hz=100_000_000, f_max_hz=20_000_000_000),
            min_wavelength_fraction=0.05,
            min_cell_size_nm=10_000,
            max_cell_size_nm=200_000,
        )
        geometry = _make_test_geometry()
        density = compute_adaptive_mesh_density(config, geometry)

        # All cell sizes should be >= min
        assert density.base_cell_nm >= config.min_cell_size_nm
        assert density.trace_cell_nm >= config.min_cell_size_nm
        assert density.via_cell_nm >= config.min_cell_size_nm
        assert density.antipad_cell_nm >= config.min_cell_size_nm
        assert density.substrate_cell_nm >= config.min_cell_size_nm
        assert density.pml_cell_nm >= config.min_cell_size_nm

        # Base cell should be <= max
        assert density.base_cell_nm <= config.max_cell_size_nm


class TestMeshConfigImmutability:
    """Tests for dataclass immutability (frozen=True)."""

    def test_mesh_config_frozen(self) -> None:
        """MeshConfig is immutable."""
        config = create_default_mesh_config()
        with pytest.raises(AttributeError):
            config.smoothmesh_ratio = 2.0  # type: ignore[misc]

    def test_frequency_range_frozen(self) -> None:
        """FrequencyRange is immutable."""
        freq = FrequencyRange(f_min_hz=100_000_000, f_max_hz=20_000_000_000)
        with pytest.raises(AttributeError):
            freq.f_min_hz = 200_000_000  # type: ignore[misc]

    def test_adaptive_mesh_density_frozen(self) -> None:
        """AdaptiveMeshDensity is immutable."""
        density = AdaptiveMeshDensity(
            base_cell_nm=100_000,
            trace_cell_nm=50_000,
            via_cell_nm=25_000,
            antipad_cell_nm=25_000,
            substrate_cell_nm=75_000,
            pml_cell_nm=100_000,
            min_feature_size_nm=50_000,
        )
        with pytest.raises(AttributeError):
            density.base_cell_nm = 200_000  # type: ignore[misc]
