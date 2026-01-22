<<<<<<< HEAD
"""Tests for M2 SimConfig validation (REQ-M2-003).

These tests validate:
- Nyquist compliance validation
- PML adequacy validation
- GPU configuration validation
- sim_config.json storage
- Validation report generation
=======
"""Tests for M2 simulation config validation (REQ-M2-003).

These tests validate:
- Nyquist compliance checking for FDTD simulations
- PML adequacy validation for boundary conditions
- GPU flag configuration
- sim_config.json storage and loading
- Comprehensive validation report generation
>>>>>>> 65c0ed4 (Implement SimConfig schema validation (REQ-M2-003))
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
<<<<<<< HEAD
from typing import Any
=======
>>>>>>> 65c0ed4 (Implement SimConfig schema validation (REQ-M2-003))

import pytest

from formula_foundry.openems import (
<<<<<<< HEAD
    DEFAULT_MIN_CELLS_PER_WAVELENGTH,
    DEFAULT_MIN_PML_WAVELENGTHS,
    RECOMMENDED_CELLS_PER_WAVELENGTH,
    SimConfigValidationReport,
    ValidationResult,
    ValidationStatus,
    load_sim_config_json,
    load_simulationspec,
    validate_gpu_config,
    validate_nyquist_compliance,
    validate_pml_adequacy,
    validate_sim_config,
    write_sim_config_json,
    write_validation_report,
=======
    BoundarySpec,
    EngineSpec,
    ExcitationSpec,
    FrequencySpec,
    GeometryRefSpec,
    MeshSpec,
    OpenEMSToolchainSpec,
    PortSpec,
    SimulationControlSpec,
    SimulationSpec,
    ToolchainSpec,
    load_simulationspec,
)
from formula_foundry.openems.sim_config_validation import (
    SimConfigValidationReport,
    ValidationResult,
    ValidationStatus,
    compute_fdtd_timestep_limit_ps,
    get_frequency_range_category,
    get_pml_layers,
    load_sim_config,
    validate_nyquist_compliance,
    validate_pml_adequacy,
    validate_sim_config,
    write_sim_config,
>>>>>>> 65c0ed4 (Implement SimConfig schema validation (REQ-M2-003))
)


# =============================================================================
# Test Fixtures
# =============================================================================


<<<<<<< HEAD
def _minimal_simulation_spec_dict() -> dict[str, Any]:
    """Return a minimal valid SimulationSpec dictionary.

    Uses a realistic frequency range for via transition coupons
    (1 GHz to 20 GHz - about 1.3 decades, typical for SI applications).
    """
    return {
        "schema_version": 1,
        "toolchain": {
            "openems": {
                "version": "0.0.35",
                "docker_image": "ghcr.io/thliebig/openems:0.0.35@sha256:abc123",
            }
        },
        "geometry_ref": {
            "design_hash": "abc123def456",
        },
        "excitation": {
            "f0_hz": "5GHz",
            "fc_hz": "10GHz",
        },
        "frequency": {
            "f_start_hz": "1GHz",  # Realistic start for SI applications
            "f_stop_hz": "20GHz",
            "n_points": 401,
        },
        "ports": [
            {
                "id": "port1",
                "position_nm": [0, 0, 0],
                "direction": "x",
                "excite": True,
            },
            {
                "id": "port2",
                "position_nm": [10_000_000, 0, 0],
                "direction": "-x",
            },
        ],
    }
=======
def _minimal_simulation_spec() -> SimulationSpec:
    """Create a minimal valid SimulationSpec for testing."""
    return SimulationSpec(
        toolchain=ToolchainSpec(
            openems=OpenEMSToolchainSpec(
                version="0.0.35",
                docker_image="ghcr.io/thliebig/openems:0.0.35@sha256:abc123",
            )
        ),
        geometry_ref=GeometryRefSpec(design_hash="abc123def456"),
        excitation=ExcitationSpec(f0_hz=5_000_000_000, fc_hz=10_000_000_000),
        frequency=FrequencySpec(
            f_start_hz=1_000_000_000, f_stop_hz=20_000_000_000, n_points=401
        ),
        ports=[
            PortSpec(id="P1", position_nm=(0, 0, 0), direction="x", excite=True),
            PortSpec(id="P2", position_nm=(10_000_000, 0, 0), direction="-x"),
        ],
    )


def _high_frequency_spec() -> SimulationSpec:
    """Create a spec with high frequencies for PML testing."""
    return SimulationSpec(
        toolchain=ToolchainSpec(
            openems=OpenEMSToolchainSpec(
                version="0.0.35",
                docker_image="ghcr.io/thliebig/openems:0.0.35@sha256:abc123",
            )
        ),
        geometry_ref=GeometryRefSpec(design_hash="abc123def456"),
        excitation=ExcitationSpec(f0_hz=30_000_000_000, fc_hz=50_000_000_000),
        frequency=FrequencySpec(
            f_start_hz=10_000_000_000, f_stop_hz=50_000_000_000, n_points=201
        ),
        ports=[
            PortSpec(id="P1", position_nm=(0, 0, 0), direction="x", excite=True),
            PortSpec(id="P2", position_nm=(10_000_000, 0, 0), direction="-x"),
        ],
        boundaries=BoundarySpec(
            x_min="PML_8", x_max="PML_8", y_min="PEC", y_max="PEC", z_min="PEC", z_max="PML_8"
        ),
    )


# =============================================================================
# GPU Flag Tests
# =============================================================================


class TestGPUConfiguration:
    """Tests for GPU configuration in EngineSpec."""

    def test_gpu_disabled_by_default(self) -> None:
        """GPU should be disabled by default."""
        spec = _minimal_simulation_spec()
        assert spec.control.engine.use_gpu is False
        assert spec.control.engine.gpu_device_id is None
        assert spec.control.engine.gpu_memory_limit_mb is None

    def test_gpu_enabled(self) -> None:
        """GPU can be enabled with optional settings."""
        engine = EngineSpec(use_gpu=True, gpu_device_id=0, gpu_memory_limit_mb=4096)
        assert engine.use_gpu is True
        assert engine.gpu_device_id == 0
        assert engine.gpu_memory_limit_mb == 4096

    def test_gpu_device_id_validation(self) -> None:
        """GPU device ID must be non-negative."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            EngineSpec(use_gpu=True, gpu_device_id=-1)

    def test_gpu_memory_limit_validation(self) -> None:
        """GPU memory limit must be at least 256 MB."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="greater than or equal to 256"):
            EngineSpec(use_gpu=True, gpu_memory_limit_mb=128)

    def test_gpu_serialization_roundtrip(self) -> None:
        """GPU settings should survive serialization roundtrip."""
        spec = _minimal_simulation_spec()
        spec_dict = spec.model_dump()
        spec_dict["control"]["engine"]["use_gpu"] = True
        spec_dict["control"]["engine"]["gpu_device_id"] = 1
        spec_dict["control"]["engine"]["gpu_memory_limit_mb"] = 8192

        reloaded = load_simulationspec(spec_dict)
        assert reloaded.control.engine.use_gpu is True
        assert reloaded.control.engine.gpu_device_id == 1
        assert reloaded.control.engine.gpu_memory_limit_mb == 8192
>>>>>>> 65c0ed4 (Implement SimConfig schema validation (REQ-M2-003))


# =============================================================================
# Nyquist Compliance Tests
# =============================================================================


class TestNyquistCompliance:
    """Tests for Nyquist compliance validation."""

<<<<<<< HEAD
    def test_default_resolution_passes(self) -> None:
        """Default lambda_resolution=20 should pass Nyquist check."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        result = validate_nyquist_compliance(spec)
        assert result.status == ValidationStatus.PASSED
        assert result.name == "nyquist_compliance"

    def test_high_resolution_passes(self) -> None:
        """High lambda_resolution should pass with flying colors."""
        data = _minimal_simulation_spec_dict()
        data["mesh"] = {"resolution": {"lambda_resolution": 50}}
        spec = load_simulationspec(data)
        result = validate_nyquist_compliance(spec)
        assert result.status == ValidationStatus.PASSED
        assert result.value == 50.0

    def test_minimum_resolution_passes(self) -> None:
        """lambda_resolution=10 (minimum) should pass."""
        data = _minimal_simulation_spec_dict()
        data["mesh"] = {"resolution": {"lambda_resolution": 10}}
        spec = load_simulationspec(data)
        result = validate_nyquist_compliance(spec)
        # At minimum, it passes but might be warning level
        assert result.passed

    def test_low_resolution_warns(self) -> None:
        """lambda_resolution below recommended but above minimum should warn."""
        data = _minimal_simulation_spec_dict()
        data["mesh"] = {"resolution": {"lambda_resolution": 15}}
        spec = load_simulationspec(data)
        result = validate_nyquist_compliance(spec)
        assert result.status == ValidationStatus.WARNING
        assert result.value == 15.0
        assert result.threshold == float(RECOMMENDED_CELLS_PER_WAVELENGTH)

    def test_wavelength_calculation(self) -> None:
        """Wavelength calculation should be included in details."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        result = validate_nyquist_compliance(spec, epsilon_r=4.0)
        assert "wavelength_nm" in result.details
        assert result.details["max_freq_hz"] == 20_000_000_000

    def test_higher_epsilon_shorter_wavelength(self) -> None:
        """Higher epsilon_r should result in shorter wavelength."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)

        result_low_eps = validate_nyquist_compliance(spec, epsilon_r=1.0)
        result_high_eps = validate_nyquist_compliance(spec, epsilon_r=4.0)

        # Higher epsilon means shorter wavelength
        assert result_high_eps.details["wavelength_nm"] < result_low_eps.details["wavelength_nm"]
=======
    def test_compute_timestep_limit(self) -> None:
        """Timestep limit should be correctly computed from cell size."""
        # 50um cell in vacuum with Courant factor 0.5:
        # dt = dx / (c * sqrt(3)) * 0.5
        # dt = 50e-6 / (3e8 * 1.732) * 0.5 ~ 48 fs = 0.048 ps
        dt_ps = compute_fdtd_timestep_limit_ps(
            min_cell_size_nm=50_000, epsilon_r=1.0, courant_factor=0.5
        )
        # Should be around 0.05 ps for 50um cells
        assert 0.03 < dt_ps < 0.1

    def test_compute_timestep_with_dielectric(self) -> None:
        """Timestep should be smaller in higher permittivity medium."""
        dt_vacuum = compute_fdtd_timestep_limit_ps(min_cell_size_nm=50_000, epsilon_r=1.0)
        dt_substrate = compute_fdtd_timestep_limit_ps(min_cell_size_nm=50_000, epsilon_r=4.0)
        # Higher epsilon = slower waves = larger allowed timestep
        assert dt_substrate > dt_vacuum

    def test_nyquist_compliance_passes(self) -> None:
        """Nyquist should pass with fine mesh and moderate frequency."""
        spec = _minimal_simulation_spec()
        # With default 50um resolution and 20GHz max, should easily pass
        result = validate_nyquist_compliance(spec)
        assert result.status == ValidationStatus.PASSED
        assert result.passed

    def test_nyquist_compliance_warning(self) -> None:
        """Nyquist should warn with marginal settings."""
        spec = _minimal_simulation_spec()
        # Force marginal conditions by using extremely coarse mesh
        # With 10mm cells and 20GHz, the ratio is about 5.2x which is between 2x and 10x
        result = validate_nyquist_compliance(
            spec, min_cell_size_nm=10_000_000, safety_factor=10.0  # 10mm cells
        )
        # Should give warning (ratio ~5.2x is above minimum 2x but below recommended 10x)
        assert result.status in (ValidationStatus.WARNING, ValidationStatus.FAILED)

    def test_nyquist_compliance_fails(self) -> None:
        """Nyquist should fail with extremely coarse mesh."""
        spec = _minimal_simulation_spec()
        # Extremely coarse mesh - 100mm cells
        result = validate_nyquist_compliance(
            spec, min_cell_size_nm=100_000_000  # 100mm cells
        )
        # At 100mm cells, the ratio is about 0.52x which is below the 2x minimum
        assert result.status == ValidationStatus.FAILED
        assert not result.passed

    def test_nyquist_result_has_details(self) -> None:
        """Nyquist result should include diagnostic details."""
        spec = _minimal_simulation_spec()
        result = validate_nyquist_compliance(spec)
        assert result.details is not None
        assert "f_max_hz" in result.details
        assert "sampling_freq_hz" in result.details
        assert "timestep_ps" in result.details
        assert "actual_ratio" in result.details
>>>>>>> 65c0ed4 (Implement SimConfig schema validation (REQ-M2-003))


# =============================================================================
# PML Adequacy Tests
# =============================================================================


class TestPMLAdequacy:
<<<<<<< HEAD
    """Tests for PML adequacy validation."""

    def test_default_pml_passes(self) -> None:
        """Default PML_8 boundaries should pass."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        result = validate_pml_adequacy(spec)
        # With high frequency, PML_8 should be adequate
        assert result.passed
        assert result.name == "pml_adequacy"

    def test_thick_pml_passes(self) -> None:
        """PML_32 should pass easily."""
        data = _minimal_simulation_spec_dict()
        data["boundaries"] = {
            "x_min": "PML_32",
            "x_max": "PML_32",
            "y_min": "PML_32",
            "y_max": "PML_32",
            "z_min": "PML_32",
            "z_max": "PML_32",
        }
        spec = load_simulationspec(data)
        result = validate_pml_adequacy(spec)
        assert result.status == ValidationStatus.PASSED

    def test_no_pml_warns(self) -> None:
        """No PML boundaries should generate warning."""
        data = _minimal_simulation_spec_dict()
        data["boundaries"] = {
            "x_min": "PEC",
            "x_max": "PEC",
            "y_min": "PEC",
            "y_max": "PEC",
            "z_min": "PEC",
            "z_max": "PEC",
        }
        spec = load_simulationspec(data)
        result = validate_pml_adequacy(spec)
        assert result.status == ValidationStatus.WARNING
        assert "No PML boundaries" in result.message

    def test_pml_details_reported(self) -> None:
        """PML check should report detailed boundary info."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        result = validate_pml_adequacy(spec)
        assert "pml_boundaries" in result.details
        assert "min_freq_hz" in result.details
        assert "wavelength_at_fmin_nm" in result.details

    def test_mixed_pml_boundaries(self) -> None:
        """Mixed PML and PEC boundaries should work correctly."""
        data = _minimal_simulation_spec_dict()
        data["boundaries"] = {
            "x_min": "PML_16",
            "x_max": "PML_16",
            "y_min": "PEC",
            "y_max": "PEC",
            "z_min": "PEC",
            "z_max": "PML_8",
        }
        spec = load_simulationspec(data)
        result = validate_pml_adequacy(spec)
        # Should check only PML boundaries
        pml_boundaries = result.details.get("pml_boundaries", {})
        assert "x_min" in pml_boundaries
        assert "x_max" in pml_boundaries
        assert "z_max" in pml_boundaries
        assert "y_min" not in pml_boundaries


# =============================================================================
# GPU Configuration Tests
# =============================================================================


class TestGPUConfig:
    """Tests for GPU configuration validation."""

    def test_gpu_disabled_passes(self) -> None:
        """GPU disabled should pass validation."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        result = validate_gpu_config(spec)
        assert result.status == ValidationStatus.PASSED
        assert "CPU" in result.message

    def test_gpu_enabled_passes(self) -> None:
        """GPU enabled with valid settings should pass."""
        data = _minimal_simulation_spec_dict()
        data["control"] = {
            "engine": {
                "use_gpu": True,
                "gpu_device_id": 0,
                "gpu_memory_fraction": 0.8,
            }
        }
        spec = load_simulationspec(data)
        result = validate_gpu_config(spec)
        assert result.passed
        assert result.details["use_gpu"] is True

    def test_gpu_low_memory_warns(self) -> None:
        """GPU with low memory fraction should warn."""
        data = _minimal_simulation_spec_dict()
        data["control"] = {
            "engine": {
                "use_gpu": True,
                "gpu_memory_fraction": 0.2,
            }
        }
        spec = load_simulationspec(data)
        result = validate_gpu_config(spec)
        assert result.status == ValidationStatus.WARNING
        assert "memory fraction" in result.message.lower()

    def test_gpu_with_multithreaded_warns(self) -> None:
        """GPU with multithreaded CPU engine should warn."""
        data = _minimal_simulation_spec_dict()
        data["control"] = {
            "engine": {
                "type": "multithreaded",
                "use_gpu": True,
            }
        }
        spec = load_simulationspec(data)
        result = validate_gpu_config(spec)
        assert result.status == ValidationStatus.WARNING


# =============================================================================
# Full Validation Tests
# =============================================================================


class TestFullValidation:
    """Tests for complete SimConfig validation."""

    def test_full_validation_default_passes(self) -> None:
        """Full validation with defaults should pass."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        report = validate_sim_config(spec)
        assert isinstance(report, SimConfigValidationReport)
        assert report.all_passed or report.overall_status == ValidationStatus.WARNING

    def test_full_validation_includes_all_checks(self) -> None:
        """Full validation should include all check types."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        report = validate_sim_config(spec)
        check_names = {c.name for c in report.checks}
        assert "nyquist_compliance" in check_names
        assert "pml_adequacy" in check_names
        assert "gpu_config" in check_names

    def test_full_validation_to_dict(self) -> None:
        """Validation report should be serializable."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        report = validate_sim_config(spec, spec_hash="test_hash")
        report_dict = report.to_dict()
        assert "overall_status" in report_dict
        assert "checks" in report_dict
        assert "spec_hash" in report_dict
        assert report_dict["spec_hash"] == "test_hash"

    def test_full_validation_canonical_hash(self) -> None:
        """Validation report should have canonical hash."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        report = validate_sim_config(spec)
        assert report.canonical_hash
        assert len(report.canonical_hash) == 64  # SHA256 hex

    def test_failed_check_fails_report(self) -> None:
        """A failed check should fail the overall report."""
        data = _minimal_simulation_spec_dict()
        # Use very low frequency to make PML inadequate
        data["frequency"]["f_start_hz"] = "1kHz"
        data["frequency"]["f_stop_hz"] = "10kHz"
        data["mesh"] = {"resolution": {"lambda_resolution": 10}}
        spec = load_simulationspec(data)
        report = validate_sim_config(spec, min_pml_wavelengths=10.0)
        # Should have at least one failure
        assert report.n_failed > 0 or report.overall_status == ValidationStatus.FAILED
=======
    """Tests for PML boundary condition adequacy validation."""

    def test_get_pml_layers(self) -> None:
        """PML layer count extraction should work correctly."""
        assert get_pml_layers("PML_8") == 8
        assert get_pml_layers("PML_16") == 16
        assert get_pml_layers("PML_32") == 32
        assert get_pml_layers("PEC") is None
        assert get_pml_layers("PMC") is None
        assert get_pml_layers("MUR") is None

    def test_frequency_range_categories(self) -> None:
        """Frequency range categorization should be correct."""
        assert get_frequency_range_category(1e9) == "low"  # 1 GHz
        assert get_frequency_range_category(4e9) == "low"  # 4 GHz
        assert get_frequency_range_category(5e9) == "mid"  # 5 GHz (boundary)
        assert get_frequency_range_category(10e9) == "mid"  # 10 GHz
        assert get_frequency_range_category(20e9) == "mid"  # 20 GHz (boundary)
        assert get_frequency_range_category(30e9) == "high"  # 30 GHz
        assert get_frequency_range_category(100e9) == "high"  # 100 GHz

    def test_pml_adequacy_passes_mid_frequency(self) -> None:
        """PML_8 should pass for mid-frequency simulations."""
        spec = _minimal_simulation_spec()
        # Default spec is 1-20 GHz, should be 'mid' category
        result = validate_pml_adequacy(spec)
        assert result.status in (ValidationStatus.PASSED, ValidationStatus.WARNING)

    def test_pml_adequacy_fails_high_frequency_with_low_pml(self) -> None:
        """PML_8 should fail/warn for high frequency simulations."""
        spec = _high_frequency_spec()  # 10-50 GHz
        result = validate_pml_adequacy(spec)
        # High frequency needs PML_16 or PML_32
        assert result.status in (ValidationStatus.WARNING, ValidationStatus.FAILED)

    def test_pml_adequacy_high_frequency_with_adequate_pml(self) -> None:
        """PML_32 should pass for high frequency simulations."""
        spec = _high_frequency_spec()
        spec_dict = spec.model_dump()
        spec_dict["boundaries"] = {
            "x_min": "PML_32",
            "x_max": "PML_32",
            "y_min": "PEC",
            "y_max": "PEC",
            "z_min": "PEC",
            "z_max": "PML_32",
        }
        spec_with_pml32 = load_simulationspec(spec_dict)
        result = validate_pml_adequacy(spec_with_pml32)
        assert result.passed

    def test_pml_adequacy_warns_missing_port_pml(self) -> None:
        """Should warn if port faces (x_min, x_max) don't have PML."""
        spec = _minimal_simulation_spec()
        spec_dict = spec.model_dump()
        spec_dict["boundaries"] = {
            "x_min": "PEC",  # No PML on port face
            "x_max": "PEC",  # No PML on port face
            "y_min": "PML_8",
            "y_max": "PML_8",
            "z_min": "PML_8",
            "z_max": "PML_8",
        }
        spec_no_port_pml = load_simulationspec(spec_dict)
        result = validate_pml_adequacy(spec_no_port_pml)
        # Should warn about missing port PML
        assert result.status == ValidationStatus.WARNING
        assert "port" in result.message.lower()


# =============================================================================
# Comprehensive Validation Tests
# =============================================================================


class TestComprehensiveValidation:
    """Tests for the comprehensive validation report."""

    def test_validate_sim_config_passes(self) -> None:
        """Valid config should pass all validation checks."""
        spec = _minimal_simulation_spec()
        report = validate_sim_config(spec)
        # Should pass overall (may have warnings)
        assert isinstance(report, SimConfigValidationReport)
        assert len(report.results) >= 4  # At least 4 checks

    def test_validate_sim_config_detects_no_excited_port(self) -> None:
        """Should fail if no port has excite=True."""
        spec = _minimal_simulation_spec()
        spec_dict = spec.model_dump()
        # Set all ports to excite=False
        for port in spec_dict["ports"]:
            port["excite"] = False
        spec_no_excite = load_simulationspec(spec_dict)

        report = validate_sim_config(spec_no_excite)
        assert not report.overall_passed
        assert any("excite" in e.lower() for e in report.errors)

    def test_validate_sim_config_detects_invalid_frequency_sweep(self) -> None:
        """Should fail if f_start >= f_stop."""
        spec = _minimal_simulation_spec()
        spec_dict = spec.model_dump()
        spec_dict["frequency"]["f_start_hz"] = 30_000_000_000  # 30 GHz
        spec_dict["frequency"]["f_stop_hz"] = 10_000_000_000  # 10 GHz
        spec_invalid_freq = load_simulationspec(spec_dict)

        report = validate_sim_config(spec_invalid_freq)
        assert not report.overall_passed
        assert any("f_start" in e.lower() or "frequency" in e.lower() for e in report.errors)

    def test_validate_sim_config_gpu_warning(self) -> None:
        """Should warn about GPU mode with non-standard engine type."""
        spec = _minimal_simulation_spec()
        spec_dict = spec.model_dump()
        spec_dict["control"]["engine"]["use_gpu"] = True
        spec_dict["control"]["engine"]["type"] = "sse-compressed"
        spec_gpu = load_simulationspec(spec_dict)

        report = validate_sim_config(spec_gpu)
        # Should have warning about GPU compatibility
        assert any("gpu" in w.lower() for w in report.warnings)

    def test_validation_report_serialization(self) -> None:
        """Validation report should be serializable to dict."""
        spec = _minimal_simulation_spec()
        report = validate_sim_config(spec)
        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert "overall_passed" in report_dict
        assert "warnings" in report_dict
        assert "errors" in report_dict
        assert "results" in report_dict
        assert isinstance(report_dict["results"], list)
>>>>>>> 65c0ed4 (Implement SimConfig schema validation (REQ-M2-003))


# =============================================================================
# sim_config.json Storage Tests
# =============================================================================


class TestSimConfigStorage:
<<<<<<< HEAD
    """Tests for sim_config.json storage."""

    def test_write_sim_config_json(self) -> None:
        """write_sim_config_json should create valid JSON file."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = write_sim_config_json(spec, output_dir)

            assert path.exists()
            assert path.name == "sim_config.json"

            # Should be valid JSON
            content = json.loads(path.read_text())
            assert "schema_version" in content
            assert "spec" in content

    def test_write_sim_config_with_validation(self) -> None:
        """write_sim_config_json should include validation report if provided."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        report = validate_sim_config(spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = write_sim_config_json(spec, output_dir, validation_report=report)

            content = json.loads(path.read_text())
            assert "validation" in content
            assert "overall_status" in content["validation"]
            assert "checks" in content["validation"]

    def test_load_sim_config_json(self) -> None:
        """load_sim_config_json should read back written config."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = write_sim_config_json(spec, output_dir)

            loaded = load_sim_config_json(path)
            assert loaded["schema_version"] == spec.schema_version
            assert "spec" in loaded

    def test_sim_config_creates_parent_dirs(self) -> None:
        """write_sim_config_json should create parent directories."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "path"
            path = write_sim_config_json(spec, output_dir)

            assert path.exists()
            assert output_dir.exists()


# =============================================================================
# Validation Report I/O Tests
# =============================================================================


class TestValidationReportIO:
    """Tests for validation report read/write."""

    def test_write_validation_report(self) -> None:
        """write_validation_report should create valid JSON file."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        report = validate_sim_config(spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation.json"
            write_validation_report(report, output_path)

            assert output_path.exists()
            content = json.loads(output_path.read_text())
            assert content["overall_status"] == report.overall_status.value

    def test_validation_report_n_passed(self) -> None:
        """Validation report should correctly count passed checks."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        report = validate_sim_config(spec)

        assert report.n_passed == sum(1 for c in report.checks if c.passed)
        assert report.n_failed == sum(1 for c in report.checks if c.status == ValidationStatus.FAILED)

    def test_get_check_by_name(self) -> None:
        """Validation report should allow getting check by name."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)
        report = validate_sim_config(spec)

        nyquist_check = report.get_check("nyquist_compliance")
        assert nyquist_check is not None
        assert nyquist_check.name == "nyquist_compliance"

        nonexistent = report.get_check("nonexistent")
        assert nonexistent is None


# =============================================================================
# GPU Engine Spec Tests
# =============================================================================


class TestGPUEngineSpec:
    """Tests for GPU-related engine spec fields."""

    def test_gpu_fields_in_spec(self) -> None:
        """EngineSpec should accept GPU-related fields."""
        data = _minimal_simulation_spec_dict()
        data["control"] = {
            "engine": {
                "type": "multithreaded",
                "use_gpu": True,
                "gpu_device_id": 0,
                "gpu_memory_fraction": 0.9,
            }
        }
        spec = load_simulationspec(data)

        assert spec.control.engine.use_gpu is True
        assert spec.control.engine.gpu_device_id == 0
        assert spec.control.engine.gpu_memory_fraction == 0.9

    def test_gpu_defaults(self) -> None:
        """GPU fields should have sensible defaults."""
        data = _minimal_simulation_spec_dict()
        spec = load_simulationspec(data)

        assert spec.control.engine.use_gpu is False
        assert spec.control.engine.gpu_device_id is None
        assert spec.control.engine.gpu_memory_fraction is None

    def test_gpu_memory_fraction_bounds(self) -> None:
        """GPU memory fraction should be bounded 0.1-1.0."""
        data = _minimal_simulation_spec_dict()

        # Too low
        data["control"] = {"engine": {"gpu_memory_fraction": 0.05}}
        with pytest.raises(Exception):  # ValidationError
            load_simulationspec(data)

        # Too high
        data["control"] = {"engine": {"gpu_memory_fraction": 1.5}}
        with pytest.raises(Exception):
            load_simulationspec(data)

    def test_gpu_device_id_non_negative(self) -> None:
        """GPU device ID should be non-negative."""
        data = _minimal_simulation_spec_dict()
        data["control"] = {"engine": {"gpu_device_id": -1}}
        with pytest.raises(Exception):
            load_simulationspec(data)
=======
    """Tests for sim_config.json file storage and loading."""

    def test_write_sim_config_creates_file(self) -> None:
        """write_sim_config should create sim_config.json file."""
        spec = _minimal_simulation_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "sim_outputs"
            config_path = write_sim_config(spec, output_dir, validate=False)

            assert config_path.exists()
            assert config_path.name == "sim_config.json"

    def test_write_sim_config_custom_filename(self) -> None:
        """write_sim_config should support custom filename."""
        spec = _minimal_simulation_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "sim_outputs"
            config_path = write_sim_config(
                spec, output_dir, validate=False, filename="simulation_config.json"
            )

            assert config_path.exists()
            assert config_path.name == "simulation_config.json"

    def test_write_sim_config_validates_by_default(self) -> None:
        """write_sim_config should validate config by default."""
        spec = _minimal_simulation_spec()
        spec_dict = spec.model_dump()
        for port in spec_dict["ports"]:
            port["excite"] = False
        spec_no_excite = load_simulationspec(spec_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "sim_outputs"
            with pytest.raises(ValueError, match="validation failed"):
                write_sim_config(spec_no_excite, output_dir, validate=True)

    def test_load_sim_config_roundtrip(self) -> None:
        """Loaded config should match original spec."""
        spec = _minimal_simulation_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "sim_outputs"
            config_path = write_sim_config(spec, output_dir, validate=False)

            loaded_spec = load_sim_config(config_path)
            assert loaded_spec.frequency.f_start_hz == spec.frequency.f_start_hz
            assert loaded_spec.frequency.f_stop_hz == spec.frequency.f_stop_hz
            assert len(loaded_spec.ports) == len(spec.ports)

    def test_load_sim_config_file_not_found(self) -> None:
        """load_sim_config should raise if file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "nonexistent.json"
            with pytest.raises(FileNotFoundError):
                load_sim_config(fake_path)

    def test_written_config_is_canonical_json(self) -> None:
        """Written config should be deterministic canonical JSON."""
        spec = _minimal_simulation_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "sim_outputs"
            config_path = write_sim_config(spec, output_dir, validate=False)

            content = config_path.read_text()
            # Should be parseable JSON
            data = json.loads(content)
            assert isinstance(data, dict)

            # Write again and verify identical
            config_path2 = output_dir / "sim_config_2.json"
            write_sim_config(spec, output_dir, validate=False, filename="sim_config_2.json")
            content2 = config_path2.read_text()
            assert content == content2


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_passed_property(self) -> None:
        """passed property should reflect status correctly."""
        passed = ValidationResult(
            name="test", status=ValidationStatus.PASSED, message="OK"
        )
        warning = ValidationResult(
            name="test", status=ValidationStatus.WARNING, message="Marginal"
        )
        failed = ValidationResult(
            name="test", status=ValidationStatus.FAILED, message="Error"
        )

        assert passed.passed is True
        assert warning.passed is True  # Warnings still count as passed
        assert failed.passed is False

    def test_result_to_dict(self) -> None:
        """ValidationResult should serialize to dictionary."""
        result = ValidationResult(
            name="test_check",
            status=ValidationStatus.PASSED,
            message="All good",
            value=42.0,
            threshold=10.0,
            details={"extra": "info"},
        )
        d = result.to_dict()

        assert d["name"] == "test_check"
        assert d["status"] == "passed"
        assert d["message"] == "All good"
        assert d["value"] == 42.0
        assert d["threshold"] == 10.0
        assert d["details"]["extra"] == "info"
>>>>>>> 65c0ed4 (Implement SimConfig schema validation (REQ-M2-003))
