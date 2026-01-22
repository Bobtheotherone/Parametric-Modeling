"""Tests for M2 SimConfig validation (REQ-M2-003).

These tests validate:
- Nyquist compliance validation
- PML adequacy validation
- GPU configuration validation
- sim_config.json storage
- Validation report generation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.openems import (
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
)


# =============================================================================
# Test Fixtures
# =============================================================================


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


# =============================================================================
# Nyquist Compliance Tests
# =============================================================================


class TestNyquistCompliance:
    """Tests for Nyquist compliance validation."""

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


# =============================================================================
# PML Adequacy Tests
# =============================================================================


class TestPMLAdequacy:
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


# =============================================================================
# sim_config.json Storage Tests
# =============================================================================


class TestSimConfigStorage:
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
