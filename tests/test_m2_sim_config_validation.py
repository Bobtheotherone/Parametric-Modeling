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

import pytest

from formula_foundry.openems import (
    BoundarySpec,
    EngineSpec,
    ExcitationSpec,
    FrequencySpec,
    GeometryRefSpec,
    MeshSpec,
    OpenEMSToolchainSpec,
    PortSpec,
    SimConfigValidationReport,
    SimulationControlSpec,
    SimulationSpec,
    ToolchainSpec,
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
        frequency=FrequencySpec(f_start_hz=1_000_000_000, f_stop_hz=20_000_000_000, n_points=401),
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
        frequency=FrequencySpec(f_start_hz=10_000_000_000, f_stop_hz=50_000_000_000, n_points=201),
        ports=[
            PortSpec(id="P1", position_nm=(0, 0, 0), direction="x", excite=True),
            PortSpec(id="P2", position_nm=(10_000_000, 0, 0), direction="-x"),
        ],
        boundaries=BoundarySpec(x_min="PML_8", x_max="PML_8", y_min="PEC", y_max="PEC", z_min="PEC", z_max="PML_8"),
    )


# =============================================================================
# GPU Configuration Tests
# =============================================================================


class TestGPUConfiguration:
    """Tests for GPU configuration in EngineSpec."""

    def test_gpu_disabled_by_default(self) -> None:
        """GPU should be disabled by default."""
        spec = _minimal_simulation_spec()
        assert spec.control.engine.use_gpu is False
        assert spec.control.engine.gpu_device_id is None

    def test_gpu_enabled(self) -> None:
        """GPU can be enabled with optional settings."""
        engine = EngineSpec(use_gpu=True, gpu_device_id=0)
        assert engine.use_gpu is True
        assert engine.gpu_device_id == 0

    def test_gpu_device_id_validation(self) -> None:
        """GPU device ID must be non-negative."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            EngineSpec(use_gpu=True, gpu_device_id=-1)

    def test_gpu_serialization_roundtrip(self) -> None:
        """GPU settings should survive serialization roundtrip."""
        spec = _minimal_simulation_spec()
        spec_dict = spec.model_dump()
        spec_dict["control"]["engine"]["use_gpu"] = True
        spec_dict["control"]["engine"]["gpu_device_id"] = 1

        reloaded = load_simulationspec(spec_dict)
        assert reloaded.control.engine.use_gpu is True
        assert reloaded.control.engine.gpu_device_id == 1


# =============================================================================
# Nyquist Compliance Tests
# =============================================================================


class TestNyquistCompliance:
    """Tests for Nyquist compliance validation."""

    def test_default_resolution_passes(self) -> None:
        """Default mesh settings should pass Nyquist check."""
        spec = _minimal_simulation_spec()
        result = validate_nyquist_compliance(spec)
        assert result.status == ValidationStatus.PASSED
        assert result.name == "nyquist_compliance"

    def test_nyquist_result_has_details(self) -> None:
        """Nyquist result should include diagnostic details."""
        spec = _minimal_simulation_spec()
        result = validate_nyquist_compliance(spec)
        assert result.details is not None
        assert "max_freq_hz" in result.details or "f_max_hz" in result.details


# =============================================================================
# PML Adequacy Tests
# =============================================================================


class TestPMLAdequacy:
    """Tests for PML adequacy validation."""

    def test_default_pml_passes(self) -> None:
        """Default PML_8 boundaries should pass."""
        spec = _minimal_simulation_spec()
        result = validate_pml_adequacy(spec)
        # With moderate frequency, PML_8 should be adequate
        assert result.passed
        assert result.name == "pml_adequacy"

    def test_thick_pml_passes(self) -> None:
        """PML_32 should pass easily."""
        spec = _minimal_simulation_spec()
        spec_dict = spec.model_dump()
        spec_dict["boundaries"] = {
            "x_min": "PML_32",
            "x_max": "PML_32",
            "y_min": "PML_32",
            "y_max": "PML_32",
            "z_min": "PML_32",
            "z_max": "PML_32",
        }
        spec_with_pml32 = load_simulationspec(spec_dict)
        result = validate_pml_adequacy(spec_with_pml32)
        assert result.status == ValidationStatus.PASSED

    def test_no_pml_warns(self) -> None:
        """No PML boundaries should generate warning."""
        spec = _minimal_simulation_spec()
        spec_dict = spec.model_dump()
        spec_dict["boundaries"] = {
            "x_min": "PEC",
            "x_max": "PEC",
            "y_min": "PEC",
            "y_max": "PEC",
            "z_min": "PEC",
            "z_max": "PEC",
        }
        spec_no_pml = load_simulationspec(spec_dict)
        result = validate_pml_adequacy(spec_no_pml)
        assert result.status == ValidationStatus.WARNING
        assert "No PML boundaries" in result.message or "pml" in result.message.lower()


# =============================================================================
# GPU Config Validation Tests
# =============================================================================


class TestGPUConfigValidation:
    """Tests for GPU configuration validation."""

    def test_gpu_disabled_passes(self) -> None:
        """GPU disabled should pass validation."""
        spec = _minimal_simulation_spec()
        result = validate_gpu_config(spec)
        assert result.status == ValidationStatus.PASSED
        assert "CPU" in result.message or "disabled" in result.message.lower()

    def test_gpu_enabled_passes(self) -> None:
        """GPU enabled with valid settings should pass."""
        spec = _minimal_simulation_spec()
        spec_dict = spec.model_dump()
        spec_dict["control"] = {
            "engine": {
                "use_gpu": True,
                "gpu_device_id": 0,
            }
        }
        spec_gpu = load_simulationspec(spec_dict)
        result = validate_gpu_config(spec_gpu)
        assert result.passed
        assert result.details["use_gpu"] is True


# =============================================================================
# Full Validation Tests
# =============================================================================


class TestFullValidation:
    """Tests for complete SimConfig validation."""

    def test_full_validation_default_passes(self) -> None:
        """Full validation with defaults should pass."""
        spec = _minimal_simulation_spec()
        report = validate_sim_config(spec)
        assert isinstance(report, SimConfigValidationReport)
        assert report.all_passed or report.overall_status == ValidationStatus.WARNING

    def test_full_validation_includes_all_checks(self) -> None:
        """Full validation should include all check types."""
        spec = _minimal_simulation_spec()
        report = validate_sim_config(spec)
        check_names = {c.name for c in report.checks}
        assert "nyquist_compliance" in check_names
        assert "pml_adequacy" in check_names
        assert "gpu_config" in check_names

    def test_full_validation_to_dict(self) -> None:
        """Validation report should be serializable."""
        spec = _minimal_simulation_spec()
        report = validate_sim_config(spec, spec_hash="test_hash")
        report_dict = report.to_dict()
        assert "overall_status" in report_dict
        assert "checks" in report_dict
        assert "spec_hash" in report_dict
        assert report_dict["spec_hash"] == "test_hash"

    def test_full_validation_canonical_hash(self) -> None:
        """Validation report should have canonical hash."""
        spec = _minimal_simulation_spec()
        report = validate_sim_config(spec)
        assert report.canonical_hash
        assert len(report.canonical_hash) == 64  # SHA256 hex


# =============================================================================
# sim_config.json Storage Tests
# =============================================================================


class TestSimConfigStorage:
    """Tests for sim_config.json storage."""

    def test_write_sim_config_json(self) -> None:
        """write_sim_config_json should create valid JSON file."""
        spec = _minimal_simulation_spec()

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
        spec = _minimal_simulation_spec()
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
        spec = _minimal_simulation_spec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = write_sim_config_json(spec, output_dir)

            loaded = load_sim_config_json(path)
            assert loaded["schema_version"] == spec.schema_version
            assert "spec" in loaded

    def test_sim_config_creates_parent_dirs(self) -> None:
        """write_sim_config_json should create parent directories."""
        spec = _minimal_simulation_spec()

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
        spec = _minimal_simulation_spec()
        report = validate_sim_config(spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation.json"
            write_validation_report(report, output_path)

            assert output_path.exists()
            content = json.loads(output_path.read_text())
            assert content["overall_status"] == report.overall_status.value

    def test_validation_report_n_passed(self) -> None:
        """Validation report should correctly count passed checks."""
        spec = _minimal_simulation_spec()
        report = validate_sim_config(spec)

        assert report.n_passed == sum(1 for c in report.checks if c.passed)
        assert report.n_failed == sum(1 for c in report.checks if c.status == ValidationStatus.FAILED)

    def test_get_check_by_name(self) -> None:
        """Validation report should allow getting check by name."""
        spec = _minimal_simulation_spec()
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
        spec = _minimal_simulation_spec()
        spec_dict = spec.model_dump()
        spec_dict["control"] = {
            "engine": {
                "type": "multithreaded",
                "use_gpu": True,
                "gpu_device_id": 0,
                "gpu_memory_fraction": 0.9,
            }
        }
        spec_gpu = load_simulationspec(spec_dict)

        assert spec_gpu.control.engine.use_gpu is True
        assert spec_gpu.control.engine.gpu_device_id == 0
        assert spec_gpu.control.engine.gpu_memory_fraction == 0.9

    def test_gpu_defaults(self) -> None:
        """GPU fields should have sensible defaults."""
        spec = _minimal_simulation_spec()

        assert spec.control.engine.use_gpu is False
        assert spec.control.engine.gpu_device_id is None
        assert spec.control.engine.gpu_memory_fraction is None

    def test_gpu_memory_fraction_bounds(self) -> None:
        """GPU memory fraction should be bounded 0.1-1.0."""
        spec = _minimal_simulation_spec()
        spec_dict = spec.model_dump()

        # Too low
        spec_dict["control"] = {"engine": {"gpu_memory_fraction": 0.05}}
        with pytest.raises(Exception):  # ValidationError
            load_simulationspec(spec_dict)

        # Too high
        spec_dict["control"] = {"engine": {"gpu_memory_fraction": 1.5}}
        with pytest.raises(Exception):
            load_simulationspec(spec_dict)

    def test_gpu_device_id_non_negative(self) -> None:
        """GPU device ID should be non-negative."""
        spec = _minimal_simulation_spec()
        spec_dict = spec.model_dump()
        spec_dict["control"] = {"engine": {"gpu_device_id": -1}}
        with pytest.raises(Exception):
            load_simulationspec(spec_dict)


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_passed_property(self) -> None:
        """passed property should reflect status correctly."""
        passed = ValidationResult(name="test", status=ValidationStatus.PASSED, message="OK")
        warning = ValidationResult(name="test", status=ValidationStatus.WARNING, message="Marginal")
        failed = ValidationResult(name="test", status=ValidationStatus.FAILED, message="Error")

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
