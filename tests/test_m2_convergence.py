"""Tests for openEMS convergence checking module (REQ-M2-008).

This module tests the convergence validation functionality including:
- Energy decay checking
- Port power balance validation
- Passivity verification
- Frequency resolution adequacy
- Overall convergence report generation
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from formula_foundry.em.touchstone import SParameterData
from formula_foundry.openems.convergence import (
    ConvergenceCheckResult,
    ConvergenceConfig,
    ConvergenceReport,
    ConvergenceStatus,
    EnergyDecayData,
    MeshInfo,
    build_convergence_manifest_entry,
    check_energy_decay,
    check_frequency_resolution,
    check_passivity,
    check_port_power_balance,
    convergence_gates_passed,
    load_energy_decay_json,
    validate_convergence,
    write_convergence_report,
)

# =============================================================================
# EnergyDecayData Tests
# =============================================================================


class TestEnergyDecayData:
    """Tests for EnergyDecayData dataclass."""

    def test_basic_creation(self):
        """Test creating EnergyDecayData with valid data."""
        time_ps = np.array([0.0, 10.0, 20.0, 30.0])
        energy_db = np.array([-10.0, -20.0, -30.0, -40.0])
        data = EnergyDecayData(time_ps=time_ps, energy_db=energy_db)

        assert data.n_timesteps == 4
        assert data.final_energy_db == -40.0
        assert data.total_time_ps == 30.0

    def test_decay_rate(self):
        """Test decay rate calculation."""
        time_ps = np.array([0.0, 10.0, 20.0, 30.0])
        energy_db = np.array([0.0, -10.0, -20.0, -30.0])
        data = EnergyDecayData(time_ps=time_ps, energy_db=energy_db)

        # -30 dB over 30 ps = -1 dB/ps
        assert abs(data.decay_rate_db_per_ps() - (-1.0)) < 1e-10

    def test_length_mismatch_raises(self):
        """Test that mismatched array lengths raise ValueError."""
        time_ps = np.array([0.0, 10.0, 20.0])
        energy_db = np.array([-10.0, -20.0])

        with pytest.raises(ValueError, match="same length"):
            EnergyDecayData(time_ps=time_ps, energy_db=energy_db)

    def test_empty_data(self):
        """Test handling of empty data arrays."""
        data = EnergyDecayData(
            time_ps=np.array([], dtype=np.float64),
            energy_db=np.array([], dtype=np.float64),
        )
        assert data.n_timesteps == 0
        assert data.final_energy_db == 0.0
        assert data.total_time_ps == 0.0
        assert data.decay_rate_db_per_ps() == 0.0


class TestLoadEnergyDecayJson:
    """Tests for load_energy_decay_json function."""

    def test_load_valid_json(self, tmp_path):
        """Test loading valid energy decay JSON."""
        json_path = tmp_path / "energy_decay.json"
        data = {
            "time_ps": [0, 50, 100, 150, 200],
            "energy_db": [-5, -15, -25, -35, -45],
        }
        json_path.write_text(json.dumps(data))

        result = load_energy_decay_json(json_path)
        assert result.n_timesteps == 5
        assert result.final_energy_db == -45.0

    def test_file_not_found(self, tmp_path):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_energy_decay_json(tmp_path / "nonexistent.json")

    def test_missing_keys(self, tmp_path):
        """Test error when required keys are missing."""
        json_path = tmp_path / "bad.json"
        json_path.write_text('{"time_ps": [0, 10]}')

        with pytest.raises(ValueError, match="energy_db"):
            load_energy_decay_json(json_path)


# =============================================================================
# Energy Decay Check Tests
# =============================================================================


class TestCheckEnergyDecay:
    """Tests for check_energy_decay function."""

    def test_converged_simulation(self):
        """Test detection of converged simulation."""
        time_ps = np.linspace(0, 200, 21)
        energy_db = np.linspace(0, -55, 21)
        data = EnergyDecayData(time_ps=time_ps, energy_db=energy_db)
        config = ConvergenceConfig(
            energy_decay_target_db=-50.0,
            energy_decay_margin_db=3.0,
        )

        result = check_energy_decay(data, config)
        assert result.status == ConvergenceStatus.PASSED
        assert result.passed
        assert result.value == -55.0
        assert result.name == "energy_decay"

    def test_unconverged_simulation(self):
        """Test detection of unconverged simulation."""
        time_ps = np.linspace(0, 200, 21)
        energy_db = np.linspace(0, -30, 21)  # Only reaches -30 dB
        data = EnergyDecayData(time_ps=time_ps, energy_db=energy_db)
        config = ConvergenceConfig(energy_decay_target_db=-50.0)

        result = check_energy_decay(data, config)
        assert result.status == ConvergenceStatus.FAILED
        assert not result.passed
        assert result.value == -30.0

    def test_warning_near_threshold(self):
        """Test warning when energy is close but above threshold."""
        time_ps = np.linspace(0, 200, 21)
        # Reaches -50 dB exactly (target) but not -53 dB (target - margin)
        energy_db = np.linspace(0, -50, 21)
        data = EnergyDecayData(time_ps=time_ps, energy_db=energy_db)
        config = ConvergenceConfig(
            energy_decay_target_db=-50.0,
            energy_decay_margin_db=3.0,
        )

        result = check_energy_decay(data, config)
        assert result.status == ConvergenceStatus.WARNING

    def test_details_included(self):
        """Test that details are included in result."""
        time_ps = np.array([0.0, 100.0])
        energy_db = np.array([0.0, -60.0])
        data = EnergyDecayData(time_ps=time_ps, energy_db=energy_db)
        config = ConvergenceConfig()

        result = check_energy_decay(data, config)
        assert "final_energy_db" in result.details
        assert "target_db" in result.details
        assert "decay_rate_db_per_ps" in result.details


# =============================================================================
# Port Power Balance Tests
# =============================================================================


def make_passive_sparam_data(
    n_freqs: int = 10,
    s11_mag: float = 0.1,
    s21_mag: float = 0.9,
) -> SParameterData:
    """Create passive 2-port S-parameter data for testing."""
    frequencies = np.linspace(1e9, 10e9, n_freqs)
    s_params = np.zeros((n_freqs, 2, 2), dtype=np.complex128)

    # Create reciprocal passive network
    s_params[:, 0, 0] = s11_mag * np.exp(1j * np.pi / 4)  # S11
    s_params[:, 1, 0] = s21_mag * np.exp(-1j * np.pi / 8)  # S21
    s_params[:, 0, 1] = s_params[:, 1, 0]  # S12 = S21 (reciprocal)
    s_params[:, 1, 1] = s11_mag * np.exp(1j * np.pi / 4)  # S22

    return SParameterData(
        frequencies_hz=frequencies,
        s_parameters=s_params,
        n_ports=2,
        reference_impedance_ohm=50.0,
    )


def make_non_passive_sparam_data(n_freqs: int = 10) -> SParameterData:
    """Create non-passive S-parameter data for testing (power gain)."""
    frequencies = np.linspace(1e9, 10e9, n_freqs)
    s_params = np.zeros((n_freqs, 2, 2), dtype=np.complex128)

    # Create active network (|S21|^2 + |S11|^2 > 1)
    s_params[:, 0, 0] = 0.3  # S11
    s_params[:, 1, 0] = 1.2  # S21 > 1 (active gain)
    s_params[:, 0, 1] = 0.3
    s_params[:, 1, 1] = 0.3

    return SParameterData(
        frequencies_hz=frequencies,
        s_parameters=s_params,
        n_ports=2,
        reference_impedance_ohm=50.0,
    )


class TestCheckPortPowerBalance:
    """Tests for check_port_power_balance function."""

    def test_passive_device_passes(self):
        """Test that passive device passes power balance check."""
        s_params = make_passive_sparam_data()
        config = ConvergenceConfig(power_balance_tolerance=0.1)

        result = check_port_power_balance(s_params, config)
        assert result.passed
        assert result.status in (ConvergenceStatus.PASSED, ConvergenceStatus.WARNING)

    def test_active_device_fails(self):
        """Test that active device fails power balance check."""
        s_params = make_non_passive_sparam_data()
        config = ConvergenceConfig(power_balance_tolerance=0.1)

        result = check_port_power_balance(s_params, config)
        assert result.status == ConvergenceStatus.FAILED
        assert not result.passed
        assert result.details["n_violations"] > 0

    def test_lossless_device(self):
        """Test lossless device (|S11|^2 + |S21|^2 = 1)."""
        frequencies = np.linspace(1e9, 10e9, 10)
        s_params = np.zeros((10, 2, 2), dtype=np.complex128)

        # Lossless: |S11|^2 + |S21|^2 = 1
        s11_mag = 0.3
        s21_mag = np.sqrt(1 - s11_mag**2)
        s_params[:, 0, 0] = s11_mag
        s_params[:, 1, 0] = s21_mag
        s_params[:, 0, 1] = s21_mag
        s_params[:, 1, 1] = s11_mag

        sparam_data = SParameterData(
            frequencies_hz=frequencies,
            s_parameters=s_params,
            n_ports=2,
        )
        config = ConvergenceConfig(power_balance_tolerance=0.01)

        result = check_port_power_balance(sparam_data, config)
        assert result.passed


class TestCheckPassivity:
    """Tests for check_passivity function."""

    def test_passive_sparam_passes(self):
        """Test that passive S-parameters pass eigenvalue check."""
        s_params = make_passive_sparam_data()
        config = ConvergenceConfig(passivity_tolerance=1e-6)

        result = check_passivity(s_params, config)
        assert result.passed
        assert result.details["max_eigenvalue"] <= 1.0 + 1e-6

    def test_non_passive_fails(self):
        """Test that non-passive S-parameters fail eigenvalue check."""
        s_params = make_non_passive_sparam_data()
        config = ConvergenceConfig(passivity_tolerance=1e-6)

        result = check_passivity(s_params, config)
        assert result.status == ConvergenceStatus.FAILED
        assert result.details["max_eigenvalue"] > 1.0


# =============================================================================
# Frequency Resolution Tests
# =============================================================================


class TestCheckFrequencyResolution:
    """Tests for check_frequency_resolution function."""

    def test_adequate_resolution_passes(self):
        """Test that adequate mesh resolution passes."""
        max_freq = 10e9  # 10 GHz
        # Wavelength at 10 GHz in vacuum: c/f = 3e8/10e9 = 0.03m = 30mm = 30e6 nm
        # With 20 cells/wavelength: cell size = 30e6/20 = 1.5e6 nm
        mesh = MeshInfo(
            min_cell_size_nm=1e6,
            max_cell_size_nm=1e6,  # 1mm cells
            n_cells_x=100,
            n_cells_y=50,
            n_cells_z=10,
        )
        config = ConvergenceConfig(min_cells_per_wavelength=20)

        result = check_frequency_resolution(max_freq, mesh, config, epsilon_r=1.0)
        assert result.passed
        assert result.details["actual_cells_per_wavelength"] >= 20

    def test_insufficient_resolution_fails(self):
        """Test that insufficient mesh resolution fails."""
        max_freq = 10e9  # 10 GHz
        # Very coarse mesh - 10mm cells (10e7 nm)
        mesh = MeshInfo(
            min_cell_size_nm=1e7,
            max_cell_size_nm=1e7,
            n_cells_x=10,
            n_cells_y=5,
            n_cells_z=2,
        )
        config = ConvergenceConfig(min_cells_per_wavelength=20)

        result = check_frequency_resolution(max_freq, mesh, config, epsilon_r=1.0)
        assert result.status == ConvergenceStatus.FAILED
        assert result.details["actual_cells_per_wavelength"] < 20

    def test_dielectric_effect(self):
        """Test that higher epsilon_r requires finer mesh."""
        max_freq = 10e9
        # With epsilon_r=4, wavelength is halved
        mesh = MeshInfo(
            min_cell_size_nm=1e6,
            max_cell_size_nm=1e6,
            n_cells_x=100,
            n_cells_y=50,
            n_cells_z=10,
        )
        config = ConvergenceConfig(min_cells_per_wavelength=20)

        # In vacuum - passes
        result_vacuum = check_frequency_resolution(max_freq, mesh, config, epsilon_r=1.0)

        # In dielectric (eps=4) - wavelength halved, may fail
        result_diel = check_frequency_resolution(max_freq, mesh, config, epsilon_r=4.0)

        # Dielectric should have fewer cells per wavelength
        assert result_diel.details["actual_cells_per_wavelength"] < result_vacuum.details["actual_cells_per_wavelength"]

    def test_no_mesh_info_skipped(self):
        """Test that check is skipped when mesh info not available."""
        config = ConvergenceConfig()
        result = check_frequency_resolution(10e9, None, config)
        assert result.status == ConvergenceStatus.SKIPPED


# =============================================================================
# Convergence Report Tests
# =============================================================================


class TestConvergenceReport:
    """Tests for ConvergenceReport dataclass."""

    def test_all_passed(self):
        """Test report with all checks passed."""
        checks = [
            ConvergenceCheckResult(
                name="energy_decay",
                status=ConvergenceStatus.PASSED,
                message="OK",
                value=-55.0,
            ),
            ConvergenceCheckResult(
                name="passivity",
                status=ConvergenceStatus.PASSED,
                message="OK",
                value=0.9,
            ),
        ]
        report = ConvergenceReport(
            checks=checks,
            overall_status=ConvergenceStatus.PASSED,
            simulation_hash="abc123",
            canonical_hash="def456",
            config=ConvergenceConfig(),
        )

        assert report.all_passed
        assert report.n_passed == 2
        assert report.n_failed == 0

    def test_some_failed(self):
        """Test report with some checks failed."""
        checks = [
            ConvergenceCheckResult(
                name="energy_decay",
                status=ConvergenceStatus.PASSED,
                message="OK",
            ),
            ConvergenceCheckResult(
                name="passivity",
                status=ConvergenceStatus.FAILED,
                message="Failed",
            ),
        ]
        report = ConvergenceReport(
            checks=checks,
            overall_status=ConvergenceStatus.FAILED,
            simulation_hash="abc123",
            canonical_hash="def456",
            config=ConvergenceConfig(),
        )

        assert not report.all_passed
        assert report.n_passed == 1
        assert report.n_failed == 1

    def test_get_check(self):
        """Test retrieving check by name."""
        checks = [
            ConvergenceCheckResult(name="energy_decay", status=ConvergenceStatus.PASSED, message="OK"),
            ConvergenceCheckResult(name="passivity", status=ConvergenceStatus.FAILED, message="Bad"),
        ]
        report = ConvergenceReport(
            checks=checks,
            overall_status=ConvergenceStatus.FAILED,
            simulation_hash="",
            canonical_hash="",
            config=ConvergenceConfig(),
        )

        assert report.get_check("energy_decay") is not None
        assert report.get_check("energy_decay").passed
        assert report.get_check("nonexistent") is None

    def test_to_dict(self):
        """Test converting report to dictionary."""
        checks = [
            ConvergenceCheckResult(
                name="energy_decay",
                status=ConvergenceStatus.PASSED,
                message="Converged",
                value=-55.0,
                threshold=-53.0,
            ),
        ]
        report = ConvergenceReport(
            checks=checks,
            overall_status=ConvergenceStatus.PASSED,
            simulation_hash="abc123",
            canonical_hash="def456",
            config=ConvergenceConfig(),
        )

        d = report.to_dict()
        assert d["overall_status"] == "passed"
        assert d["simulation_hash"] == "abc123"
        assert len(d["checks"]) == 1
        assert d["checks"][0]["name"] == "energy_decay"


# =============================================================================
# Validate Convergence Tests
# =============================================================================


class TestValidateConvergence:
    """Tests for validate_convergence function."""

    def test_all_checks_pass(self):
        """Test validation with all checks passing."""
        energy_data = EnergyDecayData(
            time_ps=np.linspace(0, 200, 21),
            energy_db=np.linspace(0, -60, 21),
        )
        s_params = make_passive_sparam_data()
        mesh = MeshInfo(
            min_cell_size_nm=1e6,
            max_cell_size_nm=1e6,
            n_cells_x=100,
            n_cells_y=50,
            n_cells_z=10,
        )

        report = validate_convergence(
            energy_data=energy_data,
            s_parameters=s_params,
            mesh_info=mesh,
            max_freq_hz=5e9,
            simulation_hash="test123",
        )

        assert report.overall_status == ConvergenceStatus.PASSED
        assert report.all_passed

    def test_energy_decay_fails(self):
        """Test validation when energy decay fails."""
        energy_data = EnergyDecayData(
            time_ps=np.linspace(0, 200, 21),
            energy_db=np.linspace(0, -20, 21),  # Only reaches -20 dB
        )
        s_params = make_passive_sparam_data()

        report = validate_convergence(
            energy_data=energy_data,
            s_parameters=s_params,
            simulation_hash="test456",
        )

        assert report.overall_status == ConvergenceStatus.FAILED
        assert report.get_check("energy_decay").status == ConvergenceStatus.FAILED

    def test_no_data_skips_all(self):
        """Test that all checks are skipped when no data provided."""
        report = validate_convergence()

        assert report.overall_status == ConvergenceStatus.SKIPPED
        for check in report.checks:
            assert check.status == ConvergenceStatus.SKIPPED

    def test_partial_data(self):
        """Test validation with only some data available."""
        energy_data = EnergyDecayData(
            time_ps=np.linspace(0, 200, 21),
            energy_db=np.linspace(0, -60, 21),
        )

        config = ConvergenceConfig(
            check_energy_decay=True,
            check_port_power=True,
            check_passivity=False,
            check_frequency_resolution=False,
        )

        report = validate_convergence(
            energy_data=energy_data,
            config=config,
        )

        # Energy decay should pass
        assert report.get_check("energy_decay").status == ConvergenceStatus.PASSED
        # Port power should be skipped (no S-param data)
        assert report.get_check("port_power_balance").status == ConvergenceStatus.SKIPPED

    def test_canonical_hash_computed(self):
        """Test that canonical hash is computed."""
        energy_data = EnergyDecayData(
            time_ps=np.array([0.0, 100.0]),
            energy_db=np.array([0.0, -60.0]),
        )

        report = validate_convergence(
            energy_data=energy_data,
            simulation_hash="test",
        )

        assert report.canonical_hash != ""
        assert len(report.canonical_hash) == 64  # SHA256 hex


# =============================================================================
# Report Writing Tests
# =============================================================================


class TestWriteConvergenceReport:
    """Tests for write_convergence_report function."""

    def test_write_and_read_back(self, tmp_path):
        """Test writing report and reading it back."""
        checks = [
            ConvergenceCheckResult(
                name="energy_decay",
                status=ConvergenceStatus.PASSED,
                message="OK",
                value=-55.0,
            ),
        ]
        report = ConvergenceReport(
            checks=checks,
            overall_status=ConvergenceStatus.PASSED,
            simulation_hash="abc123",
            canonical_hash="def456",
            config=ConvergenceConfig(),
        )

        output_path = tmp_path / "convergence_report.json"
        write_convergence_report(report, output_path)

        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert loaded["overall_status"] == "passed"
        assert loaded["simulation_hash"] == "abc123"


class TestBuildConvergenceManifestEntry:
    """Tests for build_convergence_manifest_entry function."""

    def test_manifest_entry_structure(self):
        """Test that manifest entry has correct structure."""
        checks = [
            ConvergenceCheckResult(
                name="energy_decay",
                status=ConvergenceStatus.PASSED,
                message="OK",
                value=-55.0,
                threshold=-53.0,
            ),
            ConvergenceCheckResult(
                name="passivity",
                status=ConvergenceStatus.PASSED,
                message="OK",
                value=0.9,
                threshold=1.0,
            ),
        ]
        report = ConvergenceReport(
            checks=checks,
            overall_status=ConvergenceStatus.PASSED,
            simulation_hash="abc123",
            canonical_hash="def456",
            config=ConvergenceConfig(),
        )

        entry = build_convergence_manifest_entry(report)

        assert "convergence" in entry
        assert entry["convergence"]["overall_status"] == "passed"
        assert entry["convergence"]["n_passed"] == 2
        assert "energy_decay" in entry["convergence"]["checks"]
        assert "passivity" in entry["convergence"]["checks"]


class TestConvergenceGatesPassed:
    """Tests for convergence_gates_passed function."""

    def test_returns_passed_gates(self):
        """Test that function returns list of passed gate names."""
        checks = [
            ConvergenceCheckResult(name="energy_decay", status=ConvergenceStatus.PASSED, message="OK"),
            ConvergenceCheckResult(name="passivity", status=ConvergenceStatus.FAILED, message="Bad"),
            ConvergenceCheckResult(name="power_balance", status=ConvergenceStatus.WARNING, message="OK"),
        ]
        report = ConvergenceReport(
            checks=checks,
            overall_status=ConvergenceStatus.FAILED,
            simulation_hash="",
            canonical_hash="",
            config=ConvergenceConfig(),
        )

        gates = convergence_gates_passed(report)
        assert "energy_decay" in gates
        assert "power_balance" in gates  # Warning counts as passed
        assert "passivity" not in gates


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestConvergenceIntegration:
    """Integration-like tests for the full convergence workflow."""

    def test_full_workflow_with_files(self, tmp_path):
        """Test full workflow with file I/O."""
        # Create energy decay file
        energy_path = tmp_path / "energy_decay.json"
        energy_data = {
            "time_ps": list(range(0, 210, 10)),
            "energy_db": [-3.0 * i for i in range(21)],  # Goes to -60 dB
        }
        energy_path.write_text(json.dumps(energy_data))

        # Load and validate
        loaded_energy = load_energy_decay_json(energy_path)
        s_params = make_passive_sparam_data()

        report = validate_convergence(
            energy_data=loaded_energy,
            s_parameters=s_params,
            max_freq_hz=10e9,
            simulation_hash="integration_test",
        )

        # Write report
        report_path = tmp_path / "convergence_report.json"
        write_convergence_report(report, report_path)

        # Verify
        assert report.overall_status == ConvergenceStatus.PASSED
        assert report_path.exists()

        # Check manifest entry
        entry = build_convergence_manifest_entry(report)
        assert entry["convergence"]["overall_status"] == "passed"

    def test_config_from_spec_default_values(self):
        """Test that ConvergenceConfig.from_spec uses spec values."""
        from formula_foundry.openems.spec import (
            ExcitationSpec,
            FrequencySpec,
            GeometryRefSpec,
            MeshResolutionSpec,
            MeshSpec,
            OpenEMSToolchainSpec,
            PortSpec,
            SimulationControlSpec,
            SimulationSpec,
            TerminationSpec,
            ToolchainSpec,
        )

        spec = SimulationSpec(
            schema_version=1,
            toolchain=ToolchainSpec(
                openems=OpenEMSToolchainSpec(
                    version="0.0.35",
                    docker_image="ghcr.io/openems:0.0.35",
                )
            ),
            geometry_ref=GeometryRefSpec(design_hash="test"),
            excitation=ExcitationSpec(f0_hz=5e9, fc_hz=10e9),
            frequency=FrequencySpec(f_start_hz=1e9, f_stop_hz=10e9),
            ports=[
                PortSpec(
                    id="P1",
                    type="lumped",
                    excite=True,
                    position_nm=(0, 0, 0),
                    direction="x",
                )
            ],
            control=SimulationControlSpec(
                termination=TerminationSpec(end_criteria_db=-40.0),
            ),
            mesh=MeshSpec(
                resolution=MeshResolutionSpec(lambda_resolution=25),
            ),
        )

        config = ConvergenceConfig.from_spec(spec)
        assert config.energy_decay_target_db == -40.0
        assert config.min_cells_per_wavelength == 25
