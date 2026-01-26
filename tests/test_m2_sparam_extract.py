"""Tests for S-parameter extraction and post-processing (REQ-M2-006, REQ-M2-007).

This module tests the sparam_extract module which provides:
- Windowing functions for time-domain signal processing (REQ-M2-006)
- FFT-based S-parameter extraction from port signals (REQ-M2-006)
- Touchstone and CSV file parsing
- S-parameter computation from port signals
- Frequency sweep interpolation
- De-embedding for reference plane shifting
- Structured output for manifest inclusion
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from formula_foundry.em.touchstone import (
    FrequencyUnit,
    SParameterData,
    SParameterFormat,
    TouchstoneOptions,
    write_touchstone,
)
from formula_foundry.openems import (
    ExtractionConfig,
    ExtractionResult,
    FrequencySpec,
    MultiPortSignalSet,
    PortSignalData,
    PortSpec,
    WindowConfig,
    WindowType,
    apply_deembedding,
    apply_window,
    build_manifest_entry,
    compute_window_metrics,
    create_window,
    extract_full_sparam_matrix,
    extract_sparams,
    extract_sparams_from_csv,
    extract_sparams_from_port_signals,
    extract_sparams_from_touchstone,
    load_port_signals_json,
    write_extraction_result,
)


def _make_test_frequency_spec() -> FrequencySpec:
    """Create a test frequency specification."""
    return FrequencySpec(
        f_start_hz=1_000_000_000,  # 1 GHz
        f_stop_hz=10_000_000_000,  # 10 GHz
        n_points=11,
    )


def _make_test_port_specs() -> list[PortSpec]:
    """Create test port specifications."""
    return [
        PortSpec(
            id="P1",
            type="lumped",
            impedance_ohm=50.0,
            excite=True,
            position_nm=(0, 0, 0),
            direction="x",
        ),
        PortSpec(
            id="P2",
            type="lumped",
            impedance_ohm=50.0,
            excite=False,
            position_nm=(10_000_000, 0, 0),
            direction="-x",
        ),
    ]


def _make_test_extraction_config() -> ExtractionConfig:
    """Create a test extraction configuration."""
    return ExtractionConfig(
        frequency_spec=_make_test_frequency_spec(),
        port_specs=_make_test_port_specs(),
        reference_impedance_ohm=50.0,
        deembed_enabled=False,
        output_format="both",
    )


def _make_test_sparam_data(n_freq: int = 11) -> SParameterData:
    """Create test S-parameter data."""
    frequencies_hz = np.linspace(1e9, 10e9, n_freq)
    s_parameters = np.zeros((n_freq, 2, 2), dtype=np.complex128)

    # S11: return loss - start low and increase with frequency
    for i in range(n_freq):
        s_parameters[i, 0, 0] = -0.1 - 0.01 * i + 0.05j * i / n_freq

    # S21: insertion loss - high transmission, decreasing with frequency
    for i in range(n_freq):
        s_parameters[i, 1, 0] = 0.95 - 0.02 * i - 0.01j * i

    # S12: reverse transmission (reciprocal network)
    s_parameters[:, 0, 1] = s_parameters[:, 1, 0]

    # S22: output return loss
    for i in range(n_freq):
        s_parameters[i, 1, 1] = -0.09 - 0.01 * i + 0.04j * i / n_freq

    return SParameterData(
        frequencies_hz=frequencies_hz,
        s_parameters=s_parameters,
        n_ports=2,
        reference_impedance_ohm=50.0,
    )


def _write_test_touchstone(path: Path, sparam_data: SParameterData) -> None:
    """Write test Touchstone file."""
    write_touchstone(
        sparam_data,
        path,
        TouchstoneOptions(
            frequency_unit=FrequencyUnit.HZ,
            parameter_format=SParameterFormat.RI,
            reference_impedance_ohm=50.0,
        ),
    )


def _write_test_csv(path: Path, sparam_data: SParameterData) -> None:
    """Write test CSV file."""
    lines = ["freq_hz,s11_re,s11_im,s21_re,s21_im,s12_re,s12_im,s22_re,s22_im"]
    for i, freq in enumerate(sparam_data.frequencies_hz):
        s11 = sparam_data.s_parameters[i, 0, 0]
        s21 = sparam_data.s_parameters[i, 1, 0]
        s12 = sparam_data.s_parameters[i, 0, 1]
        s22 = sparam_data.s_parameters[i, 1, 1]
        lines.append(
            f"{freq:.9g},{s11.real:.9g},{s11.imag:.9g},"
            f"{s21.real:.9g},{s21.imag:.9g},"
            f"{s12.real:.9g},{s12.imag:.9g},"
            f"{s22.real:.9g},{s22.imag:.9g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_test_port_signals(path: Path) -> None:
    """Write test port signals JSON file."""
    n_samples = 1000
    dt = 1e-12  # 1 ps timestep
    time_ps = [i * dt * 1e12 for i in range(n_samples)]

    # Create simple test signals
    v1 = [float(np.sin(2 * np.pi * 1e9 * i * dt)) for i in range(n_samples)]
    i1 = [v / 50.0 for v in v1]
    v2 = [v * 0.9 for v in v1]  # 10% loss
    i2 = [v / 50.0 for v in v2]

    payload = {
        "time_ps": time_ps,
        "ports": {
            "P1": {"voltage_v": v1, "current_a": i1},
            "P2": {"voltage_v": v2, "current_a": i2},
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# =============================================================================
# ExtractionConfig Tests
# =============================================================================


def test_extraction_config_n_ports() -> None:
    """REQ-M2-007: ExtractionConfig.n_ports returns port count."""
    config = _make_test_extraction_config()
    assert config.n_ports == 2


def test_extraction_config_n_frequencies() -> None:
    """REQ-M2-007: ExtractionConfig.n_frequencies returns frequency point count."""
    config = _make_test_extraction_config()
    assert config.n_frequencies == 11


def test_extraction_config_frequencies_hz() -> None:
    """REQ-M2-007: ExtractionConfig.frequencies_hz generates frequency array."""
    config = _make_test_extraction_config()
    freqs = config.frequencies_hz()

    assert len(freqs) == 11
    assert freqs[0] == pytest.approx(1e9)
    assert freqs[-1] == pytest.approx(10e9)


# =============================================================================
# Touchstone Extraction Tests
# =============================================================================


def test_extract_sparams_from_touchstone(tmp_path: Path) -> None:
    """REQ-M2-007: Extract S-parameters from Touchstone file."""
    config = _make_test_extraction_config()
    sparam_data = _make_test_sparam_data()

    ts_path = tmp_path / "test.s2p"
    _write_test_touchstone(ts_path, sparam_data)

    result = extract_sparams_from_touchstone(ts_path, config)

    assert isinstance(result, ExtractionResult)
    assert result.s_parameters.n_ports == 2
    assert result.s_parameters.n_frequencies == 11
    assert len(result.canonical_hash) == 64  # SHA256


def test_extract_sparams_from_touchstone_interpolation(tmp_path: Path) -> None:
    """REQ-M2-007: Interpolate Touchstone to requested frequency grid."""
    # Create data with different frequency grid
    original_data = _make_test_sparam_data(n_freq=21)

    ts_path = tmp_path / "test.s2p"
    _write_test_touchstone(ts_path, original_data)

    # Request different frequency grid
    config = _make_test_extraction_config()  # 11 points
    result = extract_sparams_from_touchstone(ts_path, config)

    assert result.s_parameters.n_frequencies == 11


def test_extract_sparams_from_touchstone_file_not_found(tmp_path: Path) -> None:
    """REQ-M2-007: Raise FileNotFoundError for missing Touchstone file."""
    config = _make_test_extraction_config()

    with pytest.raises(FileNotFoundError):
        extract_sparams_from_touchstone(tmp_path / "nonexistent.s2p", config)


# =============================================================================
# CSV Extraction Tests
# =============================================================================


def test_extract_sparams_from_csv(tmp_path: Path) -> None:
    """REQ-M2-007: Extract S-parameters from CSV file."""
    config = _make_test_extraction_config()
    sparam_data = _make_test_sparam_data()

    csv_path = tmp_path / "sparams.csv"
    _write_test_csv(csv_path, sparam_data)

    result = extract_sparams_from_csv(csv_path, config)

    assert isinstance(result, ExtractionResult)
    assert result.s_parameters.n_ports == 2
    assert result.s_parameters.n_frequencies == 11


def test_extract_sparams_from_csv_file_not_found(tmp_path: Path) -> None:
    """REQ-M2-007: Raise FileNotFoundError for missing CSV file."""
    config = _make_test_extraction_config()

    with pytest.raises(FileNotFoundError):
        extract_sparams_from_csv(tmp_path / "nonexistent.csv", config)


# =============================================================================
# Port Signal Extraction Tests
# =============================================================================


def test_load_port_signals_json(tmp_path: Path) -> None:
    """REQ-M2-007: Load port signals from JSON file."""
    json_path = tmp_path / "port_signals.json"
    _write_test_port_signals(json_path)

    signals = load_port_signals_json(json_path)

    assert len(signals) == 2
    assert all(isinstance(s, PortSignalData) for s in signals)


def test_load_port_signals_json_file_not_found(tmp_path: Path) -> None:
    """REQ-M2-007: Raise FileNotFoundError for missing port signals file."""
    with pytest.raises(FileNotFoundError):
        load_port_signals_json(tmp_path / "nonexistent.json")


def test_extract_sparams_from_port_signals() -> None:
    """REQ-M2-007: Extract S-parameters from port signals."""
    config = _make_test_extraction_config()

    # Create simple test signals
    n_samples = 1000
    dt = 1e-12
    time_s = np.arange(n_samples) * dt

    # Excitation at port 1
    v1 = np.sin(2 * np.pi * 5e9 * time_s)
    i1 = v1 / 50.0

    # Output at port 2 with some loss
    v2 = 0.9 * v1
    i2 = v2 / 50.0

    signals = [
        PortSignalData(port_id="P1", time_s=time_s, voltage_v=v1, current_a=i1),
        PortSignalData(port_id="P2", time_s=time_s, voltage_v=v2, current_a=i2),
    ]

    result = extract_sparams_from_port_signals(signals, "P1", config)

    assert isinstance(result, ExtractionResult)
    assert result.s_parameters.n_ports == 2


def test_port_signal_data_validation() -> None:
    """REQ-M2-007: PortSignalData validates array lengths."""
    time_s = np.array([0.0, 1e-12, 2e-12])
    voltage_v = np.array([0.0, 1.0, 0.0])
    current_a = np.array([0.0, 0.02])  # Wrong length

    with pytest.raises(ValueError, match="same length"):
        PortSignalData(port_id="P1", time_s=time_s, voltage_v=voltage_v, current_a=current_a)


# =============================================================================
# De-embedding Tests
# =============================================================================


def test_apply_deembedding() -> None:
    """REQ-M2-007: Apply de-embedding to shift reference planes."""
    sparam_data = _make_test_sparam_data()

    deembedded = apply_deembedding(
        sparam_data,
        deembed_distance_nm=1_000_000,  # 1 mm
        epsilon_r_eff=4.0,
    )

    assert isinstance(deembedded, SParameterData)
    assert deembedded.n_ports == sparam_data.n_ports
    assert deembedded.n_frequencies == sparam_data.n_frequencies

    # S-parameters should be different after de-embedding
    assert not np.allclose(deembedded.s_parameters, sparam_data.s_parameters)


def test_apply_deembedding_specific_ports() -> None:
    """REQ-M2-007: De-embed only specific ports."""
    sparam_data = _make_test_sparam_data()

    deembedded = apply_deembedding(
        sparam_data,
        deembed_distance_nm=1_000_000,
        epsilon_r_eff=4.0,
        port_indices=[1],  # Only port 1
    )

    assert isinstance(deembedded, SParameterData)


# =============================================================================
# Output Writing Tests
# =============================================================================


def test_write_extraction_result_touchstone(tmp_path: Path) -> None:
    """REQ-M2-007: Write extraction result as Touchstone."""
    config = ExtractionConfig(
        frequency_spec=_make_test_frequency_spec(),
        port_specs=_make_test_port_specs(),
        output_format="touchstone",
    )
    sparam_data = _make_test_sparam_data()

    result = ExtractionResult(
        s_parameters=sparam_data,
        extraction_config=config,
        source_files=["test.s2p"],
        canonical_hash="a" * 64,
    )

    output_paths = write_extraction_result(result, tmp_path)

    assert "touchstone" in output_paths
    assert output_paths["touchstone"].exists()
    assert output_paths["touchstone"].suffix == ".s2p"


def test_write_extraction_result_csv(tmp_path: Path) -> None:
    """REQ-M2-007: Write extraction result as CSV."""
    config = ExtractionConfig(
        frequency_spec=_make_test_frequency_spec(),
        port_specs=_make_test_port_specs(),
        output_format="csv",
    )
    sparam_data = _make_test_sparam_data()

    result = ExtractionResult(
        s_parameters=sparam_data,
        extraction_config=config,
        source_files=["test.s2p"],
        canonical_hash="a" * 64,
    )

    output_paths = write_extraction_result(result, tmp_path)

    assert "csv" in output_paths
    assert output_paths["csv"].exists()


def test_write_extraction_result_both(tmp_path: Path) -> None:
    """REQ-M2-007: Write extraction result in both formats."""
    config = ExtractionConfig(
        frequency_spec=_make_test_frequency_spec(),
        port_specs=_make_test_port_specs(),
        output_format="both",
    )
    sparam_data = _make_test_sparam_data()

    result = ExtractionResult(
        s_parameters=sparam_data,
        extraction_config=config,
        source_files=["test.s2p"],
        canonical_hash="a" * 64,
    )

    output_paths = write_extraction_result(result, tmp_path)

    assert "touchstone" in output_paths
    assert "csv" in output_paths
    assert "metrics" in output_paths


def test_write_extraction_result_metrics(tmp_path: Path) -> None:
    """REQ-M2-007: Metrics JSON includes hash and statistics."""
    config = _make_test_extraction_config()
    sparam_data = _make_test_sparam_data()

    result = ExtractionResult(
        s_parameters=sparam_data,
        extraction_config=config,
        source_files=["test.s2p"],
        canonical_hash="a" * 64,
        metrics={"s11_min_db": -20.0},
    )

    output_paths = write_extraction_result(result, tmp_path)

    metrics_path = output_paths["metrics"]
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["canonical_hash"] == "a" * 64


# =============================================================================
# Manifest Entry Tests
# =============================================================================


def test_build_manifest_entry() -> None:
    """REQ-M2-007: Build manifest entry from extraction result."""
    config = _make_test_extraction_config()
    sparam_data = _make_test_sparam_data()

    result = ExtractionResult(
        s_parameters=sparam_data,
        extraction_config=config,
        source_files=["test.s2p"],
        canonical_hash="a" * 64,
        metrics={"s11_min_db": -20.0, "s21_max_db": -1.0},
    )

    entry = build_manifest_entry(result)

    assert "s_parameters" in entry
    assert entry["s_parameters"]["canonical_hash"] == "a" * 64
    assert entry["s_parameters"]["n_ports"] == 2
    assert entry["s_parameters"]["n_frequencies"] == 11
    assert "extraction" in entry
    assert "metrics" in entry


# =============================================================================
# High-Level API Tests
# =============================================================================


def test_extract_sparams_touchstone_preference(tmp_path: Path) -> None:
    """REQ-M2-007: extract_sparams prefers Touchstone when available."""
    config = _make_test_extraction_config()
    sparam_data = _make_test_sparam_data()

    # Create both Touchstone and CSV
    ts_path = tmp_path / "sparams.s2p"
    csv_path = tmp_path / "sparams.csv"
    _write_test_touchstone(ts_path, sparam_data)
    _write_test_csv(csv_path, sparam_data)

    result = extract_sparams(tmp_path, config, prefer_touchstone=True)

    assert str(ts_path) in result.source_files


def test_extract_sparams_csv_fallback(tmp_path: Path) -> None:
    """REQ-M2-007: extract_sparams falls back to CSV."""
    config = _make_test_extraction_config()
    sparam_data = _make_test_sparam_data()

    # Create only CSV
    csv_path = tmp_path / "sparams.csv"
    _write_test_csv(csv_path, sparam_data)

    result = extract_sparams(tmp_path, config)

    assert str(csv_path) in result.source_files


def test_extract_sparams_port_signals_fallback(tmp_path: Path) -> None:
    """REQ-M2-007: extract_sparams falls back to port signals."""
    config = _make_test_extraction_config()

    # Create only port signals
    signals_path = tmp_path / "port_signals.json"
    _write_test_port_signals(signals_path)

    result = extract_sparams(tmp_path, config)

    assert isinstance(result, ExtractionResult)


def test_extract_sparams_no_output_found(tmp_path: Path) -> None:
    """REQ-M2-007: extract_sparams raises FileNotFoundError when no output."""
    config = _make_test_extraction_config()

    with pytest.raises(FileNotFoundError, match="No S-parameter output found"):
        extract_sparams(tmp_path, config)


# =============================================================================
# Metrics Computation Tests
# =============================================================================


def test_extraction_metrics_basic(tmp_path: Path) -> None:
    """REQ-M2-007: Extraction computes basic metrics."""
    config = _make_test_extraction_config()
    sparam_data = _make_test_sparam_data()

    ts_path = tmp_path / "test.s2p"
    _write_test_touchstone(ts_path, sparam_data)

    result = extract_sparams_from_touchstone(ts_path, config)

    assert "n_ports" in result.metrics
    assert "n_frequencies" in result.metrics
    assert "s11_min_db" in result.metrics
    assert "s21_max_db" in result.metrics


def test_extraction_metrics_passivity(tmp_path: Path) -> None:
    """REQ-M2-007: Extraction checks passivity."""
    config = _make_test_extraction_config()
    sparam_data = _make_test_sparam_data()

    ts_path = tmp_path / "test.s2p"
    _write_test_touchstone(ts_path, sparam_data)

    result = extract_sparams_from_touchstone(ts_path, config)

    assert "is_passive" in result.metrics
    assert "passivity_violations" in result.metrics


def test_extraction_metrics_symmetry(tmp_path: Path) -> None:
    """REQ-M2-007: Extraction checks reciprocity/symmetry."""
    config = _make_test_extraction_config()
    sparam_data = _make_test_sparam_data()

    ts_path = tmp_path / "test.s2p"
    _write_test_touchstone(ts_path, sparam_data)

    result = extract_sparams_from_touchstone(ts_path, config)

    assert "symmetry_max_error" in result.metrics


# =============================================================================
# Canonical Hash Tests
# =============================================================================


def test_canonical_hash_deterministic(tmp_path: Path) -> None:
    """REQ-M2-007: Canonical hash is deterministic."""
    config = _make_test_extraction_config()
    sparam_data = _make_test_sparam_data()

    ts_path = tmp_path / "test.s2p"
    _write_test_touchstone(ts_path, sparam_data)

    result1 = extract_sparams_from_touchstone(ts_path, config)
    result2 = extract_sparams_from_touchstone(ts_path, config)

    assert result1.canonical_hash == result2.canonical_hash


def test_canonical_hash_different_data(tmp_path: Path) -> None:
    """REQ-M2-007: Different data produces different hash."""
    config = _make_test_extraction_config()

    # Create two different datasets
    data1 = _make_test_sparam_data()
    data2 = _make_test_sparam_data()
    data2.s_parameters[0, 0, 0] = 0.5 + 0.5j  # Modify first value

    ts_path1 = tmp_path / "test1.s2p"
    ts_path2 = tmp_path / "test2.s2p"
    _write_test_touchstone(ts_path1, data1)
    _write_test_touchstone(ts_path2, data2)

    result1 = extract_sparams_from_touchstone(ts_path1, config)
    result2 = extract_sparams_from_touchstone(ts_path2, config)

    assert result1.canonical_hash != result2.canonical_hash


# =============================================================================
# Windowing Function Tests (REQ-M2-006)
# =============================================================================


class TestWindowFunctions:
    """Tests for window function generation and application."""

    def test_create_window_none(self) -> None:
        """REQ-M2-006: Rectangular (none) window is all ones."""
        config = WindowConfig(window_type=WindowType.NONE, normalize=False)
        window = create_window(100, config)

        assert len(window) == 100
        assert np.allclose(window, 1.0)

    def test_create_window_hann(self) -> None:
        """REQ-M2-006: Hann window has correct shape."""
        config = WindowConfig(window_type=WindowType.HANN, normalize=False)
        window = create_window(100, config)

        assert len(window) == 100
        # Hann window starts and ends at 0
        assert window[0] < 0.01
        assert window[-1] < 0.01
        # Maximum at center
        assert window[50] > 0.99

    def test_create_window_hamming(self) -> None:
        """REQ-M2-006: Hamming window has non-zero edges."""
        config = WindowConfig(window_type=WindowType.HAMMING, normalize=False)
        window = create_window(100, config)

        assert len(window) == 100
        # Hamming window doesn't go to zero at edges
        assert 0.05 < window[0] < 0.15
        assert 0.05 < window[-1] < 0.15

    def test_create_window_blackman(self) -> None:
        """REQ-M2-006: Blackman window has very small edges."""
        config = WindowConfig(window_type=WindowType.BLACKMAN, normalize=False)
        window = create_window(100, config)

        assert len(window) == 100
        # Blackman window edges very close to zero
        assert window[0] < 0.01
        assert window[-1] < 0.01

    def test_create_window_kaiser(self) -> None:
        """REQ-M2-006: Kaiser window beta controls shape."""
        config_low = WindowConfig(window_type=WindowType.KAISER, kaiser_beta=5.0, normalize=False)
        config_high = WindowConfig(window_type=WindowType.KAISER, kaiser_beta=20.0, normalize=False)

        window_low = create_window(100, config_low)
        window_high = create_window(100, config_high)

        assert len(window_low) == 100
        assert len(window_high) == 100
        # Higher beta should have narrower main lobe (edges closer to 0)
        assert window_high[0] < window_low[0]

    def test_create_window_tukey(self) -> None:
        """REQ-M2-006: Tukey window alpha controls taper."""
        config_rect = WindowConfig(window_type=WindowType.TUKEY, tukey_alpha=0.0, normalize=False)
        config_mid = WindowConfig(window_type=WindowType.TUKEY, tukey_alpha=0.5, normalize=False)
        config_hann = WindowConfig(window_type=WindowType.TUKEY, tukey_alpha=1.0, normalize=False)

        window_rect = create_window(100, config_rect)
        window_mid = create_window(100, config_mid)
        window_hann = create_window(100, config_hann)

        # Alpha=0 should be all ones
        assert np.allclose(window_rect, 1.0)
        # Alpha=1 should match Hann
        expected_hann = np.hanning(100)
        assert np.allclose(window_hann, expected_hann)
        # Mid should be flat in center, tapered at edges
        assert window_mid[50] > 0.99
        assert window_mid[0] < 0.5

    def test_create_window_normalization(self) -> None:
        """REQ-M2-006: Normalized window has mean closer to 1."""
        config_norm = WindowConfig(window_type=WindowType.HANN, normalize=True)
        config_unnorm = WindowConfig(window_type=WindowType.HANN, normalize=False)

        window_norm = create_window(100, config_norm)
        window_unnorm = create_window(100, config_unnorm)

        # Normalized window should have mean closer to 1
        assert np.abs(np.mean(window_norm) - 1.0) < np.abs(np.mean(window_unnorm) - 1.0)

    def test_apply_window(self) -> None:
        """REQ-M2-006: apply_window multiplies signal by window."""
        signal = np.ones(100)
        config = WindowConfig(window_type=WindowType.HANN, normalize=False)

        windowed = apply_window(signal, config)

        # Should equal the window itself when applied to ones
        expected = create_window(100, config)
        assert np.allclose(windowed, expected)


class TestWindowMetrics:
    """Tests for window function metrics computation."""

    def test_compute_metrics_rectangular(self) -> None:
        """REQ-M2-006: Rectangular window has coherent gain of 1."""
        config = WindowConfig(window_type=WindowType.NONE)
        metrics = compute_window_metrics(config, n_samples=1024)

        assert "coherent_gain" in metrics
        assert "noise_bandwidth_bins" in metrics
        assert "processing_gain_db" in metrics
        assert "scalloping_loss_db" in metrics

        assert np.isclose(metrics["coherent_gain"], 1.0, atol=0.01)
        assert np.isclose(metrics["noise_bandwidth_bins"], 1.0, atol=0.1)

    def test_compute_metrics_hann(self) -> None:
        """REQ-M2-006: Hann window has coherent gain of 0.5."""
        config = WindowConfig(window_type=WindowType.HANN)
        metrics = compute_window_metrics(config, n_samples=1024)

        assert np.isclose(metrics["coherent_gain"], 0.5, atol=0.01)
        assert np.isclose(metrics["noise_bandwidth_bins"], 1.5, atol=0.1)


# =============================================================================
# Windowed S-Parameter Extraction Tests (REQ-M2-006)
# =============================================================================


class TestWindowedSParameterExtraction:
    """Tests for S-parameter extraction with windowing."""

    def test_extract_with_default_window(self) -> None:
        """REQ-M2-006: Extraction uses default window from config."""
        config = ExtractionConfig(
            frequency_spec=_make_test_frequency_spec(),
            port_specs=_make_test_port_specs(),
            window_config=WindowConfig(window_type=WindowType.HANN),
        )

        n_samples = 1000
        dt = 1e-12
        time_s = np.arange(n_samples) * dt
        v1 = np.sin(2 * np.pi * 5e9 * time_s)
        i1 = v1 / 50.0

        signals = [
            PortSignalData(port_id="P1", time_s=time_s, voltage_v=v1, current_a=i1),
            PortSignalData(port_id="P2", time_s=time_s, voltage_v=v1 * 0.9, current_a=i1 * 0.9),
        ]

        result = extract_sparams_from_port_signals(signals, "P1", config)

        assert "windowing" in result.metrics
        assert result.metrics["windowing"]["window_type"] == "hann"

    def test_extract_with_explicit_window(self) -> None:
        """REQ-M2-006: Explicit window config overrides default."""
        config = ExtractionConfig(
            frequency_spec=_make_test_frequency_spec(),
            port_specs=_make_test_port_specs(),
            window_config=WindowConfig(window_type=WindowType.HANN),  # Default
        )

        n_samples = 1000
        dt = 1e-12
        time_s = np.arange(n_samples) * dt
        v1 = np.sin(2 * np.pi * 5e9 * time_s)
        i1 = v1 / 50.0

        signals = [
            PortSignalData(port_id="P1", time_s=time_s, voltage_v=v1, current_a=i1),
            PortSignalData(port_id="P2", time_s=time_s, voltage_v=v1 * 0.9, current_a=i1 * 0.9),
        ]

        # Override with Blackman
        window_config = WindowConfig(window_type=WindowType.BLACKMAN)
        result = extract_sparams_from_port_signals(signals, "P1", config, window_config=window_config)

        assert result.metrics["windowing"]["window_type"] == "blackman"

    def test_extract_different_windows_produce_different_results(self) -> None:
        """REQ-M2-006: Different windows produce different S-parameters."""
        config = _make_test_extraction_config()

        n_samples = 1000
        dt = 1e-12
        time_s = np.arange(n_samples) * dt
        v1 = np.sin(2 * np.pi * 5e9 * time_s)
        i1 = v1 / 50.0

        signals = [
            PortSignalData(port_id="P1", time_s=time_s, voltage_v=v1, current_a=i1),
            PortSignalData(port_id="P2", time_s=time_s, voltage_v=v1 * 0.9, current_a=i1 * 0.9),
        ]

        result_none = extract_sparams_from_port_signals(
            signals, "P1", config, window_config=WindowConfig(window_type=WindowType.NONE)
        )
        result_hann = extract_sparams_from_port_signals(
            signals, "P1", config, window_config=WindowConfig(window_type=WindowType.HANN)
        )

        # Results should be different
        assert not np.allclose(result_none.s_parameters.s_parameters, result_hann.s_parameters.s_parameters)

    def test_extract_kaiser_includes_beta_in_metrics(self) -> None:
        """REQ-M2-006: Kaiser window includes beta in metrics."""
        config = _make_test_extraction_config()

        n_samples = 1000
        dt = 1e-12
        time_s = np.arange(n_samples) * dt
        v1 = np.sin(2 * np.pi * 5e9 * time_s)
        i1 = v1 / 50.0

        signals = [
            PortSignalData(port_id="P1", time_s=time_s, voltage_v=v1, current_a=i1),
            PortSignalData(port_id="P2", time_s=time_s, voltage_v=v1 * 0.9, current_a=i1 * 0.9),
        ]

        window_config = WindowConfig(window_type=WindowType.KAISER, kaiser_beta=10.5)
        result = extract_sparams_from_port_signals(signals, "P1", config, window_config=window_config)

        assert result.metrics["windowing"]["window_type"] == "kaiser"
        assert result.metrics["windowing"]["kaiser_beta"] == 10.5

    def test_extract_tukey_includes_alpha_in_metrics(self) -> None:
        """REQ-M2-006: Tukey window includes alpha in metrics."""
        config = _make_test_extraction_config()

        n_samples = 1000
        dt = 1e-12
        time_s = np.arange(n_samples) * dt
        v1 = np.sin(2 * np.pi * 5e9 * time_s)
        i1 = v1 / 50.0

        signals = [
            PortSignalData(port_id="P1", time_s=time_s, voltage_v=v1, current_a=i1),
            PortSignalData(port_id="P2", time_s=time_s, voltage_v=v1 * 0.9, current_a=i1 * 0.9),
        ]

        window_config = WindowConfig(window_type=WindowType.TUKEY, tukey_alpha=0.3)
        result = extract_sparams_from_port_signals(signals, "P1", config, window_config=window_config)

        assert result.metrics["windowing"]["window_type"] == "tukey"
        assert result.metrics["windowing"]["tukey_alpha"] == 0.3


# =============================================================================
# Multi-Excitation Extraction Tests (REQ-M2-006)
# =============================================================================


class TestMultiExcitationExtraction:
    """Tests for complete S-matrix extraction from multi-port excitation."""

    def test_extract_full_matrix(self) -> None:
        """REQ-M2-006: Extract full S-matrix from multi-excitation data."""
        config = _make_test_extraction_config()

        n_samples = 1000
        dt = 1e-12
        time_s = np.arange(n_samples) * dt
        z0 = 50.0  # Reference impedance

        # Create a signal waveform
        v_wave = np.sin(2 * np.pi * 5e9 * time_s)

        # S-parameter wave conventions:
        # V+ = (V + I*Z0)/2 = forward/incident wave (toward load)
        # V- = (V - I*Z0)/2 = reflected/outgoing wave (away from DUT)
        # For S11 = V1-/V1+ and S21 = V2-/V1+

        # P1 excited scenario:
        # At P1: incident wave V1+ and reflected wave V1- (S11 related)
        # At P2: outgoing/transmitted wave V2- (S21 related), no incident V2+=0
        v1_plus = v_wave  # Incident at P1
        v1_minus = 0.2 * np.sin(2 * np.pi * 5e9 * time_s + 0.3)  # Reflected at P1 (~20%)
        v2_plus = 0.0  # No incident at P2
        v2_minus = 0.8 * v_wave  # Transmitted to P2 (~80%)

        # Convert back to V, I: V = V+ + V-, I = (V+ - V-)/Z0
        v1_p1_exc = v1_plus + v1_minus
        i1_p1_exc = (v1_plus - v1_minus) / z0
        v2_p1_exc = v2_plus + v2_minus
        i2_p1_exc = (v2_plus - v2_minus) / z0  # Negative since V- dominates

        signals_p1 = [
            PortSignalData(port_id="P1", time_s=time_s, voltage_v=v1_p1_exc, current_a=i1_p1_exc),
            PortSignalData(port_id="P2", time_s=time_s, voltage_v=v2_p1_exc, current_a=i2_p1_exc),
        ]

        # P2 excited scenario (reciprocal):
        v2_plus_p2 = v_wave  # Incident at P2
        v2_minus_p2 = 0.2 * np.sin(2 * np.pi * 5e9 * time_s + 0.3)  # Reflected at P2
        v1_plus_p2 = 0.0  # No incident at P1
        v1_minus_p2 = 0.8 * v_wave  # Transmitted to P1

        v1_p2_exc = v1_plus_p2 + v1_minus_p2
        i1_p2_exc = (v1_plus_p2 - v1_minus_p2) / z0
        v2_p2_exc = v2_plus_p2 + v2_minus_p2
        i2_p2_exc = (v2_plus_p2 - v2_minus_p2) / z0

        signals_p2 = [
            PortSignalData(port_id="P1", time_s=time_s, voltage_v=v1_p2_exc, current_a=i1_p2_exc),
            PortSignalData(port_id="P2", time_s=time_s, voltage_v=v2_p2_exc, current_a=i2_p2_exc),
        ]

        signal_sets = MultiPortSignalSet(excitation_sets={"P1": signals_p1, "P2": signals_p2})
        result = extract_full_sparam_matrix(signal_sets, config)

        assert result.s_parameters.n_ports == 2
        assert result.metrics["extraction_method"] == "multi_excitation"
        assert "P1" in result.metrics["excitation_ports"]
        assert "P2" in result.metrics["excitation_ports"]

        # Check all S-parameters have values
        s = result.s_parameters.s_parameters
        assert not np.allclose(s[:, 0, 0], 0)  # S11
        assert not np.allclose(s[:, 1, 0], 0)  # S21
        assert not np.allclose(s[:, 0, 1], 0)  # S12
        assert not np.allclose(s[:, 1, 1], 0)  # S22

    def test_multi_port_signal_set_access(self) -> None:
        """REQ-M2-006: MultiPortSignalSet provides proper access methods."""
        n_samples = 100
        time_s = np.arange(n_samples) * 1e-12
        v1 = np.zeros(n_samples)
        i1 = np.zeros(n_samples)

        signals = [
            PortSignalData(port_id="P1", time_s=time_s, voltage_v=v1, current_a=i1),
            PortSignalData(port_id="P2", time_s=time_s, voltage_v=v1, current_a=i1),
        ]

        signal_sets = MultiPortSignalSet(excitation_sets={"P1": signals, "P2": signals})

        assert signal_sets.excitation_port_ids == ["P1", "P2"]
        assert len(signal_sets.get_signals_for_excitation("P1")) == 2

        with pytest.raises(KeyError):
            signal_sets.get_signals_for_excitation("P_INVALID")
