"""Tests for renormalization exports (REQ-M2-017)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from formula_foundry.postprocess import renormalize_sparameters

from formula_foundry.em.touchstone import SParameterData
from formula_foundry.openems import (
    ExtractionConfig,
    ExtractionResult,
    FrequencySpec,
    PortSpec,
    write_extraction_result,
)


def _make_frequency_spec() -> FrequencySpec:
    return FrequencySpec(
        f_start_hz=1_000_000_000,
        f_stop_hz=3_000_000_000,
        n_points=3,
    )


def _make_port_specs() -> list[PortSpec]:
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


def _make_sparam_data() -> SParameterData:
    frequencies_hz = np.array([1e9, 2e9, 3e9], dtype=np.float64)
    s_parameters = np.zeros((3, 2, 2), dtype=np.complex128)
    s_parameters[:, 0, 0] = -0.1 + 0.02j
    s_parameters[:, 1, 1] = -0.05 + 0.01j
    s_parameters[:, 1, 0] = 0.9 - 0.05j
    s_parameters[:, 0, 1] = 0.9 - 0.05j

    return SParameterData(
        frequencies_hz=frequencies_hz,
        s_parameters=s_parameters,
        n_ports=2,
        reference_impedance_ohm=50.0,
    )


def test_renormalize_sparameters_updates_reference() -> None:
    """REQ-M2-017: Renormalization updates impedance and S-parameters."""
    sparam_data = _make_sparam_data()
    renorm = renormalize_sparameters(sparam_data, 75.0)

    assert renorm.reference_impedance_ohm == 75.0
    assert renorm.n_ports == sparam_data.n_ports
    assert renorm.n_frequencies == sparam_data.n_frequencies
    assert not np.allclose(renorm.s_parameters, sparam_data.s_parameters)
    assert "Renormalized" in renorm.comment


def test_write_extraction_result_writes_renormalized_outputs(tmp_path: Path) -> None:
    """REQ-M2-017: Write native and renormalized S-parameter exports."""
    config = ExtractionConfig(
        frequency_spec=_make_frequency_spec(),
        port_specs=_make_port_specs(),
        reference_impedance_ohm=50.0,
        renormalize_to_ohms=75.0,
        output_format="touchstone",
    )
    sparam_data = _make_sparam_data()

    result = ExtractionResult(
        s_parameters=sparam_data,
        extraction_config=config,
        source_files=["test.s2p"],
        canonical_hash="a" * 64,
    )

    output_paths = write_extraction_result(result, tmp_path)

    assert "touchstone" in output_paths
    assert "touchstone_renormalized" in output_paths

    native_path = output_paths["touchstone"]
    renorm_path = output_paths["touchstone_renormalized"]

    assert native_path.exists()
    assert renorm_path.exists()
    assert native_path.name != renorm_path.name
    assert "renorm" in renorm_path.name
    assert "75" in renorm_path.name

    native_text = native_path.read_text()
    renorm_text = renorm_path.read_text()
    assert "R 50.0" in native_text
    assert "R 75.0" in renorm_text
