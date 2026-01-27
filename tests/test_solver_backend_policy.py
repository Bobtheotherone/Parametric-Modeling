from __future__ import annotations

from pathlib import Path

import pytest
from formula_foundry.solver.backend import StubRunNotAllowedError
from formula_foundry.solver.runner import SolverRunner

from formula_foundry.openems.geometry import (
    BoardOutlineSpec,
    DiscontinuitySpec,
    GeometrySpec,
    LayerSpec,
    StackupMaterialsSpec,
    StackupSpec,
    TransmissionLineSpec,
)
from formula_foundry.openems.sim_runner import SimulationRunner
from formula_foundry.openems.spec import (
    ExcitationSpec,
    FrequencySpec,
    GeometryRefSpec,
    OpenEMSToolchainSpec,
    PortSpec,
    SimulationSpec,
    ToolchainSpec,
)


def _make_minimal_spec() -> SimulationSpec:
    return SimulationSpec(
        schema_version=1,
        simulation_id="solver-run",
        toolchain=ToolchainSpec(
            openems=OpenEMSToolchainSpec(
                version="0.0.35",
                docker_image="ghcr.io/openems:0.0.35",
            )
        ),
        geometry_ref=GeometryRefSpec(design_hash="design_hash"),
        excitation=ExcitationSpec(f0_hz=5e9, fc_hz=10e9),
        frequency=FrequencySpec(f_start_hz=1e9, f_stop_hz=10e9, n_points=5),
        ports=[
            PortSpec(
                id="P1",
                type="lumped",
                excite=True,
                position_nm=(0, 0, 0),
                direction="x",
            ),
        ],
    )


def _make_minimal_geometry() -> GeometrySpec:
    return GeometrySpec(
        design_hash="test_design_hash_12345",
        coupon_family="F1_SINGLE_ENDED_VIA",
        board=BoardOutlineSpec(
            width_nm=20_000_000,
            length_nm=40_000_000,
            corner_radius_nm=2_000_000,
        ),
        stackup=StackupSpec(
            copper_layers=4,
            thicknesses_nm={
                "L1_to_L2": 180_000,
                "L2_to_L3": 800_000,
                "L3_to_L4": 180_000,
            },
            materials=StackupMaterialsSpec(er=4.1, loss_tangent=0.02),
        ),
        layers=[
            LayerSpec(id="L1", z_nm=0, role="signal"),
            LayerSpec(id="L2", z_nm=180_000, role="ground"),
            LayerSpec(id="L3", z_nm=980_000, role="ground"),
            LayerSpec(id="L4", z_nm=1_160_000, role="signal"),
        ],
        transmission_line=TransmissionLineSpec(
            type="CPWG",
            layer="F.Cu",
            w_nm=300_000,
            gap_nm=180_000,
            length_left_nm=10_000_000,
            length_right_nm=10_000_000,
        ),
        discontinuity=DiscontinuitySpec(
            type="VIA_TRANSITION",
            parameters_nm={
                "signal_via.drill_nm": 300_000,
                "signal_via.diameter_nm": 650_000,
            },
        ),
    )


def test_solver_runner_disallows_stub_without_allow(tmp_path: Path) -> None:
    spec = _make_minimal_spec()
    geometry = _make_minimal_geometry()
    runner = SolverRunner(simulation_runner=SimulationRunner(mode="stub"), allow_stub=False)

    with pytest.raises(StubRunNotAllowedError, match="Stub solver backend"):
        runner.run(spec, geometry, output_dir=tmp_path / "run")
