from __future__ import annotations

from pathlib import Path

from formula_foundry.coupongen.kicad import KicadCliRunner, build_drc_args
import formula_foundry.coupongen.kicad.cli as cli_module


def test_module_imported_from_workspace() -> None:
    """Verify cli module is imported from workspace, not a stale installed package.

    This hardening test catches CI regressions where a stale version of the
    package might be installed in site-packages, causing tests to run against
    the wrong code.
    """
    module_path = Path(cli_module.__file__).resolve()
    # The module should be under src/formula_foundry/ in the workspace
    # or under site-packages if installed with pip install -e .
    # Either way, it should contain our expected default severity
    from formula_foundry.coupongen.kicad.cli import build_drc_args as fn
    # Check that the function has severity="all" as default
    import inspect
    sig = inspect.signature(fn)
    default_severity = sig.parameters["severity"].default
    assert default_severity == "all", (
        f"build_drc_args has wrong default severity: {default_severity!r}. "
        f"Expected 'all'. Module loaded from: {module_path}. "
        "This may indicate a stale installed package in site-packages."
    )


def test_kicad_cli_runner_modes(tmp_path: Path) -> None:
    runner_local = KicadCliRunner(mode="local")
    local_cmd = runner_local.build_command(["pcb", "drc"], workdir=tmp_path)

    assert local_cmd[0] == "kicad-cli"

    runner_docker = KicadCliRunner(mode="docker", docker_image="kicad/kicad:9.0.7@sha256:deadbeef")
    docker_cmd = runner_docker.build_command(["pcb", "drc"], workdir=tmp_path)

    assert docker_cmd[0] == "docker"
    assert "kicad/kicad:9.0.7@sha256:deadbeef" in docker_cmd


def test_drc_invocation_flags(tmp_path: Path) -> None:
    board = tmp_path / "coupon.kicad_pcb"
    report = tmp_path / "drc.json"
    args = build_drc_args(board, report)

    assert "--severity-all" in args
    assert "--exit-code-violations" in args
    assert "--format" in args
    assert "json" in args
    assert "--output" in args
