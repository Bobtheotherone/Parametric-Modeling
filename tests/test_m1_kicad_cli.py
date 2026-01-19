from __future__ import annotations

from pathlib import Path

from formula_foundry.coupongen.kicad import KicadCliRunner, build_drc_args


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
