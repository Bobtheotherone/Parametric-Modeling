from __future__ import annotations

import stat
from pathlib import Path

from formula_foundry.openems.runner import OpenEMSRunner, parse_openems_version_output


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR)


def test_openems_runner_build_command_modes(tmp_path: Path) -> None:
    runner_local = OpenEMSRunner(mode="local", openems_bin="openEMS")
    local_cmd = runner_local.build_command(["--version"], workdir=tmp_path)
    assert local_cmd[0] == "openEMS"

    runner_docker = OpenEMSRunner(
        mode="docker",
        docker_image="ghcr.io/thliebig/openems:0.0.35@sha256:" + ("deadbeef" * 8),
    )
    docker_cmd = runner_docker.build_command(["--version"], workdir=tmp_path)
    assert docker_cmd[0] == "docker"
    assert runner_docker.docker_image in docker_cmd


def test_parse_openems_version_output() -> None:
    output = "openEMS version: 0.0.35\nCSXCAD version: 0.6.3\n"
    parsed = parse_openems_version_output(output)
    assert parsed["openems_version"] == "0.0.35"
    assert parsed["csxcad_version"] == "0.6.3"


def test_openems_runner_version_metadata(tmp_path: Path) -> None:
    stub = tmp_path / "openEMS"
    _write_executable(
        stub,
        "#!/usr/bin/env bash\n"
        "echo 'openEMS version: 0.0.35'\n"
        "echo 'CSXCAD version: 0.6.3'\n",
    )
    runner = OpenEMSRunner(mode="local", openems_bin=str(stub))
    payload = runner.version_metadata(workdir=tmp_path)
    assert payload["returncode"] == 0
    assert payload["openems_version"] == "0.0.35"
    assert payload["csxcad_version"] == "0.6.3"
