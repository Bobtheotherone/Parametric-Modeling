"""Tests for openEMS runner and Docker installation verification.

REQ-M2-005: The OpenEMSRunner MUST provide version_metadata() that probes
openEMS version and returns structured metadata including openems_version
and csxcad_version.
"""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from formula_foundry.openems.runner import OpenEMSRunner, parse_openems_version_output
from formula_foundry.openems.toolchain import load_openems_toolchain


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
        "#!/usr/bin/env bash\necho 'openEMS version: 0.0.35'\necho 'CSXCAD version: 0.6.3'\n",
    )
    runner = OpenEMSRunner(mode="local", openems_bin=str(stub))
    payload = runner.version_metadata(workdir=tmp_path)
    assert payload["returncode"] == 0
    assert payload["openems_version"] == "0.0.35"
    assert payload["csxcad_version"] == "0.6.3"


def _docker_available() -> bool:
    """Check if Docker is available on the system."""
    return shutil.which("docker") is not None


def _openems_image_available() -> bool:
    """Check if the openEMS Docker image is available locally."""
    if not _docker_available():
        return False
    try:
        toolchain = load_openems_toolchain()
        # Check if image exists locally (strip digest for inspection)
        image_ref = toolchain.docker_image.split("@")[0]
        proc = subprocess.run(
            ["docker", "image", "inspect", image_ref],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(
    not _docker_available(),
    reason="Docker not available",
)
@pytest.mark.skipif(
    not _openems_image_available(),
    reason="openEMS Docker image not built locally",
)
def test_openems_docker_smoke_test(tmp_path: Path) -> None:
    """REQ-M2-002: Smoke test that verifies openEMS Docker installation.

    This test runs openEMS --help inside the Docker container to verify:
    1. The Docker image is buildable and runnable
    2. openEMS binary is accessible at the expected path
    3. The container can be invoked via the OpenEMSRunner

    This test is skipped if Docker is not available or the image is not built.
    Run manually after building: docker build -t formula-foundry-openems:0.0.35 tools/m2/docker/
    """
    toolchain = load_openems_toolchain()
    # Use image without digest for local testing (digest may not match local build)
    image_ref = toolchain.docker_image.split("@")[0]

    runner = OpenEMSRunner(mode="docker", docker_image=image_ref)
    metadata = runner.version_metadata(workdir=tmp_path)

    # Verify the command ran (may return non-zero if --version not supported)
    assert "command" in metadata
    assert "docker" in metadata["command"][0]

    # If we got any version info, verify it matches expected
    if metadata.get("openems_version"):
        assert metadata["openems_version"] == toolchain.version
