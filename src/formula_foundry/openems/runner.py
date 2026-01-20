from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

OpenEMSMode = Literal["local", "docker"]

_OPENEMS_VERSION_RE = re.compile(r"openems[^0-9]*([0-9]+(?:\.[0-9]+)+)", re.IGNORECASE)
_CSXCAD_VERSION_RE = re.compile(r"csxcad[^0-9]*([0-9]+(?:\.[0-9]+)+)", re.IGNORECASE)


def parse_openems_version_output(text: str) -> dict[str, str | None]:
    openems_version = _match_version(_OPENEMS_VERSION_RE, text)
    csxcad_version = _match_version(_CSXCAD_VERSION_RE, text)
    return {
        "openems_version": openems_version,
        "csxcad_version": csxcad_version,
    }


def _match_version(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None


@dataclass(frozen=True)
class OpenEMSRunner:
    mode: OpenEMSMode
    docker_image: str | None = None
    openems_bin: str = "openEMS"
    docker_bin: str = "docker"

    def build_command(self, args: Iterable[str], *, workdir: Path) -> list[str]:
        args_list = list(args)
        if self.mode == "local":
            return [self.openems_bin, *args_list]
        if not self.docker_image:
            raise ValueError("docker_image is required for docker mode")
        return [
            self.docker_bin,
            "run",
            "--rm",
            "-v",
            f"{workdir}:/workspace",
            "-w",
            "/workspace",
            self.docker_image,
            self.openems_bin,
            *args_list,
        ]

    def run(self, args: Iterable[str], *, workdir: Path) -> subprocess.CompletedProcess[str]:
        cmd = self.build_command(args, workdir=workdir)
        return subprocess.run(cmd, cwd=workdir, text=True, capture_output=True, check=False)

    def version_metadata(self, *, workdir: Path) -> dict[str, Any]:
        attempts = [["--version"], ["-v"], ["--help"]]
        last_proc: subprocess.CompletedProcess[str] | None = None
        last_cmd: list[str] | None = None
        for args in attempts:
            cmd = self.build_command(args, workdir=workdir)
            proc = subprocess.run(cmd, cwd=workdir, text=True, capture_output=True, check=False)
            output = (proc.stdout or "") + (proc.stderr or "")
            versions = parse_openems_version_output(output)
            if output.strip() or any(versions.values()):
                return _build_metadata(self, proc, cmd, versions)
            last_proc = proc
            last_cmd = cmd
        if last_proc is None or last_cmd is None:
            raise RuntimeError("openEMS version probe failed to run")
        return _build_metadata(self, last_proc, last_cmd, parse_openems_version_output(""))


def _build_metadata(
    runner: OpenEMSRunner,
    proc: subprocess.CompletedProcess[str],
    cmd: list[str],
    versions: dict[str, str | None],
) -> dict[str, Any]:
    return {
        "mode": runner.mode,
        "docker_image": runner.docker_image,
        "openems_bin": runner.openems_bin,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        **versions,
    }
