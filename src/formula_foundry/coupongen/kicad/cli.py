"""KiCad CLI runner with local binary and Docker support.

This module provides a runner for kicad-cli that can execute:
- Locally using the system-installed kicad-cli binary
- Via Docker using a pinned KiCad Docker image (e.g., kicad/kicad:9.0.7)

Satisfies REQ-M1-015 through REQ-M1-017:
- REQ-M1-015: Runner supports local binary or pinned Docker image execution
- REQ-M1-016: DRC with severity-all, JSON report output, and exit-code gating
- REQ-M1-017: Gerber and drill file export via KiCad CLI
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

KicadCliMode = Literal["local", "docker"]


@dataclass(frozen=True)
class KicadCliRunner:
    mode: KicadCliMode
    docker_image: str | None = None
    kicad_bin: str = "kicad-cli"

    def _to_container_path(self, host_path: Path, workdir: Path) -> str:
        """Convert a host path to a container path relative to /workspace.

        When running in docker mode, paths must be translated from host
        absolute paths to container paths inside /workspace.

        Args:
            host_path: Absolute host path to convert.
            workdir: The host directory mounted as /workspace.

        Returns:
            Container-relative path string (e.g., "board.kicad_pcb" or
            "subdir/board.kicad_pcb" for paths within workdir).

        Raises:
            ValueError: If host_path is not within workdir.
        """
        host_abs = host_path.resolve()
        workdir_abs = workdir.resolve()

        try:
            relative = host_abs.relative_to(workdir_abs)
            # Return the relative path as a string (works as /workspace/relative)
            return str(relative)
        except ValueError as e:
            raise ValueError(
                f"Path {host_path} is not within workdir {workdir}. "
                f"Docker mount mapping requires all paths to be within the mounted directory."
            ) from e

    def build_command(self, args: Iterable[str], *, workdir: Path) -> list[str]:
        if self.mode == "local":
            return [self.kicad_bin, *args]
        if not self.docker_image:
            raise ValueError("docker_image is required for docker mode")
        return [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{workdir.resolve()}:/workspace",
            "-w",
            "/workspace",
            self.docker_image,
            self.kicad_bin,
            *args,
        ]

    def run(self, args: Iterable[str], *, workdir: Path) -> subprocess.CompletedProcess[str]:
        cmd = self.build_command(args, workdir=workdir)
        result = subprocess.run(cmd, cwd=workdir, text=True, capture_output=True, check=False)

        # Add debug info on failure (returncode 3 typically means file not found in KiCad)
        if result.returncode == 3 and self.mode == "docker":
            debug_msg = (
                f"\n[DEBUG] Docker KiCad CLI returned code 3 (invalid input file).\n"
                f"  Command: {' '.join(cmd)}\n"
                f"  Workdir (host): {workdir.resolve()}\n"
                f"  Workdir (container): /workspace\n"
                f"  This usually means the board file path was not correctly "
                f"translated to a container-relative path."
            )
            result = subprocess.CompletedProcess(
                args=result.args,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr + debug_msg,
            )
        return result

    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
        workdir = board_path.parent.resolve()

        if self.mode == "docker":
            # Translate paths to container-relative paths
            board_container = self._to_container_path(board_path, workdir)
            report_container = self._to_container_path(report_path, workdir)
            args = build_drc_args(Path(board_container), Path(report_container))
        else:
            args = build_drc_args(board_path, report_path)

        return self.run(args, workdir=workdir)

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        workdir = board_path.parent.resolve()

        if self.mode == "docker":
            # Translate paths to container-relative paths
            board_container = self._to_container_path(board_path, workdir)
            out_container = self._to_container_path(out_dir, workdir)
            args = [
                "pcb",
                "export",
                "gerbers",
                "--output",
                out_container,
                board_container,
            ]
        else:
            args = [
                "pcb",
                "export",
                "gerbers",
                "--output",
                str(out_dir),
                str(board_path),
            ]

        return self.run(args, workdir=workdir)

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        workdir = board_path.parent.resolve()

        if self.mode == "docker":
            # Translate paths to container-relative paths
            board_container = self._to_container_path(board_path, workdir)
            out_container = self._to_container_path(out_dir, workdir)
            args = [
                "pcb",
                "export",
                "drill",
                "--output",
                out_container,
                board_container,
            ]
        else:
            args = [
                "pcb",
                "export",
                "drill",
                "--output",
                str(out_dir),
                str(board_path),
            ]

        return self.run(args, workdir=workdir)


def build_drc_args(board_path: Path, report_path: Path) -> list[str]:
    """Build kicad-cli DRC command arguments.

    Satisfies REQ-M1-016:
    - --severity-all: Report violations at all severity levels
    - --format json: Output in JSON format for programmatic parsing
    - --exit-code-violations: Return non-zero exit code if violations exist

    Args:
        board_path: Path to the .kicad_pcb file to check.
        report_path: Path where the JSON DRC report will be written.

    Returns:
        List of command-line arguments for kicad-cli pcb drc.
    """
    return [
        "pcb",
        "drc",
        "--severity-all",
        "--exit-code-violations",
        "--format",
        "json",
        "--output",
        str(report_path),
        str(board_path),
    ]


def get_kicad_cli_version(runner: KicadCliRunner, workdir: Path | None = None) -> str:
    """Get the kicad-cli version string.

    This helper function executes `kicad-cli --version` and parses the output
    to extract the version string. It works with both local and Docker runners.

    Args:
        runner: A KicadCliRunner instance (local or Docker mode).
        workdir: Working directory for the command. If None, uses current directory.
            Required for Docker mode to mount the volume.

    Returns:
        Version string (e.g., "9.0.7").

    Raises:
        RuntimeError: If version cannot be determined.

    Example:
        >>> runner = KicadCliRunner(mode="local")
        >>> version = get_kicad_cli_version(runner)
        >>> print(version)
        '9.0.7'
    """
    if workdir is None:
        workdir = Path.cwd()

    result = runner.run(["--version"], workdir=workdir)

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to get kicad-cli version: {result.stderr or result.stdout}"
        )

    return _parse_version_output(result.stdout)


def _parse_version_output(version_output: str) -> str:
    """Parse the version string from kicad-cli --version output.

    Args:
        version_output: Raw output from kicad-cli --version.

    Returns:
        Cleaned version string (e.g., "9.0.7").
    """
    output = version_output.strip()

    # Try to extract version pattern (e.g., "9.0.7" or "9.0.7-1")
    match = re.search(r"(\d+\.\d+\.\d+(?:-\d+)?)", output)
    if match:
        return match.group(1)

    # Fall back to returning the full output if no pattern matches
    return output


__all__ = [
    "KicadCliMode",
    "KicadCliRunner",
    "build_drc_args",
    "get_kicad_cli_version",
]
