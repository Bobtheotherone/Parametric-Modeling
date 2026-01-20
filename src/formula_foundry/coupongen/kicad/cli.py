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
            f"{workdir}:/workspace",
            "-w",
            "/workspace",
            self.docker_image,
            self.kicad_bin,
            *args,
        ]

    def run(self, args: Iterable[str], *, workdir: Path) -> subprocess.CompletedProcess[str]:
        cmd = self.build_command(args, workdir=workdir)
        return subprocess.run(cmd, cwd=workdir, text=True, capture_output=True, check=False)

    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
        args = build_drc_args(board_path, report_path)
        return self.run(args, workdir=board_path.parent)

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        args = [
            "pcb",
            "export",
            "gerbers",
            "--output",
            str(out_dir),
            str(board_path),
        ]
        return self.run(args, workdir=board_path.parent)

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        args = [
            "pcb",
            "export",
            "drill",
            "--output",
            str(out_dir),
            str(board_path),
        ]
        return self.run(args, workdir=board_path.parent)


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


__all__ = [
    "KicadCliMode",
    "KicadCliRunner",
    "build_drc_args",
]
