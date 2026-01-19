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
