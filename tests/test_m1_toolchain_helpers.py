"""Tests for toolchain helper scripts that enforce digest pinning."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "print_kicad_image_ref.py"


def _run_print_script(tmp_path: Path, lock_data: dict[str, str]) -> subprocess.CompletedProcess[str]:
    lock_path = tmp_path / "kicad.lock.json"
    lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--lock-file", str(lock_path)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_print_kicad_image_ref_missing_digest(tmp_path: Path) -> None:
    proc = _run_print_script(
        tmp_path,
        {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
        },
    )
    assert proc.returncode == 2
    assert "Docker digest missing" in proc.stderr


def test_print_kicad_image_ref_placeholder_digest(tmp_path: Path) -> None:
    proc = _run_print_script(
        tmp_path,
        {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:PLACEHOLDER",
        },
    )
    assert proc.returncode == 2
    assert "placeholder" in proc.stderr


def test_print_kicad_image_ref_valid_digest(tmp_path: Path) -> None:
    digest = "sha256:" + "a" * 64
    proc = _run_print_script(
        tmp_path,
        {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": digest,
        },
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == f"kicad/kicad:9.0.7@{digest}"
