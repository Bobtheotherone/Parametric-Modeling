"""Tests for toolchain lock validation gate in tools.verify."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import verify


def _write_lock(tmp_path: Path, lock_data: dict[str, str]) -> Path:
    toolchain_dir = tmp_path / "toolchain"
    toolchain_dir.mkdir(parents=True, exist_ok=True)
    lock_path = toolchain_dir / "kicad.lock.json"
    lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
    return lock_path


@pytest.mark.parametrize(
    "lock_data, expected",
    [
        (
            {
                "kicad_version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7",
            },
            "docker_digest",
        ),
        (
            {
                "kicad_version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7",
                "docker_digest": "sha256:PLACEHOLDER",
            },
            "placeholder",
        ),
    ],
)
def test_gate_toolchain_lock_rejects_invalid_digest(
    tmp_path: Path,
    lock_data: dict[str, str],
    expected: str,
) -> None:
    _write_lock(tmp_path, lock_data)
    result = verify._gate_toolchain_lock(tmp_path)
    assert result.passed is False
    assert expected in result.stderr


def test_gate_toolchain_lock_accepts_valid_digest(tmp_path: Path) -> None:
    _write_lock(
        tmp_path,
        {
            "kicad_version": "9.0.7",
            "docker_image": "kicad/kicad:9.0.7",
            "docker_digest": "sha256:" + "a" * 64,
        },
    )
    result = verify._gate_toolchain_lock(tmp_path)
    assert result.passed is True
