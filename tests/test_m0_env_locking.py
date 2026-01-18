from __future__ import annotations

import re
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_toml_scalar(path: Path, key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}\s*=\s*(.+)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = pattern.match(stripped)
        if match:
            value = match.group(1).strip()
            if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
                return value[1:-1]
            return value
    return None


def test_uv_lock_and_frozen_install() -> None:
    lock_path = ROOT / "uv.lock"
    assert lock_path.is_file()
    assert lock_path.read_text(encoding="utf-8").strip()

    lock_version = _read_toml_scalar(lock_path, "version")
    assert lock_version == "1"

    lock_python = _read_toml_scalar(lock_path, "requires-python")
    assert lock_python

    pyproject_requires = _read_toml_scalar(ROOT / "pyproject.toml", "requires-python")
    assert lock_python == pyproject_requires


def test_bootstrap_venv_script_contract() -> None:
    script_path = ROOT / "scripts" / "bootstrap_venv.sh"
    assert script_path.is_file()

    content = script_path.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env bash")
    assert "uv venv" in content
    assert "uv sync" in content
    assert "--frozen" in content
    assert ".venv" in content or "UV_PROJECT_ENVIRONMENT" in content
    assert script_path.stat().st_mode & stat.S_IXUSR
