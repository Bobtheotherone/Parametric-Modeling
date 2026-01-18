from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_dependencies(text: str) -> list[str]:
    deps: list[str] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if not in_block and stripped.startswith("dependencies") and "[" in stripped:
            in_block = True
            tail = stripped.split("[", 1)[1]
            if "]" in tail:
                tail = tail.split("]", 1)[0]
                in_block = False
            for item in re.findall(r'"([^"]+)"', tail):
                deps.append(item)
            continue
        if not in_block:
            continue
        if stripped.startswith("]"):
            break
        for item in re.findall(r'"([^"]+)"', stripped):
            deps.append(item)
    return deps


def test_cupy_dependency_locked() -> None:
    pyproject = ROOT / "pyproject.toml"
    deps = _parse_dependencies(pyproject.read_text(encoding="utf-8"))
    assert any(dep.startswith("cupy-cuda12x") for dep in deps), "CuPy must be a default dependency"

    lock_path = ROOT / "uv.lock"
    lock_text = lock_path.read_text(encoding="utf-8")
    assert 'name = "cupy-cuda12x"' in lock_text
