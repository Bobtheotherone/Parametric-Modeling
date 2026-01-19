from __future__ import annotations

import re
from pathlib import Path


def test_dockerfile_pins_base_image_by_digest() -> None:
    dockerfile = Path("docker") / "Dockerfile"
    assert dockerfile.is_file()

    content = dockerfile.read_text(encoding="utf-8")
    from_lines = [line.strip() for line in content.splitlines() if line.strip().startswith("FROM ")]
    assert from_lines, "Dockerfile must declare a base image"
    assert any("@sha256:" in line for line in from_lines), "Base image must be digest-pinned"
    assert any("nvidia/cuda" in line for line in from_lines), "Base image must be NVIDIA CUDA"
    assert "tools.m0" in content, "Dockerfile must expose tools.m0 command surface"


def test_dockerfile_enables_formula_foundry_import() -> None:
    """Regression: Dockerfile must enable formula_foundry imports for tools.m0.

    The project uses src-layout; either set PYTHONPATH=/app/src or perform
    an editable install so that `python -m tools.m0` can import formula_foundry.
    """
    dockerfile = Path("docker") / "Dockerfile"
    content = dockerfile.read_text(encoding="utf-8")

    # Accept either PYTHONPATH approach or an editable install
    has_pythonpath = re.search(r'PYTHONPATH\s*=\s*["\']?/app/src["\']?', content)
    has_editable_install = re.search(r'(pip|uv)\s+.*install\s+-e\s+\.', content)

    assert has_pythonpath or has_editable_install, (
        "Dockerfile must either set PYTHONPATH=/app/src or do an editable install "
        "so that tools.m0 can import formula_foundry (src-layout)"
    )
