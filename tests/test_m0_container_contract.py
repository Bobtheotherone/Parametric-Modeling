from __future__ import annotations

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
