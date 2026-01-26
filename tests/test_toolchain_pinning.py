from __future__ import annotations

from pathlib import Path

import pytest
from formula_foundry.meta import toolchain as toolchain_meta
from formula_foundry.toolchain import pinning


def test_dockerfile_base_image_is_digest_pinned() -> None:
    dockerfile = Path("docker") / "Dockerfile"
    base_image = pinning.resolve_pinned_base_image(dockerfile)
    assert base_image.digest
    assert base_image.digest.startswith("sha256:")
    assert base_image.pinned_ref.endswith(base_image.digest)


def test_toolchain_is_digest_pinned_and_recorded_in_meta() -> None:
    metadata = toolchain_meta.build_toolchain_metadata()
    docker = metadata["docker"]
    versions = metadata["versions"]

    assert docker["digest"].startswith("sha256:")
    assert docker["image"]
    assert docker["pinned_ref"].endswith(docker["digest"])
    assert versions["image_tag"]
    assert versions["cuda"]
    assert versions["ubuntu"]


def test_unpinned_base_image_rejected(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM ubuntu:22.04\n", encoding="utf-8")

    with pytest.raises(pinning.DockerPinningError, match="digest-pinned"):
        pinning.resolve_pinned_base_image(dockerfile)


def test_missing_versions_rejected(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f"FROM ubuntu@sha256:{'a' * 64}\n", encoding="utf-8")

    with pytest.raises(toolchain_meta.ToolchainMetadataError, match="versions"):
        toolchain_meta.build_toolchain_metadata(dockerfile_path=dockerfile)
