from __future__ import annotations

import json
from pathlib import Path

import pytest

from formula_foundry.openems.toolchain import (
    DEFAULT_OPENEMS_LOCKFILE_PATH,
    DEFAULT_OPENEMS_TOOLCHAIN_PATH,
    is_digest_pinned,
    load_openems_lockfile,
    load_openems_toolchain,
    parse_docker_image_ref,
    validate_openems_version,
)


def test_openems_toolchain_loads_config() -> None:
    """Test that the toolchain config loads successfully."""
    assert DEFAULT_OPENEMS_TOOLCHAIN_PATH.exists()
    toolchain = load_openems_toolchain()
    assert toolchain.version
    assert toolchain.docker_image


def test_openems_toolchain_version_is_pinned() -> None:
    """Test that the toolchain has a specific pinned version."""
    toolchain = load_openems_toolchain()
    # Version should be a valid semver-like string
    assert toolchain.version == "0.0.35"


def test_openems_lockfile_loads_extended_metadata() -> None:
    """Test that the lockfile loads with extended metadata."""
    assert DEFAULT_OPENEMS_LOCKFILE_PATH.exists()
    toolchain = load_openems_lockfile()
    assert toolchain.version == "0.0.35"
    assert toolchain.csxcad_version == "0.6.3"
    assert toolchain.appcsxcad_version == "0.2.3"
    assert toolchain.toolchain_hash


def test_openems_lockfile_base_image_digest_pinned() -> None:
    """Test that the base image in lockfile is digest-pinned."""
    toolchain = load_openems_lockfile()
    assert toolchain.base_image_digest
    assert toolchain.base_image_digest.startswith("sha256:")
    assert len(toolchain.base_image_digest) == 71  # "sha256:" + 64 hex chars


def test_validate_openems_version() -> None:
    """Test version validation."""
    assert validate_openems_version("0.0.35", "0.0.35")
    assert validate_openems_version(" 0.0.35 ", "0.0.35")
    assert not validate_openems_version("0.0.34", "0.0.35")
    assert not validate_openems_version("0.0.36", "0.0.35")


def test_parse_docker_image_ref_simple() -> None:
    """Test parsing simple image references."""
    result = parse_docker_image_ref("ubuntu:22.04")
    assert result["repository"] == "ubuntu"
    assert result["tag"] == "22.04"
    assert result["registry"] is None
    assert result["digest"] is None


def test_parse_docker_image_ref_with_registry() -> None:
    """Test parsing image references with registry."""
    result = parse_docker_image_ref("ghcr.io/thliebig/openems:0.0.35")
    assert result["registry"] == "ghcr.io"
    assert result["repository"] == "thliebig/openems"
    assert result["tag"] == "0.0.35"


def test_parse_docker_image_ref_with_digest() -> None:
    """Test parsing image references with digest."""
    digest = "sha256:77906da86b60585ce12215807090eb327e7386c8fafb5402369e421f44eff17e"
    result = parse_docker_image_ref(f"ubuntu:22.04@{digest}")
    assert result["repository"] == "ubuntu"
    assert result["tag"] == "22.04"
    assert result["digest"] == digest


def test_is_digest_pinned() -> None:
    """Test digest pinning detection."""
    digest = "sha256:77906da86b60585ce12215807090eb327e7386c8fafb5402369e421f44eff17e"
    assert is_digest_pinned(f"ubuntu:22.04@{digest}")
    assert is_digest_pinned(f"ghcr.io/thliebig/openems:0.0.35@{digest}")
    assert not is_digest_pinned("ubuntu:22.04")
    assert not is_digest_pinned("formula-foundry-openems:0.0.35")


def test_openems_toolchain_to_dict() -> None:
    """Test toolchain serialization."""
    toolchain = load_openems_toolchain()
    data = toolchain.to_dict()
    assert "version" in data
    assert "docker_image" in data


def test_openems_lockfile_to_dict_includes_extended_fields() -> None:
    """Test lockfile toolchain serialization includes extended fields."""
    toolchain = load_openems_lockfile()
    data = toolchain.to_dict()
    assert data["version"] == "0.0.35"
    assert data["csxcad_version"] == "0.6.3"
    assert data["appcsxcad_version"] == "0.2.3"
    assert "toolchain_hash" in data


def test_openems_toolchain_missing_file_raises(tmp_path: Path) -> None:
    """Test that missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_openems_toolchain(tmp_path / "nonexistent.json")


def test_openems_lockfile_missing_file_raises(tmp_path: Path) -> None:
    """Test that missing lockfile raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_openems_lockfile(tmp_path / "nonexistent.json")


def test_openems_toolchain_invalid_json_raises(tmp_path: Path) -> None:
    """Test that invalid JSON raises ValueError."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("not json", encoding="utf-8")
    with pytest.raises(Exception):
        load_openems_toolchain(invalid_file)


def test_openems_toolchain_missing_version_raises(tmp_path: Path) -> None:
    """Test that missing version field raises ValueError."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"docker_image": "test:1.0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_openems_toolchain(config_file)


def test_openems_toolchain_missing_docker_image_raises(tmp_path: Path) -> None:
    """Test that missing docker_image field raises ValueError."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"version": "0.0.35"}), encoding="utf-8")
    with pytest.raises(ValueError, match="docker_image"):
        load_openems_toolchain(config_file)
