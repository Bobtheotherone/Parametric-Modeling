"""Tests for Docker digest pinning in Dockerfiles.

This module verifies that all Dockerfiles in the repository use digest-pinned
base images to prevent silent drift in container contracts.

Gate Commands:
    pytest tests/test_docker_pinning.py -v

Requirements:
    REQ-M2-005: Toolchain is digest-pinned and recorded in every run's metadata;
                strict mode fails if it is unpinned or missing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pattern to match FROM lines with digest pinning:
# FROM image@sha256:hex64 [AS alias]
# FROM image:tag@sha256:hex64 [AS alias]
DIGEST_PINNED_FROM_PATTERN = re.compile(
    r"^FROM\s+"
    r"(?P<image>[^\s@]+)"  # image name (may include registry, port, tag)
    r"@(?P<digest>sha256:[a-fA-F0-9]{64})"  # @sha256:64hexchars
    r"(?:\s+AS\s+\w+)?"  # optional AS alias
    r"\s*$",
    re.IGNORECASE,
)

# Pattern to match any FROM line
FROM_LINE_PATTERN = re.compile(
    r"^FROM\s+(?P<image>\S+)(?:\s+AS\s+\w+)?\s*$",
    re.IGNORECASE,
)

# Known Dockerfiles in the repository
DOCKERFILES = [
    Path("docker/Dockerfile"),
    Path("tools/m2/docker/Dockerfile"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_from_lines(dockerfile_path: Path) -> list[tuple[int, str]]:
    """Extract all FROM lines from a Dockerfile with line numbers.

    Args:
        dockerfile_path: Path to the Dockerfile.

    Returns:
        List of (line_number, line_content) tuples.
    """
    content = dockerfile_path.read_text(encoding="utf-8")
    from_lines = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
            from_lines.append((lineno, stripped))
    return from_lines


def is_digest_pinned(from_line: str) -> bool:
    """Check if a FROM line uses digest pinning.

    Args:
        from_line: The FROM line content.

    Returns:
        True if the line uses @sha256:... format.
    """
    return DIGEST_PINNED_FROM_PATTERN.match(from_line) is not None


def extract_digest(from_line: str) -> str | None:
    """Extract the digest from a pinned FROM line.

    Args:
        from_line: The FROM line content.

    Returns:
        The digest string (sha256:...) or None if not pinned.
    """
    match = DIGEST_PINNED_FROM_PATTERN.match(from_line)
    if match:
        return match.group("digest")
    return None


def extract_image(from_line: str) -> str | None:
    """Extract the image name from a FROM line.

    Args:
        from_line: The FROM line content.

    Returns:
        The image name or None if parsing failed.
    """
    match = FROM_LINE_PATTERN.match(from_line)
    if match:
        image = match.group("image")
        # Remove digest if present
        if "@" in image:
            image = image.split("@")[0]
        return image
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDockerfileDigestPinning:
    """Tests for Dockerfile digest pinning enforcement."""

    @pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: str(p))
    def test_dockerfile_exists(self, dockerfile: Path, project_root: Path) -> None:
        """Verify that expected Dockerfiles exist."""
        full_path = project_root / dockerfile
        assert full_path.exists(), f"Dockerfile not found: {dockerfile}"

    @pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: str(p))
    def test_all_from_lines_are_digest_pinned(self, dockerfile: Path, project_root: Path) -> None:
        """Verify all FROM lines in Dockerfiles use digest pinning.

        This prevents silent drift in base images which could break
        reproducibility and introduce security vulnerabilities.
        """
        full_path = project_root / dockerfile
        if not full_path.exists():
            pytest.skip(f"Dockerfile not found: {dockerfile}")

        from_lines = parse_from_lines(full_path)
        assert from_lines, f"No FROM lines found in {dockerfile}"

        unpinned = []
        for lineno, line in from_lines:
            if not is_digest_pinned(line):
                unpinned.append((lineno, line))

        if unpinned:
            msg_parts = [f"Unpinned FROM lines in {dockerfile}:"]
            for lineno, line in unpinned:
                msg_parts.append(f"  Line {lineno}: {line}")
            msg_parts.append("")
            msg_parts.append("All base images must use digest pinning:")
            msg_parts.append("  FROM image@sha256:...")
            msg_parts.append("  FROM image:tag@sha256:...")
            pytest.fail("\n".join(msg_parts))

    @pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: str(p))
    def test_digests_are_valid_sha256(self, dockerfile: Path, project_root: Path) -> None:
        """Verify that digests are valid SHA256 hashes (64 hex chars)."""
        full_path = project_root / dockerfile
        if not full_path.exists():
            pytest.skip(f"Dockerfile not found: {dockerfile}")

        from_lines = parse_from_lines(full_path)

        for lineno, line in from_lines:
            digest = extract_digest(line)
            if digest:
                # Check format: sha256: followed by 64 hex chars
                assert digest.startswith("sha256:"), f"Line {lineno}: Digest must start with 'sha256:': {digest}"
                hex_part = digest[7:]  # Remove 'sha256:' prefix
                assert len(hex_part) == 64, f"Line {lineno}: SHA256 digest must be 64 hex chars, got {len(hex_part)}: {digest}"
                assert all(c in "0123456789abcdefABCDEF" for c in hex_part), (
                    f"Line {lineno}: Digest contains non-hex characters: {digest}"
                )


class TestDockerPinningHelpers:
    """Tests for the parsing helper functions."""

    def test_is_digest_pinned_with_digest(self) -> None:
        """Test detection of digest-pinned FROM lines."""
        pinned_lines = [
            "FROM ubuntu@sha256:77906da86b60585ce12215807090eb327e7386c8fafb5402369e421f44eff17e",
            "FROM ubuntu:22.04@sha256:77906da86b60585ce12215807090eb327e7386c8fafb5402369e421f44eff17e",
            "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:517da2300c184c9999ec203c2665244bdebd3578d12fcc7065e83667932643d9",
            "FROM ubuntu@sha256:77906da86b60585ce12215807090eb327e7386c8fafb5402369e421f44eff17e AS builder",
            "from ubuntu@sha256:77906da86b60585ce12215807090eb327e7386c8fafb5402369e421f44eff17e as runtime",
        ]
        for line in pinned_lines:
            assert is_digest_pinned(line), f"Should be pinned: {line}"

    def test_is_digest_pinned_without_digest(self) -> None:
        """Test detection of unpinned FROM lines."""
        unpinned_lines = [
            "FROM ubuntu",
            "FROM ubuntu:22.04",
            "FROM ubuntu:latest",
            "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04",
            "FROM ubuntu AS builder",
            "FROM ubuntu:22.04 AS builder",
        ]
        for line in unpinned_lines:
            assert not is_digest_pinned(line), f"Should NOT be pinned: {line}"

    def test_extract_digest(self) -> None:
        """Test digest extraction from FROM lines."""
        line = "FROM ubuntu:22.04@sha256:77906da86b60585ce12215807090eb327e7386c8fafb5402369e421f44eff17e AS builder"
        digest = extract_digest(line)
        assert digest == "sha256:77906da86b60585ce12215807090eb327e7386c8fafb5402369e421f44eff17e"

    def test_extract_digest_unpinned(self) -> None:
        """Test digest extraction returns None for unpinned lines."""
        line = "FROM ubuntu:22.04"
        assert extract_digest(line) is None

    def test_extract_image(self) -> None:
        """Test image name extraction from FROM lines."""
        cases = [
            ("FROM ubuntu", "ubuntu"),
            ("FROM ubuntu:22.04", "ubuntu:22.04"),
            ("FROM nvidia/cuda:12.4.1", "nvidia/cuda:12.4.1"),
            (
                "FROM ubuntu@sha256:77906da86b60585ce12215807090eb327e7386c8fafb5402369e421f44eff17e",
                "ubuntu",
            ),
            (
                "FROM ubuntu:22.04@sha256:77906da86b60585ce12215807090eb327e7386c8fafb5402369e421f44eff17e AS builder",
                "ubuntu:22.04",
            ),
        ]
        for line, expected in cases:
            assert extract_image(line) == expected, f"Failed for: {line}"


class TestMultiStageDockerfiles:
    """Tests for multi-stage Dockerfile handling."""

    def test_m2_dockerfile_has_multiple_stages(self, project_root: Path) -> None:
        """Verify the M2 Dockerfile has multiple build stages."""
        dockerfile = project_root / "tools/m2/docker/Dockerfile"
        if not dockerfile.exists():
            pytest.skip("M2 Dockerfile not found")

        from_lines = parse_from_lines(dockerfile)
        # M2 Dockerfile should have at least builder and runtime stages
        assert len(from_lines) >= 2, f"Expected multi-stage Dockerfile with at least 2 FROM lines, got {len(from_lines)}"

    def test_all_stages_use_same_base_digest_when_appropriate(self, project_root: Path) -> None:
        """Verify multi-stage builds use consistent base image digests.

        When builder and runtime stages use the same base image (e.g., ubuntu),
        they should use the same digest for reproducibility.
        """
        dockerfile = project_root / "tools/m2/docker/Dockerfile"
        if not dockerfile.exists():
            pytest.skip("M2 Dockerfile not found")

        from_lines = parse_from_lines(dockerfile)

        # Group by base image (without tag/digest)
        image_digests: dict[str, set[str]] = {}
        for _lineno, line in from_lines:
            image = extract_image(line)
            digest = extract_digest(line)
            if image and digest:
                # Normalize image name (remove tag)
                base = image.split(":")[0]
                if base not in image_digests:
                    image_digests[base] = set()
                image_digests[base].add(digest)

        # Check each base image only has one digest
        for base, digests in image_digests.items():
            if len(digests) > 1:
                pytest.fail(f"Base image '{base}' has inconsistent digests across stages: {sorted(digests)}")
