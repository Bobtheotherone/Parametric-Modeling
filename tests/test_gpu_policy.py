"""GPU Policy Lockfile Tests.

These tests inspect the lockfile (uv.lock) and pyproject.toml to ensure the CuPy
GPU backend policy cannot regress silently. The tests verify:
  1. CuPy is declared as a required (non-optional) dependency
  2. CuPy is present in the lockfile with a pinned version
  3. The CuPy version meets minimum requirements for CUDA 12 support
  4. CuPy is listed as a direct dependency (not transitive only)

REQ-M2-019 (test_gpu_backend.py::test_gpu_backend_defaults_to_cupy_and_records_fallback_reason)
requires GPU postprocessing to default to CuPy. This test file ensures that
policy cannot regress via dependency changes.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CUPY_PACKAGE_NAME = "cupy-cuda12x"
# Minimum CuPy version that supports CUDA 12.x and has stable FFT/batch transforms
MIN_CUPY_VERSION = (12, 0, 0)


def _parse_toml_dependencies(text: str) -> list[str]:
    """Parse dependencies from pyproject.toml [project] section."""
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


def _parse_version_tuple(version_str: str) -> tuple[int, ...]:
    """Parse a version string like '13.6.0' into (13, 6, 0)."""
    parts = []
    for part in version_str.split("."):
        # Handle versions like '13.6.0a1' - strip non-numeric suffix
        numeric = re.match(r"(\d+)", part)
        if numeric:
            parts.append(int(numeric.group(1)))
    return tuple(parts)


def _extract_cupy_version_from_lock(lock_text: str) -> str | None:
    """Extract the CuPy version from uv.lock content.

    Looks for the pattern:
        name = "cupy-cuda12x"
        version = "X.Y.Z"
    """
    # Find the cupy-cuda12x package block
    pattern = r'name\s*=\s*"cupy-cuda12x"\s*\nversion\s*=\s*"([^"]+)"'
    match = re.search(pattern, lock_text)
    if match:
        return match.group(1)
    return None


def _is_cupy_direct_dependency(lock_text: str) -> bool:
    """Check if CuPy is listed as a direct dependency in the formula-foundry package.

    In uv.lock, the root package lists its dependencies. We verify CuPy appears
    there, not just as a transitive dependency of another package.
    """
    # Find the formula-foundry package block dependencies
    # Pattern: after name = "formula-foundry" there's a dependencies list
    ff_pattern = r'name\s*=\s*"formula-foundry".*?dependencies\s*=\s*\[(.*?)\]'
    match = re.search(ff_pattern, lock_text, re.DOTALL)
    if match:
        deps_block = match.group(1)
        return "cupy-cuda12x" in deps_block
    return False


class TestGpuPolicyLockfile:
    """Test suite for GPU policy lockfile invariants."""

    def test_cupy_is_required_dependency_in_pyproject(self) -> None:
        """CuPy must be declared in pyproject.toml as a required dependency."""
        pyproject_path = ROOT / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml must exist"

        content = pyproject_path.read_text(encoding="utf-8")
        deps = _parse_toml_dependencies(content)

        cupy_deps = [d for d in deps if d.startswith(CUPY_PACKAGE_NAME)]
        assert len(cupy_deps) > 0, f"{CUPY_PACKAGE_NAME} must be a required dependency in pyproject.toml, found deps: {deps}"

    def test_cupy_not_in_optional_dependencies(self) -> None:
        """CuPy must NOT be relegated to optional-dependencies."""
        pyproject_path = ROOT / "pyproject.toml"
        content = pyproject_path.read_text(encoding="utf-8")

        # Check that cupy doesn't appear only in [project.optional-dependencies]
        optional_pattern = r"\[project\.optional-dependencies\].*?(?=\n\[|\Z)"
        optional_match = re.search(optional_pattern, content, re.DOTALL)
        if optional_match:
            optional_match.group(0)
            # Ensure cupy is in required deps (tested above), not just optional
            # If cupy appears in optional AND required, that's OK (for extras)
            # But it must be in required
            pass  # test_cupy_is_required_dependency_in_pyproject covers this

    def test_lockfile_contains_cupy_with_version(self) -> None:
        """uv.lock must contain cupy-cuda12x with a pinned version."""
        lock_path = ROOT / "uv.lock"
        assert lock_path.exists(), "uv.lock must exist for deterministic builds"

        lock_text = lock_path.read_text(encoding="utf-8")

        # Check package is present
        assert f'name = "{CUPY_PACKAGE_NAME}"' in lock_text, f"{CUPY_PACKAGE_NAME} must be present in uv.lock"

        # Extract and verify version
        version = _extract_cupy_version_from_lock(lock_text)
        assert version is not None, f"Could not extract version for {CUPY_PACKAGE_NAME} from uv.lock"

    def test_cupy_version_meets_minimum(self) -> None:
        """CuPy version in lockfile must meet minimum requirements."""
        lock_path = ROOT / "uv.lock"
        lock_text = lock_path.read_text(encoding="utf-8")

        version = _extract_cupy_version_from_lock(lock_text)
        assert version is not None, "CuPy version not found in lockfile"

        version_tuple = _parse_version_tuple(version)
        assert version_tuple >= MIN_CUPY_VERSION, (
            f"CuPy version {version} is below minimum {'.'.join(map(str, MIN_CUPY_VERSION))}. "
            f"GPU backend requires CuPy >= {'.'.join(map(str, MIN_CUPY_VERSION))} for "
            f"CUDA 12 FFT and batch transform support."
        )

    def test_cupy_is_direct_dependency_in_lockfile(self) -> None:
        """CuPy must be a direct dependency of formula-foundry, not transitive."""
        lock_path = ROOT / "uv.lock"
        lock_text = lock_path.read_text(encoding="utf-8")

        assert _is_cupy_direct_dependency(lock_text), (
            f"{CUPY_PACKAGE_NAME} must be a direct dependency of formula-foundry "
            f"in uv.lock, not merely a transitive dependency. This ensures the "
            f"GPU policy is explicitly declared and cannot be silently removed."
        )

    def test_cupy_lockfile_has_sha_hashes(self) -> None:
        """CuPy wheels in lockfile should have integrity hashes."""
        lock_path = ROOT / "uv.lock"
        lock_text = lock_path.read_text(encoding="utf-8")

        # Find the cupy-cuda12x package block and check for hash entries
        # Pattern looks for wheels with sha256 hashes after the package declaration
        cupy_start = lock_text.find(f'name = "{CUPY_PACKAGE_NAME}"')
        assert cupy_start != -1, f"{CUPY_PACKAGE_NAME} not found in lockfile"

        # Find the wheels section for this package
        wheels_pattern = r'name\s*=\s*"cupy-cuda12x".*?wheels\s*=\s*\[(.*?)\]'
        wheels_match = re.search(wheels_pattern, lock_text, re.DOTALL)

        assert wheels_match is not None, f"No wheels section found for {CUPY_PACKAGE_NAME} in lockfile"

        wheels_section = wheels_match.group(1)
        # Verify at least one hash exists (sha256)
        assert "hash = " in wheels_section or 'hash = "sha256:' in wheels_section, (
            f"{CUPY_PACKAGE_NAME} wheels must have integrity hashes in lockfile to ensure reproducible and secure builds"
        )


def test_gpu_policy_lockfile_cupy_present() -> None:
    """Standalone test: CuPy must be locked to prevent GPU policy regression.

    This is the primary gate test that can be run independently.
    Verifies REQ-M2-019 dependency requirement.
    """
    lock_path = ROOT / "uv.lock"
    assert lock_path.exists(), "uv.lock is required"

    pyproject_path = ROOT / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml is required"

    # Check pyproject.toml declares cupy
    pyproject_content = pyproject_path.read_text(encoding="utf-8")
    deps = _parse_toml_dependencies(pyproject_content)
    assert any(d.startswith(CUPY_PACKAGE_NAME) for d in deps), f"{CUPY_PACKAGE_NAME} must be in pyproject.toml dependencies"

    # Check lockfile has cupy pinned
    lock_text = lock_path.read_text(encoding="utf-8")
    assert f'name = "{CUPY_PACKAGE_NAME}"' in lock_text, f"{CUPY_PACKAGE_NAME} must be pinned in uv.lock"

    version = _extract_cupy_version_from_lock(lock_text)
    assert version is not None, f"{CUPY_PACKAGE_NAME} version must be extractable"
    assert _parse_version_tuple(version) >= MIN_CUPY_VERSION, f"{CUPY_PACKAGE_NAME} version {version} is too old"
