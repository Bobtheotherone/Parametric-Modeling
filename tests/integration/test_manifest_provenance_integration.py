# SPDX-License-Identifier: MIT
"""Manifest provenance integration tests (CP-1.2).

This module implements provenance verification tests:

For each golden spec (F0 and F1):
- Assert toolchain.kicad.version, toolchain.docker.image_ref, and
  toolchain.kicad.cli_version_output are present in manifests
- Verify values match expected pinned toolchain from lock file
- Ensure no 'unknown' values appear for docker builds
- Test both F0 and F1 golden specs produce complete provenance

Tests are marked with @pytest.mark.kicad_integration.
These tests require Docker with the pinned KiCad image.

Per Section 13.5.1 of the design document:
- Manifest must contain complete toolchain metadata
- For docker mode: kicad.version, kicad.cli_version_output, docker.image_ref
- No 'unknown' values allowed for docker builds (per CP-5.3)

References:
- CP-1.2: Add real integration tests (docker mode)
- CP-5.3: Ensure toolchain provenance always captured
- D5: Toolchain provenance incomplete in manifest (fix)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

# Root of the repository
REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SPECS_DIR = REPO_ROOT / "tests" / "golden_specs"
TOOLCHAIN_LOCK_PATH = REPO_ROOT / "toolchain" / "kicad.lock.json"

# Expected spec patterns (we expect at least 10 of each family per M1 requirements)
F0_PATTERN = "f0_cal_*.yaml"
F1_PATTERN = "f1_via_*.yaml"


def _load_toolchain_lock() -> dict[str, Any]:
    """Load the toolchain lock file.

    Returns:
        Parsed toolchain lock data.

    Raises:
        FileNotFoundError: If lock file doesn't exist.
        json.JSONDecodeError: If lock file is invalid JSON.
    """
    if not TOOLCHAIN_LOCK_PATH.exists():
        raise FileNotFoundError(f"Toolchain lock file not found: {TOOLCHAIN_LOCK_PATH}")
    return json.loads(TOOLCHAIN_LOCK_PATH.read_text(encoding="utf-8"))


def _collect_golden_specs() -> list[Path]:
    """Collect all golden specs for F0 and F1 families.

    Returns:
        List of paths to golden spec files (YAML only to avoid duplicates).
    """
    specs: list[Path] = []
    specs.extend(sorted(GOLDEN_SPECS_DIR.glob(F0_PATTERN)))
    specs.extend(sorted(GOLDEN_SPECS_DIR.glob(F1_PATTERN)))
    return specs


def _get_spec_ids() -> list[str]:
    """Get spec IDs for pytest parameterization."""
    return [spec.stem for spec in _collect_golden_specs()]


def _get_f0_spec_ids() -> list[str]:
    """Get F0 spec IDs for pytest parameterization."""
    return [spec.stem for spec in GOLDEN_SPECS_DIR.glob(F0_PATTERN)]


def _get_f1_spec_ids() -> list[str]:
    """Get F1 spec IDs for pytest parameterization."""
    return [spec.stem for spec in GOLDEN_SPECS_DIR.glob(F1_PATTERN)]


@pytest.fixture(scope="module")
def docker_available() -> bool:
    """Check if Docker is available on the system."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.fixture(scope="module")
def kicad_image_available(docker_available: bool) -> bool:
    """Check if the KiCad Docker image is available.

    This pulls the image if not present (may take time on first run).
    """
    if not docker_available:
        return False

    # Check for kicad/kicad:9.0.7 image
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "kicad/kicad:9.0.7"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.fixture(scope="module")
def toolchain_lock() -> dict[str, Any]:
    """Load the toolchain lock file for verification."""
    return _load_toolchain_lock()


def _run_coupongen_build(
    spec_path: Path,
    output_dir: Path,
    mode: str = "docker",
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    """Run coupongen build command for a spec file.

    Args:
        spec_path: Path to the spec YAML/JSON file.
        output_dir: Directory for build outputs.
        mode: KiCad mode ("local" or "docker").
        timeout: Command timeout in seconds.

    Returns:
        CompletedProcess with exit code, stdout, stderr.
    """
    cmd = [
        "coupongen",
        "build",
        str(spec_path),
        "--out",
        str(output_dir),
        "--mode",
        mode,
    ]

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )


def _find_manifest_json(output_dir: Path) -> Path | None:
    """Find the manifest.json file in the build output directory.

    The build creates a subdirectory named <coupon_id>-<design_hash>/
    and places manifest.json there.

    Args:
        output_dir: Root output directory passed to coupongen build.

    Returns:
        Path to manifest.json if found, None otherwise.
    """
    manifest_files = list(output_dir.rglob("manifest.json"))
    if manifest_files:
        return manifest_files[0]
    return None


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and parse a manifest.json file.

    Args:
        manifest_path: Path to the manifest.json file.

    Returns:
        Parsed manifest as a dictionary.

    Raises:
        FileNotFoundError: If manifest doesn't exist.
        json.JSONDecodeError: If manifest is invalid JSON.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


class TestToolchainLockPresent:
    """Verify that the toolchain lock file exists and is valid."""

    def test_lock_file_exists(self) -> None:
        """REQ-CP1-1: Toolchain lock file must exist."""
        assert TOOLCHAIN_LOCK_PATH.exists(), (
            f"Toolchain lock file not found at {TOOLCHAIN_LOCK_PATH}"
        )

    def test_lock_file_valid_json(self) -> None:
        """REQ-CP1-1: Toolchain lock file must be valid JSON."""
        lock_data = _load_toolchain_lock()
        assert isinstance(lock_data, dict), "Lock file must be a JSON object"

    def test_lock_file_has_required_fields(self) -> None:
        """REQ-CP1-1: Lock file must contain required fields."""
        lock_data = _load_toolchain_lock()

        required_fields = ["kicad_version", "docker_image", "docker_digest"]
        for field in required_fields:
            assert field in lock_data, f"Lock file missing required field: {field}"
            assert lock_data[field], f"Lock file field {field} must not be empty"

        docker_digest = lock_data["docker_digest"]
        assert "PLACEHOLDER" not in str(docker_digest).upper(), "docker_digest must be resolved"
        assert isinstance(docker_digest, str)
        assert len(docker_digest) == 71, "docker_digest must be sha256:<64-hex>"
        assert docker_digest.startswith("sha256:")


@pytest.mark.kicad_integration
class TestManifestProvenanceIntegration:
    """Integration tests for manifest provenance verification (CP-1.2).

    These tests verify that for each golden spec:
    1. Manifest contains complete toolchain metadata
    2. toolchain.kicad.version is present and matches lock file
    3. toolchain.docker.image_ref is present for docker builds
    4. toolchain.kicad.cli_version_output is present
    5. No 'unknown' values appear for docker builds

    Requirements:
    - Docker must be installed and running
    - KiCad Docker image (kicad/kicad:9.0.7) must be available
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_docker(
        self,
        docker_available: bool,
        kicad_image_available: bool,
    ) -> None:
        """Skip tests if Docker or KiCad image is not available."""
        if not docker_available:
            pytest.skip("Docker is not available")
        if not kicad_image_available:
            pytest.skip("KiCad Docker image not available (kicad/kicad:9.0.7)")

    @pytest.mark.parametrize(
        "spec_name",
        _get_spec_ids(),
        ids=_get_spec_ids(),
    )
    def test_manifest_toolchain_present(
        self,
        spec_name: str,
        tmp_path: Path,
        toolchain_lock: dict[str, Any],
    ) -> None:
        """Verify toolchain metadata is present in manifest.

        Per CP-1.2 and Section 13.5.1:
        - toolchain.kicad.version must be present
        - toolchain.docker.image_ref must be present for docker builds
        - toolchain.kicad.cli_version_output must be present
        - toolchain.mode must be "docker"

        Args:
            spec_name: Name of the golden spec file (without extension).
            tmp_path: Pytest temporary directory fixture.
            toolchain_lock: Loaded toolchain lock data.
        """
        # Find the spec file (prefer YAML)
        spec_path = GOLDEN_SPECS_DIR / f"{spec_name}.yaml"
        if not spec_path.exists():
            spec_path = GOLDEN_SPECS_DIR / f"{spec_name}.json"

        assert spec_path.exists(), f"Golden spec not found: {spec_name}"

        # Create output directory
        output_dir = tmp_path / "build_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run coupongen build with docker mode
        result = _run_coupongen_build(
            spec_path=spec_path,
            output_dir=output_dir,
            mode="docker",
            timeout=180,
        )

        # Check build succeeded
        assert result.returncode == 0, (
            f"coupongen build failed for {spec_name}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Find and load manifest
        manifest_path = _find_manifest_json(output_dir)
        assert manifest_path is not None, (
            f"Manifest not found for {spec_name} in {output_dir}"
        )

        manifest = _load_manifest(manifest_path)

        # Verify toolchain is present
        assert "toolchain" in manifest, "Manifest must contain 'toolchain' field"
        toolchain = manifest["toolchain"]

        # Verify toolchain.kicad.version is present
        assert "kicad" in toolchain, "Manifest toolchain must contain 'kicad' field"
        kicad_info = toolchain["kicad"]
        assert "version" in kicad_info, (
            "Manifest toolchain.kicad must contain 'version' field"
        )

        # Verify toolchain.mode is docker
        assert "mode" in toolchain, "Manifest toolchain must contain 'mode' field"
        assert toolchain["mode"] == "docker", (
            f"Expected mode='docker', got mode='{toolchain['mode']}'"
        )

        # Verify toolchain.docker.image_ref is present for docker mode
        assert "docker" in toolchain, (
            "Manifest toolchain must contain 'docker' field for docker builds"
        )
        docker_info = toolchain["docker"]
        assert "image_ref" in docker_info, (
            "Manifest toolchain.docker must contain 'image_ref' field"
        )

        # Verify toolchain.kicad.cli_version_output is present
        assert "cli_version_output" in kicad_info, (
            "Manifest toolchain.kicad must contain 'cli_version_output' field"
        )

    @pytest.mark.parametrize(
        "spec_name",
        _get_spec_ids(),
        ids=_get_spec_ids(),
    )
    def test_manifest_toolchain_matches_lock(
        self,
        spec_name: str,
        tmp_path: Path,
        toolchain_lock: dict[str, Any],
    ) -> None:
        """Verify toolchain values match pinned toolchain from lock file.

        Per CP-1.2:
        - toolchain.kicad.version must match lock file kicad_version
        - toolchain.docker.image_ref must reference the pinned docker image

        Args:
            spec_name: Name of the golden spec file (without extension).
            tmp_path: Pytest temporary directory fixture.
            toolchain_lock: Loaded toolchain lock data.
        """
        # Find the spec file
        spec_path = GOLDEN_SPECS_DIR / f"{spec_name}.yaml"
        if not spec_path.exists():
            spec_path = GOLDEN_SPECS_DIR / f"{spec_name}.json"

        assert spec_path.exists(), f"Golden spec not found: {spec_name}"

        # Create output directory
        output_dir = tmp_path / "build_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run coupongen build
        result = _run_coupongen_build(
            spec_path=spec_path,
            output_dir=output_dir,
            mode="docker",
            timeout=180,
        )

        assert result.returncode == 0, (
            f"coupongen build failed for {spec_name}\n"
            f"stderr: {result.stderr}"
        )

        # Find and load manifest
        manifest_path = _find_manifest_json(output_dir)
        assert manifest_path is not None, f"Manifest not found for {spec_name}"

        manifest = _load_manifest(manifest_path)
        toolchain = manifest["toolchain"]

        # Verify kicad version matches lock file
        expected_version = toolchain_lock["kicad_version"]
        actual_version = toolchain["kicad"]["version"]
        assert actual_version == expected_version, (
            f"Manifest kicad.version ({actual_version}) does not match "
            f"lock file ({expected_version})"
        )

        # Verify docker image reference contains the expected image
        expected_image = toolchain_lock["docker_image"]
        actual_image_ref = toolchain["docker"]["image_ref"]
        assert expected_image in actual_image_ref, (
            f"Manifest docker.image_ref ({actual_image_ref}) does not reference "
            f"expected image ({expected_image})"
        )

    @pytest.mark.parametrize(
        "spec_name",
        _get_spec_ids(),
        ids=_get_spec_ids(),
    )
    def test_manifest_no_unknown_values(
        self,
        spec_name: str,
        tmp_path: Path,
    ) -> None:
        """Verify no 'unknown' values appear in manifest for docker builds.

        Per CP-5.3 (D5 fix):
        - Docker builds must never have 'unknown' values for toolchain fields
        - kicad.version must not be 'unknown'
        - kicad.cli_version_output must not be 'unknown'
        - docker.image_ref must not be 'unknown'

        Args:
            spec_name: Name of the golden spec file (without extension).
            tmp_path: Pytest temporary directory fixture.
        """
        # Find the spec file
        spec_path = GOLDEN_SPECS_DIR / f"{spec_name}.yaml"
        if not spec_path.exists():
            spec_path = GOLDEN_SPECS_DIR / f"{spec_name}.json"

        assert spec_path.exists(), f"Golden spec not found: {spec_name}"

        # Create output directory
        output_dir = tmp_path / "build_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run coupongen build
        result = _run_coupongen_build(
            spec_path=spec_path,
            output_dir=output_dir,
            mode="docker",
            timeout=180,
        )

        assert result.returncode == 0, (
            f"coupongen build failed for {spec_name}\n"
            f"stderr: {result.stderr}"
        )

        # Find and load manifest
        manifest_path = _find_manifest_json(output_dir)
        assert manifest_path is not None, f"Manifest not found for {spec_name}"

        manifest = _load_manifest(manifest_path)
        toolchain = manifest["toolchain"]

        # Check kicad.version is not 'unknown'
        kicad_version = toolchain["kicad"]["version"]
        assert kicad_version != "unknown", (
            "Docker builds must not have 'unknown' kicad.version"
        )
        assert kicad_version, "kicad.version must not be empty"

        # Check kicad.cli_version_output is not 'unknown'
        cli_version = toolchain["kicad"]["cli_version_output"]
        assert cli_version != "unknown", (
            "Docker builds must not have 'unknown' kicad.cli_version_output"
        )
        assert cli_version, "kicad.cli_version_output must not be empty"

        # Check docker.image_ref is not 'unknown'
        image_ref = toolchain["docker"]["image_ref"]
        assert image_ref != "unknown", (
            "Docker builds must not have 'unknown' docker.image_ref"
        )
        assert image_ref, "docker.image_ref must not be empty"


@pytest.mark.kicad_integration
class TestF0ProvenanceComplete:
    """Test that F0 golden specs produce complete provenance."""

    @pytest.fixture(autouse=True)
    def skip_if_no_docker(
        self,
        docker_available: bool,
        kicad_image_available: bool,
    ) -> None:
        """Skip tests if Docker or KiCad image is not available."""
        if not docker_available:
            pytest.skip("Docker is not available")
        if not kicad_image_available:
            pytest.skip("KiCad Docker image not available (kicad/kicad:9.0.7)")

    def test_f0_first_spec_complete_provenance(
        self,
        tmp_path: Path,
        toolchain_lock: dict[str, Any],
    ) -> None:
        """Verify first F0 spec produces complete provenance.

        Tests comprehensive provenance fields:
        - schema_version, coupon_family, design_hash, coupon_id
        - resolved_design, derived_features, dimensionless_groups
        - fab_profile, stackup
        - toolchain (complete with kicad + docker)
        - toolchain_hash
        - exports list with hashes
        - verification (constraints + drc)
        - lineage (git_sha + timestamp_utc)
        """
        f0_specs = sorted(GOLDEN_SPECS_DIR.glob(F0_PATTERN))
        if not f0_specs:
            pytest.skip("No F0 golden specs available")

        spec_path = f0_specs[0]
        output_dir = tmp_path / "build_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = _run_coupongen_build(
            spec_path=spec_path,
            output_dir=output_dir,
            mode="docker",
            timeout=180,
        )

        assert result.returncode == 0, f"Build failed: {result.stderr}"

        manifest_path = _find_manifest_json(output_dir)
        assert manifest_path is not None, "Manifest not found"

        manifest = _load_manifest(manifest_path)

        # Check all required top-level fields per manifest schema
        required_fields = [
            "schema_version",
            "coupon_family",
            "design_hash",
            "coupon_id",
            "resolved_design",
            "derived_features",
            "dimensionless_groups",
            "fab_profile",
            "stackup",
            "toolchain",
            "toolchain_hash",
            "exports",
            "verification",
            "lineage",
        ]

        for field in required_fields:
            assert field in manifest, f"Manifest missing required field: {field}"

        # Verify coupon_family is F0
        assert manifest["coupon_family"].startswith("F0"), (
            f"Expected F0 family, got {manifest['coupon_family']}"
        )

        # Verify design_hash format (64-char hex)
        assert len(manifest["design_hash"]) == 64, "design_hash must be 64 hex chars"
        assert all(c in "0123456789abcdef" for c in manifest["design_hash"]), (
            "design_hash must be lowercase hex"
        )

        # Verify toolchain_hash format (64-char hex)
        assert len(manifest["toolchain_hash"]) == 64, (
            "toolchain_hash must be 64 hex chars"
        )

        # Verify exports is a list with entries
        assert isinstance(manifest["exports"], list), "exports must be a list"
        assert len(manifest["exports"]) > 0, "exports must not be empty"

        # Verify each export has path and hash
        for export in manifest["exports"]:
            assert "path" in export, "Export entry must have 'path'"
            assert "hash" in export, "Export entry must have 'hash'"
            assert len(export["hash"]) == 64, "Export hash must be 64 hex chars"

        # Verify verification structure
        verification = manifest["verification"]
        assert "constraints" in verification, "verification must have 'constraints'"
        assert "drc" in verification, "verification must have 'drc'"
        assert verification["constraints"]["passed"] is True, (
            "Golden spec constraints must pass"
        )
        assert verification["drc"]["returncode"] == 0, "Golden spec DRC must pass"

        # Verify lineage structure
        lineage = manifest["lineage"]
        assert "git_sha" in lineage, "lineage must have 'git_sha'"
        assert "timestamp_utc" in lineage, "lineage must have 'timestamp_utc'"
        assert len(lineage["git_sha"]) == 40, "git_sha must be 40 hex chars"


@pytest.mark.kicad_integration
class TestF1ProvenanceComplete:
    """Test that F1 golden specs produce complete provenance."""

    @pytest.fixture(autouse=True)
    def skip_if_no_docker(
        self,
        docker_available: bool,
        kicad_image_available: bool,
    ) -> None:
        """Skip tests if Docker or KiCad image is not available."""
        if not docker_available:
            pytest.skip("Docker is not available")
        if not kicad_image_available:
            pytest.skip("KiCad Docker image not available (kicad/kicad:9.0.7)")

    def test_f1_first_spec_complete_provenance(
        self,
        tmp_path: Path,
        toolchain_lock: dict[str, Any],
    ) -> None:
        """Verify first F1 spec produces complete provenance.

        Tests comprehensive provenance fields:
        - schema_version, coupon_family, design_hash, coupon_id
        - resolved_design, derived_features, dimensionless_groups
        - fab_profile, stackup
        - toolchain (complete with kicad + docker)
        - toolchain_hash
        - exports list with hashes
        - verification (constraints + drc)
        - lineage (git_sha + timestamp_utc)
        """
        f1_specs = sorted(GOLDEN_SPECS_DIR.glob(F1_PATTERN))
        if not f1_specs:
            pytest.skip("No F1 golden specs available")

        spec_path = f1_specs[0]
        output_dir = tmp_path / "build_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = _run_coupongen_build(
            spec_path=spec_path,
            output_dir=output_dir,
            mode="docker",
            timeout=180,
        )

        assert result.returncode == 0, f"Build failed: {result.stderr}"

        manifest_path = _find_manifest_json(output_dir)
        assert manifest_path is not None, "Manifest not found"

        manifest = _load_manifest(manifest_path)

        # Check all required top-level fields per manifest schema
        required_fields = [
            "schema_version",
            "coupon_family",
            "design_hash",
            "coupon_id",
            "resolved_design",
            "derived_features",
            "dimensionless_groups",
            "fab_profile",
            "stackup",
            "toolchain",
            "toolchain_hash",
            "exports",
            "verification",
            "lineage",
        ]

        for field in required_fields:
            assert field in manifest, f"Manifest missing required field: {field}"

        # Verify coupon_family is F1
        assert manifest["coupon_family"].startswith("F1"), (
            f"Expected F1 family, got {manifest['coupon_family']}"
        )

        # Verify design_hash format (64-char hex)
        assert len(manifest["design_hash"]) == 64, "design_hash must be 64 hex chars"
        assert all(c in "0123456789abcdef" for c in manifest["design_hash"]), (
            "design_hash must be lowercase hex"
        )

        # Verify toolchain_hash format (64-char hex)
        assert len(manifest["toolchain_hash"]) == 64, (
            "toolchain_hash must be 64 hex chars"
        )

        # Verify exports is a list with entries
        assert isinstance(manifest["exports"], list), "exports must be a list"
        assert len(manifest["exports"]) > 0, "exports must not be empty"

        # Verify each export has path and hash
        for export in manifest["exports"]:
            assert "path" in export, "Export entry must have 'path'"
            assert "hash" in export, "Export entry must have 'hash'"
            assert len(export["hash"]) == 64, "Export hash must be 64 hex chars"

        # Verify verification structure
        verification = manifest["verification"]
        assert "constraints" in verification, "verification must have 'constraints'"
        assert "drc" in verification, "verification must have 'drc'"
        assert verification["constraints"]["passed"] is True, (
            "Golden spec constraints must pass"
        )
        assert verification["drc"]["returncode"] == 0, "Golden spec DRC must pass"

        # Verify lineage structure
        lineage = manifest["lineage"]
        assert "git_sha" in lineage, "lineage must have 'git_sha'"
        assert "timestamp_utc" in lineage, "lineage must have 'timestamp_utc'"
        assert len(lineage["git_sha"]) == 40, "git_sha must be 40 hex chars"


@pytest.mark.kicad_integration
class TestToolchainHashConsistency:
    """Test toolchain_hash consistency across builds."""

    @pytest.fixture(autouse=True)
    def skip_if_no_docker(
        self,
        docker_available: bool,
        kicad_image_available: bool,
    ) -> None:
        """Skip tests if Docker or KiCad image is not available."""
        if not docker_available:
            pytest.skip("Docker is not available")
        if not kicad_image_available:
            pytest.skip("KiCad Docker image not available (kicad/kicad:9.0.7)")

    def test_toolchain_hash_matches_across_builds(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify toolchain_hash is consistent across different spec builds.

        The toolchain_hash should be deterministic based on the toolchain
        configuration, not the specific spec being built.
        """
        specs = _collect_golden_specs()
        if len(specs) < 2:
            pytest.skip("Need at least 2 golden specs for comparison")

        toolchain_hashes: list[str] = []

        for i, spec_path in enumerate(specs[:2]):
            output_dir = tmp_path / f"build_output_{i}"
            output_dir.mkdir(parents=True, exist_ok=True)

            result = _run_coupongen_build(
                spec_path=spec_path,
                output_dir=output_dir,
                mode="docker",
                timeout=180,
            )

            assert result.returncode == 0, (
                f"Build failed for {spec_path.name}: {result.stderr}"
            )

            manifest_path = _find_manifest_json(output_dir)
            assert manifest_path is not None, f"Manifest not found for {spec_path.name}"

            manifest = _load_manifest(manifest_path)
            toolchain_hashes.append(manifest["toolchain_hash"])

        # All toolchain hashes should be the same for same toolchain config
        assert len(set(toolchain_hashes)) == 1, (
            f"Toolchain hash inconsistent across builds: {toolchain_hashes}"
        )
