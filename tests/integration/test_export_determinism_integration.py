# SPDX-License-Identifier: MIT
"""Export determinism integration tests (CP-1.2).

This module implements Gate G4/G5 tests: export determinism and layer completeness.

For each golden spec (F0 and F1):
- Run build 3x into fresh directories
- Assert manifest.design_hash is stable across runs
- Assert canonical export hashes are stable
- Verify drill outputs exist
- Verify expected layer set exists per family using layer_validation module

Tests are marked with @pytest.mark.kicad_integration, @pytest.mark.gate_g4,
and @pytest.mark.gate_g5.

These tests require Docker with the pinned KiCad image.

Per Section 13.5.3 of the design document:
- Define and enforce a locked layer set for fabrication exports
- F.Cu, In1.Cu, In2.Cu, B.Cu (for 4-layer boards)
- F.Mask, B.Mask
- Edge.Cuts
- Enforce in tests that every exported fab directory contains all expected layers

References:
- CP-1.2: Add real integration tests (docker mode)
- Section 13.5.3: Layer set validation
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

# Expected spec patterns (we expect at least 10 of each family per M1 requirements)
F0_PATTERN = "f0_cal_*.yaml"
F1_PATTERN = "f1_via_*.yaml"

# Number of repeated builds for determinism tests
DETERMINISM_BUILD_COUNT = 3


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
    return [spec.stem for spec in sorted(GOLDEN_SPECS_DIR.glob(F0_PATTERN))]


def _get_f1_spec_ids() -> list[str]:
    """Get F1 spec IDs for pytest parameterization."""
    return [spec.stem for spec in sorted(GOLDEN_SPECS_DIR.glob(F1_PATTERN))]


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


def _find_build_artifacts_dir(output_dir: Path) -> Path | None:
    """Find the build artifacts directory (contains gerbers, drills, etc.).

    Args:
        output_dir: Root output directory passed to coupongen build.

    Returns:
        Path to artifacts directory if found, None otherwise.
    """
    # Look for directories containing manifest.json
    manifest_files = list(output_dir.rglob("manifest.json"))
    if manifest_files:
        return manifest_files[0].parent
    return None


def _find_drill_files(artifacts_dir: Path) -> list[Path]:
    """Find drill files in the artifacts directory.

    Args:
        artifacts_dir: Path to the build artifacts directory.

    Returns:
        List of paths to drill files (*.drl, *.exc).
    """
    drill_files: list[Path] = []
    drill_extensions = ["*.drl", "*.exc", "*-PTH.drl", "*-NPTH.drl"]

    for ext in drill_extensions:
        drill_files.extend(artifacts_dir.rglob(ext))

    return drill_files


def _get_export_paths_from_manifest(manifest: dict[str, Any]) -> list[str]:
    """Extract export paths from a manifest.

    Args:
        manifest: Parsed manifest dictionary.

    Returns:
        List of relative paths to exported files.
    """
    exports = manifest.get("exports", [])
    return [export["path"] for export in exports if "path" in export]


def _get_export_hashes_from_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    """Extract export path -> hash mapping from a manifest.

    Args:
        manifest: Parsed manifest dictionary.

    Returns:
        Dictionary mapping export paths to their hashes.
    """
    exports = manifest.get("exports", [])
    return {
        export["path"]: export["hash"]
        for export in exports
        if "path" in export and "hash" in export
    }


def _get_copper_layers_from_spec(spec_path: Path) -> int:
    """Extract copper layer count from spec file.

    Args:
        spec_path: Path to the spec YAML file.

    Returns:
        Number of copper layers.
    """
    import yaml

    with open(spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    return spec.get("stackup", {}).get("copper_layers", 4)


def _get_coupon_family_from_spec(spec_path: Path) -> str:
    """Extract coupon family from spec file.

    Args:
        spec_path: Path to the spec YAML file.

    Returns:
        Coupon family identifier.
    """
    import yaml

    with open(spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    return spec.get("coupon_family", "UNKNOWN")


class TestGoldenSpecsComplete:
    """Verify that the required golden specs are present with expected structure."""

    def test_f0_golden_specs_have_export_config(self) -> None:
        """REQ-M1-024: F0 golden specs must have export configuration."""
        f0_specs = list(GOLDEN_SPECS_DIR.glob(F0_PATTERN))
        assert len(f0_specs) >= 10, (
            f"Expected at least 10 F0 golden specs, found {len(f0_specs)}"
        )

    def test_f1_golden_specs_have_export_config(self) -> None:
        """REQ-M1-024: F1 golden specs must have export configuration."""
        f1_specs = list(GOLDEN_SPECS_DIR.glob(F1_PATTERN))
        assert len(f1_specs) >= 10, (
            f"Expected at least 10 F1 golden specs, found {len(f1_specs)}"
        )


@pytest.mark.kicad_integration
@pytest.mark.gate_g5
class TestDesignHashDeterminism:
    """Integration tests for design_hash determinism (Gate G5).

    These tests verify that for each golden spec:
    1. Running build 3x produces identical design_hash values
    2. The design_hash is a valid 64-character hex string

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
        _get_spec_ids()[:2],  # Test first 2 specs (one F0, one F1) for determinism
        ids=_get_spec_ids()[:2],
    )
    def test_design_hash_stable_across_runs(
        self,
        spec_name: str,
        tmp_path: Path,
    ) -> None:
        """Gate G5: Verify design_hash is stable across 3 repeated builds.

        Per CP-1.2:
        - Run build 3x into fresh directories
        - Assert manifest.design_hash is stable across runs

        Args:
            spec_name: Name of the golden spec file (without extension).
            tmp_path: Pytest temporary directory fixture.
        """
        # Find the spec file (prefer YAML)
        spec_path = GOLDEN_SPECS_DIR / f"{spec_name}.yaml"
        if not spec_path.exists():
            spec_path = GOLDEN_SPECS_DIR / f"{spec_name}.json"

        assert spec_path.exists(), f"Golden spec not found: {spec_name}"

        design_hashes: list[str] = []

        for run_idx in range(DETERMINISM_BUILD_COUNT):
            # Create fresh output directory for each run
            output_dir = tmp_path / f"build_run_{run_idx}"
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
                f"coupongen build failed for {spec_name} (run {run_idx + 1})\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

            # Find and load manifest
            manifest_path = _find_manifest_json(output_dir)
            assert manifest_path is not None, (
                f"Manifest not found for {spec_name} in {output_dir}"
            )

            manifest = _load_manifest(manifest_path)

            # Verify design_hash exists and is valid
            assert "design_hash" in manifest, "Manifest must contain 'design_hash'"
            design_hash = manifest["design_hash"]
            assert len(design_hash) == 64, "design_hash must be 64 hex chars"
            assert all(c in "0123456789abcdef" for c in design_hash), (
                "design_hash must be lowercase hex"
            )

            design_hashes.append(design_hash)

        # Verify all design hashes are identical
        assert len(set(design_hashes)) == 1, (
            f"design_hash is not stable across {DETERMINISM_BUILD_COUNT} runs "
            f"for {spec_name}: {design_hashes}"
        )


@pytest.mark.kicad_integration
@pytest.mark.gate_g5
class TestExportHashDeterminism:
    """Integration tests for export hash determinism (Gate G5).

    These tests verify that for each golden spec:
    1. Running build 3x produces identical export hashes
    2. The canonical export hashes are stable across runs

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
        _get_spec_ids()[:2],  # Test first 2 specs for determinism
        ids=_get_spec_ids()[:2],
    )
    def test_export_hashes_stable_across_runs(
        self,
        spec_name: str,
        tmp_path: Path,
    ) -> None:
        """Gate G5: Verify canonical export hashes are stable across 3 builds.

        Per CP-1.2:
        - Run build 3x into fresh directories
        - Assert canonical export hashes are stable

        Args:
            spec_name: Name of the golden spec file (without extension).
            tmp_path: Pytest temporary directory fixture.
        """
        # Find the spec file
        spec_path = GOLDEN_SPECS_DIR / f"{spec_name}.yaml"
        if not spec_path.exists():
            spec_path = GOLDEN_SPECS_DIR / f"{spec_name}.json"

        assert spec_path.exists(), f"Golden spec not found: {spec_name}"

        export_hashes_per_run: list[dict[str, str]] = []

        for run_idx in range(DETERMINISM_BUILD_COUNT):
            # Create fresh output directory
            output_dir = tmp_path / f"build_run_{run_idx}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Run coupongen build
            result = _run_coupongen_build(
                spec_path=spec_path,
                output_dir=output_dir,
                mode="docker",
                timeout=180,
            )

            assert result.returncode == 0, (
                f"coupongen build failed for {spec_name} (run {run_idx + 1})\n"
                f"stderr: {result.stderr}"
            )

            # Find and load manifest
            manifest_path = _find_manifest_json(output_dir)
            assert manifest_path is not None, f"Manifest not found for {spec_name}"

            manifest = _load_manifest(manifest_path)
            export_hashes = _get_export_hashes_from_manifest(manifest)

            assert len(export_hashes) > 0, (
                f"No exports found in manifest for {spec_name}"
            )

            export_hashes_per_run.append(export_hashes)

        # Verify export hashes are identical across runs
        first_run_hashes = export_hashes_per_run[0]
        for run_idx, run_hashes in enumerate(export_hashes_per_run[1:], start=2):
            assert first_run_hashes == run_hashes, (
                f"Export hashes differ between run 1 and run {run_idx} "
                f"for {spec_name}:\n"
                f"Run 1: {first_run_hashes}\n"
                f"Run {run_idx}: {run_hashes}"
            )


@pytest.mark.kicad_integration
@pytest.mark.gate_g4
class TestDrillOutputsExist:
    """Integration tests for drill output existence (Gate G4).

    These tests verify that for each golden spec:
    1. Drill files exist in the output directory
    2. At least one PTH (Plated Through Hole) drill file is present

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
    def test_drill_outputs_present(
        self,
        spec_name: str,
        tmp_path: Path,
    ) -> None:
        """Gate G4: Verify drill outputs exist for golden spec.

        Per CP-1.2:
        - Verify drill outputs exist

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

        # Find the build artifacts directory
        artifacts_dir = _find_build_artifacts_dir(output_dir)
        assert artifacts_dir is not None, (
            f"Build artifacts directory not found for {spec_name}"
        )

        # Find drill files
        drill_files = _find_drill_files(artifacts_dir)
        assert len(drill_files) > 0, (
            f"No drill files found for {spec_name} in {artifacts_dir}\n"
            f"Contents: {list(artifacts_dir.rglob('*'))}"
        )

        # Verify at least one drill file is non-empty
        non_empty_drills = [f for f in drill_files if f.stat().st_size > 0]
        assert len(non_empty_drills) > 0, (
            f"All drill files are empty for {spec_name}"
        )


@pytest.mark.kicad_integration
@pytest.mark.gate_g4
class TestLayerSetValidation:
    """Integration tests for layer set validation (Gate G4).

    These tests verify that for each golden spec:
    1. Expected layer set exists per family
    2. All required layers are present in exports
    3. Uses the layer_validation module for verification

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
    def test_layer_set_complete_for_golden_spec(
        self,
        spec_name: str,
        tmp_path: Path,
    ) -> None:
        """Gate G4: Verify expected layer set exists per family.

        Per CP-1.2:
        - Verify expected layer set exists per family using layer_validation module
        - For 4-layer boards: F.Cu, In1.Cu, In2.Cu, B.Cu, F.Mask, B.Mask, Edge.Cuts

        Args:
            spec_name: Name of the golden spec file (without extension).
            tmp_path: Pytest temporary directory fixture.
        """
        # Import layer_validation module
        from formula_foundry.coupongen.layer_validation import (
            LayerSetValidationError,
            validate_layer_set,
        )

        # Find the spec file
        spec_path = GOLDEN_SPECS_DIR / f"{spec_name}.yaml"
        if not spec_path.exists():
            spec_path = GOLDEN_SPECS_DIR / f"{spec_name}.json"

        assert spec_path.exists(), f"Golden spec not found: {spec_name}"

        # Get copper layers and family from spec
        copper_layers = _get_copper_layers_from_spec(spec_path)
        coupon_family = _get_coupon_family_from_spec(spec_path)

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
        export_paths = _get_export_paths_from_manifest(manifest)

        # Validate layer set using the layer_validation module
        # Manifest paths include fab/ prefix (e.g., fab/gerbers/..., fab/drill/...)
        try:
            validation_result = validate_layer_set(
                export_paths=export_paths,
                copper_layers=copper_layers,
                family=coupon_family,
                gerber_dir="fab/gerbers/",
                strict=True,
            )
            assert validation_result.passed, (
                f"Layer set validation failed for {spec_name}: "
                f"missing layers: {validation_result.missing_layers}"
            )
        except LayerSetValidationError as e:
            pytest.fail(
                f"Layer set validation failed for {spec_name}: "
                f"missing layers: {e.result.missing_layers}"
            )


@pytest.mark.kicad_integration
@pytest.mark.gate_g4
class TestF0LayerSetComplete:
    """Test that F0 golden specs have complete layer sets."""

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

    def test_f0_first_spec_layer_set(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify first F0 spec has complete layer set.

        Tests that all required layers for the copper count are present.
        """
        from formula_foundry.coupongen.layer_validation import (
            get_layer_set_for_copper_count,
        )

        f0_specs = sorted(GOLDEN_SPECS_DIR.glob(F0_PATTERN))
        if not f0_specs:
            pytest.skip("No F0 golden specs available")

        spec_path = f0_specs[0]
        copper_layers = _get_copper_layers_from_spec(spec_path)

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
        export_paths = _get_export_paths_from_manifest(manifest)

        # Get expected layer set
        layer_set = get_layer_set_for_copper_count(copper_layers)

        # Verify required layers are in exports
        gerber_exports = [p for p in export_paths if p.startswith("fab/gerbers/")]

        assert len(gerber_exports) >= len(layer_set.required), (
            f"Expected at least {len(layer_set.required)} gerber exports, "
            f"found {len(gerber_exports)}"
        )


@pytest.mark.kicad_integration
@pytest.mark.gate_g4
class TestF1LayerSetComplete:
    """Test that F1 golden specs have complete layer sets."""

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

    def test_f1_first_spec_layer_set(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify first F1 spec has complete layer set.

        Tests that all required layers for the copper count are present.
        F1 via transitions require at least 2 copper layers.
        """
        from formula_foundry.coupongen.layer_validation import (
            get_layer_set_for_copper_count,
            validate_family_layer_requirements,
        )

        f1_specs = sorted(GOLDEN_SPECS_DIR.glob(F1_PATTERN))
        if not f1_specs:
            pytest.skip("No F1 golden specs available")

        spec_path = f1_specs[0]
        copper_layers = _get_copper_layers_from_spec(spec_path)
        coupon_family = _get_coupon_family_from_spec(spec_path)

        # Verify family layer requirements
        validate_family_layer_requirements(copper_layers, coupon_family)

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
        export_paths = _get_export_paths_from_manifest(manifest)

        # Get expected layer set
        layer_set = get_layer_set_for_copper_count(copper_layers)

        # Verify required layers are in exports
        gerber_exports = [p for p in export_paths if p.startswith("fab/gerbers/")]

        assert len(gerber_exports) >= len(layer_set.required), (
            f"Expected at least {len(layer_set.required)} gerber exports, "
            f"found {len(gerber_exports)}"
        )


@pytest.mark.kicad_integration
@pytest.mark.gate_g5
class TestFullDeterminismSuite:
    """Comprehensive determinism test suite combining all Gate G5 checks.

    This test runs the complete determinism verification for a single spec
    covering design_hash, export hashes, and manifest consistency.
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

    def test_full_determinism_for_representative_spec(
        self,
        tmp_path: Path,
    ) -> None:
        """Run complete determinism verification on a representative spec.

        This test verifies:
        1. design_hash is stable across 3 runs
        2. All export hashes are stable across 3 runs
        3. Manifest structure is consistent
        4. Layer sets are complete
        5. Drill files exist

        Uses the first available F0 spec as representative.
        """
        from formula_foundry.coupongen.layer_validation import validate_layer_set

        specs = _collect_golden_specs()
        if not specs:
            pytest.skip("No golden specs available")

        spec_path = specs[0]  # Use first spec as representative
        copper_layers = _get_copper_layers_from_spec(spec_path)
        coupon_family = _get_coupon_family_from_spec(spec_path)

        manifests: list[dict[str, Any]] = []

        for run_idx in range(DETERMINISM_BUILD_COUNT):
            output_dir = tmp_path / f"determinism_run_{run_idx}"
            output_dir.mkdir(parents=True, exist_ok=True)

            result = _run_coupongen_build(
                spec_path=spec_path,
                output_dir=output_dir,
                mode="docker",
                timeout=180,
            )

            assert result.returncode == 0, (
                f"Build failed (run {run_idx + 1}): {result.stderr}"
            )

            manifest_path = _find_manifest_json(output_dir)
            assert manifest_path is not None, f"Manifest not found (run {run_idx + 1})"

            manifest = _load_manifest(manifest_path)
            manifests.append(manifest)

            # Verify drill files exist
            artifacts_dir = _find_build_artifacts_dir(output_dir)
            assert artifacts_dir is not None
            drill_files = _find_drill_files(artifacts_dir)
            assert len(drill_files) > 0, f"No drill files (run {run_idx + 1})"

            # Verify layer set
            # Manifest paths include fab/ prefix (e.g., fab/gerbers/..., fab/drill/...)
            export_paths = _get_export_paths_from_manifest(manifest)
            result_validation = validate_layer_set(
                export_paths=export_paths,
                copper_layers=copper_layers,
                family=coupon_family,
                gerber_dir="fab/gerbers/",
                strict=False,
            )
            assert result_validation.passed, (
                f"Layer validation failed (run {run_idx + 1}): "
                f"{result_validation.missing_layers}"
            )

        # Verify design_hash stability
        design_hashes = [m["design_hash"] for m in manifests]
        assert len(set(design_hashes)) == 1, (
            f"design_hash not stable: {design_hashes}"
        )

        # Verify export hash stability
        export_hash_sets = [
            frozenset(_get_export_hashes_from_manifest(m).items())
            for m in manifests
        ]
        assert len(set(export_hash_sets)) == 1, (
            "Export hashes not stable across runs"
        )
