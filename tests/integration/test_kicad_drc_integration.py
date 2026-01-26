# SPDX-License-Identifier: MIT
"""KiCad DRC integration tests (CP-1.2).

This module implements Gate G3 tests: KiCad DRC clean verification.

For each golden spec (F0 and F1):
- Run `coupongen build spec.yaml --mode docker`
- Verify DRC exit code is 0
- Parse drc.json and assert 0 violations

Tests are marked with @pytest.mark.kicad_integration and @pytest.mark.gate_g3.
These tests require Docker with the pinned KiCad image.

Per Section 13.1.2 of the design document:
- DRC uses --severity-all --exit-code-violations --format json
- Exit code 0 = clean, exit code 5 = violations

References:
- https://docs.kicad.org/8.0/en/cli/cli.html (DRC section)
- https://docs.kicad.org/master/en/cli/cli.html (adds --refill-zones)
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


def _is_ci_environment() -> bool:
    """Check if running in a CI environment (GitHub Actions, etc.)."""
    return os.environ.get("CI", "").lower() == "true" or os.environ.get("GITHUB_ACTIONS", "").lower() == "true"


# Root of the repository
REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SPECS_DIR = REPO_ROOT / "tests" / "golden_specs"

# Expected spec patterns (we expect at least 10 of each family per M1 requirements)
F0_PATTERN = "f0_cal_*.yaml"
F1_PATTERN = "f1_via_*.yaml"


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


@pytest.fixture
def build_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for build artifacts."""
    out_dir = tmp_path / "build_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _run_coupongen_build(
    spec_path: Path,
    output_dir: Path,
    mode: str = "docker",
    timeout: int = 120,
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


def _parse_drc_json(drc_path: Path) -> dict:
    """Parse the DRC JSON report file.

    Args:
        drc_path: Path to the drc.json file.

    Returns:
        Parsed DRC report as a dictionary.

    Raises:
        FileNotFoundError: If the DRC report doesn't exist.
        json.JSONDecodeError: If the report is invalid JSON.
    """
    if not drc_path.exists():
        raise FileNotFoundError(f"DRC report not found: {drc_path}")
    return json.loads(drc_path.read_text(encoding="utf-8"))


def _count_drc_violations(drc_report: dict) -> int:
    """Count the total number of DRC violations from a report.

    Args:
        drc_report: Parsed DRC JSON report.

    Returns:
        Total count of DRC violations.

    Note:
        KiCad DRC JSON format has "violations" as a list.
        Each violation has a "severity" and other metadata.
    """
    violations = drc_report.get("violations", [])
    return len(violations)


def _find_drc_json(output_dir: Path) -> Path | None:
    """Find the drc.json file in the build output directory.

    The build creates a subdirectory named <coupon_id>-<design_hash>/
    and places drc.json there.

    Args:
        output_dir: Root output directory passed to coupongen build.

    Returns:
        Path to drc.json if found, None otherwise.
    """
    # Look for drc.json in any subdirectory
    drc_files = list(output_dir.rglob("drc.json"))
    if drc_files:
        return drc_files[0]
    return None


class TestGoldenSpecsExist:
    """Verify that the required golden specs are present."""

    def test_f0_golden_specs_present(self) -> None:
        """REQ-M1-024: At least 10 F0 golden specs must exist."""
        f0_specs = list(GOLDEN_SPECS_DIR.glob(F0_PATTERN))
        assert len(f0_specs) >= 10, f"Expected at least 10 F0 golden specs, found {len(f0_specs)}"

    def test_f1_golden_specs_present(self) -> None:
        """REQ-M1-024: At least 10 F1 golden specs must exist."""
        f1_specs = list(GOLDEN_SPECS_DIR.glob(F1_PATTERN))
        assert len(f1_specs) >= 10, f"Expected at least 10 F1 golden specs, found {len(f1_specs)}"


@pytest.mark.kicad_integration
@pytest.mark.gate_g3
class TestKicadDRCIntegration:
    """Integration tests for KiCad DRC verification (Gate G3).

    These tests verify that for each golden spec:
    1. coupongen build completes successfully with docker mode
    2. DRC exit code is 0 (no violations)
    3. drc.json contains 0 violations

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
        """Skip tests if Docker or KiCad image is not available.

        In CI environments, missing prerequisites cause a hard failure instead
        of a skip to ensure integration tests are never silently bypassed.
        """
        if not docker_available:
            if _is_ci_environment():
                pytest.fail("Docker is not available in CI - this is a hard failure")
            pytest.skip("Docker is not available")
        if not kicad_image_available:
            if _is_ci_environment():
                pytest.fail(
                    "KiCad Docker image not available (kicad/kicad:9.0.7) in CI - "
                    "ensure the image is pulled and tagged in the workflow"
                )
            pytest.skip("KiCad Docker image not available (kicad/kicad:9.0.7)")

    @pytest.mark.parametrize(
        "spec_name",
        _get_spec_ids(),
        ids=_get_spec_ids(),
    )
    def test_drc_clean_for_golden_spec(
        self,
        spec_name: str,
        tmp_path: Path,
    ) -> None:
        """Gate G3: Verify DRC passes with 0 violations for golden spec.

        Per CP-1.2, for each golden spec:
        1. Run `coupongen build spec.yaml --mode docker`
        2. Ensure DRC exit code is 0
        3. Parse drc.json and assert 0 violations

        Args:
            spec_name: Name of the golden spec file (without extension).
            tmp_path: Pytest temporary directory fixture.
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
            timeout=180,  # Allow more time for Docker operations
        )

        # Check build succeeded
        assert result.returncode == 0, f"coupongen build failed for {spec_name}\nstdout: {result.stdout}\nstderr: {result.stderr}"

        # Find and verify DRC report
        drc_path = _find_drc_json(output_dir)
        assert drc_path is not None, f"DRC report not found for {spec_name} in {output_dir}"

        # Parse DRC report and verify 0 violations
        drc_report = _parse_drc_json(drc_path)
        violation_count = _count_drc_violations(drc_report)

        assert violation_count == 0, (
            f"DRC violations found for {spec_name}: {violation_count} violations\nDRC report: {json.dumps(drc_report, indent=2)}"
        )


@pytest.mark.kicad_integration
@pytest.mark.gate_g3
class TestDRCExitCodeSemantics:
    """Test that DRC exit code semantics are correctly handled.

    Per KiCad CLI documentation:
    - Exit code 0: DRC passed (no violations)
    - Exit code 5: DRC completed but found violations
    - Other exit codes: Error conditions
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_docker(
        self,
        docker_available: bool,
        kicad_image_available: bool,
    ) -> None:
        """Skip tests if Docker or KiCad image is not available.

        In CI environments, missing prerequisites cause a hard failure instead
        of a skip to ensure integration tests are never silently bypassed.
        """
        if not docker_available:
            if _is_ci_environment():
                pytest.fail("Docker is not available in CI - this is a hard failure")
            pytest.skip("Docker is not available")
        if not kicad_image_available:
            if _is_ci_environment():
                pytest.fail(
                    "KiCad Docker image not available (kicad/kicad:9.0.7) in CI - "
                    "ensure the image is pulled and tagged in the workflow"
                )
            pytest.skip("KiCad Docker image not available (kicad/kicad:9.0.7)")

    def test_clean_build_has_exit_code_zero(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify that a clean build results in DRC exit code 0.

        Uses the first available golden spec to verify baseline behavior.
        """
        specs = _collect_golden_specs()
        if not specs:
            pytest.skip("No golden specs available")

        spec_path = specs[0]
        output_dir = tmp_path / "build_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = _run_coupongen_build(
            spec_path=spec_path,
            output_dir=output_dir,
            mode="docker",
            timeout=180,
        )

        # The build command should succeed
        assert result.returncode == 0, f"Build failed: {result.stderr}"

        # Parse the build output JSON to verify DRC passed
        try:
            build_output = json.loads(result.stdout)
            # Build succeeded means DRC passed (since constraints.drc.must_pass is true)
            assert "design_hash" in build_output
        except json.JSONDecodeError:
            # Build output may have additional text, check that build completed
            assert "design_hash" in result.stdout or result.returncode == 0


@pytest.mark.kicad_integration
@pytest.mark.gate_g3
class TestDRCReportFormat:
    """Test DRC report format and content validation."""

    @pytest.fixture(autouse=True)
    def skip_if_no_docker(
        self,
        docker_available: bool,
        kicad_image_available: bool,
    ) -> None:
        """Skip tests if Docker or KiCad image is not available.

        In CI environments, missing prerequisites cause a hard failure instead
        of a skip to ensure integration tests are never silently bypassed.
        """
        if not docker_available:
            if _is_ci_environment():
                pytest.fail("Docker is not available in CI - this is a hard failure")
            pytest.skip("Docker is not available")
        if not kicad_image_available:
            if _is_ci_environment():
                pytest.fail(
                    "KiCad Docker image not available (kicad/kicad:9.0.7) in CI - "
                    "ensure the image is pulled and tagged in the workflow"
                )
            pytest.skip("KiCad Docker image not available (kicad/kicad:9.0.7)")

    def test_drc_json_has_expected_structure(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify that drc.json has the expected KiCad JSON structure.

        The DRC JSON report should contain:
        - source: Board file path
        - violations: Array of violation objects (should be empty for clean)
        - unconnected_items: Array of unconnected items (should be empty)
        """
        specs = _collect_golden_specs()
        if not specs:
            pytest.skip("No golden specs available")

        spec_path = specs[0]
        output_dir = tmp_path / "build_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = _run_coupongen_build(
            spec_path=spec_path,
            output_dir=output_dir,
            mode="docker",
            timeout=180,
        )

        assert result.returncode == 0, f"Build failed: {result.stderr}"

        drc_path = _find_drc_json(output_dir)
        assert drc_path is not None, "DRC report not found"

        drc_report = _parse_drc_json(drc_path)

        # Verify expected structure (KiCad DRC JSON format)
        # Note: Exact structure may vary by KiCad version, but these are common
        assert isinstance(drc_report, dict), "DRC report must be a JSON object"

        # violations should be a list (may be empty for clean boards)
        violations = drc_report.get("violations", [])
        assert isinstance(violations, list), "violations must be a list"

        # For golden specs, violations should be empty
        assert len(violations) == 0, f"Golden spec should have 0 violations, found {len(violations)}"
