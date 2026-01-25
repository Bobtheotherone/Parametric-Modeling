# SPDX-License-Identifier: MIT
"""Pytest configuration and shared fixtures for test suite.

This module provides:
- Common fixtures for golden specs and golden hashes
- Deterministic test environment setup
- Gate marker registration helpers
- M2/M3 test collection gating (temporary, controllable via env vars)
- sys.path sanitization for worktree isolation
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# sys.path Sanitization for Worktree Isolation
# ---------------------------------------------------------------------------
# Orchestrator worktrees can pollute sys.path with paths like:
#   runs/**/worktrees/**/src/...
# This can cause "not a package" and half-loaded module errors.
# Ensure the canonical repo src/ is at the front and remove worktree paths.

def _sanitize_sys_path() -> None:
    """Ensure canonical src/ is first; remove worktree src/ entries."""
    repo_root = Path(__file__).resolve().parent.parent
    canonical_src = repo_root / "src"

    # Remove any worktree src paths (paths containing /runs/ and /worktrees/)
    sys.path = [
        p for p in sys.path
        if not ("/runs/" in p and "/worktrees/" in p and "/src" in p)
    ]

    # Ensure canonical src is at the front
    canonical_src_str = str(canonical_src)
    if canonical_src_str in sys.path:
        sys.path.remove(canonical_src_str)
    sys.path.insert(0, canonical_src_str)


_sanitize_sys_path()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent
GOLDEN_SPECS_DIR = TESTS_DIR / "golden_specs"
GOLDEN_HASHES_PATH = ROOT_DIR / "golden_hashes" / "design_hashes.json"


# ---------------------------------------------------------------------------
# M2/M3 Test Collection Gating (Temporary)
# ---------------------------------------------------------------------------
# TEMPORARY: Skip M2 and M3 tests during collection to avoid import errors
# from not-yet-implemented packages (openems, em, m3, tracking).
#
# This is a collection-time skip, meaning the test files won't be imported
# at all, preventing ModuleNotFoundError from breaking the test run.
#
# Enable tests via environment variables when ready:
#   FF_ENABLE_M2_TESTS=1 pytest ...  # Enable M2 tests
#   FF_ENABLE_M3_TESTS=1 pytest ...  # Enable M3/tracking tests


def pytest_ignore_collect(
    collection_path: Path,
    config: pytest.Config,
) -> bool | None:
    """Skip M2/M3/tracking tests at collection time to avoid import errors.

    This prevents pytest from importing test files that reference not-yet-
    implemented packages like formula_foundry.openems, formula_foundry.m3, etc.

    Returns:
        True to skip collection, None to allow normal collection.
    """
    # Get the path relative to tests directory for pattern matching
    try:
        rel_path = collection_path.relative_to(TESTS_DIR)
        rel_str = str(rel_path)
    except ValueError:
        # Path is outside tests directory, don't skip
        return None

    name = collection_path.name

    # M2 tests: test_m2_*.py and tests/m2/ directory
    m2_enabled = os.environ.get("FF_ENABLE_M2_TESTS", "").strip() == "1"
    if not m2_enabled:
        if name.startswith("test_m2_") and name.endswith(".py"):
            return True
        if rel_str.startswith("m2/") or rel_str.startswith("m2\\"):
            return True

    # M3 tests: test_m3_*.py, golden_m3/, and tracking/
    m3_enabled = os.environ.get("FF_ENABLE_M3_TESTS", "").strip() == "1"
    if not m3_enabled:
        if name.startswith("test_m3_") and name.endswith(".py"):
            return True
        if rel_str.startswith("golden_m3/") or rel_str.startswith("golden_m3\\"):
            return True
        if rel_str.startswith("tracking/") or rel_str.startswith("tracking\\"):
            return True

    return None


# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest environment for determinism.

    Sets environment variables to ensure reproducible test execution.
    """
    # Set deterministic environment variables
    deterministic_env = {
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
        "PYTHONHASHSEED": "0",
    }
    for key, value in deterministic_env.items():
        os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Fixtures: Golden Specs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def golden_specs_dir() -> Path:
    """Path to the golden specs directory."""
    return GOLDEN_SPECS_DIR


@pytest.fixture(scope="session")
def all_golden_specs() -> list[Path]:
    """List of all golden spec file paths (F0 and F1)."""
    specs: list[Path] = []
    specs.extend(sorted(GOLDEN_SPECS_DIR.glob("f0_*.yaml")))
    specs.extend(sorted(GOLDEN_SPECS_DIR.glob("f1_*.yaml")))
    return specs


@pytest.fixture(scope="session")
def f0_golden_specs() -> list[Path]:
    """List of F0 golden spec file paths."""
    return sorted(GOLDEN_SPECS_DIR.glob("f0_*.yaml"))


@pytest.fixture(scope="session")
def f1_golden_specs() -> list[Path]:
    """List of F1 golden spec file paths."""
    return sorted(GOLDEN_SPECS_DIR.glob("f1_*.yaml"))


@pytest.fixture(scope="session")
def golden_hashes() -> dict[str, str]:
    """Dictionary of spec filename to expected design hash.

    Returns empty dict if golden hashes file doesn't exist.
    """
    if not GOLDEN_HASHES_PATH.exists():
        return {}
    data = json.loads(GOLDEN_HASHES_PATH.read_text(encoding="utf-8"))
    return data.get("spec_hashes", {})


# ---------------------------------------------------------------------------
# Fixtures: Test Environment
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Path to the project root directory."""
    return ROOT_DIR


@pytest.fixture
def clean_tmp_path(tmp_path: Path) -> Path:
    """Provide a clean temporary directory with deterministic naming.

    Unlike the default tmp_path, this ensures no leftover files from
    previous test runs.
    """
    test_dir = tmp_path / "test_workspace"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


# ---------------------------------------------------------------------------
# Fixtures: Spec Loading
# ---------------------------------------------------------------------------


@pytest.fixture
def load_spec_fixture():
    """Fixture providing the load_spec function."""
    from formula_foundry.coupongen import load_spec

    return load_spec


@pytest.fixture
def resolve_fixture():
    """Fixture providing the resolve function."""
    from formula_foundry.coupongen import resolve

    return resolve


@pytest.fixture
def design_hash_fixture():
    """Fixture providing the design_hash function."""
    from formula_foundry.coupongen import design_hash

    return design_hash


# ---------------------------------------------------------------------------
# Fixtures: Fake Runners
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_drc_runner():
    """Fixture providing a configurable fake DRC runner.

    Usage:
        def test_something(fake_drc_runner):
            runner = fake_drc_runner(returncode=0, violations=[])
            result = runner.run_drc(board_path, report_path)
    """
    import subprocess

    class FakeDrcRunner:
        def __init__(
            self,
            *,
            returncode: int = 0,
            violations: list[dict[str, Any]] | None = None,
        ) -> None:
            self.returncode = returncode
            self.violations = violations or []
            self.calls: list[tuple[Path, Path]] = []

        def run_drc(
            self, board_path: Path, report_path: Path
        ) -> subprocess.CompletedProcess[str]:
            self.calls.append((board_path, report_path))
            report = {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "source": str(board_path),
                "violations": self.violations,
                "unconnected_items": [],
                "schematic_parity": [],
                "coordinate_units": "mm",
            }
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return subprocess.CompletedProcess(
                args=["kicad-cli", "pcb", "drc"],
                returncode=self.returncode,
                stdout="",
                stderr="" if self.returncode == 0 else "DRC violations found",
            )

        def export_gerbers(
            self, board_path: Path, out_dir: Path
        ) -> subprocess.CompletedProcess[str]:
            out_dir.mkdir(parents=True, exist_ok=True)
            board_name = board_path.stem
            for layer in ["F_Cu", "B_Cu", "In1_Cu", "In2_Cu", "F_Mask", "B_Mask", "Edge_Cuts"]:
                (out_dir / f"{board_name}-{layer}.gbr").write_text(
                    f"G04 Fake {layer}*\n", encoding="utf-8"
                )
            return subprocess.CompletedProcess(
                args=["kicad-cli"], returncode=0, stdout="", stderr=""
            )

        def export_drill(
            self, board_path: Path, out_dir: Path
        ) -> subprocess.CompletedProcess[str]:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "drill.drl").write_text("M48\n", encoding="utf-8")
            return subprocess.CompletedProcess(
                args=["kicad-cli"], returncode=0, stdout="", stderr=""
            )

    def factory(**kwargs: Any) -> FakeDrcRunner:
        return FakeDrcRunner(**kwargs)

    return factory


# ---------------------------------------------------------------------------
# Markers Collection Helper
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark tests appropriately based on requirements.

    - Skip kicad_integration tests if Docker is not available
    """
    import shutil

    has_docker = shutil.which("docker") is not None

    for item in items:
        if "kicad_integration" in item.keywords and not has_docker:
            item.add_marker(
                pytest.mark.skip(reason="Docker not available for KiCad integration tests")
            )
