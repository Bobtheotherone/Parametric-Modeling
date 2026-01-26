# SPDX-License-Identifier: MIT
"""Regression tests for M1 gate fixes.

This module ensures that fixes to failing gates remain locked in place
by testing critical invariants and deterministic behaviors.

These tests serve as regression guards to prevent reintroduction of issues
that caused tools.verify to fail during M1 compliance work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
GOLDEN_SPECS_DIR = TESTS_DIR / "golden_specs"
GOLDEN_HASHES_PATH = ROOT / "golden_hashes" / "design_hashes.json"
PYPROJECT_PATH = ROOT / "pyproject.toml"


# ---------------------------------------------------------------------------
# Regression Tests: Gate Marker Configuration
# ---------------------------------------------------------------------------


class TestGateMarkerConfiguration:
    """Regression tests for pytest marker configuration.

    These tests ensure that all gate markers are properly declared
    in pyproject.toml to prevent "unknown marker" warnings.
    """

    def test_all_gate_markers_declared(self) -> None:
        """All gate markers (gate_g1 through gate_g5) must be declared."""
        content = PYPROJECT_PATH.read_text(encoding="utf-8")

        required_markers = ["gate_g1", "gate_g2", "gate_g3", "gate_g4", "gate_g5"]
        for marker in required_markers:
            assert marker in content, f"Marker '{marker}' not declared in pyproject.toml"

    def test_kicad_integration_marker_declared(self) -> None:
        """kicad_integration marker must be declared."""
        content = PYPROJECT_PATH.read_text(encoding="utf-8")
        assert "kicad_integration" in content, "kicad_integration marker not declared"

    def test_strict_markers_enabled(self) -> None:
        """--strict-markers must be enabled in addopts."""
        content = PYPROJECT_PATH.read_text(encoding="utf-8")
        assert "--strict-markers" in content, "pytest addopts must include --strict-markers for consistent enforcement"


# ---------------------------------------------------------------------------
# Regression Tests: Golden Spec Coverage
# ---------------------------------------------------------------------------


class TestGoldenSpecCoverage:
    """Regression tests for golden spec coverage requirements.

    Per ECO-M1-ALIGN-0001: >=10 golden specs per family (F0, F1)
    """

    def test_minimum_f0_specs_exist(self) -> None:
        """Must have >=10 F0 golden specs for M1 compliance."""
        f0_specs = list(GOLDEN_SPECS_DIR.glob("f0_*.yaml"))
        assert len(f0_specs) >= 10, f"Need >=10 F0 specs, have {len(f0_specs)}"

    def test_minimum_f1_specs_exist(self) -> None:
        """Must have >=10 F1 golden specs for M1 compliance."""
        f1_specs = list(GOLDEN_SPECS_DIR.glob("f1_*.yaml"))
        assert len(f1_specs) >= 10, f"Need >=10 F1 specs, have {len(f1_specs)}"

    def test_golden_hashes_file_exists(self) -> None:
        """golden_hashes/design_hashes.json must exist."""
        assert GOLDEN_HASHES_PATH.exists(), "Golden hashes file missing"

    def test_golden_hashes_covers_all_specs(self) -> None:
        """All golden specs must have corresponding golden hashes."""
        if not GOLDEN_HASHES_PATH.exists():
            pytest.skip("Golden hashes file not found")

        data = json.loads(GOLDEN_HASHES_PATH.read_text(encoding="utf-8"))
        spec_hashes = data.get("spec_hashes", {})

        all_specs = list(GOLDEN_SPECS_DIR.glob("f0_*.yaml")) + list(GOLDEN_SPECS_DIR.glob("f1_*.yaml"))

        missing = []
        for spec_path in all_specs:
            if spec_path.name not in spec_hashes:
                missing.append(spec_path.name)

        assert not missing, f"Missing golden hashes for specs: {missing}"

    def test_golden_hashes_are_sha256_format(self) -> None:
        """All golden hashes must be valid SHA256 hex strings."""
        if not GOLDEN_HASHES_PATH.exists():
            pytest.skip("Golden hashes file not found")

        data = json.loads(GOLDEN_HASHES_PATH.read_text(encoding="utf-8"))
        spec_hashes = data.get("spec_hashes", {})

        for spec_name, hash_value in spec_hashes.items():
            assert len(hash_value) == 64, f"Hash for {spec_name} is not 64 chars"
            assert all(c in "0123456789abcdef" for c in hash_value), f"Hash for {spec_name} is not valid hex"


# ---------------------------------------------------------------------------
# Regression Tests: Golden Spec Content Validation
# ---------------------------------------------------------------------------


class TestGoldenSpecContent:
    """Regression tests for golden spec content requirements.

    These tests ensure that golden specs meet M1 requirements for
    toolchain pinning, DRC configuration, and export settings.
    """

    def _collect_golden_specs(self) -> list[Path]:
        """Collect all golden spec files."""
        specs: list[Path] = []
        specs.extend(sorted(GOLDEN_SPECS_DIR.glob("f0_*.yaml")))
        specs.extend(sorted(GOLDEN_SPECS_DIR.glob("f1_*.yaml")))
        return specs

    def test_all_specs_have_schema_version(self) -> None:
        """All golden specs must have schema_version: 1."""
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")

        for spec_path in self._collect_golden_specs():
            data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
            assert data.get("schema_version") == 1, f"{spec_path.name} missing schema_version: 1"

    def test_all_specs_have_digest_pinned_docker(self) -> None:
        """All golden specs must use digest-pinned Docker images."""
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")

        for spec_path in self._collect_golden_specs():
            data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
            toolchain = data.get("toolchain", {})
            kicad = toolchain.get("kicad", {})
            docker_image = kicad.get("docker_image", "")

            assert "@sha256:" in docker_image, f"{spec_path.name} must use digest-pinned Docker image, got: {docker_image}"

    def test_all_specs_have_drc_must_pass(self) -> None:
        """All golden specs must have constraints.drc.must_pass: true."""
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")

        for spec_path in self._collect_golden_specs():
            data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
            constraints = data.get("constraints", {})
            drc = constraints.get("drc", {})

            assert drc.get("must_pass") is True, f"{spec_path.name} must have constraints.drc.must_pass: true"

    def test_all_specs_have_export_enabled(self) -> None:
        """All golden specs must have Gerber and drill export enabled."""
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")

        for spec_path in self._collect_golden_specs():
            data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
            export = data.get("export", {})

            gerbers = export.get("gerbers", {})
            assert gerbers.get("enabled") is True, f"{spec_path.name} must have export.gerbers.enabled: true"

            drill = export.get("drill", {})
            assert drill.get("enabled") is True, f"{spec_path.name} must have export.drill.enabled: true"


# ---------------------------------------------------------------------------
# Regression Tests: Verify Harness Integration
# ---------------------------------------------------------------------------


class TestVerifyHarnessRegression:
    """Regression tests for tools.verify harness behavior.

    These tests ensure that the verify harness correctly invokes
    gates and reports failures.
    """

    def test_verify_module_importable(self) -> None:
        """tools.verify must be importable."""
        from tools import verify

        assert hasattr(verify, "main")
        assert hasattr(verify, "GateResult")
        assert hasattr(verify, "DETERMINISTIC_ENV")

    def test_deterministic_env_has_required_keys(self) -> None:
        """DETERMINISTIC_ENV must have all required environment variables."""
        from tools.verify import DETERMINISTIC_ENV

        required_keys = ["LC_ALL", "LANG", "TZ", "PYTHONHASHSEED"]
        for key in required_keys:
            assert key in DETERMINISTIC_ENV, f"Missing env key: {key}"

        # Verify values are deterministic
        assert DETERMINISTIC_ENV["TZ"] == "UTC"
        assert DETERMINISTIC_ENV["PYTHONHASHSEED"] == "0"

    def test_gate_result_dataclass_fields(self) -> None:
        """GateResult must have all required fields."""
        from tools.verify import GateResult

        result = GateResult(name="test", passed=True)
        assert hasattr(result, "name")
        assert hasattr(result, "passed")
        assert hasattr(result, "cmd")
        assert hasattr(result, "stdout")
        assert hasattr(result, "stderr")
        assert hasattr(result, "note")


# ---------------------------------------------------------------------------
# Regression Tests: Gate Test Module Structure
# ---------------------------------------------------------------------------


class TestGateTestModuleStructure:
    """Regression tests for gate test module organization.

    These tests ensure that all gate test modules exist and have
    the expected structure.
    """

    GATES_DIR = TESTS_DIR / "gates"

    def test_gates_package_exists(self) -> None:
        """tests/gates package must exist."""
        assert self.GATES_DIR.exists(), "tests/gates directory missing"
        assert (self.GATES_DIR / "__init__.py").exists(), "tests/gates/__init__.py missing"

    def test_all_gate_test_files_exist(self) -> None:
        """All G1-G5 test files must exist."""
        expected_files = [
            "test_g1_determinism.py",
            "test_g2_constraints.py",
            "test_g3_drc.py",
            "test_g4_export_completeness.py",
            "test_g5_hash_stability.py",
        ]
        for filename in expected_files:
            path = self.GATES_DIR / filename
            assert path.exists(), f"Gate test file missing: {filename}"

    def test_gate_tests_use_correct_markers(self) -> None:
        """Each gate test file must use the appropriate marker."""
        marker_map = {
            "test_g1_determinism.py": "gate_g1",
            "test_g2_constraints.py": "gate_g2",
            "test_g3_drc.py": "gate_g3",
            "test_g4_export_completeness.py": "gate_g4",
            "test_g5_hash_stability.py": "gate_g5",
        }

        for filename, expected_marker in marker_map.items():
            path = self.GATES_DIR / filename
            if path.exists():
                content = path.read_text(encoding="utf-8")
                assert f"@pytest.mark.{expected_marker}" in content, (
                    f"{filename} missing @pytest.mark.{expected_marker} decorator"
                )


# ---------------------------------------------------------------------------
# Regression Tests: Determinism Invariants
# ---------------------------------------------------------------------------


class TestDeterminismInvariants:
    """Regression tests for determinism invariants.

    These tests verify that critical determinism properties are maintained.
    """

    def test_sha256_bytes_is_deterministic(self) -> None:
        """sha256_bytes must produce consistent results."""
        from formula_foundry.substrate import sha256_bytes

        test_input = b"test data for hashing"
        expected = "f7eb7961d8a233e6256d3a6257548bbb9293c3a08fb3574c88c7d6b429dbb9f5"

        result1 = sha256_bytes(test_input)
        result2 = sha256_bytes(test_input)

        assert result1 == result2, "sha256_bytes not deterministic"
        assert result1 == expected, f"sha256_bytes wrong output: {result1}"

    def test_canonical_json_is_deterministic(self) -> None:
        """canonical_json_dumps must produce consistent results."""
        from formula_foundry.substrate import canonical_json_dumps

        test_data = {"z": 1, "a": 2, "m": {"nested": True}}

        result1 = canonical_json_dumps(test_data)
        result2 = canonical_json_dumps(test_data)

        assert result1 == result2, "canonical_json_dumps not deterministic"

        # Keys should be sorted
        parsed = json.loads(result1)
        keys = list(parsed.keys())
        assert keys == sorted(keys), "canonical_json_dumps does not sort keys"


# ---------------------------------------------------------------------------
# Regression Tests: Module Imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Regression tests for module import health.

    These tests ensure that all critical modules can be imported without error.
    """

    def test_coupongen_imports(self) -> None:
        """formula_foundry.coupongen must be importable with key exports."""
        from formula_foundry.coupongen import (
            design_hash,
            load_spec,
            resolve,
            resolved_design_canonical_json,
        )

        assert callable(load_spec)
        assert callable(resolve)
        assert callable(design_hash)
        assert callable(resolved_design_canonical_json)

    def test_gate_runner_imports(self) -> None:
        """formula_foundry.coupongen.gate_runner must be importable."""
        from formula_foundry.coupongen.gate_runner import (
            GATE_DESCRIPTIONS,
            GATES,
            build_pytest_marker_expr,
            parse_gates,
        )

        assert isinstance(GATES, dict)
        assert len(GATES) == 5
        assert callable(parse_gates)
        assert callable(build_pytest_marker_expr)

    def test_constraint_engine_imports(self) -> None:
        """formula_foundry.coupongen.constraints.engine must be importable."""
        from formula_foundry.coupongen.constraints.engine import (
            ConstraintEngine,
            create_constraint_engine,
        )

        assert callable(create_constraint_engine)

    def test_layer_validation_imports(self) -> None:
        """formula_foundry.coupongen.layer_validation must be importable."""
        from formula_foundry.coupongen.layer_validation import (
            get_layer_set_for_copper_count,
            validate_layer_set,
        )

        assert callable(get_layer_set_for_copper_count)
        assert callable(validate_layer_set)

    def test_hashing_imports(self) -> None:
        """formula_foundry.coupongen.hashing must be importable."""
        from formula_foundry.coupongen.hashing import (
            canonicalize_export_text,
            canonicalize_kicad_pcb_text,
        )

        assert callable(canonicalize_export_text)
        assert callable(canonicalize_kicad_pcb_text)
