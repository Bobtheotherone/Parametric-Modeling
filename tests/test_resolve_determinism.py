"""Determinism tests for resolve hashing against committed golden hashes.

REQ-M1-024: CI must prove deterministic resolve hashing against committed golden hashes.

This module verifies that:
1. All golden specs produce deterministic design hashes
2. Computed hashes match the committed golden hashes exactly
3. Hashing is stable across multiple resolves
4. Key ordering in specs does not affect the hash
5. All F0 and F1 golden specs are covered
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from formula_foundry.coupongen import load_spec
from formula_foundry.coupongen.resolve import (
    design_hash,
    resolve,
    resolved_design_canonical_json,
)
from formula_foundry.coupongen.spec import CouponSpec
from formula_foundry.substrate import sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SPECS_DIR = ROOT / "tests" / "golden_specs"
GOLDEN_HASHES_PATH = ROOT / "tests" / "golden_hashes" / "design_hashes.json"


def _load_golden_hashes() -> dict[str, str]:
    """Load the committed golden hashes from tests/golden_hashes/design_hashes.json."""
    data = json.loads(GOLDEN_HASHES_PATH.read_text(encoding="utf-8"))
    return data.get("spec_hashes", {})


def _golden_spec_files() -> list[Path]:
    """Collect all golden spec files from tests/golden_specs/."""
    patterns = ("*.json", "*.yaml", "*.yml")
    specs: list[Path] = []
    for pattern in patterns:
        specs.extend(GOLDEN_SPECS_DIR.glob(pattern))
    return sorted(specs)


def _f0_golden_specs() -> list[Path]:
    """Get F0 calibration thru-line golden specs."""
    return sorted(GOLDEN_SPECS_DIR.glob("f0_cal_*.json"))


def _f1_golden_specs() -> list[Path]:
    """Get F1 single-ended via transition golden specs."""
    return sorted(GOLDEN_SPECS_DIR.glob("f1_via_*.json"))


class TestGoldenHashesExist:
    """Verify that golden hash infrastructure exists."""

    def test_golden_hashes_file_exists(self) -> None:
        """Golden hashes file must exist at tests/golden_hashes/design_hashes.json."""
        assert GOLDEN_HASHES_PATH.exists(), f"Missing: {GOLDEN_HASHES_PATH}"

    def test_golden_hashes_file_is_valid_json(self) -> None:
        """Golden hashes file must be valid JSON with spec_hashes key."""
        data = json.loads(GOLDEN_HASHES_PATH.read_text(encoding="utf-8"))
        assert "spec_hashes" in data
        assert isinstance(data["spec_hashes"], dict)

    def test_golden_specs_directory_exists(self) -> None:
        """Golden specs directory must exist at tests/golden_specs/."""
        assert GOLDEN_SPECS_DIR.exists(), f"Missing: {GOLDEN_SPECS_DIR}"
        assert GOLDEN_SPECS_DIR.is_dir()

    def test_sufficient_f0_golden_specs(self) -> None:
        """At least 10 F0 golden specs must be present."""
        specs = _f0_golden_specs()
        assert len(specs) >= 10, f"Found only {len(specs)} F0 golden specs, need >= 10"

    def test_sufficient_f1_golden_specs(self) -> None:
        """At least 10 F1 golden specs must be present."""
        specs = _f1_golden_specs()
        assert len(specs) >= 10, f"Found only {len(specs)} F1 golden specs, need >= 10"


class TestDeterministicResolveHashing:
    """REQ-M1-024: Prove deterministic resolve hashing against committed golden hashes."""

    @pytest.fixture
    def golden_hashes(self) -> dict[str, str]:
        """Load golden hashes fixture."""
        return _load_golden_hashes()

    @pytest.mark.parametrize(
        "spec_path",
        _golden_spec_files(),
        ids=lambda p: p.name,
    )
    def test_resolve_hash_matches_golden(self, spec_path: Path, golden_hashes: dict[str, str]) -> None:
        """Each golden spec's design_hash must match the committed golden hash."""
        spec = load_spec(spec_path)
        resolved = resolve(spec)
        computed_hash = design_hash(resolved)

        key = spec_path.name
        assert key in golden_hashes, f"No golden hash for {key}"
        expected_hash = golden_hashes[key]
        assert computed_hash == expected_hash, f"Hash mismatch for {key}: computed={computed_hash}, expected={expected_hash}"

    @pytest.mark.parametrize(
        "spec_path",
        _golden_spec_files(),
        ids=lambda p: p.name,
    )
    def test_resolve_is_deterministic_across_runs(self, spec_path: Path) -> None:
        """Resolving the same spec multiple times must produce identical hashes."""
        spec = load_spec(spec_path)

        resolved_a = resolve(spec)
        resolved_b = resolve(spec)
        resolved_c = resolve(spec)

        hash_a = design_hash(resolved_a)
        hash_b = design_hash(resolved_b)
        hash_c = design_hash(resolved_c)

        assert hash_a == hash_b == hash_c, "Hash varies across resolves"

    @pytest.mark.parametrize(
        "spec_path",
        _golden_spec_files(),
        ids=lambda p: p.name,
    )
    def test_canonical_json_is_deterministic(self, spec_path: Path) -> None:
        """Canonical JSON serialization must be byte-identical across resolves."""
        spec = load_spec(spec_path)

        resolved_a = resolve(spec)
        resolved_b = resolve(spec)

        json_a = resolved_design_canonical_json(resolved_a)
        json_b = resolved_design_canonical_json(resolved_b)

        assert json_a == json_b, "Canonical JSON varies across resolves"

    @pytest.mark.parametrize(
        "spec_path",
        _golden_spec_files(),
        ids=lambda p: p.name,
    )
    def test_design_hash_equals_sha256_of_canonical_json(self, spec_path: Path) -> None:
        """design_hash must equal SHA256 of canonical JSON representation."""
        spec = load_spec(spec_path)
        resolved = resolve(spec)

        canonical = resolved_design_canonical_json(resolved)
        expected = sha256_bytes(canonical.encode("utf-8"))
        actual = design_hash(resolved)

        assert actual == expected, "design_hash != sha256(canonical_json)"


class TestResolvedDesignProperties:
    """Verify properties of resolved designs from golden specs."""

    @pytest.mark.parametrize(
        "spec_path",
        _golden_spec_files(),
        ids=lambda p: p.name,
    )
    def test_resolved_parameters_are_integers(self, spec_path: Path) -> None:
        """All parameters_nm values must be integers."""
        spec = load_spec(spec_path)
        resolved = resolve(spec)

        for key, value in resolved.parameters_nm.items():
            assert isinstance(value, int), f"{key} is not int: {type(value)}"

    @pytest.mark.parametrize(
        "spec_path",
        _golden_spec_files(),
        ids=lambda p: p.name,
    )
    def test_resolved_derived_features_are_integers(self, spec_path: Path) -> None:
        """All derived_features values must be integers."""
        spec = load_spec(spec_path)
        resolved = resolve(spec)

        for key, value in resolved.derived_features.items():
            assert isinstance(value, int), f"{key} is not int: {type(value)}"

    @pytest.mark.parametrize(
        "spec_path",
        _golden_spec_files(),
        ids=lambda p: p.name,
    )
    def test_resolved_units_is_nm(self, spec_path: Path) -> None:
        """ResolvedDesign.units must always be 'nm'."""
        spec = load_spec(spec_path)
        resolved = resolve(spec)
        assert resolved.units == "nm"

    @pytest.mark.parametrize(
        "spec_path",
        _golden_spec_files(),
        ids=lambda p: p.name,
    )
    def test_resolved_schema_version_preserved(self, spec_path: Path) -> None:
        """ResolvedDesign.schema_version must match input spec."""
        spec = load_spec(spec_path)
        resolved = resolve(spec)
        assert resolved.schema_version == spec.schema_version


class TestF0FamilyDeterminism:
    """Specific tests for F0 calibration thru-line family determinism."""

    @pytest.fixture
    def golden_hashes(self) -> dict[str, str]:
        """Load golden hashes fixture."""
        return _load_golden_hashes()

    def test_all_f0_specs_have_golden_hashes(self, golden_hashes: dict[str, str]) -> None:
        """Every F0 golden spec must have a corresponding golden hash."""
        for spec_path in _f0_golden_specs():
            assert spec_path.name in golden_hashes, f"No hash for {spec_path.name}"

    def test_f0_specs_have_correct_family(self) -> None:
        """All F0 golden specs must have coupon_family F0_CAL_THRU_LINE."""
        for spec_path in _f0_golden_specs():
            spec = load_spec(spec_path)
            assert spec.coupon_family == "F0_CAL_THRU_LINE", f"{spec_path.name}: family={spec.coupon_family}"

    def test_f0_specs_have_no_discontinuity(self) -> None:
        """F0 calibration thru-lines must have discontinuity=null."""
        for spec_path in _f0_golden_specs():
            spec = load_spec(spec_path)
            assert spec.discontinuity is None, f"{spec_path.name}: discontinuity should be null"


class TestF1FamilyDeterminism:
    """Specific tests for F1 single-ended via transition family determinism."""

    @pytest.fixture
    def golden_hashes(self) -> dict[str, str]:
        """Load golden hashes fixture."""
        return _load_golden_hashes()

    def test_all_f1_specs_have_golden_hashes(self, golden_hashes: dict[str, str]) -> None:
        """Every F1 golden spec must have a corresponding golden hash."""
        for spec_path in _f1_golden_specs():
            assert spec_path.name in golden_hashes, f"No hash for {spec_path.name}"

    def test_f1_specs_have_correct_family(self) -> None:
        """All F1 golden specs must have coupon_family F1_SINGLE_ENDED_VIA."""
        for spec_path in _f1_golden_specs():
            spec = load_spec(spec_path)
            assert spec.coupon_family == "F1_SINGLE_ENDED_VIA", f"{spec_path.name}: family={spec.coupon_family}"

    def test_f1_specs_have_discontinuity(self) -> None:
        """F1 via transition specs must have a discontinuity section."""
        for spec_path in _f1_golden_specs():
            spec = load_spec(spec_path)
            assert spec.discontinuity is not None, f"{spec_path.name}: discontinuity should not be null"

    def test_f1_specs_have_signal_via(self) -> None:
        """F1 via transition specs must have signal_via in discontinuity."""
        for spec_path in _f1_golden_specs():
            spec = load_spec(spec_path)
            assert spec.discontinuity is not None
            assert spec.discontinuity.signal_via is not None, f"{spec_path.name}: signal_via is missing"


class TestKeyOrderingIndependence:
    """Verify that key ordering in input specs does not affect hashing."""

    def test_reordered_spec_produces_same_hash(self) -> None:
        """Loading a spec with reordered keys must produce the same design hash."""
        if not _golden_spec_files():
            pytest.skip("No golden specs available")

        spec_path = _golden_spec_files()[0]
        original_data = json.loads(spec_path.read_text(encoding="utf-8"))

        # Reorder top-level keys
        reordered_keys = list(reversed(original_data.keys()))
        reordered_data = {k: original_data[k] for k in reordered_keys}

        original_spec = CouponSpec.model_validate(original_data)
        reordered_spec = CouponSpec.model_validate(reordered_data)

        original_resolved = resolve(original_spec)
        reordered_resolved = resolve(reordered_spec)

        assert design_hash(original_resolved) == design_hash(reordered_resolved)


class TestGoldenHashIntegrity:
    """Verify integrity of the golden hashes file itself."""

    def test_all_golden_hashes_are_valid_sha256(self) -> None:
        """All golden hashes must be valid 64-character hex SHA256 strings."""
        hashes = _load_golden_hashes()
        for key, digest in hashes.items():
            assert len(digest) == 64, f"{key}: hash length != 64"
            assert all(c in "0123456789abcdef" for c in digest), f"{key}: invalid hex characters"

    def test_golden_hashes_cover_all_golden_specs(self) -> None:
        """Every golden spec file must have a corresponding golden hash."""
        hashes = _load_golden_hashes()
        for spec_path in _golden_spec_files():
            assert spec_path.name in hashes, f"Missing hash for {spec_path.name}"

    def test_no_orphan_golden_hashes(self) -> None:
        """Every golden hash must correspond to an existing golden spec."""
        hashes = _load_golden_hashes()
        spec_names = {p.name for p in _golden_spec_files()}
        for key in hashes:
            assert key in spec_names, f"Orphan hash: {key}"
