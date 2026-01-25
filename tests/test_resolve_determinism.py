"""Determinism tests for resolve hashing against committed golden hashes.

REQ-M1-024: CI must prove deterministic resolve hashing against committed golden hashes.
REQ-M1-014/015: Mutation tests prove that changing key fields (gap_nm, fence pitch,
connector footprint id, corner radius) changes resolved design or artifact hashes.

This module verifies that:
1. All golden specs produce deterministic design hashes
2. Computed hashes match the committed golden hashes exactly
3. Hashing is stable across multiple resolves
4. Key ordering in specs does not affect the hash
5. All F0 and F1 golden specs are covered
6. Mutations to key fields produce different hashes (no identical board for distinct specs)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

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
    def test_derived_features_are_deterministic(self, spec_path: Path) -> None:
        """REQ-M1-014: derived_features must be deterministic across multiple resolves.

        Verifies that derived features (CPWG/via/fence/launch-relevant) are emitted
        deterministically, with identical keys and values on each resolve.
        """
        spec = load_spec(spec_path)

        resolved_a = resolve(spec)
        resolved_b = resolve(spec)
        resolved_c = resolve(spec)

        # Keys must be identical and sorted
        assert list(resolved_a.derived_features.keys()) == list(resolved_b.derived_features.keys())
        assert list(resolved_b.derived_features.keys()) == list(resolved_c.derived_features.keys())

        # Keys should be sorted (deterministic ordering)
        assert list(resolved_a.derived_features.keys()) == sorted(resolved_a.derived_features.keys())

        # Values must be identical
        assert resolved_a.derived_features == resolved_b.derived_features == resolved_c.derived_features

    @pytest.mark.parametrize(
        "spec_path",
        _golden_spec_files(),
        ids=lambda p: p.name,
    )
    def test_dimensionless_groups_are_deterministic(self, spec_path: Path) -> None:
        """REQ-M1-014: dimensionless_groups must be deterministic across multiple resolves.

        Verifies that dimensionless groups (CPWG/via/fence/launch-relevant ratios) are
        emitted deterministically, with identical keys and values on each resolve.
        """
        spec = load_spec(spec_path)

        resolved_a = resolve(spec)
        resolved_b = resolve(spec)
        resolved_c = resolve(spec)

        # Keys must be identical and sorted
        assert list(resolved_a.dimensionless_groups.keys()) == list(resolved_b.dimensionless_groups.keys())
        assert list(resolved_b.dimensionless_groups.keys()) == list(resolved_c.dimensionless_groups.keys())

        # Keys should be sorted (deterministic ordering)
        assert list(resolved_a.dimensionless_groups.keys()) == sorted(resolved_a.dimensionless_groups.keys())

        # Values must be identical
        assert resolved_a.dimensionless_groups == resolved_b.dimensionless_groups == resolved_c.dimensionless_groups

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

    @pytest.mark.parametrize(
        "spec_path",
        _golden_spec_files(),
        ids=lambda p: p.name,
    )
    def test_connector_footprints_are_deterministic(self, spec_path: Path) -> None:
        """REQ-M1-015: connector_footprints provenance must be deterministic.

        Verifies that connector footprint provenance (footprint_id and metadata_hash)
        is deterministic across multiple resolves, ensuring footprint changes affect
        the design hash.
        """
        spec = load_spec(spec_path)

        resolved_a = resolve(spec)
        resolved_b = resolve(spec)
        resolved_c = resolve(spec)

        # connector_footprints must be identical across resolves
        assert resolved_a.connector_footprints == resolved_b.connector_footprints
        assert resolved_b.connector_footprints == resolved_c.connector_footprints

        # Verify structure: each connector should have footprint_id and metadata_hash
        for position, provenance in resolved_a.connector_footprints.items():
            assert "footprint_id" in provenance, f"{position}: missing footprint_id"
            assert "metadata_hash" in provenance, f"{position}: missing metadata_hash"
            assert isinstance(provenance["metadata_hash"], str), f"{position}: metadata_hash not a string"
            # metadata_hash should be a valid SHA256 (64 hex chars)
            assert len(provenance["metadata_hash"]) == 64, f"{position}: metadata_hash length != 64"


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


def _deep_set(data: dict[str, Any], path: str, value: Any) -> None:
    """Set a nested value using dot-separated path notation.

    Args:
        data: Dictionary to modify (mutated in place).
        path: Dot-separated path to the key (e.g., "transmission_line.gap_nm").
        value: Value to set.
    """
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def _deep_get(data: dict[str, Any], path: str) -> Any:
    """Get a nested value using dot-separated path notation.

    Args:
        data: Dictionary to read from.
        path: Dot-separated path to the key (e.g., "transmission_line.gap_nm").

    Returns:
        The value at the specified path.
    """
    keys = path.split(".")
    current = data
    for key in keys:
        current = current[key]
    return current


class TestMutationDetection:
    """REQ-M1-015: Mutation tests proving key field changes affect design/artifact hashes.

    These tests verify that changing key fields (gap_nm, fence pitch, connector footprint id,
    corner radius) results in different resolved design hashes, ensuring no identical board
    is produced for distinct specs.
    """

    def _get_baseline_spec_data(self) -> dict[str, Any]:
        """Load baseline spec data from first golden spec for mutation testing."""
        spec_path = _golden_spec_files()[0]
        return json.loads(spec_path.read_text(encoding="utf-8"))

    def test_gap_nm_mutation_changes_hash(self) -> None:
        """Changing transmission_line.gap_nm must change the design hash.

        REQ-M1-015: gap_nm is a key CPWG parameter that affects physical geometry.
        A different gap results in a different board design.
        """
        baseline_data = self._get_baseline_spec_data()
        baseline_spec = CouponSpec.model_validate(baseline_data)
        baseline_resolved = resolve(baseline_spec)
        baseline_hash = design_hash(baseline_resolved)

        # Mutate gap_nm
        mutated_data = copy.deepcopy(baseline_data)
        original_gap = _deep_get(mutated_data, "transmission_line.gap_nm")
        new_gap = original_gap + 10000  # Add 10um
        _deep_set(mutated_data, "transmission_line.gap_nm", new_gap)

        mutated_spec = CouponSpec.model_validate(mutated_data)
        mutated_resolved = resolve(mutated_spec)
        mutated_hash = design_hash(mutated_resolved)

        assert baseline_hash != mutated_hash, (
            f"gap_nm mutation ({original_gap} -> {new_gap}) did not change design hash. "
            f"REQ-M1-015 requires distinct specs to produce distinct hashes."
        )

    def test_corner_radius_mutation_changes_hash(self) -> None:
        """Changing board.outline.corner_radius_nm must change the design hash.

        REQ-M1-015: corner_radius_nm affects board outline geometry.
        A different radius results in a different board design.
        """
        baseline_data = self._get_baseline_spec_data()
        baseline_spec = CouponSpec.model_validate(baseline_data)
        baseline_resolved = resolve(baseline_spec)
        baseline_hash = design_hash(baseline_resolved)

        # Mutate corner_radius_nm
        mutated_data = copy.deepcopy(baseline_data)
        original_radius = _deep_get(mutated_data, "board.outline.corner_radius_nm")
        new_radius = original_radius + 500000  # Add 0.5mm
        _deep_set(mutated_data, "board.outline.corner_radius_nm", new_radius)

        mutated_spec = CouponSpec.model_validate(mutated_data)
        mutated_resolved = resolve(mutated_spec)
        mutated_hash = design_hash(mutated_resolved)

        assert baseline_hash != mutated_hash, (
            f"corner_radius_nm mutation ({original_radius} -> {new_radius}) did not change design hash. "
            f"REQ-M1-015 requires distinct specs to produce distinct hashes."
        )

    def test_connector_footprint_mutation_changes_hash(self) -> None:
        """Changing connector footprint identifier must change the design hash.

        REQ-M1-015: connector footprint ID affects physical layout and pad positions.
        A different footprint results in a different board design.
        """
        baseline_data = self._get_baseline_spec_data()
        baseline_spec = CouponSpec.model_validate(baseline_data)
        baseline_resolved = resolve(baseline_spec)
        baseline_hash = design_hash(baseline_resolved)

        # Mutate left connector footprint to alternative SMA connector with different pad dimensions
        mutated_data = copy.deepcopy(baseline_data)
        original_footprint = _deep_get(mutated_data, "connectors.left.footprint")
        new_footprint = "Coupongen_Connectors:SMA_EndLaunch_Alt"
        _deep_set(mutated_data, "connectors.left.footprint", new_footprint)

        mutated_spec = CouponSpec.model_validate(mutated_data)
        mutated_resolved = resolve(mutated_spec)
        mutated_hash = design_hash(mutated_resolved)

        assert baseline_hash != mutated_hash, (
            f"connector footprint mutation ({original_footprint} -> {new_footprint}) did not change design hash. "
            f"REQ-M1-015 requires distinct specs to produce distinct hashes."
        )

    def test_ground_via_fence_pitch_mutation_changes_hash(self) -> None:
        """Changing ground_via_fence.pitch_nm must change the design hash.

        REQ-M1-015: fence pitch affects via placement geometry.
        Different pitch values result in different board designs.
        """
        baseline_data = self._get_baseline_spec_data()

        # Create spec with ground_via_fence enabled
        fence_spec_data = copy.deepcopy(baseline_data)
        _deep_set(fence_spec_data, "transmission_line.ground_via_fence", {
            "enabled": True,
            "pitch_nm": 1000000,  # 1mm pitch
            "offset_from_gap_nm": 200000,  # 0.2mm offset
            "via": {
                "drill_nm": 300000,
                "diameter_nm": 600000,
            }
        })

        fence_spec = CouponSpec.model_validate(fence_spec_data)
        fence_resolved = resolve(fence_spec)
        fence_hash = design_hash(fence_resolved)

        # Mutate pitch_nm
        mutated_fence_data = copy.deepcopy(fence_spec_data)
        original_pitch = 1000000
        new_pitch = 1500000  # Change to 1.5mm pitch
        _deep_set(mutated_fence_data, "transmission_line.ground_via_fence.pitch_nm", new_pitch)

        mutated_fence_spec = CouponSpec.model_validate(mutated_fence_data)
        mutated_fence_resolved = resolve(mutated_fence_spec)
        mutated_fence_hash = design_hash(mutated_fence_resolved)

        assert fence_hash != mutated_fence_hash, (
            f"fence pitch mutation ({original_pitch} -> {new_pitch}) did not change design hash. "
            f"REQ-M1-015 requires distinct specs to produce distinct hashes."
        )

    def test_enabling_ground_via_fence_changes_hash(self) -> None:
        """Enabling ground_via_fence must change the design hash vs null/disabled.

        REQ-M1-015: presence/absence of via fence affects physical layout.
        """
        baseline_data = self._get_baseline_spec_data()

        # Ensure baseline has no via fence
        baseline_spec = CouponSpec.model_validate(baseline_data)
        assert baseline_spec.transmission_line.ground_via_fence is None, (
            "Baseline spec should have ground_via_fence=null for this test"
        )
        baseline_resolved = resolve(baseline_spec)
        baseline_hash = design_hash(baseline_resolved)

        # Create spec with ground_via_fence enabled
        fence_spec_data = copy.deepcopy(baseline_data)
        _deep_set(fence_spec_data, "transmission_line.ground_via_fence", {
            "enabled": True,
            "pitch_nm": 1000000,
            "offset_from_gap_nm": 200000,
            "via": {
                "drill_nm": 300000,
                "diameter_nm": 600000,
            }
        })

        fence_spec = CouponSpec.model_validate(fence_spec_data)
        fence_resolved = resolve(fence_spec)
        fence_hash = design_hash(fence_resolved)

        assert baseline_hash != fence_hash, (
            "Enabling ground_via_fence did not change design hash. "
            "REQ-M1-015 requires distinct specs to produce distinct hashes."
        )

    @pytest.mark.parametrize("field_path,delta", [
        ("transmission_line.w_nm", 10000),  # trace width
        ("transmission_line.length_left_nm", 1000000),  # trace length
        ("board.outline.width_nm", 1000000),  # board width
        ("board.outline.length_nm", 1000000),  # board length
    ])
    def test_additional_nm_mutations_change_hash(self, field_path: str, delta: int) -> None:
        """Verify other key _nm fields also produce distinct hashes when changed.

        REQ-M1-015: Any dimensional parameter change should affect the design hash.
        """
        baseline_data = self._get_baseline_spec_data()
        baseline_spec = CouponSpec.model_validate(baseline_data)
        baseline_resolved = resolve(baseline_spec)
        baseline_hash = design_hash(baseline_resolved)

        # Mutate the field
        mutated_data = copy.deepcopy(baseline_data)
        original_value = _deep_get(mutated_data, field_path)
        new_value = original_value + delta
        _deep_set(mutated_data, field_path, new_value)

        mutated_spec = CouponSpec.model_validate(mutated_data)
        mutated_resolved = resolve(mutated_spec)
        mutated_hash = design_hash(mutated_resolved)

        assert baseline_hash != mutated_hash, (
            f"{field_path} mutation ({original_value} -> {new_value}) did not change design hash. "
            f"REQ-M1-015 requires distinct specs to produce distinct hashes."
        )

    def test_multiple_mutations_compound(self) -> None:
        """Multiple mutations should produce hashes distinct from single mutations.

        REQ-M1-015: Compound changes should not accidentally cancel out.
        """
        baseline_data = self._get_baseline_spec_data()

        # Single mutation: gap_nm
        single_data = copy.deepcopy(baseline_data)
        original_gap = _deep_get(single_data, "transmission_line.gap_nm")
        _deep_set(single_data, "transmission_line.gap_nm", original_gap + 10000)
        single_spec = CouponSpec.model_validate(single_data)
        single_hash = design_hash(resolve(single_spec))

        # Compound mutation: gap_nm + corner_radius_nm
        compound_data = copy.deepcopy(baseline_data)
        _deep_set(compound_data, "transmission_line.gap_nm", original_gap + 10000)
        original_radius = _deep_get(compound_data, "board.outline.corner_radius_nm")
        _deep_set(compound_data, "board.outline.corner_radius_nm", original_radius + 500000)
        compound_spec = CouponSpec.model_validate(compound_data)
        compound_hash = design_hash(resolve(compound_spec))

        assert single_hash != compound_hash, (
            "Compound mutation produced same hash as single mutation. "
            "REQ-M1-015 requires distinct specs to produce distinct hashes."
        )

    def test_f1_mutation_via_parameters_change_hash(self) -> None:
        """F1 specs: changing discontinuity via parameters must change hash.

        REQ-M1-015: Via diameter/drill affects physical geometry.
        """
        f1_specs = _f1_golden_specs()
        if not f1_specs:
            pytest.skip("No F1 golden specs available")

        baseline_data = json.loads(f1_specs[0].read_text(encoding="utf-8"))
        baseline_spec = CouponSpec.model_validate(baseline_data)
        baseline_resolved = resolve(baseline_spec)
        baseline_hash = design_hash(baseline_resolved)

        # Mutate signal_via diameter
        mutated_data = copy.deepcopy(baseline_data)
        original_diameter = _deep_get(mutated_data, "discontinuity.signal_via.diameter_nm")
        new_diameter = original_diameter + 50000  # Add 50um
        _deep_set(mutated_data, "discontinuity.signal_via.diameter_nm", new_diameter)

        mutated_spec = CouponSpec.model_validate(mutated_data)
        mutated_resolved = resolve(mutated_spec)
        mutated_hash = design_hash(mutated_resolved)

        assert baseline_hash != mutated_hash, (
            f"signal_via.diameter_nm mutation ({original_diameter} -> {new_diameter}) did not change design hash. "
            f"REQ-M1-015 requires distinct specs to produce distinct hashes."
        )


# =============================================================================
# Module-level wrapper test for DESIGN_DOCUMENT.md Test Matrix
# =============================================================================


def test_resolve_determinism() -> None:
    """Wrapper test for REQ-M1-012, REQ-M1-014, and REQ-M1-015.

    This test aggregates key assertions for the requirements mapped to
    tests/test_resolve_determinism.py::test_resolve_determinism in Test Matrix.

    REQ-M1-012: CLI and Python APIs MUST call single canonical build pipeline.
    REQ-M1-014: Derived features and dimensionless groups MUST be emitted
                deterministically in manifest.json.
    REQ-M1-015: The test suite MUST include mutation/coverage tests proving
                that changing key fields changes geometry/artifact hashes.
    """
    # REQ-M1-012: Verify single canonical pipeline exists
    from formula_foundry.coupongen import build_coupon

    # build_coupon is the canonical entry point
    assert callable(build_coupon), "REQ-M1-012: build_coupon must be the canonical pipeline"

    # REQ-M1-014: resolve produces deterministic resolved output
    golden_specs = _golden_spec_files()
    if not golden_specs:
        pytest.skip("No golden specs available")

    spec_data = json.loads(golden_specs[0].read_text(encoding="utf-8"))
    spec = CouponSpec.model_validate(spec_data)

    # Resolve twice and verify determinism
    resolved_a = resolve(spec)
    resolved_b = resolve(spec)
    hash_a = design_hash(resolved_a)
    hash_b = design_hash(resolved_b)

    assert hash_a == hash_b, "REQ-M1-014: Resolve must be deterministic"
    assert len(hash_a) == 64, "REQ-M1-014: Design hash must be 64-char hex"

    # REQ-M1-015: Mutation test - changing gap_nm changes hash
    mutated_data = copy.deepcopy(spec_data)
    original_gap = _deep_get(mutated_data, "transmission_line.gap_nm")
    _deep_set(mutated_data, "transmission_line.gap_nm", original_gap + 10000)

    mutated_spec = CouponSpec.model_validate(mutated_data)
    mutated_hash = design_hash(resolve(mutated_spec))

    assert hash_a != mutated_hash, (
        f"REQ-M1-015: gap_nm mutation ({original_gap} -> {original_gap + 10000}) "
        "must change design hash"
    )
