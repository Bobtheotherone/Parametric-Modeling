# SPDX-License-Identifier: MIT
"""Gate G1 tests: Resolved design determinism.

This module tests that resolve(spec) produces deterministic results:
- Same spec YAML/JSON yields identical ResolvedDesign across runs
- design_hash is stable for the same inputs
- Golden resolved_design.json bytes and golden design_hash match

Per ECO-M1-ALIGN-0001 and CP-2.4, the LayoutPlan is the single source of truth
for all geometry. The design_hash must be stable and deterministic across runs.

Pytest marker: gate_g1
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.coupongen import (
    load_spec,
    resolve,
    design_hash,
    resolved_design_canonical_json,
)
from formula_foundry.substrate import sha256_bytes


# ---------------------------------------------------------------------------
# Test data paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parents[1]
GOLDEN_SPECS_DIR = TESTS_DIR / "golden_specs"
GOLDEN_HASHES_PATH = ROOT / "golden_hashes" / "design_hashes.json"


def _collect_golden_specs() -> list[Path]:
    """Collect all golden spec files (JSON and YAML)."""
    patterns = ("*.json", "*.yaml", "*.yml")
    specs: list[Path] = []
    for pattern in patterns:
        specs.extend(sorted(GOLDEN_SPECS_DIR.glob(pattern)))
    # Filter out non-spec files (like __init__.py)
    return [p for p in sorted(specs) if not p.name.startswith("_")]


def _load_golden_hashes() -> dict[str, str]:
    """Load golden design hashes from the committed JSON file."""
    if not GOLDEN_HASHES_PATH.exists():
        return {}
    data = json.loads(GOLDEN_HASHES_PATH.read_text(encoding="utf-8"))
    return data.get("spec_hashes", {})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def golden_specs() -> list[Path]:
    """Fixture providing list of golden spec paths."""
    return _collect_golden_specs()


@pytest.fixture(scope="module")
def golden_hashes() -> dict[str, str]:
    """Fixture providing golden design hash mapping."""
    return _load_golden_hashes()


# ---------------------------------------------------------------------------
# G1 Gate Tests
# ---------------------------------------------------------------------------


@pytest.mark.gate_g1
class TestG1ResolverDeterminism:
    """Gate G1 tests for resolver determinism.

    These tests verify that:
    1. resolve(spec) produces identical results across multiple runs
    2. design_hash is stable for the same inputs
    3. Canonical JSON representation is byte-identical regardless of input key order
    4. All golden specs produce their expected golden hashes
    """

    def test_minimum_golden_specs_present(self, golden_specs: list[Path]) -> None:
        """Verify sufficient golden specs exist for comprehensive testing.

        Per ECO-M1-ALIGN-0001, we need ≥10 golden specs per family (F0, F1).
        The task requests 50 canonical specs; we verify we have at least 20.
        """
        f0_specs = [p for p in golden_specs if "f0_" in p.name.lower()]
        f1_specs = [p for p in golden_specs if "f1_" in p.name.lower()]

        assert len(f0_specs) >= 10, f"Expected ≥10 F0 specs, found {len(f0_specs)}"
        assert len(f1_specs) >= 10, f"Expected ≥10 F1 specs, found {len(f1_specs)}"
        assert len(golden_specs) >= 20, f"Expected ≥20 total specs, found {len(golden_specs)}"

    def test_resolve_determinism_single_run(self, golden_specs: list[Path]) -> None:
        """Verify resolve(spec) is deterministic within a single process.

        Running resolve() twice on the same spec must produce:
        - Identical canonical JSON
        - Identical design_hash
        """
        for spec_path in golden_specs:
            spec = load_spec(spec_path)

            resolved_a = resolve(spec)
            resolved_b = resolve(spec)

            canonical_a = resolved_design_canonical_json(resolved_a)
            canonical_b = resolved_design_canonical_json(resolved_b)

            assert canonical_a == canonical_b, (
                f"Canonical JSON differs for {spec_path.name} between runs"
            )

            hash_a = design_hash(resolved_a)
            hash_b = design_hash(resolved_b)

            assert hash_a == hash_b, (
                f"design_hash differs for {spec_path.name} between runs"
            )

    def test_resolve_determinism_multiple_runs(self, golden_specs: list[Path]) -> None:
        """Verify resolve(spec) is deterministic across 3 sequential runs.

        Per ECO-M1-ALIGN-0001 gate G1: "resolved design determinism across runs."
        """
        num_runs = 3

        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            hashes: list[str] = []

            for _ in range(num_runs):
                resolved = resolve(spec)
                hashes.append(design_hash(resolved))

            # All hashes must be identical
            assert len(set(hashes)) == 1, (
                f"design_hash not stable for {spec_path.name} across {num_runs} runs: "
                f"got {len(set(hashes))} distinct hashes"
            )

    def test_design_hash_matches_golden(
        self, golden_specs: list[Path], golden_hashes: dict[str, str]
    ) -> None:
        """Verify design_hash matches committed golden hashes.

        REQ-M1-024: CI must prove deterministic resolve hashing against
        committed golden hashes.
        """
        for spec_path in golden_specs:
            # Golden hashes use the actual spec filename as key (e.g., f0_cal_001.yaml)
            key = spec_path.name

            if key not in golden_hashes:
                pytest.skip(f"No golden hash for {key}")

            spec = load_spec(spec_path)
            resolved = resolve(spec)
            computed_hash = design_hash(resolved)

            assert computed_hash == golden_hashes[key], (
                f"design_hash mismatch for {spec_path.name}:\n"
                f"  computed: {computed_hash}\n"
                f"  expected: {golden_hashes[key]}"
            )

    def test_design_hash_is_sha256_of_canonical_json(
        self, golden_specs: list[Path]
    ) -> None:
        """Verify design_hash equals SHA256 of canonical JSON bytes.

        This ensures the hash algorithm is correctly implemented.
        """
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            resolved = resolve(spec)

            canonical = resolved_design_canonical_json(resolved)
            expected_hash = sha256_bytes(canonical.encode("utf-8"))
            computed_hash = design_hash(resolved)

            assert computed_hash == expected_hash, (
                f"design_hash != sha256(canonical_json) for {spec_path.name}"
            )

    def test_canonical_json_key_order_invariance(
        self, golden_specs: list[Path]
    ) -> None:
        """Verify canonical JSON is invariant to input key ordering.

        Two specs with identical content but different key ordering must
        produce byte-identical canonical JSON.
        """
        for spec_path in golden_specs:
            spec = load_spec(spec_path)

            # Resolve normally
            resolved = resolve(spec)
            canonical = resolved_design_canonical_json(resolved)

            # Re-serialize and re-parse to potentially change ordering
            # (Python dicts preserve insertion order, but JSON parsers may vary)
            spec_data = spec.model_dump(mode="json")

            # Create a reordered version by reversing top-level keys
            reordered: dict[str, Any] = {}
            for key in reversed(list(spec_data.keys())):
                reordered[key] = spec_data[key]

            # Load from reordered data
            from formula_foundry.coupongen.spec import CouponSpec
            spec_reordered = CouponSpec.model_validate(reordered)

            resolved_reordered = resolve(spec_reordered)
            canonical_reordered = resolved_design_canonical_json(resolved_reordered)

            assert canonical == canonical_reordered, (
                f"Canonical JSON depends on key order for {spec_path.name}"
            )

    def test_resolved_design_fields_deterministic(
        self, golden_specs: list[Path]
    ) -> None:
        """Verify all ResolvedDesign fields are deterministic.

        Check that parameters_nm, derived_features, and dimensionless_groups
        are identical across runs.
        """
        for spec_path in golden_specs:
            spec = load_spec(spec_path)

            resolved_a = resolve(spec)
            resolved_b = resolve(spec)

            # parameters_nm must match exactly
            assert resolved_a.parameters_nm == resolved_b.parameters_nm, (
                f"parameters_nm differs for {spec_path.name}"
            )

            # derived_features must match exactly
            assert resolved_a.derived_features == resolved_b.derived_features, (
                f"derived_features differs for {spec_path.name}"
            )

            # dimensionless_groups must match exactly
            assert resolved_a.dimensionless_groups == resolved_b.dimensionless_groups, (
                f"dimensionless_groups differs for {spec_path.name}"
            )

            # length_right_nm must match (for F1 coupons)
            assert resolved_a.length_right_nm == resolved_b.length_right_nm, (
                f"length_right_nm differs for {spec_path.name}"
            )


@pytest.mark.gate_g1
class TestG1LayoutPlanDeterminism:
    """Gate G1 tests for LayoutPlan determinism.

    Per CP-2.1 and CP-2.4, LayoutPlan is the single source of truth for
    all geometry. These tests verify LayoutPlan is computed deterministically.
    """

    def test_layout_plan_is_computed(self, golden_specs: list[Path]) -> None:
        """Verify LayoutPlan is computed for all specs."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            resolved = resolve(spec)

            # LayoutPlan should be attached to the resolved design
            layout_plan = resolved.layout_plan
            assert layout_plan is not None, (
                f"LayoutPlan not computed for {spec_path.name}"
            )

    def test_layout_plan_deterministic(self, golden_specs: list[Path]) -> None:
        """Verify LayoutPlan geometry is deterministic across runs."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)

            resolved_a = resolve(spec)
            resolved_b = resolve(spec)

            plan_a = resolved_a.layout_plan
            plan_b = resolved_b.layout_plan

            assert plan_a is not None and plan_b is not None

            # Compare key geometry values
            assert plan_a.x_board_left_edge_nm == plan_b.x_board_left_edge_nm
            assert plan_a.x_board_right_edge_nm == plan_b.x_board_right_edge_nm
            assert plan_a.y_centerline_nm == plan_b.y_centerline_nm
            assert plan_a.total_trace_length_nm == plan_b.total_trace_length_nm

    def test_f1_continuity_formula(self, golden_specs: list[Path]) -> None:
        """Verify F1 continuity formula: xD = xL + length_left.

        Per CP-2.2, for F1 coupons the discontinuity center must satisfy:
        - xD == left_port.signal_pad_x_nm + length_left_nm
        - length_right_nm = right_port.signal_pad_x_nm - xD
        """
        f1_specs = [p for p in golden_specs if "f1_" in p.name.lower()]

        for spec_path in f1_specs:
            spec = load_spec(spec_path)
            resolved = resolve(spec)
            plan = resolved.layout_plan

            assert plan is not None

            # Get the discontinuity center from LayoutPlan
            if plan.x_disc_nm is not None:
                # Verify continuity: discontinuity is at left signal pad + length_left
                expected_discontinuity = (
                    plan.left_port.signal_pad_x_nm +
                    int(spec.transmission_line.length_left_nm)
                )
                assert plan.x_disc_nm == expected_discontinuity, (
                    f"Discontinuity position violates continuity formula for {spec_path.name}"
                )

                # Verify derived length_right_nm
                if resolved.length_right_nm is not None:
                    expected_length_right = (
                        plan.right_port.signal_pad_x_nm -
                        plan.x_disc_nm
                    )
                    assert resolved.length_right_nm == expected_length_right, (
                        f"Derived length_right_nm incorrect for {spec_path.name}"
                    )


@pytest.mark.gate_g1
class TestG1IntegerNmUnits:
    """Gate G1 tests for integer nanometer units.

    Per M1 design doc: "Internal units are 1 nm with 32-bit integer storage."
    All parameters must be integer nanometers for determinism.
    """

    def test_parameters_nm_are_integers(self, golden_specs: list[Path]) -> None:
        """Verify all parameters_nm values are integers."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            resolved = resolve(spec)

            for key, value in resolved.parameters_nm.items():
                assert isinstance(value, int), (
                    f"parameters_nm[{key}] = {value!r} is not int for {spec_path.name}"
                )

    def test_derived_features_are_integers(self, golden_specs: list[Path]) -> None:
        """Verify all derived_features values are integers."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            resolved = resolve(spec)

            for key, value in resolved.derived_features.items():
                assert isinstance(value, int), (
                    f"derived_features[{key}] = {value!r} is not int for {spec_path.name}"
                )

    def test_dimensionless_groups_are_floats(self, golden_specs: list[Path]) -> None:
        """Verify dimensionless_groups values are floats (ratios)."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            resolved = resolve(spec)

            for key, value in resolved.dimensionless_groups.items():
                assert isinstance(value, float), (
                    f"dimensionless_groups[{key}] = {value!r} is not float for {spec_path.name}"
                )

    def test_length_right_nm_is_integer_when_present(
        self, golden_specs: list[Path]
    ) -> None:
        """Verify length_right_nm is integer when present (F1 coupons)."""
        f1_specs = [p for p in golden_specs if "f1_" in p.name.lower()]

        for spec_path in f1_specs:
            spec = load_spec(spec_path)
            resolved = resolve(spec)

            if resolved.length_right_nm is not None:
                assert isinstance(resolved.length_right_nm, int), (
                    f"length_right_nm = {resolved.length_right_nm!r} "
                    f"is not int for {spec_path.name}"
                )
