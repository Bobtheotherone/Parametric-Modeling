# SPDX-License-Identifier: MIT
"""Tests for spec consumption enforcement and SpecConsumptionError handling.

This module provides focused tests for the enforce_spec_consumption function
and SpecConsumptionError exception class from formula_foundry.resolve.consumption.

Satisfies REQ-M1-001:
    - The generator MUST track and emit spec consumption (consumed paths,
      expected paths, unused provided paths)
    - MUST fail in strict mode if any provided field is unused or any
      expected field is unconsumed.
"""

from __future__ import annotations

import pytest

from formula_foundry.resolve.consumption import (
    SpecConsumptionError,
    build_spec_consumption,
    enforce_spec_consumption,
)
from formula_foundry.resolve.types import SpecConsumption


class TestEnforceSpecConsumption:
    """Tests for enforce_spec_consumption function."""

    def test_passes_when_fully_covered(self) -> None:
        """No error when all paths are consumed and no extras."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a", "b", "c"}),
            expected_paths=frozenset({"a", "b", "c"}),
            provided_paths=frozenset({"a", "b", "c"}),
        )

        # Should not raise
        enforce_spec_consumption(consumption)

    def test_passes_with_empty_sets(self) -> None:
        """No error when all sets are empty."""
        consumption = SpecConsumption(
            consumed_paths=frozenset(),
            expected_paths=frozenset(),
            provided_paths=frozenset(),
        )

        # Should not raise
        enforce_spec_consumption(consumption)

    def test_raises_on_unused_provided(self) -> None:
        """SpecConsumptionError raised when provided paths are unused."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a", "b"}),
            expected_paths=frozenset({"a", "b"}),
            provided_paths=frozenset({"a", "b", "unused1", "unused2"}),
        )

        with pytest.raises(SpecConsumptionError) as exc_info:
            enforce_spec_consumption(consumption)

        assert "unused provided paths" in str(exc_info.value)
        assert exc_info.value.unused_provided == frozenset({"unused1", "unused2"})
        assert exc_info.value.unconsumed_expected == frozenset()

    def test_raises_on_unconsumed_expected(self) -> None:
        """SpecConsumptionError raised when expected paths are unconsumed."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a"}),
            expected_paths=frozenset({"a", "b", "c"}),
            provided_paths=frozenset({"a"}),
        )

        with pytest.raises(SpecConsumptionError) as exc_info:
            enforce_spec_consumption(consumption)

        assert "unconsumed expected paths" in str(exc_info.value)
        assert exc_info.value.unconsumed_expected == frozenset({"b", "c"})
        assert exc_info.value.unused_provided == frozenset()

    def test_raises_on_both_violations(self) -> None:
        """SpecConsumptionError raised when both violations occur."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a"}),
            expected_paths=frozenset({"a", "expected1", "expected2"}),
            provided_paths=frozenset({"a", "extra1", "extra2"}),
        )

        with pytest.raises(SpecConsumptionError) as exc_info:
            enforce_spec_consumption(consumption)

        err = exc_info.value
        assert "unused provided paths" in str(err)
        assert "unconsumed expected paths" in str(err)
        assert err.unused_provided == frozenset({"extra1", "extra2"})
        assert err.unconsumed_expected == frozenset({"expected1", "expected2"})

    def test_error_message_contains_sorted_paths(self) -> None:
        """Error message contains sorted path lists for determinism."""
        consumption = SpecConsumption(
            consumed_paths=frozenset(),
            expected_paths=frozenset({"z_path", "a_path", "m_path"}),
            provided_paths=frozenset({"z_extra", "a_extra"}),
        )

        with pytest.raises(SpecConsumptionError) as exc_info:
            enforce_spec_consumption(consumption)

        error_str = str(exc_info.value)
        # Should contain sorted lists: ['a_extra', 'z_extra'] and ['a_path', 'm_path', 'z_path']
        assert "'a_extra'" in error_str
        assert "'a_path'" in error_str


class TestSpecConsumptionErrorAttributes:
    """Tests for SpecConsumptionError exception attributes."""

    def test_error_stores_unused_provided(self) -> None:
        """Error stores unused_provided as attribute."""
        unused = frozenset({"path1", "path2"})
        unconsumed = frozenset()

        err = SpecConsumptionError("test message", unused, unconsumed)

        assert err.unused_provided == unused

    def test_error_stores_unconsumed_expected(self) -> None:
        """Error stores unconsumed_expected as attribute."""
        unused = frozenset()
        unconsumed = frozenset({"expected1", "expected2"})

        err = SpecConsumptionError("test message", unused, unconsumed)

        assert err.unconsumed_expected == unconsumed

    def test_error_message_accessible(self) -> None:
        """Error message is accessible via str()."""
        message = "Custom error message"
        err = SpecConsumptionError(message, frozenset(), frozenset())

        assert message in str(err)

    def test_error_is_exception(self) -> None:
        """SpecConsumptionError is an Exception subclass."""
        err = SpecConsumptionError("msg", frozenset(), frozenset())
        assert isinstance(err, Exception)

    def test_error_catchable(self) -> None:
        """Error can be caught as generic Exception."""
        try:
            raise SpecConsumptionError("test", frozenset({"a"}), frozenset({"b"}))
        except Exception as e:
            assert isinstance(e, SpecConsumptionError)
            assert e.unused_provided == frozenset({"a"})
            assert e.unconsumed_expected == frozenset({"b"})


class TestSpecConsumptionProperties:
    """Tests for SpecConsumption computed properties related to enforcement."""

    def test_unused_provided_paths_computed(self) -> None:
        """unused_provided_paths is computed correctly."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a", "b"}),
            expected_paths=frozenset({"a", "b"}),
            provided_paths=frozenset({"a", "b", "c", "d"}),
        )

        assert consumption.unused_provided_paths == frozenset({"c", "d"})

    def test_unconsumed_expected_paths_computed(self) -> None:
        """unconsumed_expected_paths is computed correctly."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a"}),
            expected_paths=frozenset({"a", "b", "c"}),
            provided_paths=frozenset({"a"}),
        )

        assert consumption.unconsumed_expected_paths == frozenset({"b", "c"})

    def test_is_fully_covered_true(self) -> None:
        """is_fully_covered is True when no violations."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a", "b"}),
            expected_paths=frozenset({"a", "b"}),
            provided_paths=frozenset({"a", "b"}),
        )

        assert consumption.is_fully_covered is True
        assert consumption.unused_provided_paths == frozenset()
        assert consumption.unconsumed_expected_paths == frozenset()

    def test_is_fully_covered_false_unused(self) -> None:
        """is_fully_covered is False when unused provided paths exist."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a"}),
            expected_paths=frozenset({"a"}),
            provided_paths=frozenset({"a", "extra"}),
        )

        assert consumption.is_fully_covered is False

    def test_is_fully_covered_false_unconsumed(self) -> None:
        """is_fully_covered is False when unconsumed expected paths exist."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a"}),
            expected_paths=frozenset({"a", "missing"}),
            provided_paths=frozenset({"a"}),
        )

        assert consumption.is_fully_covered is False


class TestSpecConsumptionCoverageRatio:
    """Tests for coverage_ratio property."""

    def test_coverage_ratio_full_coverage(self) -> None:
        """coverage_ratio is 1.0 when all expected paths consumed."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a", "b", "c"}),
            expected_paths=frozenset({"a", "b", "c"}),
            provided_paths=frozenset({"a", "b", "c"}),
        )

        assert consumption.coverage_ratio == 1.0

    def test_coverage_ratio_no_expected(self) -> None:
        """coverage_ratio is 1.0 when no expected paths (vacuously true)."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a"}),
            expected_paths=frozenset(),
            provided_paths=frozenset({"a"}),
        )

        assert consumption.coverage_ratio == 1.0

    def test_coverage_ratio_partial(self) -> None:
        """coverage_ratio reflects partial coverage."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a", "b"}),
            expected_paths=frozenset({"a", "b", "c", "d"}),
            provided_paths=frozenset({"a", "b"}),
        )

        # 2 out of 4 expected consumed
        assert consumption.coverage_ratio == pytest.approx(0.5)

    def test_coverage_ratio_zero(self) -> None:
        """coverage_ratio is 0 when no expected paths consumed."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"x", "y"}),  # Different from expected
            expected_paths=frozenset({"a", "b"}),
            provided_paths=frozenset({"x", "y"}),
        )

        # 0 out of 2 expected consumed
        assert consumption.coverage_ratio == 0.0


class TestEnforcementEdgeCases:
    """Edge case tests for enforcement scenarios."""

    def test_consumed_superset_of_provided_is_ok(self) -> None:
        """Consumed paths can be superset of provided (e.g., defaults)."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a", "b", "default1", "default2"}),
            expected_paths=frozenset({"a", "b"}),
            provided_paths=frozenset({"a", "b"}),
        )

        # Should not raise - consumed can include defaults not from provided
        enforce_spec_consumption(consumption)

    def test_provided_subset_of_consumed_ok(self) -> None:
        """All provided paths consumed means no unused."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a", "b", "c", "d"}),
            expected_paths=frozenset({"a", "b"}),
            provided_paths=frozenset({"a", "b"}),
        )

        # All provided are consumed, all expected are consumed
        assert consumption.is_fully_covered
        enforce_spec_consumption(consumption)

    def test_consumed_empty_with_expected_fails(self) -> None:
        """Empty consumed with non-empty expected fails."""
        consumption = SpecConsumption(
            consumed_paths=frozenset(),
            expected_paths=frozenset({"a", "b"}),
            provided_paths=frozenset(),
        )

        with pytest.raises(SpecConsumptionError) as exc_info:
            enforce_spec_consumption(consumption)

        assert exc_info.value.unconsumed_expected == frozenset({"a", "b"})

    def test_disjoint_sets_both_violations(self) -> None:
        """Completely disjoint consumed/expected/provided = both violations."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"consumed1", "consumed2"}),
            expected_paths=frozenset({"expected1", "expected2"}),
            provided_paths=frozenset({"provided1", "provided2"}),
        )

        with pytest.raises(SpecConsumptionError) as exc_info:
            enforce_spec_consumption(consumption)

        err = exc_info.value
        # All expected are unconsumed (none in consumed)
        assert err.unconsumed_expected == frozenset({"expected1", "expected2"})
        # All provided are unused (none in consumed)
        assert err.unused_provided == frozenset({"provided1", "provided2"})


class TestSpecConsumptionDeterministicOutput:
    """Tests for deterministic output of SpecConsumption."""

    def test_to_summary_dict_sorted_lists(self) -> None:
        """to_summary_dict returns sorted lists for determinism."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"z", "a", "m"}),
            expected_paths=frozenset({"z", "a"}),
            provided_paths=frozenset({"z", "a", "m", "b"}),
        )

        summary = consumption.to_summary_dict()

        assert summary["consumed_paths"] == ["a", "m", "z"]
        assert summary["expected_paths"] == ["a", "z"]
        assert summary["provided_paths"] == ["a", "b", "m", "z"]

    def test_to_summary_dict_includes_computed_fields(self) -> None:
        """to_summary_dict includes computed unused/unconsumed."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"a"}),
            expected_paths=frozenset({"a", "b"}),
            provided_paths=frozenset({"a", "c"}),
        )

        summary = consumption.to_summary_dict()

        assert "unused_provided_paths" in summary
        assert "unconsumed_expected_paths" in summary
        assert summary["unused_provided_paths"] == ["c"]
        assert summary["unconsumed_expected_paths"] == ["b"]

    def test_repeated_to_summary_dict_calls_identical(self) -> None:
        """Multiple to_summary_dict calls return identical dicts."""
        consumption = SpecConsumption(
            consumed_paths=frozenset({"x", "y", "z"}),
            expected_paths=frozenset({"x", "y"}),
            provided_paths=frozenset({"x", "y", "z"}),
        )

        s1 = consumption.to_summary_dict()
        s2 = consumption.to_summary_dict()

        assert s1 == s2
