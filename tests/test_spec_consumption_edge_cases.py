# SPDX-License-Identifier: MIT
"""Edge case tests for SpecConsumption types and consumption tracking.

This module provides additional edge case coverage for the SpecConsumption
model from formula_foundry.resolve.types, complementing the tests in
test_spec_lint.py and test_schema.py.

Satisfies:
    - REQ-M1-001: The generator MUST track and emit spec consumption
                  (consumed paths, expected paths, unused provided paths)
    - REQ-M1-018: Spec consumption tracking for strict mode validation
"""

from __future__ import annotations

import json
from typing import Any

import pytest


class TestSpecConsumptionDeterminism:
    """Tests for deterministic behavior of SpecConsumption."""

    def test_to_summary_dict_keys_are_sorted(self) -> None:
        """to_summary_dict output keys should be deterministic."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset({"z_path", "a_path", "m_path"}),
            expected_paths=frozenset({"z_path", "a_path"}),
            provided_paths=frozenset({"z_path", "a_path", "m_path", "b_path"}),
        )

        summary = consumption.to_summary_dict()

        # All list values should be sorted
        assert summary["consumed_paths"] == ["a_path", "m_path", "z_path"]
        assert summary["expected_paths"] == ["a_path", "z_path"]
        assert summary["provided_paths"] == ["a_path", "b_path", "m_path", "z_path"]

    def test_to_summary_dict_json_serializable(self) -> None:
        """to_summary_dict output must be JSON serializable."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset({"path.a", "path.b"}),
            expected_paths=frozenset({"path.a", "path.b", "path.c"}),
            provided_paths=frozenset({"path.a", "path.b", "path.d"}),
        )

        summary = consumption.to_summary_dict()

        # Should not raise
        json_str = json.dumps(summary, sort_keys=True)
        parsed = json.loads(json_str)

        assert parsed["consumed_paths"] == ["path.a", "path.b"]

    def test_to_summary_dict_repeated_calls_identical(self) -> None:
        """Multiple calls to to_summary_dict produce identical output."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset({"x", "y", "z"}),
            expected_paths=frozenset({"x", "y"}),
            provided_paths=frozenset({"x", "y", "z", "w"}),
        )

        summary1 = consumption.to_summary_dict()
        summary2 = consumption.to_summary_dict()

        assert summary1 == summary2

    def test_json_deterministic_across_instances(self) -> None:
        """Two SpecConsumption instances with same data produce same JSON."""
        from formula_foundry.resolve.types import SpecConsumption

        paths_consumed = frozenset({"a", "b", "c"})
        paths_expected = frozenset({"a", "b"})
        paths_provided = frozenset({"a", "b", "c", "d"})

        c1 = SpecConsumption(
            consumed_paths=paths_consumed,
            expected_paths=paths_expected,
            provided_paths=paths_provided,
        )

        c2 = SpecConsumption(
            consumed_paths=paths_consumed,
            expected_paths=paths_expected,
            provided_paths=paths_provided,
        )

        json1 = json.dumps(c1.to_summary_dict(), sort_keys=True)
        json2 = json.dumps(c2.to_summary_dict(), sort_keys=True)

        assert json1 == json2


class TestSpecConsumptionEdgeCases:
    """Edge case tests for SpecConsumption properties."""

    def test_empty_all_sets(self) -> None:
        """SpecConsumption with all empty sets."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset(),
            expected_paths=frozenset(),
            provided_paths=frozenset(),
        )

        assert consumption.unused_provided_paths == frozenset()
        assert consumption.unconsumed_expected_paths == frozenset()
        assert consumption.is_fully_covered
        assert consumption.coverage_ratio == 1.0

    def test_only_consumed_paths(self) -> None:
        """Consumed paths with no expected or provided."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset({"a", "b", "c"}),
            expected_paths=frozenset(),
            provided_paths=frozenset(),
        )

        # All consumed but none expected or provided
        assert consumption.unused_provided_paths == frozenset()
        assert consumption.unconsumed_expected_paths == frozenset()
        assert consumption.is_fully_covered
        assert consumption.coverage_ratio == 1.0

    def test_consumed_superset_of_expected(self) -> None:
        """Consumed paths are a superset of expected paths."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset({"a", "b", "c", "d"}),
            expected_paths=frozenset({"a", "b"}),
            provided_paths=frozenset({"a", "b", "c", "d"}),
        )

        assert consumption.unconsumed_expected_paths == frozenset()
        assert consumption.unused_provided_paths == frozenset()
        assert consumption.is_fully_covered
        assert consumption.coverage_ratio == 1.0

    def test_consumed_subset_of_expected(self) -> None:
        """Consumed paths are a subset of expected paths."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset({"a"}),
            expected_paths=frozenset({"a", "b", "c", "d"}),
            provided_paths=frozenset({"a", "b", "c", "d"}),
        )

        assert consumption.unconsumed_expected_paths == frozenset({"b", "c", "d"})
        assert consumption.unused_provided_paths == frozenset({"b", "c", "d"})
        assert not consumption.is_fully_covered
        assert consumption.coverage_ratio == pytest.approx(0.25)

    def test_disjoint_consumed_and_expected(self) -> None:
        """Consumed and expected paths have no overlap."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset({"x", "y"}),
            expected_paths=frozenset({"a", "b"}),
            provided_paths=frozenset({"a", "b", "x", "y"}),
        )

        assert consumption.unconsumed_expected_paths == frozenset({"a", "b"})
        assert consumption.unused_provided_paths == frozenset({"a", "b"})
        assert not consumption.is_fully_covered
        assert consumption.coverage_ratio == 0.0

    def test_coverage_ratio_partial(self) -> None:
        """Coverage ratio with partial coverage."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset({"a", "b", "c"}),
            expected_paths=frozenset({"a", "b", "c", "d", "e", "f"}),
            provided_paths=frozenset({"a", "b", "c", "d", "e", "f"}),
        )

        # 3 out of 6 expected paths consumed
        assert consumption.coverage_ratio == pytest.approx(0.5)
        assert not consumption.is_fully_covered

    def test_is_fully_covered_requires_both_conditions(self) -> None:
        """is_fully_covered requires no unused AND no unconsumed."""
        from formula_foundry.resolve.types import SpecConsumption

        # Case 1: All expected consumed but has unused provided
        c1 = SpecConsumption(
            consumed_paths=frozenset({"a", "b"}),
            expected_paths=frozenset({"a", "b"}),
            provided_paths=frozenset({"a", "b", "extra"}),
        )
        assert c1.unconsumed_expected_paths == frozenset()
        assert c1.unused_provided_paths == frozenset({"extra"})
        assert not c1.is_fully_covered

        # Case 2: No unused but has unconsumed
        c2 = SpecConsumption(
            consumed_paths=frozenset({"a"}),
            expected_paths=frozenset({"a", "b"}),
            provided_paths=frozenset({"a"}),
        )
        assert c2.unused_provided_paths == frozenset()
        assert c2.unconsumed_expected_paths == frozenset({"b"})
        assert not c2.is_fully_covered

    def test_frozenset_immutability(self) -> None:
        """SpecConsumption fields are immutable frozensets."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset({"a"}),
            expected_paths=frozenset({"a", "b"}),
            provided_paths=frozenset({"a", "b"}),
        )

        # Properties return frozensets which are immutable
        unused = consumption.unused_provided_paths
        unconsumed = consumption.unconsumed_expected_paths

        # These should be frozen and not modifiable
        with pytest.raises(AttributeError):
            unused.add("x")  # type: ignore[attr-defined]

        with pytest.raises(AttributeError):
            unconsumed.add("y")  # type: ignore[attr-defined]


class TestSpecConsumptionPathFormats:
    """Tests for different path format handling in SpecConsumption."""

    def test_dotted_path_format(self) -> None:
        """Handles dotted path format (e.g., 'board.outline.width_nm')."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset(
                {
                    "board.outline.width_nm",
                    "board.outline.length_nm",
                    "transmission_line.w_nm",
                }
            ),
            expected_paths=frozenset(
                {
                    "board.outline.width_nm",
                    "board.outline.length_nm",
                    "transmission_line.w_nm",
                    "transmission_line.gap_nm",
                }
            ),
            provided_paths=frozenset(
                {
                    "board.outline.width_nm",
                    "board.outline.length_nm",
                    "transmission_line.w_nm",
                    "extra.path",
                }
            ),
        )

        assert "transmission_line.gap_nm" in consumption.unconsumed_expected_paths
        assert "extra.path" in consumption.unused_provided_paths

    def test_paths_with_special_characters(self) -> None:
        """Handles paths with various characters."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset(
                {
                    "path_with_underscore",
                    "path-with-dash",
                    "path.with.dots",
                }
            ),
            expected_paths=frozenset(
                {
                    "path_with_underscore",
                    "path-with-dash",
                    "path.with.dots",
                }
            ),
            provided_paths=frozenset(
                {
                    "path_with_underscore",
                    "path-with-dash",
                    "path.with.dots",
                }
            ),
        )

        assert consumption.is_fully_covered

    def test_numeric_path_segments(self) -> None:
        """Handles paths with numeric segments (e.g., array indices)."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset(
                {
                    "layers.0.thickness_nm",
                    "layers.1.thickness_nm",
                }
            ),
            expected_paths=frozenset(
                {
                    "layers.0.thickness_nm",
                    "layers.1.thickness_nm",
                    "layers.2.thickness_nm",
                }
            ),
            provided_paths=frozenset(
                {
                    "layers.0.thickness_nm",
                    "layers.1.thickness_nm",
                }
            ),
        )

        assert "layers.2.thickness_nm" in consumption.unconsumed_expected_paths
        assert consumption.coverage_ratio == pytest.approx(2 / 3)


class TestSpecConsumptionModel:
    """Tests for SpecConsumption model configuration."""

    def test_model_is_frozen(self) -> None:
        """SpecConsumption model is immutable (frozen config)."""
        from formula_foundry.resolve.types import SpecConsumption

        consumption = SpecConsumption(
            consumed_paths=frozenset({"a"}),
            expected_paths=frozenset({"a"}),
            provided_paths=frozenset({"a"}),
        )

        # Pydantic frozen models raise ValidationError on assignment
        with pytest.raises(Exception):  # ValidationError or AttributeError
            consumption.consumed_paths = frozenset({"b"})  # type: ignore[misc]

    def test_model_hashable(self) -> None:
        """SpecConsumption should be hashable (frozen model)."""
        from formula_foundry.resolve.types import SpecConsumption

        c1 = SpecConsumption(
            consumed_paths=frozenset({"a"}),
            expected_paths=frozenset({"a"}),
            provided_paths=frozenset({"a"}),
        )

        c2 = SpecConsumption(
            consumed_paths=frozenset({"a"}),
            expected_paths=frozenset({"a"}),
            provided_paths=frozenset({"a"}),
        )

        # Should be hashable
        _ = hash(c1)

        # Same data should produce same hash
        assert hash(c1) == hash(c2)

        # Can be used in sets
        consumption_set = {c1, c2}
        assert len(consumption_set) == 1
