"""Shared types for spec consumption tracking.

This module provides the SpecConsumption model which tracks consumed, expected,
and provided paths during CouponSpec resolution.

Satisfies REQ-M1-001:
    - The generator MUST track and emit spec consumption (consumed paths,
      expected paths, unused provided paths)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SpecConsumption(BaseModel):
    """Model for tracking spec consumption during resolution.

    This tracks which paths were consumed, expected, and provided during
    CouponSpec resolution. Used for validation in strict mode and emitted
    in the manifest for traceability.

    Attributes:
        consumed_paths: Paths that were actually consumed during resolution.
        expected_paths: Paths that are expected for the coupon family.
        provided_paths: Paths that were provided in the input spec.
    """

    model_config = ConfigDict(frozen=True)

    consumed_paths: frozenset[str]
    expected_paths: frozenset[str]
    provided_paths: frozenset[str]

    @property
    def unused_provided_paths(self) -> frozenset[str]:
        """Paths that were provided but not consumed."""
        return self.provided_paths - self.consumed_paths

    @property
    def unconsumed_expected_paths(self) -> frozenset[str]:
        """Expected paths that were not consumed."""
        return self.expected_paths - self.consumed_paths

    @property
    def is_fully_covered(self) -> bool:
        """True if all expected paths are consumed and no provided paths are unused."""
        return not self.unused_provided_paths and not self.unconsumed_expected_paths

    @property
    def coverage_ratio(self) -> float:
        """Ratio of consumed expected paths to total expected paths.

        Returns 1.0 if there are no expected paths.
        """
        if not self.expected_paths:
            return 1.0
        consumed_expected = self.consumed_paths & self.expected_paths
        return len(consumed_expected) / len(self.expected_paths)

    def to_summary_dict(self) -> dict[str, list[str]]:
        """Convert to a dictionary suitable for manifest emission.

        Returns sorted lists for deterministic JSON output.
        """
        return {
            "consumed_paths": sorted(self.consumed_paths),
            "expected_paths": sorted(self.expected_paths),
            "provided_paths": sorted(self.provided_paths),
            "unused_provided_paths": sorted(self.unused_provided_paths),
            "unconsumed_expected_paths": sorted(self.unconsumed_expected_paths),
        }
