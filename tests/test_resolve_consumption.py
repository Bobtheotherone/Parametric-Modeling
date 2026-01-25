# SPDX-License-Identifier: MIT
"""Unit tests for resolve/consumption module helper functions.

Tests the internal helper functions for spec consumption tracking that are not
covered by existing tests.

Satisfies REQ-M1-001:
    - The generator MUST track and emit spec consumption (consumed paths,
      expected paths, unused provided paths)
"""

from __future__ import annotations

from typing import Any

import pytest

from formula_foundry.resolve.consumption import (
    _is_path_prefix_in,
    _walk_and_collect,
    collect_provided_paths,
    get_consumed_paths,
    get_expected_paths,
)
from formula_foundry.resolve.expectations import (
    F0_EXPECTED_PATHS,
    F1_EXPECTED_PATHS,
)


class TestWalkAndCollect:
    """Tests for _walk_and_collect internal function."""

    def test_collects_flat_dict(self) -> None:
        """Collects paths from a flat dictionary."""
        obj = {"a": 1, "b": "hello", "c": 3.14}
        paths: set[str] = set()
        _walk_and_collect(obj, paths, prefix="")

        assert paths == {"a", "b", "c"}

    def test_collects_nested_dict(self) -> None:
        """Collects paths from nested dictionaries."""
        obj = {
            "outer": {
                "inner": {
                    "value": 42,
                },
            },
        }
        paths: set[str] = set()
        _walk_and_collect(obj, paths, prefix="")

        assert "outer.inner.value" in paths

    def test_collects_with_prefix(self) -> None:
        """Collects paths with existing prefix."""
        obj = {"key": "value"}
        paths: set[str] = set()
        _walk_and_collect(obj, paths, prefix="root")

        assert paths == {"root.key"}

    def test_ignores_none_values(self) -> None:
        """None values are not collected."""
        obj = {"a": 1, "b": None, "c": 3}
        paths: set[str] = set()
        _walk_and_collect(obj, paths, prefix="")

        assert paths == {"a", "c"}
        assert "b" not in paths

    def test_handles_none_root(self) -> None:
        """None as root object is handled gracefully."""
        paths: set[str] = set()
        _walk_and_collect(None, paths, prefix="")

        assert paths == set()

    def test_collects_list_items(self) -> None:
        """Collects paths from list items with indices."""
        obj = {"items": [10, 20, 30]}
        paths: set[str] = set()
        _walk_and_collect(obj, paths, prefix="")

        assert "items[0]" in paths
        assert "items[1]" in paths
        assert "items[2]" in paths

    def test_collects_nested_list_dicts(self) -> None:
        """Collects paths from dicts nested in lists."""
        obj = {
            "ports": [
                {"name": "port1", "impedance": 50},
                {"name": "port2", "impedance": 75},
            ],
        }
        paths: set[str] = set()
        _walk_and_collect(obj, paths, prefix="")

        assert "ports[0].name" in paths
        assert "ports[0].impedance" in paths
        assert "ports[1].name" in paths
        assert "ports[1].impedance" in paths

    def test_ignores_none_in_list(self) -> None:
        """None values in lists are not collected."""
        obj = {"items": [1, None, 3]}
        paths: set[str] = set()
        _walk_and_collect(obj, paths, prefix="")

        assert "items[0]" in paths
        assert "items[1]" not in paths
        assert "items[2]" in paths

    def test_empty_dict(self) -> None:
        """Empty dict yields no paths."""
        obj: dict[str, Any] = {}
        paths: set[str] = set()
        _walk_and_collect(obj, paths, prefix="")

        assert paths == set()

    def test_empty_list(self) -> None:
        """Empty list yields no paths."""
        obj = {"empty": []}
        paths: set[str] = set()
        _walk_and_collect(obj, paths, prefix="")

        # Empty list itself is not a leaf, so nothing collected
        assert paths == set()


class TestIsPathPrefixIn:
    """Tests for _is_path_prefix_in internal function."""

    def test_exact_match(self) -> None:
        """Exact path match returns True."""
        paths = frozenset({"a.b.c", "x.y.z"})
        assert _is_path_prefix_in("a.b.c", paths)

    def test_prefix_match(self) -> None:
        """Path prefix with descendants returns True."""
        paths = frozenset({"a.b.c.d", "x.y.z"})
        assert _is_path_prefix_in("a.b.c", paths)

    def test_prefix_with_list_index(self) -> None:
        """Path prefix with list index descendants returns True."""
        paths = frozenset({"items[0]", "items[1]"})
        assert _is_path_prefix_in("items", paths)

    def test_no_match(self) -> None:
        """Non-matching path returns False."""
        paths = frozenset({"a.b.c", "x.y.z"})
        assert not _is_path_prefix_in("foo.bar", paths)

    def test_partial_no_match(self) -> None:
        """Partial key overlap without proper nesting returns False."""
        paths = frozenset({"abc.def"})
        # "ab" is not a proper prefix of "abc.def" (no dot or bracket)
        assert not _is_path_prefix_in("ab", paths)

    def test_empty_paths(self) -> None:
        """Empty paths set returns False."""
        paths: frozenset[str] = frozenset()
        assert not _is_path_prefix_in("any.path", paths)

    def test_deeply_nested_prefix(self) -> None:
        """Deeply nested prefix with descendants returns True."""
        paths = frozenset({"a.b.c.d.e.f.g"})
        assert _is_path_prefix_in("a.b.c", paths)


class TestGetExpectedPaths:
    """Tests for get_expected_paths function."""

    def test_returns_f0_paths_for_f0(self) -> None:
        """get_expected_paths returns F0 paths for F0 family."""
        paths = get_expected_paths("F0")
        assert paths == F0_EXPECTED_PATHS

    def test_returns_f1_paths_for_f1(self) -> None:
        """get_expected_paths returns F1 paths for F1 family."""
        paths = get_expected_paths("F1")
        assert paths == F1_EXPECTED_PATHS

    def test_returns_empty_for_unknown(self) -> None:
        """get_expected_paths returns empty for unknown family."""
        paths = get_expected_paths("UNKNOWN")
        assert paths == frozenset()


class TestCollectProvidedPaths:
    """Tests for collect_provided_paths function with CouponSpec."""

    @pytest.fixture
    def minimal_f0_spec_data(self) -> dict[str, Any]:
        """Return a minimal F0 spec data dictionary."""
        return {
            "schema_version": 1,
            "coupon_family": "F0",
            "units": "nm",
            "toolchain": {
                "kicad": {
                    "version": "9.0.7",
                    "docker_image": "kicad/kicad:9.0.7@sha256:abc",
                }
            },
            "fab_profile": {"id": "generic_4layer"},
            "stackup": {
                "copper_layers": 4,
                "thicknesses_nm": {"core": 800000},
                "materials": {"er": 4.5, "loss_tangent": 0.015},
            },
            "board": {
                "outline": {
                    "width_nm": 10000000,
                    "length_nm": 40000000,
                    "corner_radius_nm": 1000000,
                },
                "origin": {"mode": "EDGE_L_CENTER"},
                "text": {"coupon_id": "TEST", "include_manifest_hash": False},
            },
            "connectors": {
                "left": {
                    "footprint": "SMA:SMA_Vertical",
                    "position_nm": [2000000, 5000000],
                    "rotation_deg": 0,
                },
                "right": {
                    "footprint": "SMA:SMA_Vertical",
                    "position_nm": [38000000, 5000000],
                    "rotation_deg": 180,
                },
            },
            "transmission_line": {
                "type": "CPWG",
                "layer": "F.Cu",
                "w_nm": 200000,
                "gap_nm": 150000,
                "length_left_nm": 15000000,
                "length_right_nm": 15000000,
            },
            "constraints": {
                "mode": "REJECT",
                "drc": {"must_pass": True, "severity": "all"},
                "symmetry": {"enforce": True},
                "allow_unconnected_copper": False,
            },
            "export": {
                "gerbers": {"enabled": True, "format": "gerbers"},
                "drill": {"enabled": True, "format": "excellon"},
                "outputs_dir": "output/",
            },
        }

    def test_collects_from_couponspec(self, minimal_f0_spec_data: dict[str, Any]) -> None:
        """collect_provided_paths returns frozenset from CouponSpec."""
        from formula_foundry.coupongen.spec import load_couponspec

        spec = load_couponspec(minimal_f0_spec_data)
        paths = collect_provided_paths(spec)

        assert isinstance(paths, frozenset)
        assert len(paths) > 0

    def test_includes_schema_version(self, minimal_f0_spec_data: dict[str, Any]) -> None:
        """collect_provided_paths includes schema_version."""
        from formula_foundry.coupongen.spec import load_couponspec

        spec = load_couponspec(minimal_f0_spec_data)
        paths = collect_provided_paths(spec)

        assert "schema_version" in paths

    def test_includes_nested_paths(self, minimal_f0_spec_data: dict[str, Any]) -> None:
        """collect_provided_paths includes nested paths."""
        from formula_foundry.coupongen.spec import load_couponspec

        spec = load_couponspec(minimal_f0_spec_data)
        paths = collect_provided_paths(spec)

        assert "stackup.copper_layers" in paths
        assert "transmission_line.w_nm" in paths
        assert "board.outline.width_nm" in paths


class TestGetConsumedPaths:
    """Tests for get_consumed_paths function."""

    @pytest.fixture
    def f1_spec_with_fence(self) -> dict[str, Any]:
        """Return an F1 spec with ground via fence."""
        return {
            "schema_version": 1,
            "coupon_family": "F1",
            "units": "nm",
            "toolchain": {
                "kicad": {
                    "version": "9.0.7",
                    "docker_image": "kicad/kicad:9.0.7@sha256:test",
                }
            },
            "fab_profile": {"id": "oshpark_4layer"},
            "stackup": {
                "copper_layers": 4,
                "thicknesses_nm": {"core": 800000, "prepreg": 180000},
                "materials": {"er": 4.1, "loss_tangent": 0.02},
            },
            "board": {
                "outline": {
                    "width_nm": 20000000,
                    "length_nm": 80000000,
                    "corner_radius_nm": 2000000,
                },
                "origin": {"mode": "EDGE_L_CENTER"},
                "text": {"coupon_id": "TEST", "include_manifest_hash": True},
            },
            "connectors": {
                "left": {
                    "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                    "position_nm": [5000000, 0],
                    "rotation_deg": 180,
                },
                "right": {
                    "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                    "position_nm": [75000000, 0],
                    "rotation_deg": 0,
                },
            },
            "transmission_line": {
                "type": "CPWG",
                "layer": "F.Cu",
                "w_nm": 300000,
                "gap_nm": 180000,
                "length_left_nm": 25000000,
                "length_right_nm": 25000000,
                "ground_via_fence": {
                    "enabled": True,
                    "pitch_nm": 1500000,
                    "offset_from_gap_nm": 800000,
                    "via": {"drill_nm": 300000, "diameter_nm": 600000},
                },
            },
            "discontinuity": {
                "type": "VIA_TRANSITION",
                "signal_via": {
                    "drill_nm": 300000,
                    "diameter_nm": 650000,
                    "pad_diameter_nm": 900000,
                },
                "antipads": {},
                "return_vias": {
                    "pattern": "RING",
                    "count": 4,
                    "radius_nm": 1700000,
                    "via": {"drill_nm": 300000, "diameter_nm": 650000},
                },
                "plane_cutouts": {},
            },
            "constraints": {
                "mode": "REJECT",
                "drc": {"must_pass": True, "severity": "all"},
                "symmetry": {"enforce": True},
                "allow_unconnected_copper": False,
            },
            "export": {
                "gerbers": {"enabled": True, "format": "gerbers"},
                "drill": {"enabled": True, "format": "excellon"},
                "outputs_dir": "artifacts/",
            },
        }

    def test_returns_frozenset(self, f1_spec_with_fence: dict[str, Any]) -> None:
        """get_consumed_paths returns a frozenset."""
        from formula_foundry.coupongen.spec import load_couponspec

        spec = load_couponspec(f1_spec_with_fence)
        consumed = get_consumed_paths(spec)

        assert isinstance(consumed, frozenset)

    def test_includes_expected_paths(self, f1_spec_with_fence: dict[str, Any]) -> None:
        """get_consumed_paths includes expected paths that are provided."""
        from formula_foundry.coupongen.spec import load_couponspec

        spec = load_couponspec(f1_spec_with_fence)
        consumed = get_consumed_paths(spec)

        # Some expected F1 paths should be consumed
        assert len(consumed) > 0

    def test_includes_optional_fence_paths(self, f1_spec_with_fence: dict[str, Any]) -> None:
        """get_consumed_paths includes optional fence paths when provided."""
        from formula_foundry.coupongen.spec import load_couponspec

        spec = load_couponspec(f1_spec_with_fence)
        consumed = get_consumed_paths(spec)

        # Fence paths are optional but should be consumed when present
        fence_paths = [p for p in consumed if "ground_via_fence" in p]
        assert len(fence_paths) > 0

    def test_includes_return_via_paths(self, f1_spec_with_fence: dict[str, Any]) -> None:
        """get_consumed_paths includes return via paths when provided."""
        from formula_foundry.coupongen.spec import load_couponspec

        spec = load_couponspec(f1_spec_with_fence)
        consumed = get_consumed_paths(spec)

        # Return via paths are optional but should be consumed when present
        return_via_paths = [p for p in consumed if "return_vias" in p]
        assert len(return_via_paths) > 0
