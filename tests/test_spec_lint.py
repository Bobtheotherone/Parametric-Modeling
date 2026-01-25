from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from tools.spec_lint import lint_design_document


def test_spec_lint_detects_missing_sections(tmp_path: Path) -> None:
    doc = tmp_path / "DESIGN_DOCUMENT.md"
    doc.write_text("# Doc\n\n**Milestone:** M1 — X\n\n## Normative Requirements (must)\n\n- [REQ-M1-001] hi\n", encoding="utf-8")

    res = lint_design_document(doc)
    assert not res.ok
    # Should complain about missing DoD and Test Matrix.
    joined = "\n".join(res.issues)
    assert "Definition of Done" in joined
    assert "Test Matrix" in joined


# ============================================================================
# REQ-M1-018: lint-spec-coverage tests for spec consumption tracking
# ============================================================================


def test_spec_consumption_model_computes_unused_provided() -> None:
    """REQ-M1-018: SpecConsumption computes unused_provided_paths correctly."""
    from formula_foundry.resolve.types import SpecConsumption

    consumption = SpecConsumption(
        consumed_paths=frozenset({"a", "b"}),
        expected_paths=frozenset({"a", "b", "c"}),
        provided_paths=frozenset({"a", "b", "d"}),  # 'd' is unused
    )

    assert consumption.unused_provided_paths == frozenset({"d"})
    assert consumption.unconsumed_expected_paths == frozenset({"c"})
    assert not consumption.is_fully_covered


def test_spec_consumption_model_computes_unconsumed_expected() -> None:
    """REQ-M1-018: SpecConsumption computes unconsumed_expected_paths correctly."""
    from formula_foundry.resolve.types import SpecConsumption

    consumption = SpecConsumption(
        consumed_paths=frozenset({"a"}),
        expected_paths=frozenset({"a", "b", "c"}),
        provided_paths=frozenset({"a"}),
    )

    assert consumption.unused_provided_paths == frozenset()
    assert consumption.unconsumed_expected_paths == frozenset({"b", "c"})
    assert not consumption.is_fully_covered


def test_spec_consumption_model_fully_covered() -> None:
    """REQ-M1-018: SpecConsumption reports is_fully_covered when complete."""
    from formula_foundry.resolve.types import SpecConsumption

    consumption = SpecConsumption(
        consumed_paths=frozenset({"a", "b", "c"}),
        expected_paths=frozenset({"a", "b", "c"}),
        provided_paths=frozenset({"a", "b", "c"}),
    )

    assert consumption.unused_provided_paths == frozenset()
    assert consumption.unconsumed_expected_paths == frozenset()
    assert consumption.is_fully_covered
    assert consumption.coverage_ratio == 1.0


def test_spec_consumption_model_coverage_ratio() -> None:
    """REQ-M1-018: SpecConsumption computes coverage_ratio correctly."""
    from formula_foundry.resolve.types import SpecConsumption

    # 2 out of 4 expected paths are consumed
    consumption = SpecConsumption(
        consumed_paths=frozenset({"a", "b"}),
        expected_paths=frozenset({"a", "b", "c", "d"}),
        provided_paths=frozenset({"a", "b"}),
    )

    assert consumption.coverage_ratio == 0.5


def test_spec_consumption_model_empty_expected() -> None:
    """REQ-M1-018: SpecConsumption handles empty expected_paths."""
    from formula_foundry.resolve.types import SpecConsumption

    consumption = SpecConsumption(
        consumed_paths=frozenset({"a"}),
        expected_paths=frozenset(),
        provided_paths=frozenset({"a"}),
    )

    # When no paths are expected, coverage is 100%
    assert consumption.coverage_ratio == 1.0
    # 'a' is consumed, so it's not unused (unused = provided - consumed)
    assert consumption.unused_provided_paths == frozenset()
    assert consumption.unconsumed_expected_paths == frozenset()


def test_build_spec_consumption_from_coupon_spec() -> None:
    """REQ-M1-018: build_spec_consumption creates valid SpecConsumption."""
    from formula_foundry.resolve.consumption import build_spec_consumption
    from formula_foundry.coupongen.spec import CouponSpec

    spec_dict = {
        "schema_version": 1,
        "coupon_family": "F1",
        "units": "nm",
        "toolchain": {
            "kicad": {
                "version": "9.0.7",
                "docker_image": "kicad/kicad:9.0.7@sha256:test",
            }
        },
        "fab_profile": {"id": "generic"},
        "stackup": {
            "copper_layers": 4,
            "thicknesses_nm": {
                "copper_top": 35_000,
                "core": 1_000_000,
                "prepreg": 200_000,
                "copper_inner1": 35_000,
                "copper_inner2": 35_000,
                "copper_bottom": 35_000,
            },
            "materials": {"er": 4.5, "loss_tangent": 0.02},
        },
        "board": {
            "outline": {
                "width_nm": 20_000_000,
                "length_nm": 100_000_000,
                "corner_radius_nm": 1_000_000,
            },
            "origin": {"mode": "center"},
            "text": {"coupon_id": "TEST", "include_manifest_hash": True},
        },
        "connectors": {
            "left": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [5_000_000, 0],
                "rotation_deg": 0,
            },
            "right": {
                "footprint": "Coupongen_Connectors:SMA_EndLaunch_Generic",
                "position_nm": [95_000_000, 0],
                "rotation_deg": 180,
            },
        },
        "transmission_line": {
            "type": "cpwg",
            "layer": "F.Cu",
            "w_nm": 200_000,
            "gap_nm": 150_000,
            "length_left_nm": 20_000_000,
            "length_right_nm": 20_000_000,
            "ground_via_fence": {
                "enabled": True,
                "pitch_nm": 1_500_000,
                "offset_from_gap_nm": 500_000,
                "via": {"drill_nm": 300_000, "diameter_nm": 600_000},
            },
        },
        "discontinuity": {
            "type": "single_via",
            "signal_via": {
                "drill_nm": 300_000,
                "diameter_nm": 600_000,
                "pad_diameter_nm": 900_000,
            },
            "return_vias": {
                "pattern": "ring",
                "count": 8,
                "radius_nm": 2_000_000,
                "via": {"drill_nm": 300_000, "diameter_nm": 600_000},
            },
        },
        "constraints": {
            "mode": "REPAIR",
            "drc": {"must_pass": True, "severity": "error"},
            "symmetry": {"enforce": True},
            "allow_unconnected_copper": False,
        },
        "export": {
            "gerbers": {"enabled": True, "format": "RS274X"},
            "drill": {"enabled": True, "format": "excellon"},
            "outputs_dir": "outputs",
        },
    }

    spec = CouponSpec.model_validate(spec_dict)
    consumption = build_spec_consumption(spec)

    # Should have expected, provided, and consumed paths
    assert len(consumption.expected_paths) > 0
    assert len(consumption.provided_paths) > 0
    assert len(consumption.consumed_paths) > 0

    # Coverage ratio should be between 0 and 1
    assert 0.0 <= consumption.coverage_ratio <= 1.0


def test_enforce_spec_consumption_raises_on_failure() -> None:
    """REQ-M1-018: enforce_spec_consumption raises SpecConsumptionError on failures."""
    from formula_foundry.resolve.consumption import (
        SpecConsumptionError,
        enforce_spec_consumption,
    )
    from formula_foundry.resolve.types import SpecConsumption

    consumption = SpecConsumption(
        consumed_paths=frozenset({"a"}),
        expected_paths=frozenset({"a", "b"}),  # 'b' unconsumed
        provided_paths=frozenset({"a", "c"}),  # 'c' unused
    )

    try:
        enforce_spec_consumption(consumption)
        assert False, "Should have raised SpecConsumptionError"
    except SpecConsumptionError as e:
        assert e.unused_provided == frozenset({"c"})
        assert e.unconsumed_expected == frozenset({"b"})


def test_enforce_spec_consumption_passes_when_valid() -> None:
    """REQ-M1-018: enforce_spec_consumption does not raise when valid."""
    from formula_foundry.resolve.consumption import enforce_spec_consumption
    from formula_foundry.resolve.types import SpecConsumption

    consumption = SpecConsumption(
        consumed_paths=frozenset({"a", "b"}),
        expected_paths=frozenset({"a", "b"}),
        provided_paths=frozenset({"a", "b"}),
    )

    # Should not raise
    enforce_spec_consumption(consumption)
