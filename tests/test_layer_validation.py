"""Tests for layer set validation per family.

Verifies Section 13.5.3 requirements:
    Define and enforce a locked set for fabrication exports:
    - F.Cu, In1.Cu, In2.Cu, B.Cu (for 4-layer boards)
    - F.Mask, B.Mask
    - F.SilkS, B.SilkS (optional)
    - Edge.Cuts

    Enforce in tests that every exported fab directory contains all expected layers.
"""

from __future__ import annotations

import pytest

from formula_foundry.coupongen.layer_validation import (
    FamilyOverride,
    LayerSetConfig,
    LayerSetValidationError,
    LayerValidationResult,
    clear_layer_sets_cache,
    extract_layers_from_exports,
    get_family_override,
    get_gerber_extension_map,
    get_layer_set_for_copper_count,
    layer_validation_payload,
    validate_family_layer_requirements,
    validate_layer_set,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear layer sets cache before each test."""
    clear_layer_sets_cache()
    yield
    clear_layer_sets_cache()


class TestGetLayerSetForCopperCount:
    """Tests for get_layer_set_for_copper_count function."""

    def test_2_layer_returns_correct_layers(self) -> None:
        layer_set = get_layer_set_for_copper_count(2)
        assert isinstance(layer_set, LayerSetConfig)
        assert layer_set.copper == ("F.Cu", "B.Cu")
        assert "F.Cu" in layer_set.required
        assert "B.Cu" in layer_set.required
        assert "Edge.Cuts" in layer_set.required

    def test_4_layer_returns_correct_layers(self) -> None:
        layer_set = get_layer_set_for_copper_count(4)
        assert isinstance(layer_set, LayerSetConfig)
        assert layer_set.copper == ("F.Cu", "In1.Cu", "In2.Cu", "B.Cu")
        assert "In1.Cu" in layer_set.required
        assert "In2.Cu" in layer_set.required
        assert "F.Mask" in layer_set.required
        assert "B.Mask" in layer_set.required

    def test_6_layer_returns_correct_layers(self) -> None:
        layer_set = get_layer_set_for_copper_count(6)
        assert isinstance(layer_set, LayerSetConfig)
        assert len(layer_set.copper) == 6
        assert "In3.Cu" in layer_set.copper
        assert "In4.Cu" in layer_set.copper

    def test_unsupported_layer_count_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported copper layer count"):
            get_layer_set_for_copper_count(8)

    def test_layer_set_has_mask_layers(self) -> None:
        layer_set = get_layer_set_for_copper_count(4)
        assert "F.Mask" in layer_set.mask
        assert "B.Mask" in layer_set.mask

    def test_layer_set_has_edge_cuts(self) -> None:
        layer_set = get_layer_set_for_copper_count(4)
        assert "Edge.Cuts" in layer_set.edge

    def test_layer_set_optional_includes_silkscreen(self) -> None:
        layer_set = get_layer_set_for_copper_count(4)
        assert "F.SilkS" in layer_set.optional
        assert "B.SilkS" in layer_set.optional

    def test_all_layers_property(self) -> None:
        layer_set = get_layer_set_for_copper_count(4)
        all_layers = layer_set.all_layers
        assert all(layer in all_layers for layer in layer_set.required)
        assert all(layer in all_layers for layer in layer_set.optional)


class TestGetFamilyOverride:
    """Tests for get_family_override function."""

    def test_f0_family_override(self) -> None:
        override = get_family_override("F0_CAL_THRU_LINE")
        assert override is not None
        assert isinstance(override, FamilyOverride)
        assert override.signal_layers_min == 1
        assert override.requires_via_layers is False

    def test_f1_family_override(self) -> None:
        override = get_family_override("F1_SINGLE_ENDED_VIA")
        assert override is not None
        assert isinstance(override, FamilyOverride)
        assert override.signal_layers_min == 2
        assert override.requires_via_layers is True

    def test_unknown_family_returns_none(self) -> None:
        override = get_family_override("F99_UNKNOWN_FAMILY")
        assert override is None


class TestGetGerberExtensionMap:
    """Tests for get_gerber_extension_map function."""

    def test_returns_dict(self) -> None:
        ext_map = get_gerber_extension_map()
        assert isinstance(ext_map, dict)

    def test_has_copper_layers(self) -> None:
        ext_map = get_gerber_extension_map()
        assert "F.Cu" in ext_map
        assert "B.Cu" in ext_map

    def test_extension_format(self) -> None:
        ext_map = get_gerber_extension_map()
        assert ext_map["F.Cu"].endswith(".gbr")


class TestExtractLayersFromExports:
    """Tests for extract_layers_from_exports function."""

    def test_extracts_copper_layers(self) -> None:
        export_paths = [
            "gerbers/board-F_Cu.gbr",
            "gerbers/board-B_Cu.gbr",
            "drill/drill.drl",
        ]
        layers = extract_layers_from_exports(export_paths)
        assert "F.Cu" in layers
        assert "B.Cu" in layers

    def test_extracts_mask_layers(self) -> None:
        export_paths = [
            "gerbers/board-F_Mask.gbr",
            "gerbers/board-B_Mask.gbr",
        ]
        layers = extract_layers_from_exports(export_paths)
        assert "F.Mask" in layers
        assert "B.Mask" in layers

    def test_extracts_inner_layers(self) -> None:
        export_paths = [
            "gerbers/board-In1_Cu.gbr",
            "gerbers/board-In2_Cu.gbr",
        ]
        layers = extract_layers_from_exports(export_paths)
        assert "In1.Cu" in layers
        assert "In2.Cu" in layers

    def test_ignores_non_gerber_files(self) -> None:
        export_paths = [
            "drill/drill.drl",
            "board.kicad_pcb",
        ]
        layers = extract_layers_from_exports(export_paths)
        assert len(layers) == 0

    def test_ignores_files_outside_gerber_dir(self) -> None:
        export_paths = [
            "other/board-F_Cu.gbr",
        ]
        layers = extract_layers_from_exports(export_paths)
        assert len(layers) == 0

    def test_custom_gerber_dir(self) -> None:
        export_paths = [
            "fab/board-F_Cu.gbr",
        ]
        layers = extract_layers_from_exports(export_paths, gerber_dir="fab/")
        assert "F.Cu" in layers


class TestValidateFamilyLayerRequirements:
    """Tests for validate_family_layer_requirements function."""

    def test_f0_accepts_single_layer(self) -> None:
        # Should not raise for 2+ layers
        validate_family_layer_requirements(2, "F0_CAL_THRU_LINE")

    def test_f1_requires_at_least_2_layers(self) -> None:
        # F1 requires via transition, needs at least 2 layers
        with pytest.raises(ValueError, match="requires at least 2"):
            validate_family_layer_requirements(1, "F1_SINGLE_ENDED_VIA")

    def test_f1_accepts_4_layers(self) -> None:
        # Should not raise
        validate_family_layer_requirements(4, "F1_SINGLE_ENDED_VIA")

    def test_unknown_family_does_not_raise(self) -> None:
        # Unknown families have no specific requirements
        validate_family_layer_requirements(2, "F99_UNKNOWN")


class TestValidateLayerSet:
    """Tests for validate_layer_set function."""

    def test_valid_4_layer_set_passes(self) -> None:
        export_paths = [
            "gerbers/board-F_Cu.gbr",
            "gerbers/board-In1_Cu.gbr",
            "gerbers/board-In2_Cu.gbr",
            "gerbers/board-B_Cu.gbr",
            "gerbers/board-F_Mask.gbr",
            "gerbers/board-B_Mask.gbr",
            "gerbers/board-Edge_Cuts.gbr",
        ]
        result = validate_layer_set(
            export_paths=export_paths,
            copper_layers=4,
            family="F1_SINGLE_ENDED_VIA",
            strict=False,
        )
        assert result.passed is True
        assert len(result.missing_layers) == 0

    def test_missing_layer_fails(self) -> None:
        export_paths = [
            "gerbers/board-F_Cu.gbr",
            # Missing In1.Cu, In2.Cu, B.Cu
            "gerbers/board-F_Mask.gbr",
            "gerbers/board-B_Mask.gbr",
            "gerbers/board-Edge_Cuts.gbr",
        ]
        result = validate_layer_set(
            export_paths=export_paths,
            copper_layers=4,
            family="F1_SINGLE_ENDED_VIA",
            strict=False,
        )
        assert result.passed is False
        assert "In1.Cu" in result.missing_layers
        assert "In2.Cu" in result.missing_layers
        assert "B.Cu" in result.missing_layers

    def test_strict_mode_raises_on_missing_layers(self) -> None:
        export_paths = [
            "gerbers/board-F_Cu.gbr",
        ]
        with pytest.raises(LayerSetValidationError) as exc_info:
            validate_layer_set(
                export_paths=export_paths,
                copper_layers=4,
                family="F1_SINGLE_ENDED_VIA",
                strict=True,
            )
        assert exc_info.value.result.passed is False
        assert len(exc_info.value.result.missing_layers) > 0

    def test_2_layer_set_validation(self) -> None:
        export_paths = [
            "gerbers/board-F_Cu.gbr",
            "gerbers/board-B_Cu.gbr",
            "gerbers/board-F_Mask.gbr",
            "gerbers/board-B_Mask.gbr",
            "gerbers/board-Edge_Cuts.gbr",
        ]
        result = validate_layer_set(
            export_paths=export_paths,
            copper_layers=2,
            family="F0_CAL_THRU_LINE",
            strict=False,
        )
        assert result.passed is True
        assert result.copper_layer_count == 2

    def test_extra_layers_recorded(self) -> None:
        export_paths = [
            "gerbers/board-F_Cu.gbr",
            "gerbers/board-B_Cu.gbr",
            "gerbers/board-F_Mask.gbr",
            "gerbers/board-B_Mask.gbr",
            "gerbers/board-Edge_Cuts.gbr",
            "gerbers/board-F_SilkS.gbr",  # Optional
        ]
        result = validate_layer_set(
            export_paths=export_paths,
            copper_layers=2,
            family="F0_CAL_THRU_LINE",
            strict=False,
        )
        assert result.passed is True
        # F.SilkS is optional, not extra
        assert "F.SilkS" not in result.extra_layers


class TestLayerValidationResult:
    """Tests for LayerValidationResult dataclass."""

    def test_result_immutable(self) -> None:
        result = LayerValidationResult(
            passed=True,
            missing_layers=(),
            extra_layers=(),
            expected_layers=("F.Cu", "B.Cu"),
            actual_layers=("F.Cu", "B.Cu"),
            copper_layer_count=2,
            family="F0_CAL_THRU_LINE",
        )
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore


class TestLayerValidationPayload:
    """Tests for layer_validation_payload function."""

    def test_payload_contains_required_fields(self) -> None:
        result = LayerValidationResult(
            passed=True,
            missing_layers=(),
            extra_layers=(),
            expected_layers=("F.Cu", "B.Cu"),
            actual_layers=("F.Cu", "B.Cu"),
            copper_layer_count=2,
            family="F0_CAL_THRU_LINE",
        )
        payload = layer_validation_payload(result)
        assert "passed" in payload
        assert "copper_layer_count" in payload
        assert "family" in payload
        assert "expected_layers" in payload
        assert "actual_layers" in payload
        assert "missing_layers" in payload
        assert "extra_layers" in payload

    def test_payload_converts_tuples_to_lists(self) -> None:
        result = LayerValidationResult(
            passed=True,
            missing_layers=("In1.Cu",),
            extra_layers=(),
            expected_layers=("F.Cu", "B.Cu"),
            actual_layers=("F.Cu",),
            copper_layer_count=4,
            family="F1_SINGLE_ENDED_VIA",
        )
        payload = layer_validation_payload(result)
        assert isinstance(payload["expected_layers"], list)
        assert isinstance(payload["actual_layers"], list)
        assert isinstance(payload["missing_layers"], list)


class TestLayerSetValidationError:
    """Tests for LayerSetValidationError exception."""

    def test_error_message_contains_family(self) -> None:
        result = LayerValidationResult(
            passed=False,
            missing_layers=("In1.Cu", "In2.Cu"),
            extra_layers=(),
            expected_layers=("F.Cu", "In1.Cu", "In2.Cu", "B.Cu"),
            actual_layers=("F.Cu", "B.Cu"),
            copper_layer_count=4,
            family="F1_SINGLE_ENDED_VIA",
        )
        error = LayerSetValidationError(result)
        assert "F1_SINGLE_ENDED_VIA" in str(error)
        assert "4-layer" in str(error)
        assert "In1.Cu" in str(error)

    def test_error_has_result_attribute(self) -> None:
        result = LayerValidationResult(
            passed=False,
            missing_layers=("In1.Cu",),
            extra_layers=(),
            expected_layers=("F.Cu", "In1.Cu"),
            actual_layers=("F.Cu",),
            copper_layer_count=4,
            family="F1_SINGLE_ENDED_VIA",
        )
        error = LayerSetValidationError(result)
        assert error.result is result
        assert error.result.passed is False
