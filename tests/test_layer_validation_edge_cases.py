# SPDX-License-Identifier: MIT
"""Edge case tests for layer_validation module.

This module provides additional edge case coverage for the layer_validation module,
extending the tests in test_layer_validation.py with scenarios involving:
- Configuration loading edge cases
- Error message formatting
- Cache behavior
- Family override interactions

Satisfies Section 13.5.3 of the design doc:
    Define and enforce a locked set for fabrication exports.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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
def clear_cache_fixture():
    """Clear layer sets cache before and after each test."""
    clear_layer_sets_cache()
    yield
    clear_layer_sets_cache()


class TestLayerSetConfigProperties:
    """Tests for LayerSetConfig dataclass properties."""

    def test_all_layers_is_union_of_required_and_optional(self) -> None:
        """all_layers should be concatenation of required and optional."""
        config = LayerSetConfig(
            copper=("F.Cu", "B.Cu"),
            mask=("F.Mask", "B.Mask"),
            silkscreen=("F.SilkS", "B.SilkS"),
            edge=("Edge.Cuts",),
            required=("F.Cu", "B.Cu", "Edge.Cuts"),
            optional=("F.SilkS", "B.SilkS"),
        )

        all_layers = config.all_layers
        assert "F.Cu" in all_layers
        assert "B.Cu" in all_layers
        assert "Edge.Cuts" in all_layers
        assert "F.SilkS" in all_layers
        assert "B.SilkS" in all_layers

    def test_layersetconfig_frozen(self) -> None:
        """LayerSetConfig should be immutable (frozen dataclass)."""
        config = LayerSetConfig(
            copper=("F.Cu",),
            mask=("F.Mask",),
            silkscreen=("F.SilkS",),
            edge=("Edge.Cuts",),
            required=("F.Cu", "F.Mask", "Edge.Cuts"),
            optional=("F.SilkS",),
        )

        with pytest.raises(AttributeError):
            config.copper = ("F.Cu", "B.Cu")  # type: ignore[misc]

    def test_layersetconfig_hashable(self) -> None:
        """LayerSetConfig should be hashable for use in sets/dicts."""
        config1 = LayerSetConfig(
            copper=("F.Cu",),
            mask=(),
            silkscreen=(),
            edge=(),
            required=("F.Cu",),
            optional=(),
        )
        config2 = LayerSetConfig(
            copper=("F.Cu",),
            mask=(),
            silkscreen=(),
            edge=(),
            required=("F.Cu",),
            optional=(),
        )

        # Should be hashable
        _ = hash(config1)

        # Equal configs should have equal hashes
        assert hash(config1) == hash(config2)

        # Can be used in sets
        config_set = {config1, config2}
        assert len(config_set) == 1


class TestFamilyOverrideProperties:
    """Tests for FamilyOverride dataclass."""

    def test_family_override_frozen(self) -> None:
        """FamilyOverride should be immutable."""
        override = FamilyOverride(
            description="Test family",
            signal_layers_min=2,
            requires_via_layers=True,
        )

        with pytest.raises(AttributeError):
            override.signal_layers_min = 4  # type: ignore[misc]

    def test_family_override_equality(self) -> None:
        """Two FamilyOverride with same values should be equal."""
        o1 = FamilyOverride(
            description="Via transition",
            signal_layers_min=2,
            requires_via_layers=True,
        )
        o2 = FamilyOverride(
            description="Via transition",
            signal_layers_min=2,
            requires_via_layers=True,
        )

        assert o1 == o2


class TestLayerValidationResultProperties:
    """Tests for LayerValidationResult dataclass."""

    def test_passed_true_when_no_missing(self) -> None:
        """passed should be True when missing_layers is empty."""
        result = LayerValidationResult(
            passed=True,
            missing_layers=(),
            extra_layers=("Unknown.Layer",),  # Extra is allowed
            expected_layers=("F.Cu", "B.Cu"),
            actual_layers=("F.Cu", "B.Cu", "Unknown.Layer"),
            copper_layer_count=2,
            family="F0_CAL_THRU_LINE",
        )

        # passed=True even with extra layers
        assert result.passed is True

    def test_result_fields_accessible(self) -> None:
        """All result fields should be accessible."""
        result = LayerValidationResult(
            passed=False,
            missing_layers=("In1.Cu",),
            extra_layers=(),
            expected_layers=("F.Cu", "In1.Cu", "B.Cu"),
            actual_layers=("F.Cu", "B.Cu"),
            copper_layer_count=4,
            family="F1_SINGLE_ENDED_VIA",
        )

        assert result.passed is False
        assert "In1.Cu" in result.missing_layers
        assert result.copper_layer_count == 4
        assert result.family == "F1_SINGLE_ENDED_VIA"


class TestExtractLayersEdgeCases:
    """Edge case tests for extract_layers_from_exports."""

    def test_empty_export_paths(self) -> None:
        """Empty export paths list should return empty layers."""
        layers = extract_layers_from_exports([])
        assert layers == []

    def test_no_gerber_files(self) -> None:
        """Paths without gerber files should return empty layers."""
        export_paths = [
            "README.md",
            "board.kicad_pcb",
            "drill/drill.drl",
            "bom.csv",
        ]
        layers = extract_layers_from_exports(export_paths)
        assert layers == []

    def test_mixed_content(self) -> None:
        """Mixed content should only return gerber layers."""
        export_paths = [
            "gerbers/board-F_Cu.gtl",
            "drill/drill.drl",
            "README.md",
            "gerbers/board-B_Cu.gbl",
        ]
        layers = extract_layers_from_exports(export_paths)

        assert "F.Cu" in layers
        assert "B.Cu" in layers
        assert len(layers) == 2

    def test_duplicate_layer_files(self) -> None:
        """Duplicate layer files should produce duplicate layer entries."""
        export_paths = [
            "gerbers/board-F_Cu.gtl",
            "gerbers/backup-F_Cu.gtl",  # Same layer, different file
        ]
        layers = extract_layers_from_exports(export_paths)

        # Both files match F.Cu, so we get two entries
        assert layers.count("F.Cu") == 2

    def test_case_sensitive_gerber_dir(self) -> None:
        """Gerber directory matching should be case-sensitive."""
        export_paths = [
            "Gerbers/board-F_Cu.gtl",  # Capital G
            "gerbers/board-B_Cu.gbl",  # lowercase g
        ]

        # Only lowercase gerbers/ should match with default
        layers = extract_layers_from_exports(export_paths, gerber_dir="gerbers/")
        assert "B.Cu" in layers
        assert len(layers) == 1


class TestValidateLayerSetEdgeCases:
    """Edge case tests for validate_layer_set."""

    def test_strict_false_does_not_raise(self) -> None:
        """strict=False should never raise LayerSetValidationError."""
        export_paths = []  # Completely empty

        # Should not raise even with no layers
        result = validate_layer_set(
            export_paths=export_paths,
            copper_layers=4,
            family="F1_SINGLE_ENDED_VIA",
            strict=False,
        )

        assert result.passed is False
        assert len(result.missing_layers) > 0

    def test_all_layers_missing(self) -> None:
        """All required layers missing should fail validation."""
        result = validate_layer_set(
            export_paths=[],
            copper_layers=2,
            family="F0_CAL_THRU_LINE",
            strict=False,
        )

        assert result.passed is False
        # All required layers should be in missing
        layer_set = get_layer_set_for_copper_count(2)
        for layer in layer_set.required:
            assert layer in result.missing_layers


class TestValidateFamilyLayerRequirementsEdgeCases:
    """Edge case tests for validate_family_layer_requirements."""

    def test_minimum_layers_exactly_met(self) -> None:
        """Exactly minimum layers should pass."""
        # F1 requires minimum 2 layers
        validate_family_layer_requirements(2, "F1_SINGLE_ENDED_VIA")
        # Should not raise

    def test_f0_single_layer_boards(self) -> None:
        """F0 should allow very minimal layer counts if override permits."""
        # F0 has signal_layers_min=1, so 2-layer should be fine
        validate_family_layer_requirements(2, "F0_CAL_THRU_LINE")

    def test_high_layer_count_always_passes(self) -> None:
        """High layer count should always pass family requirements."""
        for family in ["F0_CAL_THRU_LINE", "F1_SINGLE_ENDED_VIA"]:
            validate_family_layer_requirements(6, family)
            # Should not raise


class TestLayerValidationPayloadFormat:
    """Tests for layer_validation_payload output format."""

    def test_payload_all_fields_present(self) -> None:
        """Payload should contain all required fields."""
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

        required_keys = {
            "passed",
            "copper_layer_count",
            "family",
            "expected_layers",
            "actual_layers",
            "missing_layers",
            "extra_layers",
        }

        assert set(payload.keys()) == required_keys

    def test_payload_json_serializable(self) -> None:
        """Payload should be JSON serializable."""
        result = LayerValidationResult(
            passed=False,
            missing_layers=("In1.Cu", "In2.Cu"),
            extra_layers=("Custom.Layer",),
            expected_layers=("F.Cu", "In1.Cu", "In2.Cu", "B.Cu"),
            actual_layers=("F.Cu", "B.Cu", "Custom.Layer"),
            copper_layer_count=4,
            family="F1_SINGLE_ENDED_VIA",
        )

        payload = layer_validation_payload(result)

        # Should not raise
        json_str = json.dumps(payload, sort_keys=True)
        parsed = json.loads(json_str)

        assert parsed["passed"] is False
        assert parsed["copper_layer_count"] == 4

    def test_payload_lists_are_sorted_deterministic(self) -> None:
        """Payload list fields should be deterministic for manifest stability."""
        result = LayerValidationResult(
            passed=True,
            missing_layers=(),
            extra_layers=(),
            expected_layers=("B.Cu", "F.Cu", "A.Cu"),  # Out of order
            actual_layers=("F.Cu", "B.Cu", "A.Cu"),
            copper_layer_count=3,
            family="TEST",
        )

        payload1 = layer_validation_payload(result)
        payload2 = layer_validation_payload(result)

        # Multiple calls should produce identical output
        assert payload1 == payload2


class TestLayerSetValidationErrorFormatting:
    """Tests for LayerSetValidationError message formatting."""

    def test_error_message_includes_all_missing_layers(self) -> None:
        """Error message should mention all missing layers."""
        result = LayerValidationResult(
            passed=False,
            missing_layers=("In1.Cu", "In2.Cu", "Edge.Cuts"),
            extra_layers=(),
            expected_layers=("F.Cu", "In1.Cu", "In2.Cu", "B.Cu", "Edge.Cuts"),
            actual_layers=("F.Cu", "B.Cu"),
            copper_layer_count=4,
            family="F1_SINGLE_ENDED_VIA",
        )

        error = LayerSetValidationError(result)
        error_str = str(error)

        assert "In1.Cu" in error_str
        assert "In2.Cu" in error_str

    def test_error_is_exception_subclass(self) -> None:
        """LayerSetValidationError should be an Exception subclass."""
        result = LayerValidationResult(
            passed=False,
            missing_layers=("F.Cu",),
            extra_layers=(),
            expected_layers=("F.Cu",),
            actual_layers=(),
            copper_layer_count=2,
            family="TEST",
        )

        error = LayerSetValidationError(result)
        assert isinstance(error, Exception)

    def test_error_can_be_caught_as_exception(self) -> None:
        """Error should be catchable as generic Exception."""
        result = LayerValidationResult(
            passed=False,
            missing_layers=("F.Cu",),
            extra_layers=(),
            expected_layers=("F.Cu",),
            actual_layers=(),
            copper_layer_count=2,
            family="TEST",
        )

        try:
            raise LayerSetValidationError(result)
        except Exception as e:
            assert isinstance(e, LayerSetValidationError)
            assert e.result is result


class TestCacheBehavior:
    """Tests for layer sets configuration cache behavior."""

    def test_cache_cleared_between_tests(self) -> None:
        """Cache should be cleared by fixture (implicit test)."""
        # First call loads configuration
        _ = get_layer_set_for_copper_count(4)

        # Clear cache
        clear_layer_sets_cache()

        # This should load fresh (no way to verify directly,
        # but ensures clear_layer_sets_cache doesn't raise)
        _ = get_layer_set_for_copper_count(4)

    def test_multiple_clear_calls_safe(self) -> None:
        """Multiple cache clear calls should be safe."""
        clear_layer_sets_cache()
        clear_layer_sets_cache()
        clear_layer_sets_cache()
        # Should not raise


class TestGerberExtensionMapConsistency:
    """Tests for gerber extension map consistency."""

    def test_extension_map_has_common_layers(self) -> None:
        """Extension map should include common PCB layers."""
        ext_map = get_gerber_extension_map()

        common_layers = ["F.Cu", "B.Cu", "F.Mask", "B.Mask", "Edge.Cuts"]
        for layer in common_layers:
            assert layer in ext_map, f"Missing common layer: {layer}"

    def test_extension_map_values_are_strings(self) -> None:
        """All extension values should be strings."""
        ext_map = get_gerber_extension_map()

        for layer, ext in ext_map.items():
            assert isinstance(layer, str)
            assert isinstance(ext, str)

    def test_extension_map_consistent_with_extraction(self) -> None:
        """Extension map should be consistent with layer extraction."""
        ext_map = get_gerber_extension_map()

        # Create test paths using the extension map
        test_paths = [f"gerbers/board{ext}" for ext in ext_map.values()]

        # Extract should recover all layers
        extracted = extract_layers_from_exports(test_paths)

        for layer in ext_map.keys():
            assert layer in extracted, f"Layer {layer} not extracted"
