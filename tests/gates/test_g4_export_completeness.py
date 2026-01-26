# SPDX-License-Identifier: MIT
"""Gate G4 tests: Export completeness and layer set validation.

This module tests:
- Export layer set completeness per copper count
- Required Gerber layers are present (F.Cu, B.Cu, masks, edge cuts)
- Inner layer files present for 4+ layer boards
- Drill files (PTH, NPTH) present
- Manifest references all exported artifacts

Per ECO-M1-ALIGN-0001:
- REQ-M1-017: Gerber and drill file export via KiCad CLI
- REQ-M1-025: CI must prove export completeness for all golden specs

Section 13.5.3 specifies locked layer sets:
- F.Cu, In1.Cu, In2.Cu, B.Cu (for 4-layer boards)
- F.Mask, B.Mask
- F.SilkS, B.SilkS (optional)
- Edge.Cuts

Pytest marker: gate_g4

Note: Real KiCad export integration tests are in tests/integration/test_export_determinism_integration.py
and require Docker. These tests verify the export logic without Docker.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.coupongen import load_spec
from formula_foundry.coupongen.layer_validation import (
    FamilyOverride,
    LayerSetConfig,
    LayerSetValidationError,
    LayerValidationResult,
    extract_layers_from_exports,
    get_family_override,
    get_gerber_extension_map,
    get_layer_set_for_copper_count,
    validate_family_layer_requirements,
    validate_layer_set,
)

# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parents[1]
GOLDEN_SPECS_DIR = TESTS_DIR / "golden_specs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_golden_specs() -> list[Path]:
    """Collect all golden spec files (YAML only to avoid duplicates)."""
    specs: list[Path] = []
    specs.extend(sorted(GOLDEN_SPECS_DIR.glob("f0_*.yaml")))
    specs.extend(sorted(GOLDEN_SPECS_DIR.glob("f1_*.yaml")))
    return specs


def _collect_f0_specs() -> list[Path]:
    """Collect F0 golden specs."""
    return sorted(GOLDEN_SPECS_DIR.glob("f0_*.yaml"))


def _collect_f1_specs() -> list[Path]:
    """Collect F1 golden specs."""
    return sorted(GOLDEN_SPECS_DIR.glob("f1_*.yaml"))


class _FakeExportRunner:
    """Fake KiCad CLI runner for unit testing export pipeline.

    Generates deterministic fake export files without requiring KiCad.
    """

    def __init__(self, *, seed: str = "default", copper_layers: int = 4) -> None:
        self.seed = seed
        self.copper_layers = copper_layers
        self.gerber_calls: list[tuple[Path, Path]] = []
        self.drill_calls: list[tuple[Path, Path]] = []

    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
        """Simulate DRC execution."""
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"violations": []}), encoding="utf-8")
        return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        """Simulate Gerber export with all required layers."""
        self.gerber_calls.append((board_path, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        # Get board name prefix from board_path (e.g., "coupon" from "coupon.kicad_pcb")
        board_name = board_path.stem

        # Generate copper layers based on copper_layers setting
        copper_layer_names = ["F.Cu", "B.Cu"]
        if self.copper_layers >= 4:
            copper_layer_names = ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
        if self.copper_layers >= 6:
            copper_layer_names = ["F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "B.Cu"]

        # KiCad Gerber extension mapping (industry-standard extensions)
        layer_extension_map = {
            "F.Cu": ".gtl",
            "B.Cu": ".gbl",
            "In1.Cu": ".g1",
            "In2.Cu": ".g2",
            "In3.Cu": ".g3",
            "In4.Cu": ".g4",
            "F.Mask": ".gts",
            "B.Mask": ".gbs",
            "F.SilkS": ".gto",
            "B.SilkS": ".gbo",
            "Edge.Cuts": ".gm1",
        }

        layers = [
            *[(layer, f"G04 {layer}*") for layer in copper_layer_names],
            ("F.SilkS", "G04 Top Silkscreen*"),
            ("B.SilkS", "G04 Bottom Silkscreen*"),
            ("F.Mask", "G04 Top Soldermask*"),
            ("B.Mask", "G04 Bottom Soldermask*"),
            ("Edge.Cuts", "G04 Board Outline*"),
        ]

        for layer_name, content_start in layers:
            content_hash = hashlib.sha256(f"{self.seed}:{layer_name}".encode()).hexdigest()[:8]
            content = f"{content_start}\nG04 Hash={content_hash}*\nX0Y0D02*\nM02*\n"
            # Convert layer name to KiCad filename format with proper extension
            kicad_layer = layer_name.replace(".", "_")
            extension = layer_extension_map.get(layer_name, ".gbr")
            kicad_filename = f"{board_name}-{kicad_layer}{extension}"
            (out_dir / kicad_filename).write_text(content, encoding="utf-8")

        return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        """Simulate drill file export."""
        self.drill_calls.append((board_path, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        drill_files = [
            ("drill.drl", "M48\n; PTH drill file\nT1C0.3\n%\nT1\nX10Y10\nM30\n"),
            ("drill-NPTH.drl", "M48\n; NPTH drill file\n%\nM30\n"),
        ]

        for filename, content in drill_files:
            (out_dir / filename).write_text(content, encoding="utf-8")

        return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden_specs() -> list[Path]:
    """Fixture providing list of golden spec paths."""
    return _collect_golden_specs()


# ---------------------------------------------------------------------------
# G4 Gate Tests: Golden Spec Coverage
# ---------------------------------------------------------------------------


@pytest.mark.gate_g4
class TestG4GoldenSpecCoverage:
    """Gate G4 tests verifying sufficient golden specs for export testing.

    Per ECO-M1-ALIGN-0001: "For ≥10 golden specs per family, CI proves
    export completeness + stable hashes (G4)"
    """

    def test_minimum_f0_golden_specs_for_export(self) -> None:
        """Verify at least 10 F0 golden specs exist for export testing."""
        specs = _collect_f0_specs()
        assert len(specs) >= 10, f"Expected ≥10 F0 specs for export gate, found {len(specs)}"

    def test_minimum_f1_golden_specs_for_export(self) -> None:
        """Verify at least 10 F1 golden specs exist for export testing."""
        specs = _collect_f1_specs()
        assert len(specs) >= 10, f"Expected ≥10 F1 specs for export gate, found {len(specs)}"

    def test_total_golden_specs_for_export(self, golden_specs: list[Path]) -> None:
        """Verify at least 20 total golden specs exist for export testing."""
        assert len(golden_specs) >= 20, f"Expected ≥20 total golden specs for export gate, found {len(golden_specs)}"


# ---------------------------------------------------------------------------
# G4 Gate Tests: Layer Set Configuration
# ---------------------------------------------------------------------------


@pytest.mark.gate_g4
class TestG4LayerSetConfiguration:
    """Gate G4 tests for layer set configuration per copper count.

    Per Section 13.5.3: Define and enforce a locked layer set for fab exports.
    """

    def test_2_layer_copper_set(self) -> None:
        """2-layer boards should have F.Cu and B.Cu."""
        layer_set = get_layer_set_for_copper_count(2)
        assert layer_set.copper == ("F.Cu", "B.Cu")

    def test_4_layer_copper_set(self) -> None:
        """4-layer boards should have F.Cu, In1.Cu, In2.Cu, B.Cu."""
        layer_set = get_layer_set_for_copper_count(4)
        assert layer_set.copper == ("F.Cu", "In1.Cu", "In2.Cu", "B.Cu")

    def test_6_layer_copper_set(self) -> None:
        """6-layer boards should have correct copper layers."""
        layer_set = get_layer_set_for_copper_count(6)
        assert len(layer_set.copper) == 6
        assert "In3.Cu" in layer_set.copper
        assert "In4.Cu" in layer_set.copper

    def test_layer_set_includes_mask_layers(self) -> None:
        """All layer sets should include F.Mask and B.Mask."""
        for copper_count in (2, 4, 6):
            layer_set = get_layer_set_for_copper_count(copper_count)
            assert "F.Mask" in layer_set.mask
            assert "B.Mask" in layer_set.mask

    def test_layer_set_includes_edge_cuts(self) -> None:
        """All layer sets should include Edge.Cuts."""
        for copper_count in (2, 4, 6):
            layer_set = get_layer_set_for_copper_count(copper_count)
            assert "Edge.Cuts" in layer_set.edge

    def test_layer_set_optional_silkscreen(self) -> None:
        """Silkscreen layers should be optional."""
        layer_set = get_layer_set_for_copper_count(4)
        assert "F.SilkS" in layer_set.optional
        assert "B.SilkS" in layer_set.optional

    def test_layer_set_required_layers(self) -> None:
        """Required layers should include copper, mask, and edge cuts."""
        layer_set = get_layer_set_for_copper_count(4)
        required = layer_set.required
        # All copper layers required
        for copper in layer_set.copper:
            assert copper in required, f"{copper} should be required"
        # Mask layers required
        assert "F.Mask" in required
        assert "B.Mask" in required
        # Edge.Cuts required
        assert "Edge.Cuts" in required

    def test_unsupported_layer_count_raises(self) -> None:
        """Unsupported copper layer counts should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported copper layer count"):
            get_layer_set_for_copper_count(8)


# ---------------------------------------------------------------------------
# G4 Gate Tests: Family-Specific Layer Requirements
# ---------------------------------------------------------------------------


@pytest.mark.gate_g4
class TestG4FamilyLayerRequirements:
    """Gate G4 tests for family-specific layer requirements."""

    def test_f0_family_override_exists(self) -> None:
        """F0 family should have override configuration."""
        override = get_family_override("F0_CAL_THRU_LINE")
        assert override is not None
        assert isinstance(override, FamilyOverride)

    def test_f1_family_override_exists(self) -> None:
        """F1 family should have override configuration."""
        override = get_family_override("F1_SINGLE_ENDED_VIA")
        assert override is not None
        assert isinstance(override, FamilyOverride)

    def test_f0_single_layer_sufficient(self) -> None:
        """F0 calibration coupons only need 1 signal layer."""
        override = get_family_override("F0_CAL_THRU_LINE")
        assert override is not None
        assert override.signal_layers_min == 1
        assert override.requires_via_layers is False

    def test_f1_requires_multiple_layers(self) -> None:
        """F1 via transition coupons require at least 2 layers."""
        override = get_family_override("F1_SINGLE_ENDED_VIA")
        assert override is not None
        assert override.signal_layers_min == 2
        assert override.requires_via_layers is True

    def test_f1_accepts_4_layers(self) -> None:
        """F1 family should accept 4-layer boards."""
        # Should not raise
        validate_family_layer_requirements(4, "F1_SINGLE_ENDED_VIA")

    def test_f1_rejects_1_layer(self) -> None:
        """F1 family should reject 1-layer boards."""
        with pytest.raises(ValueError, match="requires at least 2"):
            validate_family_layer_requirements(1, "F1_SINGLE_ENDED_VIA")


# ---------------------------------------------------------------------------
# G4 Gate Tests: Layer Extraction from Exports
# ---------------------------------------------------------------------------


@pytest.mark.gate_g4
class TestG4LayerExtraction:
    """Gate G4 tests for layer extraction from export paths."""

    def test_extracts_copper_layers(self) -> None:
        """Should extract copper layer names from Gerber paths.

        Uses KiCad's standard Gerber extensions (.gtl, .gbl) per layer_sets.json.
        """
        export_paths = [
            "gerbers/board-F_Cu.gtl",
            "gerbers/board-B_Cu.gbl",
        ]
        layers = extract_layers_from_exports(export_paths)
        assert "F.Cu" in layers
        assert "B.Cu" in layers

    def test_extracts_inner_copper_layers(self) -> None:
        """Should extract inner copper layer names.

        Uses KiCad's standard inner layer extensions (.g1, .g2).
        """
        export_paths = [
            "gerbers/board-In1_Cu.g1",
            "gerbers/board-In2_Cu.g2",
        ]
        layers = extract_layers_from_exports(export_paths)
        assert "In1.Cu" in layers
        assert "In2.Cu" in layers

    def test_extracts_mask_layers(self) -> None:
        """Should extract mask layer names.

        Uses KiCad's standard soldermask extensions (.gts, .gbs).
        """
        export_paths = [
            "gerbers/board-F_Mask.gts",
            "gerbers/board-B_Mask.gbs",
        ]
        layers = extract_layers_from_exports(export_paths)
        assert "F.Mask" in layers
        assert "B.Mask" in layers

    def test_extracts_edge_cuts(self) -> None:
        """Should extract Edge.Cuts layer.

        Uses KiCad's standard mechanical layer extension (.gm1).
        """
        export_paths = ["gerbers/board-Edge_Cuts.gm1"]
        layers = extract_layers_from_exports(export_paths)
        assert "Edge.Cuts" in layers

    def test_ignores_non_gerber_files(self) -> None:
        """Should ignore non-Gerber files."""
        export_paths = [
            "drill/drill.drl",
            "board.kicad_pcb",
        ]
        layers = extract_layers_from_exports(export_paths)
        assert len(layers) == 0

    def test_respects_gerber_dir_filter(self) -> None:
        """Should only extract from specified gerber directory.

        Uses KiCad's standard extensions (.gtl, .gbl).
        """
        export_paths = [
            "fab/board-F_Cu.gtl",
            "other/board-B_Cu.gbl",
        ]
        layers = extract_layers_from_exports(export_paths, gerber_dir="fab/")
        assert "F.Cu" in layers
        assert "B.Cu" not in layers


# ---------------------------------------------------------------------------
# G4 Gate Tests: Layer Set Validation
# ---------------------------------------------------------------------------


@pytest.mark.gate_g4
class TestG4LayerSetValidation:
    """Gate G4 tests for layer set validation logic."""

    def test_valid_4_layer_set_passes_strict(self) -> None:
        """Complete 4-layer export set should pass validation with strict=True.

        Uses KiCad's standard Gerber extensions per layer_sets.json.
        This is the oracle pass-case: a complete export must not trigger
        LayerSetValidationError when strict=True.
        """
        export_paths = [
            "gerbers/board-F_Cu.gtl",
            "gerbers/board-In1_Cu.g1",
            "gerbers/board-In2_Cu.g2",
            "gerbers/board-B_Cu.gbl",
            "gerbers/board-F_Mask.gts",
            "gerbers/board-B_Mask.gbs",
            "gerbers/board-Edge_Cuts.gm1",
        ]
        result = validate_layer_set(
            export_paths=export_paths,
            copper_layers=4,
            family="F1_SINGLE_ENDED_VIA",
            strict=True,
        )
        assert result.passed is True
        assert len(result.missing_layers) == 0

    def test_missing_inner_layers_fails(self) -> None:
        """Missing inner copper layers should fail validation."""
        export_paths = [
            "gerbers/board-F_Cu.gtl",
            # Missing In1.Cu, In2.Cu
            "gerbers/board-B_Cu.gbl",
            "gerbers/board-F_Mask.gts",
            "gerbers/board-B_Mask.gbs",
            "gerbers/board-Edge_Cuts.gm1",
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

    def test_missing_mask_layers_fails(self) -> None:
        """Missing mask layers should fail validation.

        Uses KiCad's standard Gerber extensions per layer_sets.json.
        """
        export_paths = [
            "gerbers/board-F_Cu.gtl",
            "gerbers/board-B_Cu.gbl",
            # Missing F.Mask, B.Mask
            "gerbers/board-Edge_Cuts.gm1",
        ]
        result = validate_layer_set(
            export_paths=export_paths,
            copper_layers=2,
            family="F0_CAL_THRU_LINE",
            strict=False,
        )
        assert result.passed is False
        assert "F.Mask" in result.missing_layers
        assert "B.Mask" in result.missing_layers

    def test_missing_edge_cuts_fails(self) -> None:
        """Missing Edge.Cuts should fail validation.

        Uses KiCad's standard Gerber extensions per layer_sets.json.
        """
        export_paths = [
            "gerbers/board-F_Cu.gtl",
            "gerbers/board-B_Cu.gbl",
            "gerbers/board-F_Mask.gts",
            "gerbers/board-B_Mask.gbs",
            # Missing Edge.Cuts
        ]
        result = validate_layer_set(
            export_paths=export_paths,
            copper_layers=2,
            family="F0_CAL_THRU_LINE",
            strict=False,
        )
        assert result.passed is False
        assert "Edge.Cuts" in result.missing_layers

    def test_strict_mode_raises_exception(self) -> None:
        """Strict mode should raise LayerSetValidationError on failure."""
        export_paths = ["gerbers/board-F_Cu.gtl"]
        with pytest.raises(LayerSetValidationError) as exc_info:
            validate_layer_set(
                export_paths=export_paths,
                copper_layers=4,
                family="F1_SINGLE_ENDED_VIA",
                strict=True,
            )
        assert exc_info.value.result.passed is False

    def test_validation_result_has_all_fields_strict(self) -> None:
        """Validation result should have all required fields with strict=True.

        Uses KiCad's standard Gerber extensions per layer_sets.json.
        This is an oracle pass-case: complete exports must pass strict validation.
        """
        export_paths = [
            "gerbers/board-F_Cu.gtl",
            "gerbers/board-B_Cu.gbl",
            "gerbers/board-F_Mask.gts",
            "gerbers/board-B_Mask.gbs",
            "gerbers/board-Edge_Cuts.gm1",
        ]
        result = validate_layer_set(
            export_paths=export_paths,
            copper_layers=2,
            family="F0_CAL_THRU_LINE",
            strict=True,
        )
        assert result.passed is True
        assert hasattr(result, "passed")
        assert hasattr(result, "missing_layers")
        assert hasattr(result, "expected_layers")
        assert hasattr(result, "actual_layers")
        assert hasattr(result, "copper_layer_count")
        assert hasattr(result, "family")


# ---------------------------------------------------------------------------
# G4 Gate Tests: Golden Specs Export Configuration
# ---------------------------------------------------------------------------


@pytest.mark.gate_g4
class TestG4GoldenSpecsExportConfig:
    """Gate G4 tests verifying golden specs have valid export configuration."""

    def test_golden_specs_have_export_enabled(self, golden_specs: list[Path]) -> None:
        """All golden specs should have export.gerbers.enabled=True."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            assert spec.export.gerbers.enabled is True, f"Golden spec {spec_path.name} must have Gerber export enabled"

    def test_golden_specs_have_drill_enabled(self, golden_specs: list[Path]) -> None:
        """All golden specs should have export.drill.enabled=True."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            assert spec.export.drill.enabled is True, f"Golden spec {spec_path.name} must have drill export enabled"

    def test_golden_specs_have_valid_copper_count(self, golden_specs: list[Path]) -> None:
        """All golden specs should have valid copper layer counts (2, 4, or 6)."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            copper_layers = spec.stackup.copper_layers
            assert copper_layers in (2, 4, 6), f"Golden spec {spec_path.name} has unsupported copper count: {copper_layers}"

    def test_f0_golden_specs_have_consistent_stackup(self) -> None:
        """F0 golden specs should have consistent stackup configuration."""
        specs = _collect_f0_specs()
        copper_counts = set()
        for spec_path in specs:
            spec = load_spec(spec_path)
            copper_counts.add(spec.stackup.copper_layers)
        # F0 specs may vary, but should be valid
        for count in copper_counts:
            assert count in (2, 4, 6)

    def test_f1_golden_specs_have_via_support(self) -> None:
        """F1 golden specs should have sufficient layers for vias."""
        specs = _collect_f1_specs()
        for spec_path in specs:
            spec = load_spec(spec_path)
            copper_layers = spec.stackup.copper_layers
            assert copper_layers >= 2, f"F1 spec {spec_path.name} needs ≥2 layers for via transition"


# ---------------------------------------------------------------------------
# G4 Gate Tests: Drill File Requirements
# ---------------------------------------------------------------------------


@pytest.mark.gate_g4
class TestG4DrillFileRequirements:
    """Gate G4 tests for drill file requirements."""

    def test_export_creates_drill_files(self, tmp_path: Path) -> None:
        """Export should create drill files."""
        runner = _FakeExportRunner(seed="test")
        board_path = tmp_path / "test.kicad_pcb"
        out_dir = tmp_path / "drill"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        runner.export_drill(board_path, out_dir)

        drill_files = list(out_dir.glob("*.drl"))
        assert len(drill_files) >= 1, "Should create at least 1 drill file"

    def test_pth_drill_file_exists(self, tmp_path: Path) -> None:
        """PTH (Plated Through Hole) drill file should exist."""
        runner = _FakeExportRunner(seed="test")
        board_path = tmp_path / "test.kicad_pcb"
        out_dir = tmp_path / "drill"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        runner.export_drill(board_path, out_dir)

        # Main drill file is PTH
        pth_files = list(out_dir.glob("drill.drl"))
        assert len(pth_files) >= 1, "PTH drill file should exist"

    def test_drill_files_are_non_empty(self, tmp_path: Path) -> None:
        """Drill files should have content."""
        runner = _FakeExportRunner(seed="test")
        board_path = tmp_path / "test.kicad_pcb"
        out_dir = tmp_path / "drill"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        runner.export_drill(board_path, out_dir)

        for drill_file in out_dir.glob("*.drl"):
            assert drill_file.stat().st_size > 0, f"Drill file {drill_file.name} should be non-empty"


# ---------------------------------------------------------------------------
# G4 Gate Tests: Golden Specs Export Completeness
# ---------------------------------------------------------------------------


@pytest.mark.gate_g4
class TestG4GoldenSpecsExportCompleteness:
    """Gate G4 tests verifying golden specs produce complete exports.

    Uses fake runners to test export pipeline logic without Docker.
    """

    @pytest.mark.parametrize(
        "spec_path",
        _collect_golden_specs(),
        ids=lambda p: p.name,
    )
    def test_golden_spec_exports_complete(self, spec_path: Path, tmp_path: Path) -> None:
        """Each golden spec should produce complete exports."""
        from formula_foundry.coupongen import build_coupon

        spec = load_spec(spec_path)
        copper_layers = spec.stackup.copper_layers
        runner = _FakeExportRunner(seed=spec_path.name, copper_layers=copper_layers)

        result = build_coupon(
            spec,
            out_root=tmp_path,
            mode="docker",
            runner=runner,
            kicad_cli_version="9.0.7",
        )

        # Verify manifest contains exports
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert "exports" in manifest, "Manifest must have exports"

        exports = manifest["exports"]
        assert isinstance(exports, list)
        export_paths = [e["path"] for e in exports]

        # Verify Gerber exports
        gerber_exports = [p for p in export_paths if "gerbers/" in p]
        assert len(gerber_exports) >= 4, f"Spec {spec_path.name}: Expected ≥4 Gerber exports, got {len(gerber_exports)}"

        # Verify drill exports
        drill_exports = [p for p in export_paths if "drill/" in p]
        assert len(drill_exports) >= 1, f"Spec {spec_path.name}: Expected ≥1 drill export, got {len(drill_exports)}"

    @pytest.mark.parametrize(
        "spec_path",
        _collect_golden_specs(),
        ids=lambda p: p.name,
    )
    def test_golden_spec_layer_set_valid_strict(self, spec_path: Path, tmp_path: Path) -> None:
        """Each golden spec export should have valid layer sets with strict=True.

        This is the oracle pass-case for layer validation: ALL golden specs
        must produce exports that pass strict layer set validation.
        Per M1 oracle requirements, we validate EVERY golden spec (not just specs[0]).
        """
        from formula_foundry.coupongen import build_coupon

        spec = load_spec(spec_path)
        copper_layers = spec.stackup.copper_layers
        family = spec.coupon_family
        runner = _FakeExportRunner(seed=spec_path.name, copper_layers=copper_layers)

        result = build_coupon(
            spec,
            out_root=tmp_path,
            mode="docker",
            runner=runner,
            kicad_cli_version="9.0.7",
        )

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        export_paths = [e["path"] for e in manifest["exports"]]

        # Validate layer set with strict=True (oracle path)
        # Manifest paths include fab/ prefix (e.g., fab/gerbers/..., fab/drill/...)
        validation_result = validate_layer_set(
            export_paths=export_paths,
            copper_layers=copper_layers,
            family=family,
            gerber_dir="fab/gerbers/",
            strict=True,
        )
        assert validation_result.passed is True, (
            f"Layer validation failed for {spec_path.name}: missing {validation_result.missing_layers}"
        )


# ---------------------------------------------------------------------------
# G4 Gate Tests: Gerber Extension Mapping
# ---------------------------------------------------------------------------


@pytest.mark.gate_g4
class TestG4GerberExtensionMapping:
    """Gate G4 tests for Gerber file extension mapping."""

    def test_extension_map_exists(self) -> None:
        """Gerber extension map should be available."""
        ext_map = get_gerber_extension_map()
        assert isinstance(ext_map, dict)
        assert len(ext_map) > 0

    def test_extension_map_has_copper_layers(self) -> None:
        """Extension map should include copper layers."""
        ext_map = get_gerber_extension_map()
        assert "F.Cu" in ext_map
        assert "B.Cu" in ext_map

    def test_extension_map_has_mask_layers(self) -> None:
        """Extension map should include mask layers."""
        ext_map = get_gerber_extension_map()
        assert "F.Mask" in ext_map
        assert "B.Mask" in ext_map

    def test_extension_format(self) -> None:
        """Extensions should use standard KiCad Gerber extensions.

        KiCad uses industry-standard extensions:
        - .gtl, .gbl for top/bottom copper
        - .g1, .g2, etc. for inner copper layers
        - .gts, .gbs for soldermask
        - .gto, .gbo for silkscreen
        - .gm1 for mechanical/edge cuts
        """
        ext_map = get_gerber_extension_map()
        # Verify extensions are valid KiCad Gerber formats
        valid_extensions = {".gtl", ".gbl", ".g1", ".g2", ".g3", ".g4", ".gts", ".gbs", ".gto", ".gbo", ".gtp", ".gbp", ".gm1"}
        for layer, ext in ext_map.items():
            ext_suffix = "." + ext.split(".")[-1]
            assert ext_suffix in valid_extensions, f"Layer {layer} extension should be a valid KiCad Gerber extension, got {ext}"
