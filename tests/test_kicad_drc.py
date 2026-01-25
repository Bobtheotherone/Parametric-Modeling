"""KiCad DRC tests for golden specs.

REQ-M1-003: Connector footprints MUST be sourced from vendored in-repo `.kicad_mod` files.
REQ-M1-004: Footprint-to-net and anchor-pad mapping MUST be deterministic and explicit.
REQ-M1-005: CPWG generation MUST produce net-aware copper geometry.
REQ-M1-006: DRC MUST be run with zone refill enabled and exports with zone checks enabled.
REQ-M1-008: A launch feature MUST exist for F0/F1 that connects connector pads to CPWG.
REQ-M1-016: For golden specs, KiCad DRC MUST pass in CI on the pinned toolchain.

This module tests that:
- All golden specs can run DRC with the pinned KiCad Docker image
- DRC invocation uses correct flags (severity-all, exit-code-violations, JSON output)
- Docker mode correctly mounts workdirs and uses digest-pinned images
- DRC reports are generated in the expected format
- Connector footprints are sourced from vendored in-repo files
- Pad maps are deterministic and correct
- CPWG copper geometry includes signal trace and gap enforcement
- Launch features connect connector pads to CPWG properly
- Zone policy flags are enforced in DRC and exports

IMPORTANT: These tests use fake runners to avoid actually invoking KiCad
during CI. The real KiCad DRC is tested via integration tests that require
the KiCad Docker image.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.coupongen import (
    KicadCliRunner,
    build_drc_args,
    load_spec,
    resolve_spec,
)
from formula_foundry.coupongen.geom.footprint_meta import (
    load_footprint_meta,
    list_available_footprint_meta,
)
from formula_foundry.coupongen.geom.cpwg import (
    CPWGSpec,
    generate_cpwg_segment,
    generate_cpwg_ground_tracks,
)
from formula_foundry.coupongen.geom.launch import build_launch_plan
from formula_foundry.coupongen.geom.primitives import PositionNM
from formula_foundry.coupongen.kicad import (
    deterministic_uuid_indexed,
    parse,
)
from formula_foundry.coupongen.kicad.runners.protocol import DEFAULT_ZONE_POLICY
from formula_foundry.coupongen.paths import (
    FOOTPRINT_LIB_DIR,
    FOOTPRINT_META_DIR,
    get_footprint_module_path,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SPECS_DIR = ROOT / "tests" / "golden_specs"


class _FakeDrcRunner:
    """Fake KiCad CLI runner that simulates DRC without invoking KiCad.

    This allows testing the DRC pipeline without requiring the KiCad Docker image.
    Returns configurable DRC results for testing success and failure paths.
    """

    def __init__(
        self,
        *,
        returncode: int = 0,
        violations: list[dict[str, Any]] | None = None,
    ) -> None:
        self.returncode = returncode
        self.violations = violations or []
        self.calls: list[tuple[Path, Path]] = []

    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
        """Simulate DRC execution and write a fake report."""
        self.calls.append((board_path, report_path))

        report = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "source": str(board_path),
            "violations": self.violations,
            "unconnected_items": [],
            "schematic_parity": [],
            "coordinate_units": "mm",
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return subprocess.CompletedProcess(
            args=["kicad-cli", "pcb", "drc"],
            returncode=self.returncode,
            stdout="",
            stderr="" if self.returncode == 0 else "DRC violations found",
        )

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        """Simulate Gerber export."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "F.Cu.gbr").write_text("G04 Fake Gerber*\nX0Y0D02*\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        """Simulate drill file export."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "drill.drl").write_text("M48\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=["kicad-cli"], returncode=0, stdout="", stderr="")


class _SpyKicadCliRunner(KicadCliRunner):
    """KiCad CLI runner that records calls and writes stub outputs."""

    def __init__(
        self,
        *,
        mode: str,
        docker_image: str | None,
        returncode: int = 0,
        violations: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(mode=mode, docker_image=docker_image)
        object.__setattr__(self, "_calls", [])
        object.__setattr__(self, "_returncode", returncode)
        object.__setattr__(self, "_violations", violations or [])

    @property
    def calls(self) -> list[list[str]]:
        return self._calls  # type: ignore[return-value]

    def run(
        self,
        args: list[str],
        *,
        workdir: Path,
        timeout: float | None = None,
        variables: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        args_list = list(args)
        self._calls.append(args_list)
        self._write_stub_outputs(args_list, workdir)
        return subprocess.CompletedProcess(
            args=args_list,
            returncode=self._returncode,
            stdout="",
            stderr="" if self._returncode == 0 else "DRC violations found",
        )

    def _write_stub_outputs(self, args: list[str], workdir: Path) -> None:
        if args[:2] == ["pcb", "drc"]:
            report_path = self._resolve_output_path(args, workdir)
            if report_path is not None:
                report = {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "source": str(args[-1]),
                    "violations": list(self._violations),
                    "unconnected_items": [],
                    "schematic_parity": [],
                    "coordinate_units": "mm",
                }
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return

        if args[:3] == ["pcb", "export", "gerbers"]:
            out_dir = self._resolve_output_path(args, workdir)
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "F.Cu.gbr").write_text("G04 Fake Gerber*\nX0Y0D02*\n", encoding="utf-8")
            return

        if args[:3] == ["pcb", "export", "drill"]:
            out_dir = self._resolve_output_path(args, workdir)
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "drill.drl").write_text("M48\n", encoding="utf-8")
            return

    @staticmethod
    def _resolve_output_path(args: list[str], workdir: Path) -> Path | None:
        if "--output" not in args:
            return None
        output_idx = args.index("--output") + 1
        if output_idx >= len(args):
            return None
        output = Path(str(args[output_idx]))
        if output.is_absolute():
            return output
        return workdir / output


def _golden_specs() -> list[Path]:
    """Collect all golden spec files."""
    patterns = ("*.json", "*.yaml", "*.yml")
    specs: list[Path] = []
    for pattern in patterns:
        specs.extend(sorted(GOLDEN_SPECS_DIR.glob(pattern)))
    # Filter out __init__.py and any non-spec files
    specs = [s for s in specs if s.name != "__init__.py"]
    return sorted(specs)


def _segments_from_board(board: list[Any]) -> list[list[Any]]:
    return [node for node in board if isinstance(node, list) and node and node[0] == "segment"]


def _footprints_from_board(board: list[Any]) -> list[list[Any]]:
    return [node for node in board if isinstance(node, list) and node and node[0] == "footprint"]


def _nets_from_board(board: list[Any]) -> list[list[Any]]:
    return [node for node in board if isinstance(node, list) and node and node[0] == "net"]


def _segment_attr(segment: list[Any], key: str) -> Any | None:
    for node in segment:
        if isinstance(node, list) and node and node[0] == key and len(node) > 1:
            return node[1]
    return None


def _segment_net_id(segment: list[Any]) -> int | None:
    value = _segment_attr(segment, "net")
    return int(value) if isinstance(value, int) else None


def _segment_layer(segment: list[Any]) -> str | None:
    value = _segment_attr(segment, "layer")
    return str(value) if value is not None else None


def _segment_tstamp(segment: list[Any]) -> str | None:
    value = _segment_attr(segment, "tstamp")
    return str(value) if value is not None else None


def _net_name_map(board: list[Any]) -> dict[int, str]:
    net_map: dict[int, str] = {}
    for node in _nets_from_board(board):
        if len(node) >= 3 and isinstance(node[1], int) and isinstance(node[2], str):
            net_map[node[1]] = node[2]
    return net_map


def _footprint_names(board: list[Any]) -> list[str]:
    names: list[str] = []
    for node in _footprints_from_board(board):
        if len(node) >= 2 and isinstance(node[1], str):
            names.append(node[1])
    return names


# ============================================================================
# REQ-M1-003: Vendored Connector Footprints
# ============================================================================


class TestVendoredConnectorFootprints:
    """Tests for REQ-M1-003: Connector footprints must be sourced from vendored files.

    REQ-M1-003: Connector footprints MUST be sourced from vendored in-repo
    `.kicad_mod` files and embedded into the generated `.kicad_pcb`;
    placeholder "single pad connector" generation is disallowed for M1 compliance.
    """

    def test_footprint_library_directory_exists(self) -> None:
        """Verify vendored footprint library directory exists."""
        assert FOOTPRINT_LIB_DIR.exists(), (
            f"Vendored footprint library directory not found: {FOOTPRINT_LIB_DIR}"
        )
        assert FOOTPRINT_LIB_DIR.is_dir()

    def test_connectors_pretty_directory_exists(self) -> None:
        """Verify Coupongen_Connectors.pretty directory exists."""
        connectors_dir = FOOTPRINT_LIB_DIR / "Coupongen_Connectors.pretty"
        assert connectors_dir.exists(), (
            f"Connectors footprint library not found: {connectors_dir}"
        )
        assert connectors_dir.is_dir()

    def test_sma_endlaunch_footprint_exists(self) -> None:
        """Verify SMA_EndLaunch_Generic.kicad_mod exists in vendored library."""
        footprint_path = get_footprint_module_path(
            "Coupongen_Connectors", "SMA_EndLaunch_Generic"
        )
        assert footprint_path.exists(), (
            f"SMA_EndLaunch_Generic footprint not found: {footprint_path}"
        )

    def test_footprint_metadata_directory_exists(self) -> None:
        """Verify footprint metadata directory exists."""
        assert FOOTPRINT_META_DIR.exists(), (
            f"Footprint metadata directory not found: {FOOTPRINT_META_DIR}"
        )

    def test_sma_endlaunch_metadata_exists(self) -> None:
        """Verify SMA_EndLaunch_Generic metadata JSON exists."""
        meta_path = FOOTPRINT_META_DIR / "SMA_EndLaunch_Generic.json"
        assert meta_path.exists(), (
            f"SMA_EndLaunch_Generic metadata not found: {meta_path}"
        )

    def test_footprint_file_has_valid_sexpr(self) -> None:
        """Verify vendored footprint file contains valid KiCad S-expression."""
        footprint_path = get_footprint_module_path(
            "Coupongen_Connectors", "SMA_EndLaunch_Generic"
        )
        content = footprint_path.read_text(encoding="utf-8")
        # Basic S-expression validation: starts with (footprint or (module
        assert content.strip().startswith("("), "Footprint must start with S-expression"
        assert "footprint" in content or "module" in content, (
            "Footprint file must contain 'footprint' or 'module' keyword"
        )

    @pytest.mark.parametrize("spec_path", _golden_specs(), ids=lambda p: p.name)
    def test_golden_spec_uses_vendored_footprint(self, spec_path: Path) -> None:
        """REQ-M1-003: Each golden spec must reference a vendored footprint."""
        spec = load_spec(spec_path)

        # Check left connector
        left_fp = spec.connectors.left.footprint
        lib, name = left_fp.split(":", 1)
        fp_path = get_footprint_module_path(lib, name)
        assert fp_path.exists(), (
            f"Golden spec {spec_path.name} uses non-vendored left footprint: {left_fp}"
        )

        # Check right connector
        right_fp = spec.connectors.right.footprint
        lib, name = right_fp.split(":", 1)
        fp_path = get_footprint_module_path(lib, name)
        assert fp_path.exists(), (
            f"Golden spec {spec_path.name} uses non-vendored right footprint: {right_fp}"
        )


# ============================================================================
# REQ-M1-004: Pad-Map Correctness
# ============================================================================


class TestPadMapCorrectness:
    """Tests for REQ-M1-004: Footprint-to-net and anchor-pad mapping.

    REQ-M1-004: Footprint-to-net and anchor-pad mapping MUST be deterministic
    and explicit (via `pad_map` or documented conventions) so the launch connects
    to the true signal pad and GND pads/nets are correctly assigned.
    """

    def test_sma_footprint_has_signal_pad(self) -> None:
        """Verify SMA footprint metadata has a signal pad defined."""
        meta = load_footprint_meta("SMA_EndLaunch_Generic")
        assert meta.signal_pad is not None
        assert meta.signal_pad.pad_number

    def test_sma_footprint_has_ground_pads(self) -> None:
        """Verify SMA footprint metadata has ground pads defined."""
        meta = load_footprint_meta("SMA_EndLaunch_Generic")
        assert len(meta.ground_pads) > 0, "At least one ground pad required"

    def test_pad_net_map_is_deterministic(self) -> None:
        """Verify pad_net_map returns consistent mapping."""
        meta = load_footprint_meta("SMA_EndLaunch_Generic")
        pad_map_1 = meta.pad_net_map()
        pad_map_2 = meta.pad_net_map()
        assert pad_map_1 == pad_map_2, "Pad map must be deterministic"

    def test_signal_pad_maps_to_sig_net(self) -> None:
        """Verify signal pad maps to SIG net."""
        meta = load_footprint_meta("SMA_EndLaunch_Generic")
        pad_map = meta.pad_net_map()
        signal_pad_num = meta.signal_pad.pad_number
        assert signal_pad_num in pad_map
        assert pad_map[signal_pad_num] == "SIG", (
            f"Signal pad {signal_pad_num} should map to SIG net"
        )

    def test_ground_pads_map_to_gnd_net(self) -> None:
        """Verify ground pads map to GND net."""
        meta = load_footprint_meta("SMA_EndLaunch_Generic")
        pad_map = meta.pad_net_map()
        for ground_pad in meta.ground_pads:
            pad_num = ground_pad.pad_number
            assert pad_num in pad_map
            assert pad_map[pad_num] == "GND", (
                f"Ground pad {pad_num} should map to GND net"
            )

    def test_all_footprint_meta_have_valid_pad_maps(self) -> None:
        """Verify all available footprint metadata have valid pad maps."""
        meta_ids = list_available_footprint_meta()
        assert len(meta_ids) > 0, "No footprint metadata available"

        for meta_id in meta_ids:
            meta = load_footprint_meta(meta_id)
            pad_map = meta.pad_net_map()
            # Should have at least signal and one ground
            assert len(pad_map) >= 2, f"Footprint {meta_id} must have signal + ground pads"


# ============================================================================
# REQ-M1-005: CPWG Copper with Gap
# ============================================================================


class TestCPWGCopperWithGap:
    """Tests for REQ-M1-005: CPWG generation with net-aware copper geometry.

    REQ-M1-005: CPWG generation MUST produce net-aware copper geometry: a signal
    conductor on the declared layer plus a GND reference conductor on that layer
    with an enforced `gap_nm` (no "CPWG in schema only").
    """

    def test_cpwg_segment_has_width(self) -> None:
        """Verify CPWG signal segment has the specified width."""
        spec = CPWGSpec(w_nm=250_000, gap_nm=180_000, length_nm=24_000_000)
        start = PositionNM(0, 0)
        end = PositionNM(24_000_000, 0)
        segment = generate_cpwg_segment(start, end, spec)

        assert segment.width_nm == 250_000

    def test_cpwg_segment_has_net_id(self) -> None:
        """Verify CPWG signal segment has net ID assigned."""
        spec = CPWGSpec(w_nm=250_000, gap_nm=180_000, length_nm=24_000_000, net_id=1)
        start = PositionNM(0, 0)
        end = PositionNM(24_000_000, 0)
        segment = generate_cpwg_segment(start, end, spec)

        assert segment.net_id == 1

    def test_cpwg_ground_tracks_enforce_gap(self) -> None:
        """Verify ground tracks are offset to enforce gap_nm."""
        w_nm = 250_000
        gap_nm = 180_000
        spec = CPWGSpec(w_nm=w_nm, gap_nm=gap_nm, length_nm=24_000_000)
        start = PositionNM(0, 0)
        end = PositionNM(24_000_000, 0)

        pos_track, neg_track = generate_cpwg_ground_tracks(start, end, spec)

        # Ground tracks should be offset from center by:
        # (signal_width/2) + gap + (ground_width/2)
        expected_offset = w_nm // 2 + gap_nm + w_nm // 2

        # For horizontal segment, offset is in Y direction
        assert pos_track.start.y == expected_offset
        assert neg_track.start.y == -expected_offset

    def test_cpwg_ground_tracks_have_ground_net_id(self) -> None:
        """Verify ground tracks have ground net ID assigned."""
        spec = CPWGSpec(
            w_nm=250_000, gap_nm=180_000, length_nm=24_000_000,
            net_id=1, ground_net_id=2
        )
        start = PositionNM(0, 0)
        end = PositionNM(24_000_000, 0)

        pos_track, neg_track = generate_cpwg_ground_tracks(start, end, spec)

        assert pos_track.net_id == 2
        assert neg_track.net_id == 2

    @pytest.mark.parametrize("spec_path", _golden_specs(), ids=lambda p: p.name)
    def test_golden_spec_has_cpwg_parameters(self, spec_path: Path) -> None:
        """REQ-M1-005: Each golden spec must define CPWG with gap_nm."""
        spec = load_spec(spec_path)

        assert spec.transmission_line.type == "CPWG", (
            f"Golden spec {spec_path.name} must use CPWG transmission line"
        )
        assert spec.transmission_line.w_nm > 0, (
            f"Golden spec {spec_path.name} must have positive trace width"
        )
        assert spec.transmission_line.gap_nm > 0, (
            f"Golden spec {spec_path.name} must have positive gap"
        )


# ============================================================================
# REQ-M1-006: Zone-Policy DRC Flags
# ============================================================================


class TestZonePolicyDrcFlags:
    """Tests for REQ-M1-006: DRC with zone refill and export with zone checks.

    REQ-M1-006: If CPWG uses zones, DRC MUST be run with zone refill enabled
    and exports MUST be run with zone checks enabled (KiCad CLI flags/policy
    pinned in code and recorded in manifest).
    """

    def test_default_zone_policy_refill_zones(self) -> None:
        """Verify default zone policy enables refill for DRC."""
        assert DEFAULT_ZONE_POLICY.drc_refill_zones is True

    def test_default_zone_policy_drc_flag(self) -> None:
        """Verify default zone policy uses --refill-zones flag."""
        assert DEFAULT_ZONE_POLICY.drc_refill_flag == "--refill-zones"

    def test_default_zone_policy_export_check_zones(self) -> None:
        """Verify default zone policy enables zone checks for export."""
        assert DEFAULT_ZONE_POLICY.export_check_zones is True

    def test_default_zone_policy_export_flag(self) -> None:
        """Verify default zone policy uses --check-zones flag."""
        assert DEFAULT_ZONE_POLICY.export_check_flag == "--check-zones"

    def test_build_drc_args_includes_refill_zones(self, tmp_path: Path) -> None:
        """Verify build_drc_args includes --refill-zones flag."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert "--refill-zones" in args

    def test_build_drc_args_refill_can_be_disabled(self, tmp_path: Path) -> None:
        """Verify refill_zones=False disables --refill-zones flag."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report, refill_zones=False)

        assert "--refill-zones" not in args

    def test_zone_policy_has_policy_id(self) -> None:
        """Verify zone policy has a stable policy ID for manifesting."""
        assert DEFAULT_ZONE_POLICY.policy_id == "kicad-cli-zones-v1"

    def test_zone_policy_to_dict_structure(self) -> None:
        """Verify zone policy serializes with expected structure."""
        policy_dict = DEFAULT_ZONE_POLICY.to_dict()

        assert "policy_id" in policy_dict
        assert "drc" in policy_dict
        assert "export" in policy_dict
        assert policy_dict["drc"]["refill_zones"] is True
        assert policy_dict["export"]["check_zones"] is True

    def test_zone_policy_with_version(self) -> None:
        """Verify zone policy can include kicad_cli_version."""
        policy = DEFAULT_ZONE_POLICY.with_kicad_cli_version("9.0.7")
        assert policy.kicad_cli_version == "9.0.7"

        policy_dict = policy.to_dict()
        assert policy_dict["kicad_cli_version"] == "9.0.7"


# ============================================================================
# REQ-M1-008: Launch Feature Presence
# ============================================================================


class TestLaunchFeaturePresence:
    """Tests for REQ-M1-008: Launch feature connecting connector pads to CPWG.

    REQ-M1-008: A launch feature MUST exist for F0/F1 that deterministically
    connects connector pads to CPWG (taper or stepped transition) with correct
    nets, optional stitching, and manufacturable DFM constraints.
    """

    def test_build_launch_plan_creates_segments(self) -> None:
        """Verify build_launch_plan creates transition segments."""
        pad_center = PositionNM(5_000_000, 0)
        launch_point = PositionNM(10_000_000, 0)

        plan = build_launch_plan(
            side="left",
            pad_center=pad_center,
            launch_point=launch_point,
            launch_direction_deg=0,
            rotation_deg=0,
            pad_size_x_nm=500_000,
            pad_size_y_nm=1_500_000,
            trace_width_nm=250_000,
            trace_layer="F.Cu",
            gap_nm=180_000,
            min_trace_width_nm=100_000,
            min_gap_nm=100_000,
        )

        assert plan.side == "left"
        assert plan.pad_center == pad_center
        assert plan.launch_point == launch_point

    def test_launch_plan_has_sig_net(self) -> None:
        """Verify launch plan uses SIG net name."""
        plan = build_launch_plan(
            side="right",
            pad_center=PositionNM(75_000_000, 0),
            launch_point=PositionNM(70_000_000, 0),
            launch_direction_deg=180,
            rotation_deg=0,
            pad_size_x_nm=500_000,
            pad_size_y_nm=1_500_000,
            trace_width_nm=250_000,
            trace_layer="F.Cu",
            gap_nm=180_000,
            min_trace_width_nm=100_000,
            min_gap_nm=100_000,
        )

        assert plan.net_name == "SIG"

    def test_launch_plan_transition_length(self) -> None:
        """Verify launch plan calculates correct transition length."""
        pad_center = PositionNM(5_000_000, 0)
        launch_point = PositionNM(10_000_000, 0)

        plan = build_launch_plan(
            side="left",
            pad_center=pad_center,
            launch_point=launch_point,
            launch_direction_deg=0,
            rotation_deg=0,
            pad_size_x_nm=500_000,
            pad_size_y_nm=1_500_000,
            trace_width_nm=250_000,
            trace_layer="F.Cu",
            gap_nm=180_000,
            min_trace_width_nm=100_000,
            min_gap_nm=100_000,
        )

        expected_length = abs(launch_point.x - pad_center.x)
        assert plan.transition_length_nm == expected_length

    def test_launch_plan_side_validation(self) -> None:
        """Verify launch plan validates side parameter."""
        with pytest.raises(ValueError, match="side must be"):
            build_launch_plan(
                side="invalid",
                pad_center=PositionNM(0, 0),
                launch_point=PositionNM(1_000_000, 0),
                launch_direction_deg=0,
                rotation_deg=0,
                pad_size_x_nm=500_000,
                pad_size_y_nm=1_500_000,
                trace_width_nm=250_000,
                trace_layer="F.Cu",
                gap_nm=180_000,
                min_trace_width_nm=100_000,
                min_gap_nm=100_000,
            )

    def test_footprint_meta_has_launch_reference(self) -> None:
        """Verify footprint metadata includes launch reference point."""
        meta = load_footprint_meta("SMA_EndLaunch_Generic")
        assert meta.launch_reference is not None
        assert meta.launch_reference.direction_deg >= 0
        assert meta.launch_reference.direction_deg < 360


# ============================================================================
# REQ-M1-016: Golden Specs DRC Pass
# ============================================================================


class TestKicadCliRunnerModes:
    """Tests for KiCad CLI runner mode configuration.

    REQ-M1-016: CI must prove DRC-clean boards using the pinned KiCad toolchain.
    """

    def test_local_mode_command_structure(self, tmp_path: Path) -> None:
        """Local mode should use kicad-cli directly."""
        runner = KicadCliRunner(mode="local")
        cmd = runner.build_command(["pcb", "drc", "board.kicad_pcb"], workdir=tmp_path)

        assert cmd[0] == "kicad-cli"
        assert "pcb" in cmd
        assert "drc" in cmd
        assert "board.kicad_pcb" in cmd

    def test_docker_mode_command_structure(self, tmp_path: Path) -> None:
        """REQ-M1-016: Docker mode should use pinned digest image."""
        docker_image = "kicad/kicad:9.0.7@sha256:abc123def456"
        runner = KicadCliRunner(mode="docker", docker_image=docker_image)
        cmd = runner.build_command(["pcb", "drc", "board.kicad_pcb"], workdir=tmp_path)

        assert cmd[0] == "docker"
        assert "run" in cmd
        assert "--rm" in cmd
        assert docker_image in cmd
        assert "-v" in cmd
        # Verify workdir mount
        mount_idx = cmd.index("-v")
        mount_arg = cmd[mount_idx + 1]
        assert str(tmp_path) in mount_arg
        assert "/workspace" in mount_arg

    def test_docker_mode_requires_image(self, tmp_path: Path) -> None:
        """Docker mode must require a docker_image."""
        runner = KicadCliRunner(mode="docker", docker_image=None)
        with pytest.raises(ValueError, match="docker_image is required"):
            runner.build_command(["pcb", "drc"], workdir=tmp_path)


class TestBuildDrcArgs:
    """Tests for DRC argument construction.

    REQ-M1-016: CI must prove DRC-clean boards using correct invocation flags.
    """

    def test_drc_args_default_severity_all(self, tmp_path: Path) -> None:
        """DRC should default to all severity for M1.

        Per REQ-M1-016 and DESIGN_DOCUMENT.md: M1 uses --severity-all to catch
        all DRC violations including warnings for thorough quality checks.
        """
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert "--severity-all" in args

    def test_drc_args_severity_all_option(self, tmp_path: Path) -> None:
        """DRC should support all severity levels when requested."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report, severity="all")

        assert "--severity-all" in args

    def test_drc_args_severity_warning_option(self, tmp_path: Path) -> None:
        """DRC should support warning severity level."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report, severity="warning")

        assert "--severity-warning" in args

    def test_drc_args_include_exit_code_violations(self, tmp_path: Path) -> None:
        """DRC should return non-zero exit on violations."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert "--exit-code-violations" in args

    def test_drc_args_use_json_format(self, tmp_path: Path) -> None:
        """DRC should output JSON format for machine parsing."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert "--format" in args
        format_idx = args.index("--format")
        assert args[format_idx + 1] == "json"

    def test_drc_args_specify_output_path(self, tmp_path: Path) -> None:
        """DRC should write report to specified path."""
        board = tmp_path / "coupon.kicad_pcb"
        report = tmp_path / "drc.json"
        args = build_drc_args(board, report)

        assert "--output" in args
        output_idx = args.index("--output")
        assert args[output_idx + 1] == str(report)


class TestGoldenSpecsDrcCompatibility:
    """Tests verifying golden specs are compatible with DRC toolchain.

    REQ-M1-016: CI must prove DRC-clean boards and export completeness for all
    golden specs using the pinned KiCad toolchain.
    """

    def test_golden_specs_exist(self) -> None:
        """Verify golden specs directory contains expected specs."""
        specs = _golden_specs()
        assert len(specs) >= 20, f"Expected at least 20 golden specs, found {len(specs)}"

    def test_golden_specs_specify_pinned_docker_image(self) -> None:
        """REQ-M1-016: All golden specs must use digest-pinned Docker images."""
        specs = _golden_specs()
        for spec_path in specs:
            spec = load_spec(spec_path)
            docker_image = spec.toolchain.kicad.docker_image
            assert "@sha256:" in docker_image, (
                f"Golden spec {spec_path.name} must use digest-pinned Docker image, got: {docker_image}"
            )

    def test_golden_specs_kicad_version_pinned(self) -> None:
        """REQ-M1-016: All golden specs must pin KiCad version."""
        specs = _golden_specs()
        for spec_path in specs:
            spec = load_spec(spec_path)
            version = spec.toolchain.kicad.version
            assert version, f"Golden spec {spec_path.name} must specify KiCad version"
            # Version should be a semantic version string
            parts = version.split(".")
            assert len(parts) >= 2, f"Invalid version format: {version}"

    @pytest.mark.parametrize("spec_path", _golden_specs(), ids=lambda p: p.name)
    def test_golden_spec_drc_clean_with_fake_runner(self, spec_path: Path, tmp_path: Path) -> None:
        """REQ-M1-016: Each golden spec should be DRC-clean.

        This test uses a spy runner to validate DRC/export flags and board geometry
        without invoking the real KiCad CLI.
        """
        from formula_foundry.coupongen import build_coupon

        spec = load_spec(spec_path)
        runner = _SpyKicadCliRunner(
            mode="docker",
            docker_image=spec.toolchain.kicad.docker_image,
            returncode=0,
        )

        result = build_coupon(
            spec,
            out_root=tmp_path,
            mode="docker",
            runner=runner,
            kicad_cli_version=spec.toolchain.kicad.version,
        )

        drc_calls = [call for call in runner.calls if call[:2] == ["pcb", "drc"]]
        gerber_calls = [call for call in runner.calls if call[:3] == ["pcb", "export", "gerbers"]]
        drill_calls = [call for call in runner.calls if call[:3] == ["pcb", "export", "drill"]]

        assert len(drc_calls) == 1
        assert DEFAULT_ZONE_POLICY.drc_refill_flag in drc_calls[0]
        assert len(gerber_calls) == 1
        assert DEFAULT_ZONE_POLICY.export_check_flag in gerber_calls[0]
        assert len(drill_calls) == 1
        assert DEFAULT_ZONE_POLICY.export_check_flag in drill_calls[0]

        # Verify manifest records DRC success
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["verification"]["drc"]["returncode"] == 0

        board_path = result.output_dir / "coupon.kicad_pcb"
        board = parse(board_path.read_text(encoding="utf-8"))

        # Verify connector footprints are embedded
        footprint_names = _footprint_names(board)
        left_fp = spec.connectors.left.footprint
        right_fp = spec.connectors.right.footprint
        if left_fp == right_fp:
            assert footprint_names.count(left_fp) >= 2
        else:
            assert left_fp in footprint_names
            assert right_fp in footprint_names

        # Verify CPWG nets exist
        net_map = _net_name_map(board)
        assert net_map.get(1) == "SIG"
        assert net_map.get(2) == "GND"

        segments = _segments_from_board(board)
        signal_segments = [seg for seg in segments if _segment_net_id(seg) == 1]
        ground_segments = [seg for seg in segments if _segment_net_id(seg) == 2]

        assert signal_segments, "Signal net segments missing from board"
        assert ground_segments, "Ground net segments missing from board"

        cpwg_layer = spec.transmission_line.layer
        assert any(_segment_layer(seg) == cpwg_layer for seg in signal_segments)
        assert sum(1 for seg in ground_segments if _segment_layer(seg) == cpwg_layer) >= 2

        resolved = resolve_spec(spec)
        assert resolved.layout_plan is not None
        segment_tstamps = {
            tstamp for seg in segments if (tstamp := _segment_tstamp(seg)) is not None
        }
        for side in ("left", "right"):
            launch_plan = resolved.layout_plan.get_launch_plan(side)
            assert launch_plan is not None
            assert launch_plan.segments, f"Launch plan missing segments for {side}"
            for idx, _segment in enumerate(launch_plan.segments):
                expected = deterministic_uuid_indexed(
                    spec.schema_version,
                    f"track.launch.{side}",
                    idx,
                )
                assert expected in segment_tstamps


class TestDrcFailureHandling:
    """Tests for DRC failure scenarios."""

    def test_drc_failure_raises_when_must_pass(self, tmp_path: Path) -> None:
        """DRC failure should raise when constraints.drc.must_pass is True."""
        from formula_foundry.coupongen import build_coupon
        from formula_foundry.coupongen.spec import CouponSpec

        spec_data = {
            "schema_version": 1,
            "coupon_family": "F0_CAL_THRU_LINE",
            "units": "nm",
            "toolchain": {
                "kicad": {
                    "version": "9.0.7",
                    "docker_image": "kicad/kicad:9.0.7@sha256:deadbeef",
                }
            },
            "fab_profile": {"id": "oshpark_4layer", "overrides": {}},
            "stackup": {
                "copper_layers": 4,
                "thicknesses_nm": {
                    "L1_to_L2": 180000,
                    "L2_to_L3": 800000,
                    "L3_to_L4": 180000,
                },
                "materials": {"er": 4.1, "loss_tangent": 0.02},
            },
            "board": {
                "outline": {
                    "width_nm": 20000000,
                    "length_nm": 80000000,
                    "corner_radius_nm": 2000000,
                },
                "origin": {"mode": "EDGE_L_CENTER"},
                "text": {"coupon_id": "${COUPON_ID}", "include_manifest_hash": True},
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
                "w_nm": 250000,
                "gap_nm": 180000,
                "length_left_nm": 24000000,
                "length_right_nm": 24000000,
                "ground_via_fence": None,
            },
            "discontinuity": None,
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
        spec = CouponSpec.model_validate(spec_data)
        runner = _FakeDrcRunner(
            returncode=1,
            violations=[{"type": "clearance", "severity": "error", "description": "Test violation"}],
        )

        with pytest.raises(RuntimeError, match="KiCad DRC failed"):
            build_coupon(
                spec,
                out_root=tmp_path,
                mode="docker",
                runner=runner,
                kicad_cli_version="9.0.7",
            )

    def test_drc_report_json_structure(self, tmp_path: Path) -> None:
        """DRC report should have expected JSON structure."""
        runner = _FakeDrcRunner(returncode=0)
        board_path = tmp_path / "test.kicad_pcb"
        report_path = tmp_path / "drc.json"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")

        runner.run_drc(board_path, report_path)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "violations" in report
        assert "unconnected_items" in report
        assert isinstance(report["violations"], list)
