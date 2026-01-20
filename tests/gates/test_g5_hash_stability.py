# SPDX-License-Identifier: MIT
"""Gate G5 tests: Output hash stability across repeated runs.

This module tests:
- design_hash is stable across 3 consecutive runs for same spec
- Board file canonical hash is stable (ignoring tstamp/uuid)
- Gerber canonical hashes are stable (ignoring timestamps/dates)
- Manifest export hashes are stable
- Hash computation is deterministic

Per ECO-M1-ALIGN-0001:
- REQ-M1-024: CI must prove deterministic resolve hashing against committed golden hashes
- REQ-M1-025: Export hashes must be stable (canonical hashing)

Gate G5 verifies: "stable canonical hashes across repeated runs"
- design_hash stable across 3 runs
- Board file canonical hash stable
- Gerber canonical hashes stable
- All hashes addressable and reproducible

Pytest marker: gate_g5

Note: Real KiCad integration tests are in tests/integration/test_export_determinism_integration.py
and require Docker. These tests verify hash stability logic without Docker.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.coupongen import (
    canonical_hash_export_text,
    canonical_hash_kicad_pcb_text,
    load_spec,
    resolve,
    design_hash,
    resolved_design_canonical_json,
)
from formula_foundry.coupongen.hashing import (
    canonicalize_export_text,
    canonicalize_kicad_pcb_text,
)
from formula_foundry.substrate import sha256_bytes


# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parents[1]
GOLDEN_SPECS_DIR = TESTS_DIR / "golden_specs"
GOLDEN_HASHES_PATH = ROOT / "golden_hashes" / "design_hashes.json"

# Number of runs for stability tests
NUM_STABILITY_RUNS = 3


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


def _load_golden_hashes() -> dict[str, str]:
    """Load golden design hashes from committed JSON file."""
    if not GOLDEN_HASHES_PATH.exists():
        return {}
    data = json.loads(GOLDEN_HASHES_PATH.read_text(encoding="utf-8"))
    return data.get("spec_hashes", {})


class _FakeStabilityRunner:
    """Fake KiCad CLI runner for hash stability testing.

    Generates deterministic output based on a seed for reproducibility testing.
    """

    def __init__(self, *, seed: str = "default") -> None:
        self.seed = seed
        self.run_count = 0

    def run_drc(
        self, board_path: Path, report_path: Path
    ) -> subprocess.CompletedProcess[str]:
        """Simulate DRC execution."""
        self.run_count += 1
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"violations": []}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(
            args=["kicad-cli"], returncode=0, stdout="", stderr=""
        )

    def export_gerbers(
        self, board_path: Path, out_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        """Simulate Gerber export with deterministic content based on seed."""
        self.run_count += 1
        out_dir.mkdir(parents=True, exist_ok=True)

        layers = [
            ("F_Cu.gbr", "F.Cu"),
            ("B_Cu.gbr", "B.Cu"),
            ("In1_Cu.gbr", "In1.Cu"),
            ("In2_Cu.gbr", "In2.Cu"),
            ("F_Mask.gbr", "F.Mask"),
            ("B_Mask.gbr", "B.Mask"),
            ("Edge_Cuts.gbr", "Edge.Cuts"),
        ]

        for filename, layer in layers:
            # Deterministic content based on seed and layer
            content_hash = hashlib.sha256(
                f"{self.seed}:{layer}".encode()
            ).hexdigest()[:16]
            content = f"G04 Layer {layer}*\nG04 Hash={content_hash}*\nX0Y0D02*\nX1000Y0D01*\nM02*\n"
            (out_dir / filename).write_text(content, encoding="utf-8")

        return subprocess.CompletedProcess(
            args=["kicad-cli"], returncode=0, stdout="", stderr=""
        )

    def export_drill(
        self, board_path: Path, out_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        """Simulate drill export with deterministic content."""
        self.run_count += 1
        out_dir.mkdir(parents=True, exist_ok=True)

        content_hash = hashlib.sha256(f"{self.seed}:drill".encode()).hexdigest()[:16]
        (out_dir / "drill.drl").write_text(
            f"M48\n; Hash={content_hash}\nT1C0.3\n%\nM30\n", encoding="utf-8"
        )
        (out_dir / "drill-NPTH.drl").write_text(
            f"M48\n; NPTH Hash={content_hash}\n%\nM30\n", encoding="utf-8"
        )

        return subprocess.CompletedProcess(
            args=["kicad-cli"], returncode=0, stdout="", stderr=""
        )


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
# G5 Gate Tests: Golden Spec Coverage
# ---------------------------------------------------------------------------


@pytest.mark.gate_g5
class TestG5GoldenSpecCoverage:
    """Gate G5 tests verifying sufficient golden specs for stability testing.

    Per ECO-M1-ALIGN-0001: "For ≥10 golden specs per family, CI proves
    output hash stability across 3 runs (G5)"
    """

    def test_minimum_f0_golden_specs_for_stability(self) -> None:
        """Verify at least 10 F0 golden specs exist for stability testing."""
        specs = _collect_f0_specs()
        assert len(specs) >= 10, (
            f"Expected ≥10 F0 specs for stability gate, found {len(specs)}"
        )

    def test_minimum_f1_golden_specs_for_stability(self) -> None:
        """Verify at least 10 F1 golden specs exist for stability testing."""
        specs = _collect_f1_specs()
        assert len(specs) >= 10, (
            f"Expected ≥10 F1 specs for stability gate, found {len(specs)}"
        )

    def test_total_golden_specs_for_stability(self, golden_specs: list[Path]) -> None:
        """Verify at least 20 total golden specs exist for stability testing."""
        assert len(golden_specs) >= 20, (
            f"Expected ≥20 total golden specs for stability gate, found {len(golden_specs)}"
        )


# ---------------------------------------------------------------------------
# G5 Gate Tests: Design Hash Stability
# ---------------------------------------------------------------------------


@pytest.mark.gate_g5
class TestG5DesignHashStability:
    """Gate G5 tests for design_hash stability across repeated runs.

    design_hash must be identical for the same spec across multiple runs
    within the same process and across separate invocations.
    """

    def test_design_hash_stable_single_process(
        self, golden_specs: list[Path]
    ) -> None:
        """design_hash must be stable within a single process."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            hashes: list[str] = []

            for _ in range(NUM_STABILITY_RUNS):
                resolved_design = resolve(spec)
                hashes.append(design_hash(resolved_design))

            # All hashes must be identical
            assert len(set(hashes)) == 1, (
                f"design_hash not stable for {spec_path.name} across {NUM_STABILITY_RUNS} runs: "
                f"got {len(set(hashes))} distinct hashes"
            )

    def test_design_hash_is_sha256_hex(self, golden_specs: list[Path]) -> None:
        """design_hash must be a 64-character lowercase hex string (SHA256)."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            resolved_design = resolve(spec)
            h = design_hash(resolved_design)

            assert len(h) == 64, f"design_hash must be 64 chars, got {len(h)}"
            assert h.islower(), "design_hash must be lowercase"
            assert all(c in "0123456789abcdef" for c in h), (
                "design_hash must be valid hex"
            )

    def test_design_hash_equals_sha256_of_canonical_json(
        self, golden_specs: list[Path]
    ) -> None:
        """design_hash must equal SHA256 of canonical JSON bytes."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            resolved_design = resolve(spec)

            canonical = resolved_design_canonical_json(resolved_design)
            expected_hash = sha256_bytes(canonical.encode("utf-8"))
            computed_hash = design_hash(resolved_design)

            assert computed_hash == expected_hash, (
                f"design_hash != sha256(canonical_json) for {spec_path.name}"
            )

    def test_design_hash_matches_golden(
        self, golden_specs: list[Path], golden_hashes: dict[str, str]
    ) -> None:
        """design_hash must match committed golden hashes."""
        for spec_path in golden_specs:
            key = spec_path.with_suffix(".json").name

            if key not in golden_hashes:
                pytest.skip(f"No golden hash for {key}")

            spec = load_spec(spec_path)
            resolved_design = resolve(spec)
            computed_hash = design_hash(resolved_design)

            assert computed_hash == golden_hashes[key], (
                f"design_hash mismatch for {spec_path.name}:\n"
                f"  computed: {computed_hash}\n"
                f"  expected: {golden_hashes[key]}"
            )


# ---------------------------------------------------------------------------
# G5 Gate Tests: Canonical JSON Stability
# ---------------------------------------------------------------------------


@pytest.mark.gate_g5
class TestG5CanonicalJsonStability:
    """Gate G5 tests for canonical JSON stability.

    Canonical JSON must be byte-identical regardless of input key ordering
    or other non-semantic variations.
    """

    def test_canonical_json_stable_across_runs(
        self, golden_specs: list[Path]
    ) -> None:
        """Canonical JSON must be stable across multiple resolutions."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            canonicals: list[str] = []

            for _ in range(NUM_STABILITY_RUNS):
                resolved_design = resolve(spec)
                canonicals.append(resolved_design_canonical_json(resolved_design))

            # All canonical JSONs must be byte-identical
            assert len(set(canonicals)) == 1, (
                f"Canonical JSON not stable for {spec_path.name}"
            )

    def test_canonical_json_key_order_invariant(
        self, golden_specs: list[Path]
    ) -> None:
        """Canonical JSON must be invariant to input key ordering."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)

            resolved = resolve(spec)
            canonical = resolved_design_canonical_json(resolved)

            # Re-serialize with different key order
            spec_data = spec.model_dump(mode="json")
            reordered: dict[str, Any] = {}
            for key in reversed(list(spec_data.keys())):
                reordered[key] = spec_data[key]

            from formula_foundry.coupongen.spec import CouponSpec
            spec_reordered = CouponSpec.model_validate(reordered)
            resolved_reordered = resolve(spec_reordered)
            canonical_reordered = resolved_design_canonical_json(resolved_reordered)

            assert canonical == canonical_reordered, (
                f"Canonical JSON depends on key order for {spec_path.name}"
            )


# ---------------------------------------------------------------------------
# G5 Gate Tests: Board File Canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.gate_g5
class TestG5BoardFileCanonicalization:
    """Gate G5 tests for KiCad board file (.kicad_pcb) canonicalization.

    Board file canonical hash must ignore tstamp and uuid fields
    that vary between KiCad sessions.
    """

    def test_canonical_hash_ignores_tstamp(self) -> None:
        """Board canonical hash must ignore tstamp fields."""
        board_a = "(kicad_pcb\n  (tstamp 12345678-1234-1234-1234-123456789abc)\n  (net 1)\n)"
        board_b = "(kicad_pcb\n  (tstamp 87654321-4321-4321-4321-cba987654321)\n  (net 1)\n)"

        assert canonical_hash_kicad_pcb_text(board_a) == canonical_hash_kicad_pcb_text(board_b)

    def test_canonical_hash_ignores_uuid(self) -> None:
        """Board canonical hash must ignore uuid fields."""
        board_a = "(kicad_pcb\n  (uuid 11111111-1111-1111-1111-111111111111)\n  (net 1)\n)"
        board_b = "(kicad_pcb\n  (uuid 22222222-2222-2222-2222-222222222222)\n  (net 1)\n)"

        assert canonical_hash_kicad_pcb_text(board_a) == canonical_hash_kicad_pcb_text(board_b)

    def test_canonical_hash_sensitive_to_content(self) -> None:
        """Board canonical hash must differ for actual content changes."""
        board_a = "(kicad_pcb\n  (tstamp 12345)\n  (net 1)\n)"
        board_b = "(kicad_pcb\n  (tstamp 12345)\n  (net 2)\n)"

        assert canonical_hash_kicad_pcb_text(board_a) != canonical_hash_kicad_pcb_text(board_b)

    def test_canonicalize_removes_tstamp(self) -> None:
        """Canonicalization must remove tstamp content."""
        board = "(kicad_pcb (tstamp abc123) (net 1))"
        canonical = canonicalize_kicad_pcb_text(board)

        assert "abc123" not in canonical
        assert "(net 1)" in canonical

    def test_canonicalize_removes_uuid(self) -> None:
        """Canonicalization must remove uuid content."""
        board = "(kicad_pcb (uuid 12345678-1234-1234-1234-123456789abc) (net 1))"
        canonical = canonicalize_kicad_pcb_text(board)

        assert "12345678-1234-1234-1234-123456789abc" not in canonical
        assert "(net 1)" in canonical


# ---------------------------------------------------------------------------
# G5 Gate Tests: Gerber Canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.gate_g5
class TestG5GerberCanonicalization:
    """Gate G5 tests for Gerber file canonicalization.

    Gerber canonical hash must ignore timestamps, dates, and other
    non-deterministic content.
    """

    def test_canonical_hash_ignores_creation_date(self) -> None:
        """Gerber canonical hash must ignore CreationDate comments."""
        gerber_a = "G04 CreationDate: 2026-01-19 10:00:00*\nX0Y0D02*\n"
        gerber_b = "G04 CreationDate: 2026-12-31 23:59:59*\nX0Y0D02*\n"

        assert canonical_hash_export_text(gerber_a) == canonical_hash_export_text(gerber_b)

    def test_canonical_hash_ignores_timestamp_comments(self) -> None:
        """Gerber canonical hash must ignore timestamp comments."""
        gerber_a = ";timestamp=2026-01-19\nX0Y0D02*\n"
        gerber_b = ";timestamp=2026-12-31\nX0Y0D02*\n"

        assert canonical_hash_export_text(gerber_a) == canonical_hash_export_text(gerber_b)

    def test_canonical_hash_normalizes_line_endings(self) -> None:
        """Gerber canonical hash must normalize CRLF to LF."""
        gerber_crlf = "G04 Test*\r\nX0Y0D02*\r\n"
        gerber_lf = "G04 Test*\nX0Y0D02*\n"

        assert canonical_hash_export_text(gerber_crlf) == canonical_hash_export_text(gerber_lf)

    def test_canonical_hash_sensitive_to_content(self) -> None:
        """Gerber canonical hash must differ for actual content changes."""
        gerber_a = "X0Y0D02*\n"
        gerber_b = "X1Y1D02*\n"

        assert canonical_hash_export_text(gerber_a) != canonical_hash_export_text(gerber_b)

    def test_canonicalize_removes_dates(self) -> None:
        """Canonicalization must remove date patterns."""
        gerber = "G04 CreationDate: 2026-01-19 10:00:00*\nX0Y0D02*\n"
        canonical = canonicalize_export_text(gerber)

        assert "2026-01-19" not in canonical
        assert "X0Y0D02*" in canonical


# ---------------------------------------------------------------------------
# G5 Gate Tests: Export Hash Stability
# ---------------------------------------------------------------------------


@pytest.mark.gate_g5
class TestG5ExportHashStability:
    """Gate G5 tests for export hash stability across repeated runs."""

    def test_same_seed_produces_same_hashes(self, tmp_path: Path) -> None:
        """Same seed must produce identical export hashes."""
        from formula_foundry.coupongen import export_fab
        from formula_foundry.coupongen.spec import KicadToolchain

        board_path = tmp_path / "coupon.kicad_pcb"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")
        toolchain = KicadToolchain(
            version="9.0.7", docker_image="kicad/kicad:9.0.7@sha256:deadbeef"
        )

        # Run export twice with same seed
        runner_a = _FakeStabilityRunner(seed="test_seed")
        runner_b = _FakeStabilityRunner(seed="test_seed")

        hashes_a = export_fab(board_path, tmp_path / "fab_a", toolchain, runner=runner_a)
        hashes_b = export_fab(board_path, tmp_path / "fab_b", toolchain, runner=runner_b)

        assert hashes_a == hashes_b, "Same seed must produce identical hashes"

    def test_export_hashes_are_64_char_hex(self, tmp_path: Path) -> None:
        """All export hashes must be 64-character SHA256 hex strings."""
        from formula_foundry.coupongen import export_fab
        from formula_foundry.coupongen.spec import KicadToolchain

        board_path = tmp_path / "coupon.kicad_pcb"
        board_path.write_text("(kicad_pcb)", encoding="utf-8")
        toolchain = KicadToolchain(
            version="9.0.7", docker_image="kicad/kicad:9.0.7@sha256:deadbeef"
        )
        runner = _FakeStabilityRunner()

        hashes = export_fab(board_path, tmp_path / "fab", toolchain, runner=runner)

        for path, digest in hashes.items():
            assert len(digest) == 64, f"Hash for {path} must be 64 chars"
            assert all(c in "0123456789abcdef" for c in digest), (
                f"Hash for {path} must be valid hex"
            )


# ---------------------------------------------------------------------------
# G5 Gate Tests: Manifest Hash Stability
# ---------------------------------------------------------------------------


@pytest.mark.gate_g5
class TestG5ManifestHashStability:
    """Gate G5 tests for manifest hash stability."""

    @pytest.mark.parametrize(
        "spec_path",
        _collect_golden_specs(),
        ids=lambda p: p.name,
    )
    def test_manifest_design_hash_stable(
        self, spec_path: Path, tmp_path: Path
    ) -> None:
        """Manifest design_hash must be stable across runs."""
        from formula_foundry.coupongen import build_coupon

        spec = load_spec(spec_path)
        design_hashes: list[str] = []

        for run_idx in range(NUM_STABILITY_RUNS):
            out_dir = tmp_path / f"run_{run_idx}"
            out_dir.mkdir(parents=True)
            runner = _FakeStabilityRunner(seed=spec_path.name)

            result = build_coupon(
                spec,
                out_root=out_dir,
                mode="docker",
                runner=runner,
                kicad_cli_version="9.0.7",
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            design_hashes.append(manifest["design_hash"])

        # All design_hashes must be identical
        assert len(set(design_hashes)) == 1, (
            f"design_hash not stable for {spec_path.name} across {NUM_STABILITY_RUNS} runs"
        )

    def test_manifest_export_hashes_stable(self, tmp_path: Path) -> None:
        """Manifest export hashes must be stable across runs."""
        specs = _collect_golden_specs()
        if not specs:
            pytest.skip("No golden specs available")

        from formula_foundry.coupongen import build_coupon

        spec_path = specs[0]
        spec = load_spec(spec_path)
        export_hashes_per_run: list[dict[str, str]] = []

        for run_idx in range(NUM_STABILITY_RUNS):
            out_dir = tmp_path / f"run_{run_idx}"
            out_dir.mkdir(parents=True)
            runner = _FakeStabilityRunner(seed="stability_test")

            result = build_coupon(
                spec,
                out_root=out_dir,
                mode="docker",
                runner=runner,
                kicad_cli_version="9.0.7",
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            exports = manifest.get("exports", [])
            export_hashes = {e["path"]: e["hash"] for e in exports}
            export_hashes_per_run.append(export_hashes)

        # All export hash sets must be identical
        first_hashes = export_hashes_per_run[0]
        for run_idx, run_hashes in enumerate(export_hashes_per_run[1:], start=2):
            assert first_hashes == run_hashes, (
                f"Export hashes differ between run 1 and run {run_idx}"
            )


# ---------------------------------------------------------------------------
# G5 Gate Tests: Full Determinism Verification
# ---------------------------------------------------------------------------


@pytest.mark.gate_g5
class TestG5FullDeterminism:
    """Gate G5 comprehensive determinism tests combining all stability checks."""

    def test_full_build_determinism(self, tmp_path: Path) -> None:
        """Full build must be deterministic across 3 runs.

        This test verifies:
        1. design_hash stable
        2. All export hashes stable
        3. Manifest structure consistent
        """
        specs = _collect_golden_specs()
        if not specs:
            pytest.skip("No golden specs available")

        from formula_foundry.coupongen import build_coupon

        # Test with first F0 and first F1 spec
        f0_specs = _collect_f0_specs()
        f1_specs = _collect_f1_specs()
        test_specs = [f0_specs[0], f1_specs[0]] if f0_specs and f1_specs else specs[:2]

        for spec_path in test_specs:
            spec = load_spec(spec_path)
            manifests: list[dict[str, Any]] = []

            for run_idx in range(NUM_STABILITY_RUNS):
                out_dir = tmp_path / f"{spec_path.stem}_run_{run_idx}"
                out_dir.mkdir(parents=True)
                runner = _FakeStabilityRunner(seed=spec_path.name)

                result = build_coupon(
                    spec,
                    out_root=out_dir,
                    mode="docker",
                    runner=runner,
                    kicad_cli_version="9.0.7",
                )

                manifests.append(
                    json.loads(result.manifest_path.read_text(encoding="utf-8"))
                )

            # Verify design_hash stability
            design_hashes = [m["design_hash"] for m in manifests]
            assert len(set(design_hashes)) == 1, (
                f"design_hash not stable for {spec_path.name}"
            )

            # Verify export hash stability
            export_hash_sets = [
                frozenset((e["path"], e["hash"]) for e in m.get("exports", []))
                for m in manifests
            ]
            assert len(set(export_hash_sets)) == 1, (
                f"Export hashes not stable for {spec_path.name}"
            )

    def test_hash_format_compliance(self, golden_specs: list[Path]) -> None:
        """All hashes must comply with SHA256 format requirements."""
        for spec_path in golden_specs:
            spec = load_spec(spec_path)
            resolved_design = resolve(spec)
            h = design_hash(resolved_design)

            # SHA256 produces 256 bits = 64 hex chars
            assert len(h) == 64
            # Must be lowercase for consistency
            assert h.islower()
            # Must be valid hex
            int(h, 16)  # Should not raise
