# Milestone Design Document

**Milestone:** M1 — M1.1 Geometry Realization + Spec Coverage Hardening

## Scope
- Eliminate “silent stubs”: any accepted `CouponSpec` field must be consumed and must affect resolved design, geometry, constraints, or outputs (unless explicitly reserved and rejected).
- Make generated coupons physically-real for equation discovery: real connector footprints, real launch region, real CPWG (signal + GND copper + enforced gap), real via fences, and real rounded outlines.
- Preserve and strengthen determinism + provenance: stable hashes, stable manifests, stable exports across repeated runs (toolchain pinned).
- Make REPAIR mode fully reproducible (audited, serialized, rebuildable).
- Make spec coverage a CI gate so future schema drift cannot silently corrupt datasets.

## Normative Requirements (must)

- [REQ-M1-001] The generator MUST track and emit spec consumption (consumed paths, expected paths, unused provided paths) and MUST fail in strict mode if any provided field is unused or any expected field is unconsumed.
- [REQ-M1-002] The spec validator MUST enforce family-specific correctness (e.g., F0 cannot include F1-only blocks) and MUST reject unknown/extra fields (no silent accept) under strict mode.
- [REQ-M1-003] Connector footprints MUST be sourced from vendored in-repo `.kicad_mod` files and embedded into the generated `.kicad_pcb`; placeholder “single pad connector” generation is disallowed for M1 compliance.
- [REQ-M1-004] Footprint-to-net and anchor-pad mapping MUST be deterministic and explicit (via `pad_map` or documented conventions) so the launch connects to the true signal pad and GND pads/nets are correctly assigned.
- [REQ-M1-005] CPWG generation MUST produce net-aware copper geometry: a signal conductor on the declared layer plus a GND reference conductor on that layer with an enforced `gap_nm` (no “CPWG in schema only”).
- [REQ-M1-006] If CPWG uses zones, DRC MUST be run with zone refill enabled and exports MUST be run with zone checks enabled (KiCad CLI flags/policy pinned in code and recorded in manifest).
- [REQ-M1-007] When `ground_via_fence.enabled=true`, the generator MUST place GND via fences deterministically with correct pitch/offset policies, symmetry, and collision/edge-clearance enforcement suggests by the spec and fab profile.
- [REQ-M1-008] A launch feature MUST exist for F0/F1 that deterministically connects connector pads to CPWG (taper or stepped transition) with correct nets, optional stitching, and manufacturable DFM constraints.
- [REQ-M1-009] If `corner_radius_nm > 0` is provided, the board outline MUST be generated as a rounded-rectangle on `Edge.Cuts` using deterministic integer-nm arcs/segments and validated for feasibility.
- [REQ-M1-010] The generator MUST place deterministic board annotations on silkscreen, including `coupon_id` and a short hash marker (e.g., design/manifest hash prefix), and these annotations MUST appear in exported silkscreen Gerbers.
- [REQ-M1-011] REPAIR mode MUST emit a serialized `repair_map` plus a deterministic `repaired_spec` (or deterministic patch set) such that rebuilding from the repaired spec reproduces the same `design_hash` and artifacts.
- [REQ-M1-012] CLI and Python APIs MUST call a single canonical build pipeline (validate→resolve→generate→drc→export→manifest); divergent “parallel pipelines” are disallowed for production readiness.
- [REQ-M1-013] The manifest MUST include a spec-consumption summary, footprint provenance (paths + hashes of source footprint content), and an explicit zone policy record (refill/check behavior and toolchain versioning).
- [REQ-M1-014] Derived features and dimensionless groups MUST be expanded to include CPWG/via/fence/launch-relevant groups and MUST be emitted deterministically in `manifest.json`.
- [REQ-M1-015] The test suite MUST include mutation/coverage tests proving that changing key fields (e.g., `gap_nm`, fence pitch, connector footprint id, corner radius) changes the appropriate geometry/artifact hashes and never results in an identical board for distinct specs.
- [REQ-M1-016] For golden specs, KiCad DRC MUST pass in CI on the pinned toolchain with the zone policy applied, demonstrating that the realized copper/nets are manufacturable and DRC-clean.
- [REQ-M1-017] For golden specs, gerber/drill export completeness and canonical hash stability MUST hold across repeated builds (including silkscreen content), subject only to documented canonicalization rules.
- [REQ-M1-018] The CLI MUST provide `lint-spec-coverage` (non-zero on coverage failures) and `explain` (human-readable resolved + tightest-constraint summary) as additive commands.

## Notes
- This document is intentionally a “quality hardening” change order: it does not weaken determinism guarantees; it eliminates schema/geometry drift and improves physical realism for equation discovery.
- Preferred CPWG implementation for net-awareness is a GND zone on the CPWG layer plus deterministic keepout/rule areas that enforce `gap_nm` around the signal path; explicit rails are acceptable if net-aware and equally deterministic.
- The zone policy must be explicit: refilling/checking zones in headless flows is mandatory for correctness of DRC and exports.

## Definition of Done

- Spec consumption tracking exists, is emitted in artifacts, and strict mode fails on unused or unconsumed fields.
- Connector footprints are embedded from vendored `.kicad_mod` sources; no placeholder connector geometry remains in the M1 build path.
- CPWG is realized in copper with enforced gaps and correct nets; via fences and launch features exist and are deterministic.
- Board outline rounding is implemented when requested; board annotations include `coupon_id` and hash marker and appear in silkscreen exports.
- REPAIR mode emits `repair_map` + `repaired_spec` and rebuild-from-repaired-spec reproduces identical `design_hash` and artifacts.
- A single canonical pipeline is used by both CLI and Python API.
- Manifest includes spec coverage summary, footprint provenance hashes, and zone policy; derived features/groups are expanded and deterministic.
- CI passes for golden specs: determinism gates, DRC clean, export completeness, and export hash stability across repeated runs.

## Test Matrix

| Requirement | Pytest(s) |
|---|---|
| REQ-M1-001 | tests/test_schema.py::test_schema |
| REQ-M1-002 | tests/test_schema.py::test_schema |
| REQ-M1-003 | tests/test_kicad_drc.py::test_kicad_drc |
| REQ-M1-004 | tests/test_kicad_drc.py::test_kicad_drc |
| REQ-M1-005 | tests/test_kicad_drc.py::test_kicad_drc |
| REQ-M1-006 | tests/test_kicad_drc.py::test_kicad_drc |
| REQ-M1-007 | tests/test_constraints.py::test_constraints |
| REQ-M1-008 | tests/test_kicad_drc.py::test_kicad_drc |
| REQ-M1-009 | tests/test_constraints.py::test_constraints |
| REQ-M1-010 | tests/test_export_hashes.py::test_export_hashes |
| REQ-M1-011 | tests/test_constraints.py::test_constraints |
| REQ-M1-012 | tests/test_resolve_determinism.py::test_resolve_determinism |
| REQ-M1-013 | tests/test_export_hashes.py::test_export_hashes |
| REQ-M1-014 | tests/test_resolve_determinism.py::test_resolve_determinism |
| REQ-M1-015 | tests/test_resolve_determinism.py::test_resolve_determinism |
| REQ-M1-016 | tests/test_kicad_drc.py::test_kicad_drc |
| REQ-M1-017 | tests/test_export_hashes.py::test_export_hashes |
| REQ-M1-018 | tests/test_schema.py::test_schema |
