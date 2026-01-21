# SPDX-License-Identifier: MIT
"""Gate tests for M1 compliance verification.

This package contains tests organized by determinism gates (G1-G5) per
ECO-M1-ALIGN-0001. Each gate verifies a specific aspect of M1 compliance.

Test Modules:
- test_g1_determinism.py: Resolver determinism
- test_g2_constraints.py: Constraint proof schema and REJECT/REPAIR
- test_g3_drc.py: KiCad DRC clean verification
- test_g4_export_completeness.py: Export layer set and drill completeness
- test_g5_hash_stability.py: Output hash stability across runs

Gates:

- gate_g1: Resolved design determinism
  - Same spec YAML yields identical ResolvedDesign across runs
  - design_hash stable for same inputs
  - Canonical JSON is key-order invariant
  - Golden hashes match committed values

- gate_g2: Constraint proof completeness and reject/repair behavior
  - Tiered constraint engine produces complete proof JSON
  - REJECT mode fails on violations with deterministic failure IDs
  - REPAIR mode fixes deterministically with bounded repair distance
  - 10k seeded random u vectors verify determinism

- gate_g3: DRC clean
  - KiCad DRC passes with zero violations
  - Exit code 0 from `kicad-cli pcb drc --exit-code-violations`
  - DRC JSON report structure validation
  - All golden specs satisfy DRC requirements

- gate_g4: Export completeness
  - All required Gerber layers present (F.Cu, In1.Cu, In2.Cu, B.Cu for 4-layer)
  - Mask layers (F.Mask, B.Mask) present
  - Edge.Cuts present
  - Drill files (PTH, NPTH) present
  - Manifest references all exported artifacts
  - Layer set validation per copper count and family

- gate_g5: Stable canonical hashes across repeated runs
  - design_hash stable across 3 consecutive runs
  - Board file canonical hash stable (ignoring tstamp/uuid)
  - Gerber canonical hashes stable (ignoring timestamps/dates)
  - Manifest export hashes stable
  - All golden specs (>=10 F0, >=10 F1) verified

Coverage Requirements:
- Per M1 design doc: ≥10 golden specs per family (F0, F1)
- CI must prove all gates pass for all golden specs
- Real KiCad DRC via Docker tested in tests/integration/

Mark tests with appropriate @pytest.mark.gate_gN decorators.
"""
