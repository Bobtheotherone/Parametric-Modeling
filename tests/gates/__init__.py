# SPDX-License-Identifier: MIT
"""Gate tests for M1 compliance verification.

This package contains tests organized by determinism gates (G1-G5):

- gate_g1: Resolved design determinism
  - Same spec YAML yields identical ResolvedDesign across runs
  - design_hash stable for same inputs

- gate_g2: Constraint proof completeness and reject/repair behavior
  - Tiered constraint engine produces complete proof JSON
  - REJECT mode fails on violations, REPAIR mode fixes deterministically

- gate_g3: DRC clean
  - KiCad DRC passes with zero violations
  - Exit code 0 from `kicad-cli pcb drc --exit-code-violations`

- gate_g4: Export completeness
  - All required Gerber layers present (F.Cu, B.Cu, etc.)
  - Drill files (PTH, NPTH) present
  - Manifest references all exported artifacts

- gate_g5: Stable canonical hashes across repeated runs
  - Board file canonical hash stable
  - Gerber canonical hashes stable
  - Manifest hashes stable

Mark tests with appropriate @pytest.mark.gate_gN decorators.
"""
