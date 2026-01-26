
### `oracle_case.json` (canonical)

`oracle_case.json` is the authoritative interface. It compiles down into gerber2ems-compatible inputs while preserving stricter invariants (explicit defaults, strict schema, recorded provenance).

Key fields:
- `format_version`, `case_id`
- `frequency { start_hz, stop_hz, npoints, spacing }`
- `solver_policy { boundary, pml_cells, end_criteria, max_steps, threads, time_step_factor }`
- `grid_policy { lambda_divisor_max_cell, thirds_rule, max_ratio, roi_margins_um, pml_clearance_policy }`
- `ports { port_definitions[], reference_impedance_ohm, backend }`
- `structures { type, expected_behavior_profile, port_map }`
- `postprocess { export_touchstone, ri_format, renormalize_to_ohms, mixed_mode }`
- `verification { enabled_checks, thresholds, strict }`
- `provenance { git_commit, toolchain_digest, openems_version, gerber2ems_version }`

## Toolchain & Reproducibility

### Digest-pinned toolchain

M2.1 requires an OCI container image pinned by digest that contains openEMS + gerber2ems. Every artifact bundle records:
- toolchain image + digest
- openEMS version/commit
- gerber2ems version/commit
- any relevant build flags

### Provenance in every run

`meta.json` must include:
- case hash
- repo git commit
- toolchain digest + versions
- normalized case config (explicit defaults)
- frequency grid used
- mesh summary hash
- per-port mapping and reference plane
- runtime and termination statistics

## Golden Path Pipeline

### Stage 1 — Validation (fail fast)

Validation covers:
- required files + naming conventions
- stackup/Gerber consistency
- port marker parsing and cardinal rotations
- port geometry overlap and anti-short checks
- port length vs local mesh cell size
- resource preflight (cells + memory budget)

### Stage 2 — Geometry Builder (gerber2ems)

Gerber2ems is the canonical geometry frontend. The geometry builder produces:
- deterministic `geometry.xml`
- recorded rasterization/contour extraction settings
- intermediate logs for auditability

### Stage 3 — Mesh Planner

Mesh planning enforces:
- wavelength-based Δmax
- smooth grading via max ratio
- thirds-rule refinement near metal edges (default on)
- PML clearances and padding

It writes `mesh_summary.json` with:
- mesh lines per axis
- min/max Δ
- ratio statistics
- total cell count
- conservative memory estimate
- mesh hash

### Stage 4 — Solver Runner (openEMS)

Solver execution rules:
- one excitation per port for N-port assembly
- store raw per-port waves/traces sufficient to recompute S offline
- record termination cause (end criteria vs max steps)
- enforce timestep sufficiency policy relative to excitation length

Stub execution exists only behind an explicit flag and produces NON-GOLD output.

### Stage 5 — Postprocessing

Postprocessing produces:
- Touchstone `.s2p`/`.s4p` in RI format with frequency in Hz
- optional renormalized exports when configured
- optional mixed-mode outputs for differential cases with strict port pairing

GPU backend:
- CuPy for FFT and batch transforms when available
- explicit, recorded fallback to NumPy with reason (never silent)

### Stage 6 — Verification & Calibration Gates

Verification includes:
- passivity checks (N-port)
- reciprocity checks where expected
- causality sanity checks (group delay)
- dead port detection and sanity checks on time-domain behavior for calibration structures

Calibration suite:
- CAL-0…CAL-6 minimum
- metric-based regression comparisons (not bitwise)
- merge-blocking for oracle/mesh/port/postprocess changes

Mesh invariance:
- baseline mesh vs refined mesh
- ΔS thresholds enforced across band

### Stage 7 — Artifact Writer + Caching

Artifact bundle completeness is enforced; missing required files is a hard failure.

Caching:
- keyed by input file hashes + normalized config + toolchain digest + mesh hash
- verifies output hashes before reuse

## Testing & CI

### Test tiers

Tier 1 (fast PR):
- schema/validation, hashing, mesh determinism, touchstone I/O, postprocess unit tests, verification logic unit tests

Tier 2 (PR required minimal real solver):
- run a small CAL-2 Thru case in pinned container
- validate artifacts complete + key verification metrics

Tier 3 (nightly/scheduled):
- `oracle cal run --all`
- mesh invariance checks
- performance trending

## Performance & Storage Policies

- ROI required by default (no full-board runs unless explicitly allowed)
- conservative memory preflight hard cap
- field dumps disabled by default and gated
- openEMS threading controlled; postprocess batched on GPU where possible

## Migration & Compatibility

Existing internal geometry/mesh paths may remain for dev/test, but only the Gerber-driven pipeline is treated as “gold” unless a backend can prove identical artifact completeness, verification behavior, and calibration performance.

## Risks & Mitigations

- Port correctness: mitigate with geometry overlap/anti-short/orientation checks + calibration behavior tests + negative tests.
- Toolchain drift: mitigate with digest pinning + enforcement tests + provenance.
- Mesh sensitivity: mitigate with thirds rule + mesh invariance regression.
- Boundary reflections: mitigate with PML clearances + recorded margins.
- Performance regressions: mitigate with CI benchmark envelopes + preflight cell/memory budgeting.
- Storage blow-ups: mitigate with default-off field exports + strict gating.

## Implementation Plan (work packages)

WP0 — Make gold path non-stub and explicit  
WP1 — Toolchain digest pinning + enforcement  
WP2 — Deterministic solver input generation (no manual sim.xml)  
WP3 — Gerber2ems integration (canonical fab ingestion)  
WP4 — Port verification hardening + debug overlays  
WP5 — Calibration suite + regression runner + CI gates  
WP6 — GPU postprocessing backend + recorded fallbacks  
WP7 — Artifact completeness enforcement + caching upgrades

## Definition of Done

- The repository contains a root-level `DESIGN_DOCUMENT.md` matching this M2 spec and the workflow lints it successfully.
- `oracle run <case_dir>` executes the real, end-to-end golden pipeline by default and produces a complete artifact bundle.
- Stub execution is only possible via explicit opt-in and produces outputs that are clearly labeled NON-GOLD.
- Toolchain is digest-pinned and recorded in every run’s metadata; strict mode fails if it is unpinned or missing.
- Ports are validated at geometry level (overlap, anti-short, cardinal rotation, length vs mesh) and failures halt strict runs.
- N-port simulations run N excitations and store raw per-port waves sufficient to recompute S deterministically.
- Touchstone export is canonical (Hz, RI), and metadata contains per-port Zref and complete provenance.
- Mixed-mode outputs are produced for 4-port differential cases with strict, recorded pairing conventions.
- GPU-backed postprocessing is available; any CPU fallback is explicit, logged, and recorded with reason.
- Calibration suite (CAL-0…CAL-6) exists, runs end-to-end, and regression thresholds are enforced for oracle changes.
- Mesh invariance gate exists and is exercised in CI (nightly or required tier).
- CI runs at least one real openEMS simulation in the pinned toolchain and validates artifact completeness + key verification metrics.

## Test Matrix

| Requirement | Pytest(s) (pytest node ids) |
|---|---|
| REQ-M2-001 | tests/test_oracle_pipeline.py::test_oracle_run_executes_full_pipeline_real_default |
| REQ-M2-002 | tests/test_oracle_pipeline.py::test_stub_mode_requires_explicit_allow_and_labels_non_gold |
| REQ-M2-003 | tests/test_case_validation.py::test_validator_enforces_canonical_case_layout_and_required_fab_files |
| REQ-M2-004 | tests/test_geometry_builder_gerber2ems.py::test_gerber2ems_geometry_is_deterministic_given_fixed_settings |
| REQ-M2-005 | tests/test_toolchain_pinning.py::test_toolchain_is_digest_pinned_and_recorded_in_meta |
| REQ-M2-006 | tests/test_artifact_bundle.py::test_artifact_bundle_contains_required_files |
| REQ-M2-007 | tests/test_case_fingerprint.py::test_fingerprint_includes_all_input_bytes_normalized_config_and_toolchain_digest |
| REQ-M2-008 | tests/test_mesh_policy.py::test_mesh_policy_enforces_lambda_rule_grading_thirds_rule_and_pml_clearance |
| REQ-M2-009 | tests/test_resource_preflight.py::test_preflight_estimates_cells_and_memory_and_fails_over_budget |
| REQ-M2-010 | tests/test_field_export_policy.py::test_field_exports_disabled_by_default_and_require_explicit_flag |
| REQ-M2-011 | tests/test_port_verification.py::test_port_rotation_must_be_cardinal_by_default |
| REQ-M2-012 | tests/test_port_verification.py::test_port_geometry_overlap_no_short_and_min_length_vs_mesh |
| REQ-M2-013 | tests/test_port_map_metadata.py::test_port_map_records_mapping_orientation_reference_plane_and_backend |
| REQ-M2-014 | tests/test_solver_runner.py::test_solver_runs_one_excitation_per_port_and_persists_raw_waves |
| REQ-M2-015 | tests/test_termination_policy.py::test_requires_end_criteria_and_max_steps_and_records_termination_cause |
| REQ-M2-016 | tests/test_touchstone_export.py::test_touchstone_exports_hz_ri_and_records_zref |
| REQ-M2-017 | tests/test_renormalization.py::test_exports_native_and_renormalized_sparams_when_configured |
| REQ-M2-018 | tests/test_mixed_mode.py::test_mixed_mode_enforces_strict_pairing_and_records_in_meta |
| REQ-M2-019 | tests/test_gpu_backend.py::test_gpu_backend_defaults_to_cupy_and_records_fallback_reason |
| REQ-M2-020 | tests/test_verification_suite.py::test_verification_enforces_passivity_reciprocity_and_causality_in_strict_mode |
| REQ-M2-021 | tests/test_calibration_suite.py::test_calibration_library_contains_cal_0_through_cal_6_and_runs_end_to_end |
| REQ-M2-022 | tests/test_calibration_regression.py::test_calibration_regression_is_metric_based_and_merge_blocking |
| REQ-M2-023 | tests/test_mesh_invariance.py::test_mesh_invariance_gate_runs_baseline_vs_refined_and_enforces_thresholds |
| REQ-M2-024 | tests/test_ci_real_openems.py::test_ci_runs_minimal_real_openems_case_in_pinned_toolchain |
