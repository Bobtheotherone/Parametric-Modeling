# Milestone Design Document

**Milestone:** M2 — Formula Foundry missing-module completion (No-shims)
**Document Version:** M2.3

## Scope

- Deliver a complete, state-of-the-art implementation for all missing `formula_foundry.*` and `tools.ci.real_openems` modules required by pytest collection.
- Enforce a strict “no scaffolding / no temporary shim” policy: no stubbed solver runs unless explicitly allowed and auditable.
- Provide deterministic canonicalization + hashing for cases, meshes, and metadata.
- Implement real domain logic for oracle schema validation, mesh planning, postprocess (renormalization, mixed-mode, Touchstone), verification metrics, calibration suite + regression, artifact bundling, meta writing, and toolchain lock handling.
- Ensure thin, explicit package exports so `from formula_foundry.<pkg> import <module>` imports work exactly as tests expect.

## Normative Requirements (must)

- [REQ-M2P3-001] The repository root MUST contain a plain UTF-8 Markdown file named `DESIGN_DOCUMENT.md` whose milestone line includes the exact token `**Milestone:** M2`.
- [REQ-M2P3-002] All missing modules identified by failing pytest collection MUST exist at the exact paths implied by their dotted import names and MUST be importable without internet access.
- [REQ-M2P3-003] Every module imported by tests MUST export the required symbol names exactly (case-sensitive) as imported in tests.
- [REQ-M2P3-004] `formula_foundry.oracle` MUST implement strict schema validation that raises `OracleCaseMissingKeyError` for missing keys and `OracleCaseExtraKeyError` for unknown keys, and MUST provide `normalize_case`, `validate_oracle_case`, and `to_canonical_dict`.
- [REQ-M2P3-005] Canonicalization and hashing MUST be deterministic across runs and platforms using stable JSON serialization (sorted keys, stable float formatting rules) and SHA-256 over canonical JSON bytes.
- [REQ-M2P3-006] `compute_case_fingerprint`, `meta.writer.compute_case_hash`, and any mesh hash computations MUST be content-based and MUST NOT incorporate timestamps or other non-deterministic data.
- [REQ-M2P3-007] `postprocess.renormalize.renormalize_sparameters` MUST implement correct multiport renormalization via S↔Z conversion with numerically robust handling of ill-conditioned matrices (pinv or regularization) and MUST produce finite results or raise a domain validation error.
- [REQ-M2P3-008] `postprocess.mixed_mode` MUST implement a correct mixed-mode transform for 4-port differential cases using an orthonormal transform matrix, MUST validate pairing correctness, and MUST provide file outputs via `write_mixed_mode_outputs` using atomic write semantics.
- [REQ-M2P3-009] `postprocess.touchstone` MUST implement deterministic Touchstone export for S-parameter datasets (at minimum S2P and S4P) with stable formatting and canonical header content, and MUST write outputs atomically.
- [REQ-M2P3-010] `verification.metrics` MUST implement passivity (singular value), reciprocity (Sij−Sji), and causality (time-domain energy in negative time via FFT/IFFT with resampling as needed), and MUST provide `run_verification_suite` plus required synthetic 2-port generators used by tests.
- [REQ-M2P3-011] `mesh.planner` MUST implement a real mesh planning policy driven by maximum frequency, material εr, points-per-wavelength, and feature size constraints, and MUST provide `compute_lambda_max_cell_nm`.
- [REQ-M2P3-012] `mesh.invariance` MUST compute meaningful delta metrics (magnitude dB and phase degrees, max and RMS) between baseline and candidate S-parameters, MUST report violations, and MUST implement `run_invariance_gate`.
- [REQ-M2P3-013] `preflight.resource` MUST parse resource budget lines deterministically and MUST raise `ResourceBudgetExceeded` with actionable fields when predicted resource use exceeds limits.
- [REQ-M2P3-014] `solver.field_export.enforce_field_export_policy` MUST block field export unless an explicit environment flag is set and MUST raise `FieldExportNotAllowedError` when policy is violated.
- [REQ-M2P3-015] `solver.termination` MUST provide a `TerminationPolicy`, `TerminationCause`, and MUST write a canonical termination record via `record_termination` using atomic writes.
- [REQ-M2P3-016] `artifacts.bundle.ArtifactBundle` MUST validate that all `REQUIRED_FILES` exist for a run’s output bundle and MUST raise `ArtifactBundleError` for missing/invalid bundles.
- [REQ-M2P3-017] `calibration` MUST provide a calibration library, suite runner, and regression comparison that produce structured results and enforce thresholds via `check_regression_gate` (no silent pass on regressions).
- [REQ-M2P3-018] `tools.ci.real_openems` MUST implement toolchain lock loading/validation, digest enforcement, minimal-case generation, required artifact validation, and metrics summary construction using deterministic parsing and output ordering.

## Architecture Overview

### Target module tree (created under `src/` and `tools/`)

The following packages/modules MUST exist (directories include `__init__.py`):

- `src/formula_foundry/artifacts/{__init__.py,bundle.py}`
- `src/formula_foundry/calibration/{__init__.py,library.py,runner.py,regression.py}`
- `src/formula_foundry/oracle/{__init__.py,schema.py,fingerprint.py,run.py}`
- `src/formula_foundry/geometry/{__init__.py,gerber2ems.py,xml.py,primitives.py}`
- `src/formula_foundry/mesh/{__init__.py,planner.py,summary.py,invariance.py,grid.py}`
- `src/formula_foundry/solver/{__init__.py,field_export.py,termination.py,runner.py,backend.py,openems_backend.py}`
- `src/formula_foundry/postprocess/{__init__.py,sparams.py,renormalize.py,mixed_mode.py,touchstone.py,gpu_backend.py}`
- `src/formula_foundry/meta/{__init__.py,writer.py,toolchain.py}`
- `src/formula_foundry/toolchain/{__init__.py,pinning.py,lock.py}`
- `src/formula_foundry/ports/{__init__.py,metadata.py,verification.py}`
- `src/formula_foundry/preflight/{__init__.py,resource.py}`
- `src/formula_foundry/verification/{__init__.py,metrics.py}`
- `tools/ci/{__init__.py,real_openems.py}`

### Required public exports (as imported by tests)

All required exports listed by tests MUST be present exactly as imported (names and casing). Package-level import behavior MUST work:

- `from formula_foundry.oracle import run`
- `from formula_foundry.postprocess import gpu_backend`
- `from formula_foundry.meta import writer`
- `from formula_foundry.meta import toolchain as toolchain_meta`
- `from formula_foundry.preflight import resource`
- `from formula_foundry.toolchain import pinning`

### Deterministic canonicalization and hashing

- Canonical dict rules: recursive key sorting; dataclasses→dict; enums→value; paths→posix strings; stable float formatting; numpy arrays summarized deterministically or bytes-hashed when appropriate.
- Hashing: SHA-256 over canonical JSON bytes produced with `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.

### “No shims” enforcement

- Solver execution MUST NOT silently fall back to dummy outputs.
- If a real backend is unavailable, the pipeline MUST fail loudly and explainably unless explicitly permitted by a named allowlist mechanism (e.g., an environment flag), and must raise `StubRunNotAllowedError` in the oracle pipeline layer.

### Implementation order (dependency-minimizing)

1. `common/*` canonicalization, hashing, validation, atomic JSON writing
2. `oracle/schema.py` + package exports + normalization helpers
3. `postprocess/sparams.py` (S-parameter representation)
4. `mesh/*` planner/grid/summary
5. `verification/metrics.py`
6. `postprocess/*` renormalize/mixed_mode/touchstone/gpu_backend
7. `ports/*`
8. `preflight/resource.py`
9. `solver/*` policies, termination, backend abstraction, runner
10. `meta/*`
11. `artifacts/bundle.py`
12. `calibration/*`
13. `oracle/fingerprint.py` + `oracle/run.py`
14. `geometry/*`
15. `tools/ci/real_openems.py`

## Definition of Done

- `DESIGN_DOCUMENT.md` exists at repo root and contains the required milestone line token.
- All missing module file paths exist with correct packages (`__init__.py`) and pytest collection completes without `ModuleNotFoundError`.
- All required exported symbols are present and importable exactly as tests import them.
- Oracle schema validation correctly rejects missing/extra keys with the required error types and produces canonical dicts deterministically.
- Case and mesh hashing are deterministic across runs and do not incorporate timestamps.
- Renormalization and mixed-mode conversion produce physically correct results or raise actionable domain errors on invalid inputs.
- Touchstone export formatting is deterministic and standards-aligned for supported port counts.
- Verification metrics correctly detect passivity/reciprocity/causality failures and produce structured reports.
- Mesh planning yields realistic nonuniform grids under growth and feature constraints; invariance gate reports meaningful deltas.
- Resource preflight parses budgets deterministically and enforces exceedance with structured exceptions.
- Field export policy and termination recording operate correctly and write atomically.
- Artifacts bundles validate required files and errors are actionable.
- Calibration suite runs and regression gate fails on threshold violations; no silent “green by any means.”
- CI toolchain lock utilities validate, enforce digest refs, generate minimal cases, validate artifacts, and summarize metrics deterministically.

## Test Matrix (REDACTED)