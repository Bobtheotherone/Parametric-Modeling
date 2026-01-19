# Milestone Design Document

**Milestone:** M1 — Parametric coupon generator: Geometry DSL -> KiCad/Gerbers with strict constraints

**Repo capability unlocked:** Deterministically generate manufacturable, DRC-clean high-speed interconnect coupons (via/launch discontinuities) from a parametric geometry DSL, producing KiCad source + fabrication Gerbers/drills + a cryptographically hashed manifest suitable for simulation + equation discovery.

---

## 0. Executive intent

This milestone exists to make the physics dataset (and therefore downstream symbolic discovery) trustworthy and repeatable. If M1 is implemented correctly, then:

1. The AI can propose a design vector x (within constraints).
2. The repo compiles x into a deterministic coupon geometry (no hand edits).
3. The output is:

   - a KiCad project as the canonical CAD artifact,
   - a complete fabrication pack (Gerbers + drill),
   - a machine-readable manifest that binds geometry -> artifacts -> constraints -> toolchain versioning,
   - and a strict pass/fail validation (DRC + our own constraint proof objects).

If M1 is implemented incorrectly, you will generate noisy datasets where changes in S-parameters are caused by CAD/DRC/plotting nondeterminism or manufacturability violations, destroying the odds of discovering a real formula.

---

## 1. Non-negotiable principles (engineering doctrine)

### P1. Determinism is a first-class requirement

- Same CouponSpec + same toolchain version => same resolved geometry and same canonical hashes for artifacts.
- We do not require literal byte-identical `.kicad_pcb` unless feasible, because KiCad embeds UUIDs (`tstamp`) in many objects; but we do require canonical, stable hashes computed from normalized content (see Section 9).

KiCad board files use S-expression and include per-item timestamp/UUID fields such as `(tstamp UUID)` for tracks/vias (see KiCad Developer Documentation).

### P2. Strict constraints: designs must be valid by construction

- The generator must never hope KiCad fixes it.
- Inputs that violate hard constraints must either:
  - REJECT (strict mode), or
  - REPAIR by projecting onto feasible space with a fully audited repair map (research mode).
- Final authority: KiCad DRC must pass in CI using the CLI DRC command and exit-code gate. The CLI supports producing a DRC report and returning a non-zero exit code when violations exist.

### P3. Toolchain stability: pin KiCad version + containerize

- KiCad provides a CLI (`kicad-cli`) for automated actions such as plotting Gerbers and running DRC.
- Pin to a known KiCad 9 stable patch release (target: 9.0.7).
- Use the official KiCad Docker images intended for CLI usage in CI (GUI in Docker is explicitly not supported).

### P4. Future-proofing: SWIG pcbnew bindings are deprecated

KiCad's SWIG-based Python bindings are deprecated as of KiCad 9.0 and planned for removal in KiCad 10.0 (planned Feb 2026 per dev docs). They will be replaced by the IPC API.

The newer IPC API is designed to be stable, but:
- it currently requires a running KiCad GUI instance (no headless mode),
- and it does not provide plotting/export (CLI is recommended for plotting).

So M1 must be designed with a backend abstraction that works headlessly today and is not trapped by SWIG removal.

### P5. Hardware utilization: GPU-first where it actually matters

M1's heavy lifting is CAD + geometry + constraints (mostly CPU). The GPU still matters in one place: batch feasibility filtering for large candidate sets (active learning / adversarial design search). M1 therefore includes a GPU-vectorized constraint prefilter that can reject/repair millions of candidates cheaply before invoking KiCad or any expensive geometry operations.

---

## 2. Scope boundaries

### In scope

- A domain-specific Geometry DSL for the Phase-1 coupon family:
  - end-launch connector <-> CPWG/microstrip launch <-> via transition / discontinuity region <-> return vias / fencing <-> plane cutouts/antipads,
  - single-ended and (optionally) differential variants.
- Deterministic compilation:
  - DSL -> resolved parameter set -> KiCad project (board-centric) -> Gerbers/drill.
- Constraint system:
  - manufacturability + symmetry + spacing + keepouts + stackup constraints.
- Verification:
  - programmatic constraint proof + KiCad DRC/export sanity gates in CI.
- Artifact manifest:
  - strong linkage between geometry, constraints, tool versions, and outputs.

### Out of scope (explicitly deferred)

- Full EM simulation (M2).
- Vector fitting/macromodel building (M4).
- Active learning orchestration and SR loop (M6-M8).
- Measurement ingestion (M9).

M1 must prepare for those milestones by providing deterministic artifacts and metadata contracts.

---

## 3. Tooling decisions (deep research synthesis)

### 3.1 KiCad CLI (mandatory)

We will use `kicad-cli` for:

- DRC: `kicad-cli pcb drc ...` can generate reports (including JSON) and can return an exit code based on violations.
- Fabrication outputs:
  - `kicad-cli pcb export gerbers ...` exports Gerber files.
  - `kicad-cli pcb export drill ...` exports drill files.

### 3.2 KiCad Python API backends (strategy, not dogma)

We implement a backend interface `IKiCadBackend` with two implementations:

1. Backend A (Primary, headless, stable): S-expression board writer + footprint library vendor
   - Generate `.kicad_pcb` deterministically using the published file format structure.
   - Pros: headless, deterministic, not reliant on deprecated SWIG.
   - Cons: more engineering; must generate valid boards and stable DRC behavior.

2. Backend B (Transitional convenience): SWIG `pcbnew`
   - Optional: only while pinned to KiCad 9.x if it materially reduces engineering time (e.g., zone filling).
   - Treated as compatibility layer due to deprecation.

We do not plan to base M1 on the IPC API today because it requires running GUI instances and plotting/export is the CLI's job.

### 3.3 Internal coordinate representation: integer nanometers

KiCad stores measurements at 1 nanometer internal resolution as 32-bit integers. Therefore all geometry math in M1 uses integer nanometers, never floats, to:

- avoid cross-platform rounding drift,
- make canonical hashing meaningful,
- and support exact clearance proofs.

---

## 4. Canonical artifacts and contracts

M1 produces four classes of artifacts, each with a strict contract:

### A) CouponSpec (input)

- Human-readable DSL (YAML/JSON) describing:
  - coupon family type,
  - design vector values (or normalized vector + mapping),
  - stackup,
  - fab capability profile,
  - constraints profile,
  - rendering/export settings.

### B) ResolvedDesign (intermediate)

- Fully concrete, unit-resolved, integer-nm geometry parameters
- Derived features + dimensionless groups computed
- Constraint proof object

### C) KiCad source (CAD truth)

- `coupon.kicad_pcb` (required)
- `coupon.kicad_pro` (recommended)
- local footprint library vendored in repo
- optional jobset file `coupon.kicad_jobset` (see Section 8)

### D) Fabrication & simulation exports

- Gerbers and drill files generated by `kicad-cli pcb export gerbers` and `kicad-cli pcb export drill`.
- A packaged "fab zip" (optional; M1 may create a directory tree)

---

## 5. Geometry DSL specification

### 5.1 Why a DSL (not just JSON params)

Because we need:
- strict schema validation,
- composable substructures (connector + launch + line + discontinuity),
- derived parameters + dimensionless groups,
- safe constraints + repair policies,
- and a stable evolution story (versioning).

### 5.2 DSL versions and stability

- `schema_version` integer, semver-like policy:
  - Minor additive changes allowed within same major.
  - Major increments when semantics change.
- Every artifact includes `schema_version` and `generator_version`.

### 5.3 Core type system

All lengths are represented as:

- `LengthNM`: signed 64-bit integer nanometers (clamped to 32-bit KiCad bounds at export time; see Section 6.4).

Provide parsing helpers:
- "0.25mm", "10mil", "250um" -> integer nm.

Angles: degrees or millidegrees integer.
Frequencies: Hz integer.

### 5.4 High-level coupon families (Phase 1)

Minimum supported families for M1 completeness:

#### Family F0 — Calibration Thru Line

- end-launch connector -> CPWG straight line -> end-launch connector
- no via discontinuity
- purpose: validate launch geometry + baseline insertion loss

#### Family F1 — Single-ended via transition coupon

- end-launch -> CPWG -> via transition (top to inner or top to bottom) -> CPWG -> end-launch
- includes antipads/cutouts, return vias

#### Family F2 — Differential via transition coupon (optional for M1)

- two signal vias, symmetric return vias, mode conversion controls

M1 is complete if F0 + F1 are implemented with full constraints and deterministic output. F2 is optional.

### 5.5 CouponSpec schema (normative)

```yaml
schema_version: 1
coupon_family: F1_SINGLE_ENDED_VIA
units: nm  # always normalized to nm internally; YAML may use mm/mil for convenience

toolchain:
  kicad:
    version: "9.0.7"      # pinned tool version
    # For strict reproducibility in CI, prefer digest-pinned images.
    docker_image: "kicad/kicad:9.0.7@sha256:<digest>"

fab_profile:
  id: "oshpark_4layer"   # or pcbway/jlcpcb/... (see Section 7)
  overrides: {}          # optional

stackup:
  copper_layers: 4
  thicknesses_nm:
    L1_to_L2: 180000
    L2_to_L3: 800000
    L3_to_L4: 180000
  materials:
    er: 4.1
    loss_tangent: 0.02

board:
  outline:
    width_nm: 20000000
    length_nm: 80000000
    corner_radius_nm: 2000000
  origin:
    mode: "EDGE_L_CENTER"   # canonical origin for all coupons
  text:
    coupon_id: "${COUPON_ID}"  # may use KiCad variables (see Section 8.3)
    include_manifest_hash: true

connectors:
  left:
    footprint: "Coupongen_Connectors:SMA_EndLaunch_Generic"
    position_nm: [5000000, 0]
    rotation_deg: 180
  right:
    footprint: "Coupongen_Connectors:SMA_EndLaunch_Generic"
    position_nm: [75000000, 0]
    rotation_deg: 0

transmission_line:
  type: "CPWG"
  layer: "F.Cu"
  w_nm: 300000
  gap_nm: 180000
  length_left_nm: 25000000
  length_right_nm: 25000000
  ground_via_fence:
    enabled: true
    pitch_nm: 1500000
    offset_from_gap_nm: 800000
    via:
      drill_nm: 300000
      diameter_nm: 600000

discontinuity:
  type: "VIA_TRANSITION"
  signal_via:
    drill_nm: 300000
    diameter_nm: 650000
    pad_diameter_nm: 900000
  antipads:
    L2:
      shape: "ROUNDRECT"
      rx_nm: 1200000
      ry_nm: 900000
      corner_nm: 250000
    L3:
      shape: "CIRCLE"
      r_nm: 1100000
  return_vias:
    pattern: "RING"
    count: 4
    radius_nm: 1700000
    via:
      drill_nm: 300000
      diameter_nm: 650000
  plane_cutouts:
    L2:
      shape: "SLOT"
      length_nm: 3000000
      width_nm: 1500000
      rotation_deg: 0

constraints:
  mode: "REJECT"   # REJECT | REPAIR
  drc:
    must_pass: true
    severity: "all"
  symmetry:
    enforce: true
  allow_unconnected_copper: false

export:
  gerbers:
    enabled: true
    format: "gerbers"
  drill:
    enabled: true
    format: "excellon"
  outputs_dir: "artifacts/"
```

### 5.6 Design vector normalization and mapping

For AI/search, we define two representations:

- Normalized vector u in [0,1]^d (GPU-friendly).
- Physical vector x (nm/int) produced by mapping:
  - linear scaling, log scaling, or discrete choice mapping.

Mapping is part of the spec as a `parameterization` block (optional), or baked per family as default.

The generator must emit both:
- `design_vector_physical` (resolved integer nm)
- `design_vector_normalized` (if provided)

---

## 6. Geometry kernel and compilation

### 6.1 Canonical coordinate frame

We adopt one canonical frame across all coupons:

- Origin at left board edge center (`EDGE_L_CENTER`).
- +x direction to the right along coupon length.
- +y upward (right-handed 2D).

### 6.2 Primitive set (internal IR)

Compile DSL -> IR consisting only of:
- FootprintInstance
- Pad
- TrackSegment (straight)
- ArcTrack (optional)
- Via
- Polygon (for keepouts and copper cutouts)
- BoardOutline polygon
- Text
- RuleArea/Keepout (optional)
- Net and NetClass declarations

All coordinates in integer nm.

### 6.3 Composition graph

Coupon is compiled as a feature graph:

1. Board outline feature
2. Port features (left/right connectors)
3. Launch region feature (taper + pad transitions)
4. Transmission line feature (CPWG segment + optional fence)
5. Discontinuity feature (via + antipads + return pattern)
6. Annotation feature (silk label, hash QR/ID)

Each feature returns:
- a set of IR objects
- a set of constraints it requires (pre/post conditions)
- a set of derived metrics

### 6.4 Integer safety and KiCad limits

KiCad stores object positions in 32-bit integers at 1 nm resolution, enabling boards up to ~4m x 4m. We enforce:
- All coordinates fit in 32-bit signed range
- All polygon vertices integer nm
- All widths/drills positive and within fab profile bounds

Any overflow triggers a hard error.

---

## 7. Strict constraint system

### 7.1 Constraint categories

We define constraints in five tiers (increasing cost):

Tier 0 — Parameter bounds (GPU vectorizable)
- Example: w_nm >= min_trace_width_nm

Tier 1 — Derived scalar constraints (GPU vectorizable)
- Example: annular ring, via fence pitch constraints

Tier 2 — Analytic spatial constraints (GPU vectorizable)
- Example: return vias must not collide with signal via pad

Tier 3 — Exact geometry collision checks (CPU)
- integer polygon clipping / Minkowski sum for clearance

Tier 4 — KiCad DRC gate (authoritative)
- run `kicad-cli pcb drc --severity-all --exit-code-violations ...`

### 7.2 Fab capability profiles (DFM)

A `fab_profile` is a versioned JSON document containing:
- min trace width / spacing
- min drill
- min annular ring
- soldermask expansion limits
- silkscreen min width and clearance
- edge clearance rules

We do not hardcode vendor numbers; we ship conservative defaults and allow override.

### 7.3 Symmetry and topology constraints

For diff pair designs (F2), enforce:
- mirror symmetry about axis,
- matched lengths (within tolerance),
- mirrored return via pattern.

For single-ended (F1), enforce:
- connector placement symmetry (left/right)
- net class assignments consistent

### 7.4 Constraint modes: REJECT vs REPAIR

- REJECT: fail fast, deterministic error, no side effects.
- REPAIR: project into feasible region, returning:
  - repaired ResolvedDesign
  - repair_map
  - repair_reason list
  - repair_distance in normalized space

### 7.5 Constraint proofs (auditable)

Every generated design emits `constraint_proof.json` containing:
- constraint id
- description
- symbolic form (string)
- evaluated values
- pass boolean
- signed margin

---

## 8. KiCad project generation and export strategy

### 8.1 The reality of KiCad APIs

- SWIG pcbnew bindings allow scripting but are deprecated and unstable across major versions.
- IPC API is stable but currently requires GUI and does not cover plotting; CLI is recommended for exporting.
- CLI is stable for DRC/export and designed for automation.

Therefore M1 is built around deterministic board generation (preferably without SWIG) and CLI for DRC/export.

### 8.2 Backend A: deterministic `.kicad_pcb` generation (recommended foundation)

#### 8.2.1 Approach

- Maintain a minimal board template.
- Programmatically inject geometry objects.

#### 8.2.2 Deterministic UUID/tstamp strategy

Generate per-object UUIDs deterministically using UUIDv5:
- namespace: UUIDv5("coupongen:<schema_version>")
- name: canonical object path string

This gives stable UUIDs without global state.

#### 8.2.3 Ensuring DRC sees correct nets

Assign conductive objects to nets:
- signal tracks/vias: net SIG
- grounds: net GND

If representing planes as zones, either:
- fill via SWIG backend, or
- represent copper planes as explicit polygons and enforce clearances via our own collision engine + DRC rule areas.

### 8.3 Deterministic variable injection

KiCad CLI supports `--define-var KEY=VALUE`. Use it to inject deterministic variables like:
- COUPON_ID
- GIT_SHA
- STACKUP_ID

### 8.4 Backend B: SWIG pcbnew (optional transitional layer)

Only if needed for zone refill/fill or other hard-to-reimplement features. Treat as temporary.

### 8.5 Export via KiCad CLI (mandatory)

#### 8.5.1 DRC

Run:
- `kicad-cli pcb drc --severity-all --exit-code-violations --format json --output drc.json coupon.kicad_pcb`

Exit code is 0 if no violations, 5 if violations exist.

#### 8.5.2 Gerbers

Use:
- `kicad-cli pcb export gerbers ... coupon.kicad_pcb`

#### 8.5.3 Drill files

Use:
- `kicad-cli pcb export drill ... coupon.kicad_pcb`

### 8.6 Optional jobsets (nice-to-have)

KiCad supports jobsets to define a reusable set of outputs/actions.

---

## 9. Deterministic hashing and provenance

### 9.1 Identifiers

Define:
- design_hash = SHA256(canonical_resolved_design_json_bytes)
- coupon_id = base32(design_hash)[0:12]

### 9.2 Canonicalization

Compute stable hashes for artifacts:

- KiCad board hash:
  - preferred: hash ResolvedDesign only
  - optional: hash `.kicad_pcb` after normalization

- Gerber/drill hashes:
  - canonicalize by removing nondeterministic comment lines and normalizing line endings

### 9.3 Provenance manifest (`manifest.json`)

Every design output directory includes `manifest.json` with required fields:
- schema_version, coupon_family
- design_hash, coupon_id
- resolved_design
- derived_features + dimensionless_groups
- fab_profile_id + resolved limits
- stackup
- toolchain (KiCad version, docker image tag/digest, kicad-cli --version output)
- exports list with canonical hashes
- verification (DRC summary + constraint_proof summary)
- lineage (git commit hash, UTC timestamp; timestamp is allowed to vary and is not part of design_hash)

---

## 10. CLI interface (repo API surface)

### 10.1 Commands (M1 must implement)

1. `coupongen validate <spec.yaml>`
   - load schema
   - resolve parameters
   - run Tier 0-3 constraints
   - emit ResolvedDesign + constraint_proof.json
   - no KiCad required

2. `coupongen generate <spec.yaml> --out <dir>`
   - runs validate
   - generates KiCad project files

3. `coupongen drc <dir>/coupon.kicad_pcb`
   - invokes `kicad-cli pcb drc` with required flags
   - writes JSON report
   - fails non-zero on violations

4. `coupongen export <dir>/coupon.kicad_pcb --out <dir>/fab`
   - invokes gerber + drill export
   - canonicalizes and hashes outputs

5. `coupongen build <spec.yaml>`
   - generate + drc + export in one command
   - returns path to artifact dir keyed by design_hash

### 10.2 Machine interface (Python API)

Provide `coupongen` as an importable library:
- load_spec(path) -> CouponSpec
- resolve(spec) -> ResolvedDesign
- generate_kicad(resolved, outdir) -> KiCadProjectPaths
- run_drc(board_path) -> DrcReport
- export_fab(board_path, outdir) -> FabArtifacts

---

## 11. Computational efficiency plan (making the most of hardware)

### 11.1 Real performance bottleneck

Not geometry math; it is:
- invoking KiCad CLI repeatedly,
- filesystem IO for Gerbers,
- DRC runs.

Optimize via:
- GPU constraint filtering before generating boards,
- caching,
- minimizing CLI invocations.

### 11.2 GPU vectorized feasibility filter

Implement:

`batch_filter(u_batch: cupy.ndarray [N,d]) -> mask_feasible, u_repaired, repair_metadata`

- uses CuPy arrays for Tier 0-2 constraints
- outputs feasible mask + repaired candidates

### 11.3 Structural caching

Cache levels:
- ResolvedDesign cache: design_hash -> resolved_design.json
- KiCad cache: design_hash -> coupon.kicad_pcb
- Fab cache: design_hash -> fab/ outputs

Cache key includes toolchain_hash so upgrades trigger rebuild.

### 11.4 Parallelism strategy (single laptop)

- Use multi-process pool for external KiCad invocations.
- Throttle with semaphores (from M0 runner):
  - max concurrent KiCad CLI jobs: 1-2
  - max concurrent pure-python resolves: CPU count

---

## 12. Verification and acceptance gates (how we know we've met the bar)

### 12.1 Acceptance gates (must pass in CI)

Gate G1 — Schema + resolver determinism
- fixed set of canonical CouponSpec files
- resolve produces byte-identical resolved_design.json
- design_hash matches golden values

Gate G2 — Constraint proof completeness
- every constraint has id/description/evaluation/margin
- proof file passes schema
- REJECT/REPAIR behaviors are deterministic and auditable

Gate G3 — KiCad DRC clean
- generate board
- run `kicad-cli pcb drc --severity-all --exit-code-violations --format json ...`
- must exit 0

Gate G4 — Export completeness
- Gerbers + drill exist
- expected layer set exists
- exports have canonical hashes

Gate G5 — Output hash stability across runs
- build same spec 3x in fresh dirs
- design_hash identical
- export hashes identical
- drc report hash identical (or canonicalized to ignore timestamps)

### 12.2 Performance gates (must pass locally; optional CI)

On target laptop:
- GPU filter throughput vs CPU baseline
- build throughput budgets

### 12.3 Manual quality gates (one-time sign-off)

- Open generated boards in KiCad GUI
- Inspect connector placement, antipads/cutouts, silkscreen ID
- Load Gerbers into viewer

---

## 13. Repository structure (recommended)

Note: This repo is already a src-layout Python package rooted at `src/formula_foundry/` (M0). For M1, keep the coupongen code under the same top-level package to avoid import/packaging drift.

Recommended structure:

```
/coupongen/
  /schemas/
  /fab_profiles/
  /stackups/
  /kicad_templates/
  /libs/
    /footprints/
      Coupongen_Connectors.pretty/
      Coupongen_Test.pretty/
/src/formula_foundry/
  /coupongen/
    __init__.py
    spec.py
    units.py
    resolve.py
    constraints/
      tiers.py
      gpu_filter.py
      repair.py
    geom/
      primitives.py
      cpwg.py
      via_patterns.py
      cutouts.py
    kicad/
      sexpr.py
      board_writer.py
      pcbnew_backend.py
      cli.py
      canonicalize.py
    export.py
    manifest.py
    cli_main.py
/tests/
  /golden_specs/
  /golden_hashes/
  test_schema.py
  test_resolve_determinism.py
  test_constraints.py
  test_kicad_drc.py
  test_export_hashes.py
```

---

## 14. How M1 directly increases odds of discovering the novel equation

1. Equation discovery is only as real as your oracle labels. M1 removes geometry noise by enforcing deterministic CAD and DRC-clean manufacturable designs.
2. Symbolic discovery needs stable, meaningful variables. M1 defines the parameterization and emits dimensionless groups into the manifest.
3. Active learning needs fast feasibility filtering. The GPU constraint prefilter prevents wasted EM solves.
4. Corporate-grade trust requires auditable artifacts. The manifest + hashing + DRC reports are the evidence chain.

---

## 15. Risk register and mitigation

R1: SWIG deprecation/removal breaks scripts
- Mitigation: Backend A is primary; SWIG backend is optional.

R2: IPC API is not headless and lacks plotting
- Mitigation: do not depend on IPC API for M1 generation/export; use CLI.

R3: Gerber nondeterminism (timestamps/comments)
- Mitigation: canonicalization + canonical hashes.

R4: Constraint mismatch between our solver and KiCad DRC
- Mitigation: treat KiCad DRC as authoritative gate.

R5: Polygon/zone complexity causes DRC/export differences
- Mitigation: keep feature set minimal in M1.

---

## 16. Concrete "M1 is DONE when..." checklist (repo-level)

M1 is complete only when all below are true:

1. `coupongen build` exists and works on a fresh machine inside the pinned KiCad docker image.
2. At least two coupon families (F0 and F1) are supported end-to-end.
3. For each family, there are >= 10 golden specs committed, and CI proves:
   - deterministic resolved_design hashing,
   - deterministic export hashing,
   - `kicad-cli pcb drc --exit-code-violations` returns success.
4. A batch GPU feasibility filter exists and is used by the generator pipeline (even if optional), with unit tests.
5. The manifest schema is stable, versioned, and includes toolchain versions, stackup, fab profile, and artifact hashes.
6. The generator emits DRC-clean boards with required manufacturing outputs (Gerbers + drills).
7. All outputs are addressable by design_hash and no orphan artifacts exist.

---

## Normative Requirements (must)

### A) CouponSpec schema + units
- [REQ-M1-001] The repo must define a strict JSON Schema for CouponSpec and validate all inputs before resolve/generate.
- [REQ-M1-002] The repo must represent all geometry internally as integer nanometers and provide deterministic parsing for mm/mil/um inputs.

### B) Deterministic resolve + design hashing
- [REQ-M1-003] `resolve(spec)` must emit a ResolvedDesign with only integer-nm parameters, derived features, and dimensionless groups.
- [REQ-M1-004] `design_hash` must be computed as SHA256 over canonical ResolvedDesign JSON bytes and must be stable across runs.
- [REQ-M1-005] The repo must provide canonical hash functions for board and export artifacts that remove nondeterministic noise (timestamps/comments).

### C) Coupon families
- [REQ-M1-006] M1 must implement Family F0 (calibration thru) end-to-end.
- [REQ-M1-007] M1 must implement Family F1 (single-ended via transition) end-to-end.

### D) Constraint system
- [REQ-M1-008] The repo must implement a tiered constraint system (Tiers 0-4) and expose tiered evaluation results.
- [REQ-M1-009] In REJECT mode, infeasible specs must fail deterministically with constraint IDs and human-readable reasons.
- [REQ-M1-010] In REPAIR mode, infeasible specs must be projected into feasible space with an auditable repair_map, repair_reason list, and repair_distance.
- [REQ-M1-011] Every generated design must emit a constraint_proof.json with per-constraint evaluations and signed margins.

### E) KiCad board generation backends
- [REQ-M1-012] The repo must define `IKiCadBackend` and implement Backend A (headless S-expression board writer) as the primary path.
- [REQ-M1-013] Backend A must generate deterministic per-object UUID/tstamp values (UUIDv5 or equivalent) from canonical object paths.
- [REQ-M1-014] Footprints used by coupongen must be vendored in-repo (no external library dependency at build time).

### F) KiCad CLI integration (DRC + exports)
- [REQ-M1-015] The repo must provide a `kicad-cli` runner that can execute via local binary or via the pinned KiCad Docker image.
- [REQ-M1-016] `coupongen drc` must run KiCad DRC with severity-all, JSON report output, and exit-code gating.
- [REQ-M1-017] `coupongen export` must generate Gerbers and drill files via KiCad CLI and compute canonical hashes for all outputs.

### G) Manifest + artifact addressing
- [REQ-M1-018] The repo must emit a manifest.json for every build containing required provenance fields and export hashes.
- [REQ-M1-019] All output directories must be keyed by design_hash and coupon_id; re-running build must not create divergent outputs.
- [REQ-M1-020] The build pipeline must implement caching keyed by design_hash + toolchain_hash and must be deterministic when cache hits occur.

### H) CLI + Python API
- [REQ-M1-021] The repo must provide a `coupongen` CLI with validate/generate/drc/export/build commands and correct exit codes.
- [REQ-M1-022] The repo must provide a typed Python API equivalent for orchestration (M6-M8 tool calls).

### I) Golden specs + CI gates
- [REQ-M1-023] The repo must include >= 10 golden specs for F0 and >= 10 golden specs for F1, committed under tests/golden_specs.
- [REQ-M1-024] CI must prove deterministic resolve hashing against committed golden hashes.
- [REQ-M1-025] CI must prove DRC-clean boards and export completeness for all golden specs using the pinned KiCad toolchain.

### J) M0 regression prevention (mandatory for M1 completion)
- [REQ-M1-026] M1 completion must run `python -m tools.verify --include-m0 --strict-git` to ensure M0 substrate gates remain green.
- [REQ-M1-027] The orchestrator completion gate must enforce REQ-M1-026 when the active milestone is M1 or higher.

---

## Definition of Done

- All requirements REQ-M1-001..REQ-M1-027 are implemented.
- Every requirement is mapped to at least one pytest node id in the Test Matrix.
- `python -m tools.verify --strict-git` exits 0.
- `python -m tools.verify --include-m0 --strict-git` exits 0.
- The full Section 16 checklist can be executed successfully on the target laptop.
- The repo has no uncommitted changes (`git status --porcelain` empty).

---

## Test Matrix

| Requirement | Pytest(s) |
|---|---|
| REQ-M1-001 | tests/test_m1_schema.py::test_couponspec_schema_validation |
| REQ-M1-002 | tests/test_m1_units.py::test_lengthnm_parsing_integer_nm |
| REQ-M1-003 | tests/test_m1_resolve_determinism.py::test_resolve_emits_integer_nm_and_groups |
| REQ-M1-004 | tests/test_m1_resolve_determinism.py::test_design_hash_is_stable |
| REQ-M1-005 | tests/test_m1_hashing.py::test_canonical_hashing_removes_nondeterminism |
| REQ-M1-006 | tests/test_m1_families.py::test_family_f0_builds |
| REQ-M1-007 | tests/test_m1_families.py::test_family_f1_builds |
| REQ-M1-008 | tests/test_m1_constraints.py::test_constraint_tiers_exist |
| REQ-M1-009 | tests/test_m1_constraints.py::test_reject_mode_reports_constraint_ids |
| REQ-M1-010 | tests/test_m1_constraints.py::test_repair_mode_emits_repair_map_and_distance |
| REQ-M1-011 | tests/test_m1_constraints.py::test_constraint_proof_schema |
| REQ-M1-012 | tests/test_m1_backend_contract.py::test_backend_a_exists_and_writes_board |
| REQ-M1-013 | tests/test_m1_backend_contract.py::test_deterministic_uuid_generation |
| REQ-M1-014 | tests/test_m1_backend_contract.py::test_footprints_are_vendored |
| REQ-M1-015 | tests/test_m1_kicad_cli.py::test_kicad_cli_runner_modes |
| REQ-M1-016 | tests/test_m1_kicad_cli.py::test_drc_invocation_flags |
| REQ-M1-017 | tests/test_m1_export.py::test_export_outputs_and_hashes |
| REQ-M1-018 | tests/test_m1_manifest.py::test_manifest_required_fields |
| REQ-M1-019 | tests/test_m1_manifest.py::test_outputs_keyed_by_design_hash |
| REQ-M1-020 | tests/test_m1_cache.py::test_cache_toolchain_hash_behavior |
| REQ-M1-021 | tests/test_m1_cli.py::test_cli_commands_exist |
| REQ-M1-022 | tests/test_m1_api.py::test_python_api_contract |
| REQ-M1-023 | tests/test_m1_golden_specs.py::test_golden_specs_present |
| REQ-M1-024 | tests/test_m1_golden_specs.py::test_golden_hashes_match |
| REQ-M1-025 | tests/test_m1_kicad_integration.py::test_drc_clean_and_exports_in_pinned_toolchain |
| REQ-M1-026 | tests/test_m1_m0_regression.py::test_verify_include_m0_required_for_completion |
| REQ-M1-027 | tests/test_m1_m0_regression.py::test_orchestrator_completion_gate_includes_m0 |

---

## References (implementation guidance)

[1]: https://dev-docs.kicad.org/en/file-formats/sexpr-pcb/index.html
[2]: https://docs.kicad.org/master/en/cli/cli.pdf
[3]: https://docs.kicad.org/9.0/en/cli/cli.html
[4]: https://www.kicad.org/blog/2026/01/KiCad-9.0.7-Release/
[5]: https://www.kicad.org/download/docker/
[6]: https://dev-docs.kicad.org/en/apis-and-binding/pcbnew/index.html
[7]: https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/for-addon-developers/
[8]: https://docs.kicad.org/8.0/en/cli/cli.html
[9]: https://docs.kicad.org/8.0/en/pcbnew/pcbnew.html
[10]: https://docs.kicad.org/8.0/en/pcbnew/pcbnew.html
[11]: https://jlcpcb.com/blog/pcb-design-rules-best-practices
[12]: https://docs.kicad.org/9.0/en/kicad/kicad.html
