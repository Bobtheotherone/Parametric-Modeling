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

# M2 Design Document — Golden EM Oracle

**openEMS + gerber2ems pipeline with verified ports and calibration structures**

## 0. Mission alignment and why M2 must be “over‑engineered”

Your project’s core success condition is: **equation discovery on EM behavior must not be poisoned by oracle noise** (inconsistent ports, unstable meshing, non-reproducible solver setups, or hidden de-embedding mistakes). M2 is therefore not “just run a solver.” It is a **metrology-grade data-generation system** that produces **auditable, reproducible, physically consistent** multiport S-parameters for your parametric PCB discontinuity families.

**Golden Oracle (Phase 1)** = openEMS full-wave FDTD (EC-FDTD) simulations driven from manufacturable Gerbers via gerber2ems. openEMS is an FDTD full-wave solver (EC-FDTD variant) with graded meshes and absorbing boundaries (PML/MUR), configured via Python/Matlab/Octave. ([openEMS Documentation][1])
gerber2ems is a Python workflow that consumes PCB fabrication artifacts (Gerber/drill/stackup) and drives openEMS with automatic grid generation and postprocessing, including port voltage/current and S-parameter extraction. ([GitHub][2])

**Critical point:** M2 must be trustworthy enough that later milestones (vector fitting + symbolic regression + falsification) can treat its output as “physics truth,” within quantified uncertainty.

---

## 1. Scope, non-goals, and deliverable contract

### 1.1 In scope

M2 delivers a repository capability:

> **Given a “simulation case” (Gerbers + stackup + drill + port markers + simulation config), produce validated, reproducible Touchstone S-parameters** (2-port and 4-port), along with a complete audit bundle (geometry XML, mesh summary, solver logs, derived metrics, and verification artifacts).

Key features:

* **Gerber-driven EM simulation** pipeline (ROI/slice-based).
* **Port definition standard** and verification (orientation, reference plane, impedance, numbering, polarity).
* **Calibration structures** in-repo for regression + sanity checks (Thru/Line/Reflect, matched lines, known-delay lines, via-less baselines).
* **Deterministic meshing + simulation policies** (as deterministic as FDTD allows).
* **Quality gates**: fails fast on suspect setups.
* **Caching** keyed by content-addressed fingerprints.
* **Performance policy**: ROI bounding, adaptive grids, bounded memory.

### 1.2 Explicit non-goals (M2 does not do these)

* **Measurement ingestion** (VNA), de-embedding from physical fixtures (that’s M9).
* Full-board simulations without slicing/ROI (too expensive on laptop).
* Full EM co-simulation with active devices or nonlinear components.
* “Perfect correlation to measurement.” M2’s job is a consistent simulation oracle; M9 later reconciles sim↔measurement.

### 1.3 Hard deliverable contract

M2 must provide:

* A **Python API** (library) and **CLI** (tool) that can:

  1. validate inputs
  2. generate geometry and mesh
  3. run openEMS
  4. postprocess to S-parameters
  5. run verification/calibration checks
  6. emit an artifact bundle

* Output formats:

  * Touchstone `.s2p` and `.s4p` (complex S, frequency axis).
  * Companion `.json` metadata (units, port maps, solver versions, mesh stats).
  * Optional: CSV exports (gerber2ems already produces CSV outputs) ([GitHub][2])

---

## 2. Technical background constraints (what we must respect)

### 2.1 Solver characteristics: openEMS

openEMS is a 3D full-wave EC-FDTD solver with graded meshes and absorbing boundary options (MUR, PML). ([openEMS Documentation][1])
Relevant API controls in Python include:

* `SetBoundaryCond` (PEC/PMC/MUR/PML),
* `SetGaussExcite(f0, fc)`,
* `SetEndCriteria(val)`,
* `SetNumberOfTimeSteps`,
* time-step method/factor for stability,
* and a thread-count parameter (`numThreads`) for CPU parallelism. ([openEMS Documentation][3])

### 2.2 Pipeline characteristics: gerber2ems

gerber2ems expects:

* **Gerbers per copper layer** named to match stackup layer names,
* **stackup.json** including thickness, epsilon, loss tangent,
* a **PTH drill file**,
* a **position file** describing ports (“SP1…”) and port rotations,
* and a **simulation.json** containing frequency band, grid settings, port definitions, etc. ([GitHub][2])

It also emphasizes:

* simulating a whole PCB is too expensive; **slice/ROI** extraction is essential. ([GitHub][2])
* ports are defined with width/length/impedance and layer/plane mapping; port length should be at least **8× mesh cell size** (per README). ([GitHub][2])
* it recommends verifying that timesteps are sufficient; it reports excitation length and suggests at least **~3× excitation length** as max steps. ([GitHub][2])
* exporting fields can create **hundreds of GB** of data (must be controlled). ([GitHub][2])

Antmicro has recently improved the flow with:

* **adaptive grid generation** (denser near geometry, sparser elsewhere),
* use of the **1/3 meshing rule**,
* **differential pair simulation**,
  and validation by comparing simulation and VNA measurement trends. ([Antmicro][4])

---

## 3. M2 architecture

### 3.1 High-level component diagram

**M2 consists of 7 subsystems:**

1. **Case Spec & Fingerprinting**

* Normalize inputs (case manifest).
* Compute content hash.

2. **Input Validator**

* Validate file structure, units, naming, stackup consistency, port definition completeness.

3. **Geometry Builder (Gerber → CSXCAD)**

* Drive gerber2ems geometry generation into `geometry.xml` (+ intermediate raster/triangulation artifacts if needed).

4. **Grid/Mesh Planner**

* Either rely on gerber2ems adaptive grid or enforce a deterministic post-processing mesh policy.
* Emit mesh stats & determinism hash.

5. **Solver Runner (openEMS)**

* Run openEMS once per excited port (multiport assembly).
* Enforce resource and convergence policy.

6. **Postprocessor**

* Convert time-domain port data → frequency domain → S-parameters.
* Export Touchstone, apply renormalization and mixed-mode conversions as needed.

7. **Verification & Calibration Suite**

* Run port-verification checks, physicality checks, and calibration-structure regression tests.
* Gate acceptance.

### 3.2 Data flow

**Input**
`case_dir/`

* `fab/` (Gerbers, drill, stackup.json, pos.csv)
* `simulation.json` (or `oracle_case.json` which compiles down to simulation.json)
* optional: `netinfo.json` / ROI spec from slicing

**Intermediate**

* `ems/geometry/geometry.xml`
* `ems/simulation/*` (openEMS raw results)
* logs, mesh summary, port debug plots

**Output**
`artifacts/<case_hash>/`

* `sparams.s2p` or `sparams.s4p`
* `meta.json`
* `verification_report.json` + `verification_plots/`
* `solver_log.txt`
* `geometry.xml`
* `mesh_summary.json`

---

## 4. Input specification (authoritative contracts)

### 4.1 Required files and conventions (canonical)

Adopt gerber2ems conventions as the “lowest-friction baseline” and wrap them with stricter validation:

**Required (minimum)**:

* `fab/stackup.json` with layer list, including dielectric `epsilon` and `lossTangent` fields. ([GitHub][2])
* Gerber files for each simulated copper layer named `*-<name-from-stackup>.gbr`. ([GitHub][2])
* Drill file `*-PTH.drl` (Excellon). ([GitHub][2])
* Position file `*-pos.csv` containing “SP#” markers (ports) with X/Y/Rot/Side. ([GitHub][2])
* `simulation.json` (or our higher-level case file compiled into it). ([GitHub][2])

**Port marker rule** (must be enforced):

* Each simulation port marker designator starts with `"SP"` followed by the port number. ([GitHub][2])

**Coordinate reference rule**:

* Drill origin bottom-left, pos.csv coordinates referenced from bottom-left corner (per gerber2ems doc). ([GitHub][2])

### 4.2 Simulation config schema: baseline + our extensions

gerber2ems `simulation.json` includes:

* frequency `start/stop`,
* max timesteps,
* grid parameters (`optimal`, `diagonal`, `perpendicular`, `max`, margins, cell ratio),
* ports list with `width`, `length`, `impedance`, `layer`, `plane`, `excite`,
* and traces/differential pair definitions referencing port numbers. ([GitHub][2])

**We will keep compatibility** but define **M2 canonical case schema**: `oracle_case.json` that compiles to `simulation.json`.

#### 4.2.1 `oracle_case.json` fields (canonical)

* `format_version`: semantic version for our schema.
* `case_id`: human name, plus deterministic `case_hash`.
* `frequency`: `{ start_hz, stop_hz, npoints, grid: "lin"|"log" }`
* `solver_policy`:

  * `boundary`: `"PML_8"` default, with explicit thickness and margins
  * `end_criteria`: e.g. `1e-5` (≈ -50 dB) as a baseline, adjustable
  * `max_steps`: required
  * `threads`: explicit int (maps to openEMS `numThreads`) ([openEMS Documentation][3])
  * `time_step_factor`: optional stability knob (openEMS supports) ([openEMS Documentation][3])
* `grid_policy`:

  * `lambda_divisor_max_cell`: default 15 (openEMS rule-of-thumb) ([OpenEMS Wiki][5])
  * `metal_res_lambda_divisor`: default 20 (openEMS mesh example guidance) ([OpenEMS Wiki][5])
  * `thirds_rule`: true (enforce 1/3 rule on metal edges) ([OpenEMS Wiki][5])
  * `cell_ratio_xy_max`: default 2.0 (strict), or 1.2 if following gerber2ems default style ([OpenEMS Wiki][5])
  * `roi_margin_xy_um`, `roi_margin_z_um`
* `ports`:

  * `port_definitions`: array of logical port specs (see §5), each references an SP marker
  * `reference_impedance_ohm`: default 50
* `structures`:

  * `type`: `"cal_thru"|"cal_line"|"cal_reflect"|"coupon"|"via_transition"|"launch" …`
  * `port_map`: map from logical ports to physical SP markers
* `postprocess`:

  * `export_touchstone`: true
  * `mixed_mode`: options for diff pairs (true/false)
  * `renormalize_to_ohms`: 50 (optional)
* `verification`:

  * enabled checks + thresholds
* `provenance`:

  * git commit hash, openEMS commit hash, gerber2ems version, container image digest

---

## 5. Port standards and verification (the heart of M2)

Ports are the most common way EM datasets become unusable (wrong reference planes, swapped orientations, inconsistent impedance definitions).

### 5.1 Port definition strategy

We standardize **two layers of port definition**:

1. **Physical port marker** in PCB space (SP footprints or pos.csv entries).
2. **Logical port spec** in case config, including reference plane, expected direction, impedance, and validity checks.

#### 5.1.1 Physical port placement & orientation

gerber2ems computes port area from pos.csv coordinates + rotation:

* Rotation maps to X/Y span and **wave travel direction** (0/90/180/270 degrees). ([GitHub][2])
  Antmicro’s slicing tool places ports at trace endpoints and limits orientation to the four cardinal directions. ([Antmicro][6])

**M2 rule:** port rotation must be in {0, 90, 180, 270}. If not, fail validation unless `allow_non_cardinal_ports=true` and you provide a custom mapping (advanced mode).

#### 5.1.2 Transmission-line ports vs lumped ports

openEMS supports multiple port types:

* **LumpedPort** (compact port with feed resistance) and
* **MSLPort** (microstrip transmission line port requiring mesh definition, with start/stop ordering defining propagation direction and options like `Feed_R`, `MeasPlaneShift`). ([OpenEMS Wiki][7])

gerber2ems notes ports are “currently composed of microstripline fragments” and have minimum length constraints. ([GitHub][2])

**M2 policy**:

* Default to the gerber2ems port model for compatibility.
* Provide an **experimental alternate port backend**:

  * For grounded CPW launches and via transitions, a lumped-port feeding strategy can be more robust (especially when CPW grounds complicate quasi-TEM assumptions).
  * This backend is optional but recommended for later high-rigor validations (and for debugging port issues).

**Port backend selection** is recorded in metadata and becomes part of the run fingerprint.

### 5.2 Reference plane standard

For any port `i`, define:

* **Port reference plane** = plane where S-parameters are referenced.
* For MSL-like ports, openEMS/ports treat the start/stop box order as propagation direction; the reference plane can be shifted (e.g., `MeasPlaneShift`) and excitation plane is at start. ([OpenEMS Wiki][7])

**M2 rule:**

* The reference plane must be explicit and reproducible:

  * either by a fixed measurement plane shift (in drawing units),
  * or by a geometric anchor (e.g., “at inner edge of launch pad”) converted deterministically to a coordinate shift.

### 5.3 Port numbering and polarity standard (2-port + 4-port)

#### 5.3.1 Single-ended 2-port

* Port 1 = “source side”, Port 2 = “sink side”
* Convention: forward direction is 1→2

#### 5.3.2 Differential 4-port

Antmicro describes differential nets requiring 4 simulation ports. ([Antmicro][6])

**M2 strict port ordering (default):**

* (1,2) = near-end pair: P and N
* (3,4) = far-end pair: P and N

This ordering is chosen specifically to make mixed-mode conversion consistent and unambiguous.

We also store a `diff_pair_map`:

* `near_end: {p:1, n:2}`, `far_end: {p:3, n:4}`

### 5.4 Port verification pipeline (multi-layer defense)

We do not rely on “it ran.” We prove ports are correct using four verification layers:

#### 5.4.1 Geometry-level verification (pre-sim)

Checks:

1. **Port intersects signal metal**: The port box must overlap a signal copper region on the specified `layer`.
2. **Reference plane exists**: The specified `plane` copper layer exists and has local ground continuity near port.
3. **Port length constraint**: `length >= 8 * local_mesh_dx` (using intended mesh), aligned with gerber2ems guidance that ports should be long relative to mesh. ([GitHub][2])
4. **Orientation correctness**: port direction must point into the network, not outward.
5. **No accidental short**: port box must not intersect ground copper in ways that collapse the port.

Outputs:

* `port_debug.png` overlays: port boxes on copper layers.
* `port_validation.json` with pass/fail and measured overlaps.

#### 5.4.2 Wave-level verification (post-sim time domain)

After openEMS runs:

* Ensure `port[i].uf.inc` is non-zero over band (no dead excitation).
* Ensure the excited port shows expected initial reflection behavior:

  * For matched calibration lines, early-time reflection should be low.
  * For reflect standards, early reflection magnitude should be high.

(We compute these from saved port V/I time traces and FFT.)

#### 5.4.3 S-parameter physicality checks (post-sim frequency domain)

Using the computed S-matrix:

* **Reciprocity check**: for passive reciprocal structures, `|S21 - S12|` small across band (within tolerance).
* **Energy conservation check**: for passive structures, ensure `|S11|^2 + |S21|^2 ≤ 1 + ε` (2-port), extend to N-port with total outgoing power bounds.
* **Causality sanity**: group delay should not be wildly negative beyond tolerance (numerical noise allowed near deep nulls).

These checks don’t “prove” EM is correct, but they catch many port/sign mistakes.

#### 5.4.4 Calibration-structure regression (gold standard)

Run the calibration suite (§8) regularly; if any calibration fails, **block merges**.

---

## 6. Meshing and grid policy (determinism + efficiency)

### 6.1 What we must optimize for

* openEMS runtime scales brutally with mesh density; antmicro notes that making the grid twice as dense can increase time by **more than 8×**, and smaller cells also reduce timestep size, requiring more steps. ([Antmicro][4])
* We must therefore:

  * keep cells coarse where possible,
  * refine only near geometry/ports/discontinuities,
  * keep the mesh smooth (no huge jumps).

### 6.2 Foundational rules (enforced)

From openEMS mesh guidance:

* **Max cell size** should be about a tenth of the smallest wavelength; better `max(Δx,Δy,Δz) < λ_min / 15`. ([OpenEMS Wiki][5])
* **Smooth mesh**: neighboring cell sizes should not differ by more than ~2×. ([OpenEMS Wiki][5])
* **Thirds rule**: for metal edges, place a mesh line 1/3 inside and 2/3 outside to reduce FDTD edge error; recommended for precision microstrip results. ([OpenEMS Wiki][5])
  Antmicro’s improved flow explicitly uses the 1/3 meshing rule for adaptive grids. ([Antmicro][4])

### 6.3 Adaptive mesh generation (preferred)

We adopt antmicro’s philosophy:

* Analyze Gerber copper contours.
* Place dense grid lines around “interesting metal patterns” and ports.
* Use sparser grid lines in uniform regions. ([Antmicro][4])

Implementation requirements:

* Deterministic contour extraction and grid placement:

  * fixed pixel size and deterministic rasterization settings (gerber2ems has `pixel_size` in config). ([GitHub][2])
* Mesh lines:

  * always include all copper edges and port box boundaries,
  * apply thirds rule on edges,
  * apply smoothing to enforce ratio limits.

### 6.4 ROI and boundary placement

openEMS PML guidance:

* PML is absorbing but has warnings: the last `x` lines are PML material; structures must be far enough away; PML thickness often 6–20 cells and PML_8 is a reasonable default. ([OpenEMS Wiki][8])
  openEMS mesh rules also emphasize keeping structure away from absorbing boundaries (e.g., ~λ/4) for accuracy. ([OpenEMS Wiki][5])

**M2 default boundary policy**:

* Use `PML_8` on all 6 boundaries unless explicitly using symmetry planes.
* Enforce a minimum clearance from geometry bounding box to PML region, expressed in:

  * cells and
  * physical distance.

**Performance mode**:

* For some closed-ish structures, MUR can be faster; openEMS notes PML isn’t optimal for speed and suggests MUR for faster sims (with limitations). ([OpenEMS Wiki][8])
  M2 may allow MUR in “fast debug mode,” but the golden oracle mode defaults to PML.

---

## 7. Simulation run policy (openEMS settings that must be standardized)

### 7.1 Frequency plan

Input is a frequency band `[f_start, f_stop]` and a requested sampling grid.

**M2 standard**:

* Use a broadband Gaussian excitation (openEMS supports `SetGaussExcite(f0, fc)`). ([openEMS Documentation][3])
* Choose:

  * `f0 = 0.5*(f_start + f_stop)`
  * `fc = 0.5*(f_stop - f_start)` (or tuned so -20 dB bandwidth fully covers band)

### 7.2 Time stepping and termination criteria

We support two termination models:

1. **Energy-based termination**: openEMS supports `EndCriteria` (e.g., 1e-5 ≈ -50 dB). ([openEMS Documentation][3])
2. **Hard cap**: `max_steps`

gerber2ems shows an operational guideline: ensure max timesteps are at least ~3× excitation signal length. ([GitHub][2])

**M2 standard**:

* Require both:

  * `max_steps`
  * `end_criteria`
* Enforce:

  * `max_steps / excitation_length >= 3.0` as a warning threshold; `<2.0` is hard-fail (unless `allow_short_runs=true`).
* Record termination cause:

  * ended by end criteria vs by step cap.

### 7.3 Multiport assembly

To get a complete N-port S-matrix:

* Run one simulation per excited port, with other ports terminated.
  This matches openEMS multi-port guidance: typically only one port is active at a time to compute reflection/transmission (multiport port array, one active). ([OpenEMS Wiki][7])

**M2 requirement**:

* For N logical ports:

  * perform N runs (unless using superposition features explicitly validated).
* Store all raw port waveforms so S can be recomputed offline.

### 7.4 Threading and CPU resource governance

openEMS python interface supports a `numThreads` parameter. ([openEMS Documentation][3])

**M2 default**:

* `numThreads = min(physical_cores, user_limit)`
* Never oversubscribe beyond thermal constraints.
* Integrate with M0 scheduler (GPU and CPU semaphores), but M2 must be able to run standalone.

---

## 8. Postprocessing pipeline (turn raw data into “learnable truth”)

### 8.1 Port postprocessing: incident/reflected waves → S

openEMS port postprocessing exposes incident/reflected voltage and current in frequency domain, and S-parameters can be computed as ratios (e.g., `s11 = u_ref/u_inc`, `s21 = u_ref(out)/u_inc(in)`). ([OpenEMS Wiki][7])

**M2 requirement**:

* The pipeline must produce:

  * raw time-domain traces,
  * raw frequency-domain waves (inc/ref/tot),
  * and final S-parameters.

### 8.2 Touchstone export rules

Touchstone output must include:

* frequency in Hz,
* complex S in RI or MA or DB/angle (choose one standard and stick to it; recommended RI),
* reference impedance per port.

### 8.3 Renormalization

We will often want to compare across cases with different port reference impedances. scikit-rf supports renormalizing S-parameters to new port impedances. ([scikit-rf.readthedocs.io][9])

**M2 requirement**:

* Provide a `renormalize_to_ohms` option:

  * if set, export both:

    * native Zref S,
    * and renormalized S.

### 8.4 Mixed-mode conversion (differential)

For differential structures, we need mixed-mode S (Sdd, Scc, Sdc, Scd). scikit-rf provides mixed-mode conversion utilities (`se2gmm`, mixed-mode basics). ([scikit-rf.readthedocs.io][10])

**M2 requirement**:

* For 4-port results:

  * export `.s4p` (single-ended),
  * export mixed-mode `.s4p` or `.s2p`-equivalent representations:

    * `Sdd21`, `Sdd11`, etc. (define format in metadata).
* Strictly enforce diff port pairing to avoid ambiguity.

### 8.5 GPU-accelerated postprocessing (important for your mission)

Even if openEMS is CPU-bound, M2 should be **GPU-first where it matters**:

* FFTs and array transforms across large batches of cases can be done in CuPy.
* Mixed-mode conversion, renormalization, resampling, and quality metric computation can be batched on GPU.

**M2 requirement**:

* Implement postprocessing kernels with a backend abstraction:

  * `numpy` fallback,
  * `cupy` default when GPU available,
  * no silent fallback (must log “CPU fallback reason”).

This directly increases dataset throughput for equation discovery.

---

## 9. Calibration structures (what we simulate to prove M2 is correct)

M2 includes a **calibration library**: parameterized KiCad/Gerber sources + expected behaviors + golden results. These are used for:

* regression testing,
* port verification,
* mesh policy verification,
* and as reference structures for later measurement de-embedding (M9).

### 9.1 Why TRL, and what structures to include

TRL calibration uses **Thru, Reflect, Line** standards. ([Copper Mountain Technologies][11])
Even if M2 doesn’t perform measurement calibration, TRL-style structures are excellent because they:

* are easy to fabricate,
* are transmission-line based (good match to PCB environments),
* and provide strong sanity checks on phase/delay and port definition.

### 9.2 Calibration structure set (minimum viable, but rigorous)

#### CAL-0: Empty “airbox” sanity

* No copper, only boundary conditions.
* Expect near-zero coupling, near-zero reflected power except numerical noise.
* Purpose: boundary + solver stability baseline.

#### CAL-1: Matched uniform line (2-port)

* Straight microstrip or grounded CPW of known length L.
* Expected:

  * |S11| low in band,
  * |S21| near 0 dB minus dielectric/metal loss,
  * phase slope corresponds to group delay ≈ L / v_p.
* Compare to analytic TL model (from impedance/ε_eff estimates) within tolerance.

#### CAL-2: “Thru” (TRL T)

* Essentially a very short interconnect between ports.
* Expected:

  * S21 ≈ 1∠0 (minus small loss),
  * S11 small.

#### CAL-3: “Line” (TRL L)

* Same as Thru but with additional known length ΔL.
* Expected:

  * S21 phase difference vs Thru = β(ω)ΔL.

#### CAL-4: “Reflect” (TRL R)

* Short or open termination (in PCB context, often a short is easier/cleaner).
* Expected:

  * |S11| ≈ 1 in band (away from parasitic resonances),
  * S21 ≈ 0.

#### CAL-5: Differential pair straight line (4-port)

* Known-length diff pair in homogeneous environment.
* Expected:

  * low mode conversion (Sdc, Scd small),
  * consistent Sdd21 delay,
  * symmetry/reciprocity sanity.

#### CAL-6: Known discontinuity “signature” structures

These are not to match a simple analytic curve, but to be **stable regression fingerprints**:

* a via-less pad discontinuity,
* a stub,
* a simple via transition with a known antipad.

### 9.3 “Golden results” policy for calibration tests

For each CAL-* structure:

* store:

  * the case inputs (Gerbers/stackup/ports config),
  * and one or more golden outputs:

    * `golden.s2p`,
    * `golden_meta.json` including solver commit and mesh stats.

**But**: since solver versions and floating point can shift results slightly, the regression test must be based on **metrics**, not bitwise equality:

* max |ΔS| in band,
* max phase error in band,
* group delay error,
* passivity margin slack.

---

## 10. Verification gates (how we know M2 is *done* and trustworthy)

This is the section you asked for explicitly: **how we know for certain we’ve reached strict engineering rigor**.

### 10.1 Quality gates must be automated and blocking

M2 is “done” only if the repo can run:

```bash
m2 validate --all
```

and it passes:

* unit tests,
* integration tests,
* calibration regression suite,
* performance sanity suite.

### 10.2 Definition of Done (DoD) — required pass conditions

#### A) Deterministic build + pinned dependencies

* openEMS version pinned (or pinned commit) and recorded in output metadata.

  * gerber2ems recommends a specific openEMS commit as “latest tested” with gerber2ems. ([GitHub][2])
* gerber2ems version pinned and recorded.
* Container or environment hash recorded in `meta.json`.

#### B) Input validation correctness

The validator must correctly detect and fail:

* missing stackup layers referenced by Gerbers,
* missing drill or pos file,
* non-cardinal port rotations (unless explicitly allowed),
* port boxes that do not overlap signal copper,
* invalid grid settings (e.g., violating `max <= λmin/10` guidance). ([GitHub][2])

#### C) Port verification suite passes

On CAL-1/CAL-2:

* |S11| median in band < threshold (e.g., -15 dB) for matched lines.
* Group delay positive and within ±X% of expected from line length.
* S21 magnitude near expected (allow loss).

On CAL-4 reflect:

* |S11| median > threshold (e.g., -1 dB) in band.
* S21 small.

On reciprocity-expected structures:

* max |S21 - S12| < tolerance.

#### D) Mesh policy invariance tests

For at least one calibration structure, run **two meshes**:

* baseline mesh,
* refined mesh (e.g., tighten lambda divisor or local refine near ports),

and require:

* max |ΔS21| < tolerance in band (e.g., 0.2 dB magnitude, 5° phase)
  This is your strongest evidence that your oracle is not a mesh-artifact generator.

#### E) Solver termination sanity

For all calibration sims:

* simulation must end via end criteria *or* reach max steps with energy decreasing.
* must not end prematurely with “not enough timesteps” (hard-fail if `max_steps < 2× excitation_len`, warn if <3×). ([GitHub][2])

#### F) Artifact completeness and auditability

For every run, M2 must output:

* Touchstone,
* `meta.json`,
* `mesh_summary.json`,
* `port_map.json`,
* solver logs,
* verification report.

No exceptions: without these, data is not “gold.”

#### G) Performance sanity (single laptop)

M2 must include a benchmark target:

* CAL-2 Thru simulation must complete within a set envelope on your laptop.
* If it regresses, CI fails (local CI).

This prevents creeping inefficiency that kills dataset throughput.

---

## 11. Computational efficiency design (what we implement for throughput)

### 11.1 Cost model and early rejection

Before running openEMS:

* estimate number of Yee cells from mesh lines: Nx×Ny×Nz.
* estimate memory consumption (conservative).
* if projected memory > user budget (e.g., 24 GB of RAM), **fail early** with a diagnostic:

  * “ROI too big”
  * “grid too fine”
  * “frequency too high for requested resolution”

### 11.2 ROI minimization

gerber2ems and Antmicro stress simulating whole boards is too expensive; slices/ROI are key. ([GitHub][2])

**M2 requirement**:

* enforce ROI bounding via:

  * `margin/from_trace = true` by default (use nets-of-interest b-box). ([GitHub][2])
* store ROI bounding box in meta.

### 11.3 Adaptive grids default

Use adaptive grid settings by default (per antmicro improvements). ([Antmicro][4])
Expose a deterministic configuration for:

* edge resolution vs diagonal resolution vs perpendicular resolution, consistent with gerber2ems config fields. ([GitHub][2])

### 11.4 Parallelism policy

* openEMS run is CPU-heavy; run **one openEMS job at a time** with controlled `numThreads`.
* Postprocessing (FFT, conversions) can be GPU-parallel across cases.

### 11.5 Storage policy

Because field dumps can be enormous (hundreds of GB), field exports must be:

* disabled by default,
* gated behind explicit flags,
* auto-cleaned unless `--keep-field-dumps` is set. ([GitHub][2])

---

## 12. API and CLI specification (exact interfaces)

### 12.1 Python API

Module: `m2_em_oracle/`

#### Core classes

* `OracleCase`

  * loads `oracle_case.json`, resolves paths, validates schema.
* `CaseFingerprint`

  * computes deterministic hash over:

    * all input file bytes (Gerbers, drill, stackup, pos.csv),
    * normalized config,
    * openEMS version/commit,
    * gerber2ems version,
    * container/env hash.
* `GeometryBuilder`

  * `build(case) -> GeometryArtifact`
* `SolverRunner`

  * `run(case, geometry) -> RawSimulationArtifact`
* `PostProcessor`

  * `compute_sparams(raw, case) -> SParameterArtifact`
* `Verifier`

  * `verify(case, sparams, raw, geometry) -> VerificationReport`
* `ArtifactWriter`

  * writes the final bundle.

#### Key function signature

```python
def run_oracle_case(case_dir: Path, *, force: bool=False) -> Path:
    """
    Runs full M2 pipeline and returns artifact directory path.
    Raises OracleValidationError / OracleRuntimeError on failure.
    """
```

### 12.2 CLI

Command group: `oracle`

* `oracle validate <case_dir>`

  * schema + file checks + port/copper overlap checks.

* `oracle build-geometry <case_dir>`

  * produces `geometry.xml` and mesh summary, no sim.

* `oracle simulate <case_dir>`

  * runs openEMS and saves raw waveforms.

* `oracle postprocess <case_dir>`

  * computes Touchstone outputs.

* `oracle verify <case_dir>`

  * runs full verification suite.

* `oracle run <case_dir>`

  * full pipeline with caching.

* `oracle cal run --all`

  * runs all CAL-* structures and produces a dashboard report.

---

## 13. Logging, metadata, and reproducibility (audit bundle spec)

Every artifact bundle must contain:

### 13.1 `meta.json` (minimum)

* `case_hash`
* `timestamp`
* `git_commit`
* `openems`: `{ version, commit, build_flags }`
* `gerber2ems`: `{ version, commit }`
* `host`: `{ cpu, ram, os }`
* `solver_policy`: boundary, end criteria, max steps, threads
* `grid_policy`: all mesh settings + computed λmin
* `ports`: logical-to-physical mapping, orientations, reference planes
* `frequency_grid`: exact frequencies used
* `run_times`: geometry build, sim time per excited port, postprocess time

### 13.2 `mesh_summary.json`

* mesh lines per axis,
* min/max Δ,
* cell ratio statistics,
* total cell count,
* estimated memory.

### 13.3 `verification_report.json`

* pass/fail for each check,
* numeric metrics,
* tolerances used,
* references to plots.

### 13.4 `solver_log.txt`

* raw openEMS stdout/stderr
* gerber2ems stdout/stderr

---

## 14. Risks and mitigations (known failure modes)

### 14.1 Port misplacement or incorrect direction

Mitigation:

* geometry-level overlap checks
* calibration suite checks (delay sign, reflect behavior)

### 14.2 Mesh-induced artifacts (staircasing, edge errors)

Mitigation:

* thirds rule + mesh invariance tests ([OpenEMS Wiki][5])

### 14.3 Boundary reflections polluting S-parameters

Mitigation:

* enforce clearance to PML region
* default PML_8; record clearance metrics ([OpenEMS Wiki][8])

### 14.4 Inadequate run time (timesteps)

Mitigation:

* enforce max_steps relative to excitation length guidance ([GitHub][2])

### 14.5 Storage blow-up from field dumps

Mitigation:

* disable by default; strict gating ([GitHub][2])

### 14.6 Licensing constraints

openEMS is GPLv3 (and CSXCAD LGPLv3) per documentation; gerber2ems is Apache-2.0. ([openEMS Documentation][1])
Mitigation:

* treat openEMS as an external tool dependency, not a linked library inside proprietary components.
* keep your proprietary “secret formula” tooling separate from GPL obligations if distributing.

---

## 15. How M2 maximizes probability of discovering a *novel equation*

M2 increases discovery odds by ensuring:

* **low-variance labels**: consistent S-parameters across cases (ports + mesh invariance),
* **traceable causality**: you can attribute any model failure to geometry, not oracle noise,
* **high throughput**: adaptive grids + ROI + GPU postprocessing means you can run more cases and support active learning loops,
* **physics fidelity**: calibration structures anchor the solver to known behaviors (delay, reflection), keeping symbolic regression from learning garbage invariants.

---

## 16. M2 “Absolute Quality Standard” Checklist (print this in the repo)

M2 is acceptable only when:

1. ✅ `oracle cal run --all` passes on a clean machine with pinned toolchain.
2. ✅ Port verification catches intentionally broken port cases (negative tests).
3. ✅ Mesh refinement invariance holds on at least 2 representative structures.
4. ✅ Reciprocity/energy checks pass (within stated tolerances) for passive reciprocal structures.
5. ✅ Full artifact bundles are generated and contain required metadata.
6. ✅ Re-running the same case yields identical frequency grid + near-identical S within numeric tolerance.
7. ✅ Performance benchmark does not regress (CAL-2 Thru within envelope).
8. ✅ Documentation includes:

   * port conventions (with diagrams),
   * calibration structures meaning + expected outcomes,
   * how to add a new case,
   * how to interpret verification failures.

If even one of these fails, M2 is not “gold,” and downstream equation discovery is at risk.

---

If you want the next step to be even more implementation-ready, I can provide a **repo-ready file tree** for M2 (exact module names, test directories, golden artifact locations) plus a **full set of calibration case manifests** (CAL-0…CAL-6) and the exact numeric tolerances you should start with on a laptop-class run budget.

[1]: https://docs.openems.de/intro.html "https://docs.openems.de/intro.html"
[2]: https://github.com/antmicro/gerber2ems "https://github.com/antmicro/gerber2ems"
[3]: https://docs.openems.de/python/openEMS/openEMS.html "https://docs.openems.de/python/openEMS/openEMS.html"
[4]: https://antmicro.com/blog/2025/07/recent-improvements-to-antmicros-signal-integrity-simulation-flow/ "https://antmicro.com/blog/2025/07/recent-improvements-to-antmicros-signal-integrity-simulation-flow/"
[5]: https://wiki.openems.de/index.php/FDTD_Mesh.html "https://wiki.openems.de/index.php/FDTD_Mesh.html"
[6]: https://antmicro.com/blog/2024/07/automated-pcb-trace-selection-for-si-simulation/ "https://antmicro.com/blog/2024/07/automated-pcb-trace-selection-for-si-simulation/"
[7]: https://wiki.openems.de/index.php/Ports.html "https://wiki.openems.de/index.php/Ports.html"
[8]: https://wiki.openems.de/index.php/FDTD_Boundary_Conditions.html "https://wiki.openems.de/index.php/FDTD_Boundary_Conditions.html"
[9]: https://scikit-rf.readthedocs.io/en/latest/examples/networktheory/Renormalizing%20S-parameters.html "https://scikit-rf.readthedocs.io/en/latest/examples/networktheory/Renormalizing%20S-parameters.html"
[10]: https://scikit-rf.readthedocs.io/en/latest/examples/mixedmodeanalysis/Mixed%20Mode%20Basics.html "https://scikit-rf.readthedocs.io/en/latest/examples/mixedmodeanalysis/Mixed%20Mode%20Basics.html"
[11]: https://coppermountaintech.com/wp-content/uploads/2018/05/Design-and-Fabrication-of-a-TRL-Calibration-Kit.pdf "https://coppermountaintech.com/wp-content/uploads/2018/05/Design-and-Fabrication-of-a-TRL-Calibration-Kit.pdf"


# M3 Design Document — Data and Experiment Backbone

**Dataset versioning + pipeline DAGs + artifact store** (single‑laptop, GPU‑first, high‑rigor)

This document specifies the *complete* M3 implementation: how the repository will **store, version, index, reproduce, and audit** every dataset, simulation, fit, and discovered formula so we can run thousands of iterations without losing lineage, wasting compute, or accidentally “discovering” an artifact of sloppy data handling.

M3 is not “just experiment tracking.” It is the **foundational substrate** that makes every later milestone (M4 vector fitting, M6 symbolic regression, M7 falsification, M8 agents, M9 measurement) *scientifically defensible* and *engineering‑grade*.

---

## 0) Mission binding: why M3 exists in this project

We are building a “formula foundry” whose outputs must survive corporate scrutiny:

* A discovered symbolic law must be **traceable** down to:

  1. the exact geometry parameters (coupon design vector),
  2. the exact solver configuration/mesh/ports,
  3. the exact S‑parameter dataset used,
  4. the exact rational fit settings and passivity gates,
  5. the exact SR/search settings, and
  6. the exact falsification adversaries and failure cases.

* We must support **rapid iterative loops** (active learning, falsification, tournament selection) without recomputing expensive stages. This is how we stay within a single laptop’s physical constraints.

* We must enforce a **“no orphan artifact”** rule: nothing exists unless it is referenced by a dataset snapshot and a run record.

---

## 1) Constraints and design goals

### 1.1 Hardware & execution reality

* Single machine.
* 2.5 TB SSD (fast, finite).
* 32 GB RAM (moderate).
* GPU is abundant vs CPU for search loops; EM solver is CPU‑dominant.
* We need storage policies that keep the SSD from filling and pipelines from degenerating into “rehash everything” operations.

### 1.2 “Strict rigor” goals

* Deterministic and/or *numerically stable* reproducibility.
* Full provenance & lineage graph: **formula → fitted model → S‑parameters → geometry → commit + environment**.
* Clear definition of done (DoD) with **automated verification** and **performance budgets**.

### 1.3 Tooling choices (primary)

**DVC** for data versioning + pipeline DAGs + cache/dedup primitives. DVC pipelines are configured in `dvc.yaml` and lock reproducible dependencies/outputs via `dvc.lock`. ([Data Version Control · DVC][1])
DVC’s cache is **content‑addressable storage** in `.dvc/cache`. ([Data Version Control · DVC][2])
We will tune link types to maximize efficiency (prefer reflinks). ([Data Version Control · DVC][3])
We will use `dvc gc` for garbage collection of unused cached data. ([Data Version Control · DVC][4])

**MLflow** for experiment tracking metadata (params/metrics/tags), with a **local SQLite backend store** and a **local filesystem artifact store** (small artifacts only). MLflow supports configuring backend store and artifact store locations. ([MLflow][5])

**A project‑local Artifact Registry (SQLite)** (ours) for ultra‑fast indexing/querying of artifacts, *derived entirely from manifests*, so it can be rebuilt deterministically.

> Why not rely on just one tool?
> DVC excels at reproducible data/pipeline + dedup cache. MLflow excels at “experiment table” views and metadata. Our registry gives low‑latency queries and integrity checks at scale without forcing DVC to scan the whole tree repeatedly.

---

## 2) M3 deliverables

### 2.1 Repository structure (canonical)

```
repo/
  dvc.yaml
  dvc.lock
  params.yaml
  .dvc/
    cache/                 # DVC CAS
    config                 # DVC settings (cache dir, link type, remotes)
  mlruns/                  # MLflow default artifact root (small artifacts only)
  mlruns.db                # MLflow SQLite backend store (or configurable path)
  data/
    README.md              # contracts for datasets
    registry/              # dataset snapshots (DVC-tracked)
      datasets/
        <dataset_id>/
          dataset.json
          index.parquet
          splits.json
          manifest.sha256
    objects/               # artifact objects (DVC-tracked or DVC outputs)
      em_sparams/
      em_logs/
      vf_models/
      features/
      sr_candidates/
      reports/
  runs/
    <run_id>/
      run.json
      stdout.log
      stderr.log
      env.json
      pointers.json
  tools/
    m3/
      cli.py
      schemas/
      storage.py
      registry.py
      verify.py
      lineage.py
```

**Key point:**

* `data/registry/**` is the *source of truth for dataset snapshots*.
* `data/objects/**` is the *artifact object store* (large, DVC‑managed).
* `runs/**` is lightweight run metadata and pointers; it is Git‑tracked (or small enough to be).

### 2.2 CLI and API surface (M3 “public interface”)

#### CLI

* `m3 init`
  Initializes:

  * DVC project structure (if not already) and configures cache/link type.
  * MLflow tracking URI to local SQLite file.
  * Creates schema files and verification scaffolding.

* `m3 run <pipeline> --params ...`
  A wrapper that:

  * stamps a `run_id`,
  * sets deterministic env vars,
  * runs `dvc repro` (or a specified stage),
  * logs to MLflow,
  * writes `runs/<run_id>/run.json`.

* `m3 dataset create --name <...> --from <...>`
  Creates a dataset snapshot (content‑addressed `dataset_id`) that references artifacts by ID.

* `m3 dataset diff <dataset_a> <dataset_b>`
  Reports exact added/removed samples + derived artifacts.

* `m3 artifact show <artifact_id>`
  Shows manifest, parents, children, file sizes, hashes.

* `m3 verify`
  Runs the full M3 verification suite (integrity + reproducibility gates + performance checks).

* `m3 gc --policy <...>`
  Applies retention policy and triggers `dvc gc` appropriately. ([Data Version Control · DVC][4])

#### Python API

* `ArtifactStore.put(manifest, files...) -> artifact_id`
* `ArtifactStore.get(artifact_id) -> paths + manifest`
* `DatasetSnapshot.load(dataset_id) -> table`
* `LineageGraph.trace(artifact_id | dataset_id | run_id)`

---

## 3) Core data model

M3 defines **three fundamental ID types**:

### 3.1 `spec_id` (pre‑execution identity)

A deterministic hash of the *inputs* to a computation, including:

* parent artifact IDs,
* parameter dict (canonicalized),
* tool versions (solver/vectorfit/SR),
* schema version,
* relevant environment knobs.

**Purpose:** caching / reuse / “should I recompute?”.

### 3.2 `content_hash` (post‑execution integrity)

A deterministic hash of the *produced content bytes*, computed after writing outputs.

**Purpose:** integrity + dedup across specs (if any) + detecting nondeterminism.

### 3.3 `artifact_id` (primary reference)

For M3, **artifact_id = content_hash** (content‑addressed store).
We retain `spec_id` in the manifest as the intended computation identity.

**Why this matters:**
If later we rerun the same `spec_id` and produce a different `content_hash`, that is a *hard failure* unless explicitly allowed by a “numeric tolerance” policy (described below).

---

## 4) Artifact taxonomy (what we store)

We store artifacts as **immutable** objects, each with:

* `manifest.json` (validated by JSON Schema)
* one or more data files
* optional `meta/` (logs, plots, etc.)

### 4.1 Artifact types (initial set)

1. **Geometry / manufacturing**

* `kicad_project` (zipped project or directory)
* `gerbers` (zipped)
* `drill_files`
* `coupon_manifest` (the design vector `x`, constraints, connector type)

2. **EM oracle outputs**

* `sparams_touchstone` (`.s2p`, `.s4p`, etc.)
* `sparams_binary` (NPZ with complex64/complex128 arrays, plus frequency grid)
* `em_run_log` (compressed)
* `em_config` (ports, boundary conditions, mesh parameters)

3. **Derived network objects**

* `network_features` (normalized features, dimensionless groups, etc.)
* `vectorfit_model` (poles/residues; JSON + NPZ)
* `passivity_report` (metrics and pass/fail)

4. **Symbolic regression & discovery**

* `sr_search_trace` (compressed trace of candidate evaluations)
* `formula_candidate` (SymPy expression, codegen outputs, constraints status)
* `falsification_counterexample_set` (adversarial points + failures)

5. **Reports**

* `benchmark_report` (HTML/PDF/Markdown)
* `comparison_plots` (PNG/SVG; small)

### 4.2 Artifact location scheme

Artifacts are stored under:

```
data/objects/<type>/<prefix>/<artifact_id>/
  manifest.json
  payload/... (files)
```

Where `prefix` is first 2–3 hex characters of `artifact_id` to avoid giant directories.

---

## 5) Manifest schemas (executable contracts)

Every artifact must include a manifest with **strict schema versioning**.

### 5.1 Shared manifest fields (required)

```json
{
  "schema_version": "m3.artifact.v1",
  "artifact_id": "blake3:...",
  "content_hash": "blake3:...",
  "spec_id": "blake3:...",
  "type": "sparams_binary",
  "created_at_utc": "2026-01-19T00:00:00Z",
  "producer": {
    "name": "openems_runner",
    "version": "git:<commit>",
    "container_image": "sha256:<digest>"
  },
  "parents": [
    {"artifact_id": "blake3:...", "role": "geometry"},
    {"artifact_id": "blake3:...", "role": "em_config"}
  ],
  "files": [
    {
      "relative_path": "payload/sparams.npz",
      "bytes": 123456,
      "sha256": "...",
      "mime": "application/octet-stream"
    }
  ],
  "units": {
    "frequency": "Hz"
  },
  "parameters": { "..." : "..." },
  "quality": {
    "status": "pass|fail",
    "checks": [
      {"name": "schema_valid", "status": "pass"},
      {"name": "hash_match", "status": "pass"}
    ]
  }
}
```

### 5.2 Numeric determinism policy fields

For artifacts produced by solvers or floating computations, manifest includes:

```json
"determinism": {
  "class": "bitwise|tolerance",
  "tolerance_policy": {
    "metric": "max_abs_error",
    "threshold": 1e-9,
    "reference_artifact_id": "blake3:..."
  }
}
```

* **bitwise**: content_hash must match reference exactly.
* **tolerance**: content_hash may differ, but numeric equivalence gates must pass vs a reference.

> Note: M3 does not decide the numeric tolerance thresholds; it enforces that *a threshold exists* and is applied consistently.

---

## 6) Dataset snapshots (versioned, queryable)

A **dataset snapshot** is a content‑addressed object describing an immutable set of samples.

### 6.1 Dataset snapshot structure

```
data/registry/datasets/<dataset_id>/
  dataset.json
  index.parquet
  splits.json
  manifest.sha256
```

* `dataset.json` (small, human‑readable):

  * dataset name, domain (“via_transition_v1”)
  * creation metadata
  * pointers to `index.parquet`
  * parent dataset IDs (lineage)
* `index.parquet`:

  * the authoritative table of sample rows (fast to query in Python/pandas)
  * each row references artifact IDs for required pieces

Example required columns for an “EM S‑params dataset”:

* `sample_id` (stable: e.g., blake3 of coupon manifest)
* `geometry_artifact_id`
* `em_config_artifact_id`
* `sparams_artifact_id`
* `freq_grid_id` (if shared)
* `tags` (stackup, connector, diff/se, etc.)
* `created_at_utc`

### 6.2 Dataset ID definition

`dataset_id = blake3(canonical(dataset.json sans timestamps) + canonical(index.parquet content hash))`

This makes dataset IDs stable across machines and ensures **exact dataset identity** for downstream SR.

---

## 7) Pipeline DAGs (DVC as the canonical DAG executor)

We use **DVC pipelines** as the authoritative definition of deterministic transformations:

* Each stage declares deps/outs/metrics/params in `dvc.yaml`. ([Data Version Control · DVC][1])
* Stage outputs are cached via DVC’s content‑addressable cache. ([Data Version Control · DVC][2])
* Metrics/plots/params are first‑class for comparisons and can be integrated into experiments. ([Data Version Control · DVC][6])

### 7.1 DVC pipeline design principles

1. **Stages must be pure**
   Given the same deps + params + tool versions, they produce the same outputs (bitwise or within defined tolerances).

2. **No implicit inputs**
   No stage may read from:

* user home directories
* system temp directories
* the network (unless explicitly declared and hashed)
* uncontrolled environment variables

3. **Outputs must be deterministic in shape and location**
   DVC does not support wildcard outputs; stages must output to known paths. For dynamic outputs, track a directory output. ([DVC Community Forum][7])

4. **Shard outputs to avoid pathological file counts**
   DVC can become slow with extremely large numbers of files (hundreds of thousands to millions). ([GitHub][8])
   We therefore shard and/or bundle artifacts in predictable ways (see §10).

### 7.2 Canonical `dvc.yaml` skeleton (M3 baseline)

```yaml
stages:
  build_param_sweep:
    cmd: python -m tools.sweep.generate --params params.yaml --out data/tmp/sweep_spec.jsonl
    deps:
      - tools/sweep/generate.py
      - params.yaml
    outs:
      - data/tmp/sweep_spec.jsonl

  generate_geometry:
    cmd: python -m tools.geom.batch --spec data/tmp/sweep_spec.jsonl --out data/tmp/geom_out/
    deps:
      - data/tmp/sweep_spec.jsonl
      - tools/geom/
    outs:
      - data/tmp/geom_out/

  run_openems:
    cmd: python -m tools.em.run_batch --in data/tmp/geom_out/ --out data/tmp/em_out/
    deps:
      - data/tmp/geom_out/
      - tools/em/
    outs:
      - data/tmp/em_out/

  register_dataset:
    cmd: python -m tools.m3.register --em data/tmp/em_out/ --out data/registry/datasets/
    deps:
      - data/tmp/em_out/
      - tools/m3/
    outs:
      - data/registry/datasets/
    metrics:
      - data/registry/metrics/dataset_build.json
```

**Policy:** `data/tmp/**` is a staging area; only `data/registry/**` and `data/objects/**` become canonical.

---

## 8) Artifact store implementation details (the “object database”)

### 8.1 Storage backend

* Primary: local filesystem under `data/objects/**`
* Managed by DVC (outs tracked in pipeline stages)
* DVC cache provides dedup and supports efficient link types (reflinks preferred). ([Data Version Control · DVC][3])

### 8.2 DVC cache configuration (performance + safety)

DVC supports different file link types for cache efficiency; reflink is preferred when available, with hardlink/symlink as alternatives depending on filesystem layout. ([Data Version Control · DVC][3])

**We set:**

* `cache.type = reflink,copy` (prefer reflink, fallback to copy)
* If cache moved to a different filesystem: `symlink,copy` may be appropriate. ([Data Version Control · DVC][3])

**Why:**
Reflinks provide copy‑on‑write safety and avoid cache corruption risks from in‑place edits. ([Data Version Control · DVC][3])

### 8.3 Remote storage (optional, future‑proof)

Even on a single laptop, you want an abstraction for “remote”:

* external SSD
* NAS
* S3/minio

DVC supports remotes and can garbage collect remote objects with `dvc gc --cloud`. ([Data Version Control · DVC][4])
**Default** for M3: local remote path (e.g., external drive mount) for backups.

### 8.4 Atomic writes (mandatory)

To prevent partial artifacts from being referenced:

1. write to `.../<artifact_id>.tmp/`
2. fsync files
3. write manifest last
4. rename to `.../<artifact_id>/` (atomic rename on same filesystem)
5. only then update dataset snapshot / registry

### 8.5 Immutability enforcement

Any attempt to modify files under an artifact directory must fail:

* enforce read‑only permissions post‑commit
* verify content_hash matches stored hash in `m3 verify`

---

## 9) Experiment tracking (MLflow integration)

### 9.1 What MLflow is used for (and what it is NOT used for)

**MLflow used for:**

* run tables: params/metrics/tags
* quick comparisons across SR seeds, objective weights, model orders
* storing small artifacts: plots, small JSON summaries, candidate formula text

MLflow provides tracking and an artifact store concept; it defaults to local filesystem but supports other storage backends. ([MLflow][9])

**MLflow NOT used for:**

* storing large EM datasets, Touchstone corpora, or heavy binary arrays
  (because MLflow doesn’t dedup like DVC; you will blow the SSD).

### 9.2 MLflow backend store: SQLite (local)

MLflow can log to a local SQLite backend store via `MLFLOW_TRACKING_URI=sqlite:///...`. ([MLflow][10])
If we run an MLflow server, backend store URI can be configured with `--backend-store-uri`. ([MLflow][5])

### 9.3 MLflow run metadata contract

Every MLflow run must include tags:

* `git_commit`
* `dvc_lock_hash` (hash of `dvc.lock`)
* `dataset_id_in`
* `dataset_id_out`
* `pipeline_stage` or `campaign_id`
* `container_digest`

And must log:

* `metrics.json` (machine‑readable)
* `pointers.json` (artifact IDs produced)

This bridges MLflow’s view to the DVC‑managed object store.

---

## 10) Scaling strategy: file count, sharding, and I/O efficiency

### 10.1 Why we care

DVC and filesystem operations degrade with huge numbers of files; DVC has reported issues with very large file counts (e.g., millions of files) being slow or problematic. ([GitHub][8])
Even if we never reach millions, we design so we *don’t paint ourselves into a corner*.

### 10.2 Sharding policy (mandatory)

We define *artifact payload policies* by type:

#### Type A: “small per‑sample objects” (default)

* `sparams_binary` per sample (NPZ)
* `vectorfit_model` per sample (JSON+NPZ)
* `coupon_manifest` per sample

This is fine up to ~50k–150k samples (depending on file count per sample).

#### Type B: “bundled shards”

For high‑volume derived outputs (e.g., features used for SR loops):

* bundle into deterministic shards:

  * shard size: e.g., 1024 samples
  * shard file: `features_shard_<shard_idx>.npz.zst`
  * include an index table mapping sample_id → offset

**Rule:** shards are immutable. Add new samples ⇒ create new shard, never rewrite old shards.

#### Type C: “debug‑heavy logs”

EM logs can be large and numerous; store:

* a compressed log per sample only when needed
* otherwise store:

  * a summary (`stderr_tail`, warnings, solver status, runtime)
  * and keep full logs only for failed / selected samples

### 10.3 Memory layout for GPU‑first evaluation loops

For SR/falsification you’ll repeatedly evaluate many samples on GPU. Therefore:

* prefer storing dense numeric arrays in contiguous formats that load quickly (NPZ/NPY)
* maintain a “compiled dataset” artifact:

  * `compiled_features_v1` that contains:

    * `X` (float32)
    * `Y` (float32/complex64)
    * plus index arrays
  * chunked by shard for VRAM‑safe streaming

**Goal:** minimize Python object overhead and disk seeks, maximize sequential reads.

---

## 11) Lineage graph (auditable provenance)

M3 must be able to answer, deterministically:

* “Which exact S‑parameter samples produced this formula?”
* “Which openEMS settings produced those S‑parameters?”
* “Which geometry parameters define those samples?”
* “Which commit and container image created this?”

### 11.1 Lineage graph model

We represent lineage as a DAG of nodes:

* artifact nodes (`artifact_id`)
* dataset nodes (`dataset_id`)
* run nodes (`run_id`)

Edges:

* artifact → artifact (parent relationships)
* dataset → artifact (contains)
* run → artifact (produced)
* run → dataset (output)
* dataset → dataset (derived_from)

### 11.2 Where lineage is stored

* Canonical lineage is implicit via manifests + dataset index.
* A derived `lineage.sqlite` index is rebuilt from manifests (fast queries).
* A derived `lineage.graphml` export can be generated for visualization.

**Never** make the SQLite index the source of truth.

---

## 12) Integrity, validation, and “no orphan artifacts” enforcement

### 12.1 Invariants (hard requirements)

1. Every artifact directory contains a valid `manifest.json`.
2. Every `artifact_id` in the manifest matches:

   * directory name
   * content hash computed by `m3 verify`
3. Every dataset snapshot references only existing artifact IDs.
4. Every artifact is reachable from at least one dataset snapshot **OR** is explicitly pinned by a retention policy (e.g., “keep failed logs for 14 days”).
5. Every run references:

   * git commit
   * dvc lock hash
   * environment stamp

### 12.2 Schema validation

* JSON schema files live in `tools/m3/schemas/`
* `m3 verify` validates every manifest and dataset JSON

### 12.3 Tamper detection

* each dataset snapshot has `manifest.sha256`
* each artifact file entry has `sha256`
* optional future: sign dataset snapshots (Ed25519) if you start sharing externally

---

## 13) Caching & recomputation rules (compute efficiency)

### 13.1 Stage caching (DVC)

DVC will skip stage recomputation when deps/outs unchanged, because the pipeline is defined in `dvc.yaml` and locked in `dvc.lock`. ([Data Version Control · DVC][1])

### 13.2 Artifact reuse (M3 internal)

Before running a stage for a spec:

* compute `spec_id`
* query registry: do we already have an artifact with this spec_id and matching tool versions?
* if yes: reuse and skip compute
* if no: compute and store

### 13.3 “Recompute on tool drift”

If any of these change, `spec_id` changes:

* solver version
* vector fitting implementation version
* SR engine version
* schema version
* critical params

This is how we prevent “silent invalid reuse.”

---

## 14) Retention, eviction, garbage collection

### 14.1 Storage tiers

* **Tier 0 (ephemeral)**: `data/tmp/**` — always safe to delete.
* **Tier 1 (canonical)**: `data/registry/**`, `data/objects/**` — must be reproducible and versioned.
* **Tier 2 (derived indices)**: SQLite registries — rebuildable.

### 14.2 DVC garbage collection

We rely on `dvc gc` to remove unused objects from local cache and optionally remote when configured. ([Data Version Control · DVC][4])

### 14.3 Pinning policy

Define:

* “pinned datasets” (e.g., baseline corpora used in papers/demos)
* “pinned runs” (best formula candidates)
  Pinned means:
* cannot be removed by `m3 gc`
* requires explicit unpin

### 14.4 Space budget enforcement

`m3 gc --policy laptop_default` enforces:

* keep last N datasets of each campaign
* keep all datasets that are ancestors of pinned datasets
* keep failed EM logs for X days
* keep all “published” artifacts forever

---

## 15) Concurrency, locking, and resumability

### 15.1 Requirements

* Multiple processes may run concurrently (CPU openEMS + GPU SR).
* Artifact creation must be safe under concurrency.
* Partial runs must be resumable and must not corrupt the store.

### 15.2 Locking strategy

* global lock file: `data/.locks/artifact_store.lock` for store‑wide metadata updates
* per‑artifact lock on spec_id during creation to avoid duplicate work
* atomic rename commit pattern (see §8.4)

### 15.3 Resumability

* A run writes `runs/<run_id>/run.json` early.
* Each stage appends progress records.
* On restart, `m3 run` can detect incomplete stage outputs and either:

  * resume (if safe), or
  * delete staging and recompute.

---

## 16) Verification: how we know M3 is “done” and high‑quality

This is the strict, automated Definition of Done.

### 16.1 M3 Definition of Done (DoD)

M3 is complete only when all items below are implemented and **`m3 verify` passes on CI and on a fresh clone**.

#### A) Correctness & integrity gates

1. **Schema compliance:** 100% of artifacts and dataset snapshots validate.
2. **No orphan artifacts:** every artifact is either referenced by a dataset snapshot or pinned.
3. **Hash correctness:** computed content hashes match manifest for a sampled set and for all small artifacts.
4. **Lineage closure:** tracing a formula candidate to raw S‑params must always succeed and must return:

   * geometry manifest
   * EM config
   * solver logs summary
   * commit + container digest
5. **Determinism check (smoke):**

   * rerun a tiny pipeline twice → identical dataset_id
   * for artifacts with `determinism.class=bitwise`, identical content_hash

#### B) Reproducibility gates

6. **Rebuild from scratch:** On a clean workspace:

   * `dvc pull` (if remote configured) + `dvc repro` produces the same dataset snapshot IDs.
7. **Cache correctness:** repeated `dvc repro` triggers no recomputation when inputs unchanged.

#### C) Performance gates (single‑laptop realism)

8. **Index query latency:** `m3 dataset diff` and `m3 artifact show` must run in:

   * < 200 ms for 10k samples (warm cache)
   * < 1 s for 100k samples (warm cache)
9. **Artifact ingest throughput:** storing N=1000 small artifacts must exceed a minimum throughput (define per‑type).
10. **Garbage collection:** `m3 gc` must reclaim space and leave store consistent; `dvc gc` invocation documented and tested. ([Data Version Control · DVC][4])

#### D) Operational gates

11. **Crash safety:** simulate power‑loss mid‑artifact write → no broken artifact is referenced (atomic commit).
12. **Audit report:** `m3 audit <formula_candidate_id>` generates a deterministic report listing all ancestors and key metrics.

### 16.2 The `m3 verify` test suite (mandatory)

`m3 verify` runs:

1. **Schema validation**

* validate all manifests against JSON schema
* validate dataset snapshots and parquet schema

2. **Store integrity scan**

* verify directory structure
* verify referenced files exist
* verify recorded sizes

3. **Hash audit**

* recompute sha256 for a stratified sample of large artifacts
* recompute all hashes for small artifacts
* recompute dataset snapshot manifest hashes

4. **Lineage audit**

* for a chosen set of artifacts (at least one per type), trace to roots and ensure required roles exist (geometry, config, oracle output)

5. **Pipeline cache audit**

* run `dvc repro` twice on a small pipeline and verify no stages rerun the second time

6. **GC dry‑run**

* compute what would be deleted and ensure pinned objects are preserved
* optionally run in CI with a sandbox store

---

## 17) DVC “artifact metadata” feature usage (optional, but recommended)

DVC supports an `artifacts:` section in `dvc.yaml` to declare structured metadata about artifacts. ([Data Version Control · DVC][1])
DVC can also download artifacts by name via `dvc artifacts get`. ([Data Version Control · DVC][11])

**How we use this:**

* For *published* artifacts (best formula candidates, demo models), we declare them in the root `dvc.yaml` `artifacts:` section so they’re easy to fetch and reference in external demos.
* We do **not** declare every single artifact here (too many); this is only for curated outputs.

---

## 18) Implementation plan (engineering tasks)

### Phase 1 — Foundation (weekend‑scale, but rigorous)

1. Create schemas (`artifact.v1`, `dataset.v1`, `run.v1`).
2. Implement `ArtifactStore` with atomic writes + content hash + manifest.
3. Implement `DatasetSnapshot` writer/reader with parquet index.
4. Implement basic `LineageGraph` tracer from manifests.

### Phase 2 — Tool integration

5. Configure DVC:

   * cache type preference: reflink then copy ([Data Version Control · DVC][3])
   * remote optional
6. Configure MLflow:

   * SQLite backend store path and local artifact root ([MLflow][10])
7. Implement `m3 run` wrapper that stamps run metadata + logs pointers.

### Phase 3 — Verification & hardening

8. Implement `m3 verify` end‑to‑end gates.
9. Add CI jobs:

   * unit tests (schema, hashing)
   * integration tests (tiny pipeline with DVC)
10. Implement retention policy + `m3 gc` calling `dvc gc` safely. ([Data Version Control · DVC][4])

---

## 19) How M3 directly increases odds of discovering a novel equation

A “novel equation” is only valuable if it is:

* **real** (not a bug),
* **robust** (not a cherry‑picked domain),
* **reproducible** (can be re‑run and verified),
* **auditable** (a customer can trust the provenance),
* **efficient** (you can iterate fast enough to actually search).

M3’s backbone is what lets you:

* run falsification tournaments without losing the counterexamples,
* compare formulas across dataset versions and *prove* which dataset created which improvement,
* and most importantly: **avoid wasting compute** by preventing duplicate sims and by enabling precise caching and lineage.

---

## 20) Minimal “gold standard” demo of M3 (what you should be able to do immediately)

After M3 is complete, the following must work on a fresh clone:

1. `m3 init`
2. `m3 run em_dataset_smoke`
   (creates a tiny dataset: e.g., 10 coupons)
3. `m3 dataset show <dataset_id>`
4. `m3 dataset diff <dataset_id> <dataset_id>` (no changes)
5. `m3 verify` (all green)
6. delete `data/tmp/**`, rerun `m3 run em_dataset_smoke` → no recompute of already‑stored objects
7. `m3 gc --policy laptop_default` reclaims space without breaking lineage

If all of that holds, M3 is truly “done” in the sense that the repository can now scale to thousands of runs without collapsing into ambiguity.

---

If you want, I can also provide **concrete JSON Schemas**, a fully fleshed **reference `dvc.yaml`**, and a proposed **SQLite registry schema** (tables + indexes + queries) so implementation is copy‑pasteable into the repo.

[1]: https://doc.dvc.org/user-guide/project-structure/dvcyaml-files?utm_source=chatgpt.com "dvc.yaml Files | Data Version Control · DVC"
[2]: https://doc.dvc.org/user-guide/project-structure/internal-files?utm_source=chatgpt.com "Internal Directories and Files"
[3]: https://doc.dvc.org/user-guide/data-management/large-dataset-optimization?utm_source=chatgpt.com "Large Dataset Optimization | Data Version Control · DVC"
[4]: https://doc.dvc.org/command-reference/gc?utm_source=chatgpt.com "gc | Data Version Control · DVC"
[5]: https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/?utm_source=chatgpt.com "Backend Stores"
[6]: https://doc.dvc.org/start/data-pipelines/metrics-parameters-plots?utm_source=chatgpt.com "Get Started: Metrics, Plots, and Parameters"
[7]: https://discuss.dvc.org/t/how-to-handle-dynamic-number-of-outputs/1724?utm_source=chatgpt.com "How to handle dynamic number of outputs?"
[8]: https://github.com/iterative/dvc/issues/7681?utm_source=chatgpt.com "repro: DVC is slow with million of files · Issue #7681"
[9]: https://mlflow.org/docs/latest/ml/tracking/?utm_source=chatgpt.com "MLflow Tracking"
[10]: https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database/?utm_source=chatgpt.com "Tracking Experiments with Local Database"
[11]: https://doc.dvc.org/command-reference/artifacts/get?utm_source=chatgpt.com "artifacts get - DVC Documentation - Data Version Control"

