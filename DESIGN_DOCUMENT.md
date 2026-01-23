# M4 Design Document

**Milestone:** M4 — Network processing oracle

Network processing oracle: Vector Fitting + Passivity/Causality enforcement + SPICE export

This document specifies Milestone M4 as a production-grade "network processing oracle" that turns raw frequency-domain network data (primarily S-parameters from openEMS / measurement) into compact, stable, passive, causal macromodels with auditable guarantees and exportable SPICE subcircuits. This milestone is a hard prerequisite for the overarching mission: discovering a novel parametric equation for network behavior by learning how rational-model parameters (poles/residues/sections) vary with geometry.

The central engineering idea is:

Every EM-simulated (or measured) coupon becomes a “well-posed rational object” (passive/causal/stable) that downstream symbolic regression can safely learn from.

## 1. Mission alignment and why M4 is non-negotiable
### 1.1 Role in the full discovery pipeline

M4 provides the canonical intermediate representation that makes the “formula foundry” possible:

Raw network 
𝑆
(
𝑓
)
S(f) → (causality screening/correction) → Vector Fitted Rational Macromodel → (passivity enforcement) → Guaranteed-stable & passive model → SPICE export + feature extraction for equation discovery

Vector fitting is a standard rational approximation approach used to convert sampled frequency responses (S/Y/Z matrices) into rational models suitable for circuit simulation.

### 1.2 Why “oracle-grade” rigor matters for formula discovery

Symbolic regression is adversarial in practice: it will exploit any inconsistency, non-causality, or numerical artifact. If macromodels are not strictly passive/causal/stable, you get:

False “laws” that encode solver/measurement artifacts

Instability when embedding candidate models into larger circuits

Pole/residue discontinuities that destroy learnability

M4 therefore must be designed as a gatekeeper:

Accept only models meeting strict guarantees

Reject or repair anything questionable, with traceable diagnostics

## 2. Definitions and “physics legality” requirements
### 2.1 Rational macromodel form

We represent an 
𝑁
N-port response as a rational model (common-pole multiport form):

𝐻
(
𝑠
)
=
𝐷
+
𝑠
𝐸
+
∑
𝑘
=
1
𝐾
𝑅
𝑘
𝑠
−
𝑝
𝑘
H(s)=D+sE+
k=1
∑
K
	​

s−p
k
	​

R
k
	​

	​


For S-parameters we generally enforce 
𝐸
=
0
E=0 (see passivity tooling constraints). The scikit-rf vector fitting tutorial and API describe constant/proportional terms and pole-residue structure.

### 2.2 Stability (causality prerequisite)

Causal LTI systems require stable dynamics in the macromodel:

All poles must satisfy 
ℜ
(
𝑝
𝑘
)
<
0
ℜ(p
k
	​

)<0

Complex poles occur in conjugate pairs

Vector fitting frameworks assume/produce conjugate pairing.

### 2.3 Passivity requirement for scattering models

For scattering 
𝑆
(
𝑗
𝜔
)
S(jω), passivity is equivalent to all singular values ≤ 1 for all frequencies:

𝜎
𝑖
(
𝑆
(
𝑗
𝜔
)
)
≤
1
∀
𝜔
,
 
∀
𝑖
σ
i
	​

(S(jω))≤1∀ω,∀i

This criterion is stated in the passivity enforcement literature and used by scikit-rf’s passivity test/enforcement machinery.

### 2.4 Causality requirement for tabulated data

Measured/simulated bandlimited S-parameters can be non-causal due to calibration, de-embedding, noise, and other effects. A robust causality check must account for finite bandwidth and out-of-band uncertainty. Triverio proposes a filtered Fourier transform method with a rigorous truncation error bound.

## 3. Scope, non-scope, and milestone contract
### 3.1 In scope (M4 must deliver)

Network ingestion

Touchstone (.sNp), plus internal numpy/cupy arrays from openEMS postprocessing

Metadata: port reference impedances 
𝑧
0
z
0
	​

, S-definition if applicable

Preprocessing

Frequency grid validation + optional resampling/decimation

Optional DC extrapolation / low-frequency conditioning (controlled, logged)

Reciprocity/symmetry checks (report-only in M4; enforcement optional)

Causality screening

Robust causality check with finite-bandwidth error bounds

Optional repair operations (configurable): delay alignment, time-gating, band extension heuristics (see §7)

Vector fitting

Multiport common-pole rational fitting

Deterministic initial poles policy + convergence controls

Automatic model order option (“adding & skimming”)

Passivity evaluation

Exact band detection for passivity violations using the half-size test matrix (reciprocal models)

Secondary dense frequency sampling check (belt-and-suspenders)

Passivity enforcement

Singular-value perturbation approach (with optional DC preservation)

Exports

macromodel.json (canonical pole-residue form + metadata)

macromodel.npz (fast binary form)

macromodel.sp (SPICE subcircuit, simulator-compatible)

quality_report.json (all metrics + pass/fail gates)

Auditable “oracle decision”

Every run ends in: ACCEPTED, REPAIRED_AND_ACCEPTED, or REJECTED

With explicit reasons and reproducing artifacts

### 3.2 Non-scope (explicitly deferred)

Full “parametric vector fitting” across geometry space (that is M6/M7-level)

Advanced Y/Z passivity enforcement tooling beyond scattering passivity (M4 supports conversion/reporting, but enforcement is S-only by default)

Measurement de-embedding standards implementation (M9)

## 4. External dependencies and baseline references
### 4.1 Baseline library: scikit-rf VectorFitting

We will treat scikit-rf as the baseline implementation for:

Vector fitting routine

Automatic order selection (auto_fit)

Passivity test (passivity_test) based on half-size test matrix

Passivity enforcement (passivity_enforce) via singular value perturbation

SPICE export (write_spice_subcircuit_s) via state-space circuit synthesis

scikit-rf VectorFitting provides vector fitting + passivity evaluation/enforcement + SPICE circuit export.

Important limitation to codify: scikit-rf passivity evaluation/enforcement is currently scattering-only, and raises errors for nonzero proportional terms in the model.

### 4.2 Key algorithms we rely on

Vector fitting (broadly documented in Triverio’s tutorial paper)

Half-size passivity assessment for S-parameter rational models (efficient eigenvalue-based identification of violating bands). scikit-rf explicitly uses it for reciprocal models.

Passivity enforcement via singular value perturbation (Deschrijver/Dhaene; DC-preserving variant) as implemented by scikit-rf.

Robust causality check via filtered inverse Fourier transform with truncation error bounds.

## 5. System architecture
### 5.1 Module layout (repo proposal)
m4/
  __init__.py
  types.py                 # pydantic/dataclasses schemas (NetworkData, RationalModel, Reports)
  io/
    touchstone.py          # parsing + metadata normalization
    network_adapter.py     # conversion between internal arrays and skrf.Network
  preprocess/
    freq_grid.py           # grid validation/resampling/decimation
    renormalize.py         # z0 handling, s_def policy
    smoothing.py           # optional noise smoothing (careful, logged)
  causality/
    triverio_fft.py        # filtered-FFT check + error bounds
    repairs.py             # optional repairs: delay, time-gating, etc.
  vf/
    fit.py                 # deterministic pole init + vf runner
    order_select.py        # auto_fit wrapper + manual order selection
    spurious.py            # spurious pole detection wrapper
  passivity/
    assess.py              # half-size test matrix assessment interface
    enforce.py             # enforcement loop wrapper + convergence policy
  export/
    macromodel_json.py     # canonical model serializer
    spice.py               # SPICE export + dialect tests
    plots.py               # diagnostic plots saved as artifacts
  oracle.py                # orchestrates end-to-end run; returns ACCEPT/REJECT + artifacts
  cli.py                   # `m4-fit`, `m4-check`, `m4-export`
tests/
  test_m4_*.py

### 5.2 Dataflow (single model run)

Load network data

Validate frequency grid + metadata

Causality check (and optional repair)

Vector fit (auto or fixed order)

Spurious pole detection & pruning (optional)

Passivity assessment (exact bands)

Passivity enforcement (if needed)

Re-assess passivity; final dense check

Export JSON/NPZ/SPICE

Emit quality_report.json + verdict

## 6. Canonical interfaces and data contracts
### 6.1 Python API (must exist)
from m4.oracle import process_network

result = process_network(
    network_path="artifacts/raw/design_001.s4p",
    config=M4Config(...),
)

assert result.verdict in {"ACCEPTED","REPAIRED_AND_ACCEPTED","REJECTED"}
print(result.artifacts.macromodel_json_path)

### 6.2 CLI (must exist)

m4-fit <input.sNp> --config config.yaml --out run_dir/

m4-check run_dir/macromodel.json (re-run passivity/causality checks)

m4-export run_dir/macromodel.json --spice --touchstone

### 6.3 Canonical file outputs

Required outputs (minimum):

macromodel.json

macromodel.npz

quality_report.json

macromodel.sp (if accepted)

Optional debug outputs (when verbose):

causality_impulse.png

passivity_bands.json

vf_convergence.png

s_singular_values.png

model_vs_data_s_db.png

### 6.4 macromodel.json schema (strict)

At minimum:

{
  "schema_version": "1.0",
  "parameter_type": "S",
  "s_def": "power",
  "z0_ohms": [50.0, 50.0, ...],
  "freq_band_hz": {"f_min": 1e6, "f_max": 6e9},
  "model": {
    "poles": [{"re": -1.2e9, "im": 3.1e10}, ...],
    "residues": [
      {"k":0, "matrix_re": [[...]], "matrix_im": [[...]]},
      ...
    ],
    "D": {"re": [[...]], "im": [[...]]},
    "E": {"re": [[0...]], "im": [[0...]]}
  },
  "canonicalization": {
    "pole_sort": "increasing_imag_then_real",
    "conjugate_pairing": true,
    "numerical_dtype": "float64"
  },
  "provenance": {
    "code_commit": "...",
    "tool_versions": {"scikit-rf":"1.8.0", "...":"..."},
    "input_hash": "...",
    "config_hash": "..."
  }
}

## 7. Causality screening and enforcement
### 7.1 Why we need this

Triverio documents that non-causal sampled S-parameters (from calibration, de-embedding, noise, non-causal material models, etc.) can severely degrade macromodeling, even with vector fitting, and can destabilize transient simulation.

### 7.2 Primary causality check (must implement): Triverio filtered-FFT method

We implement the robust causality check described by Triverio:

Uses a filtered inverse Fourier transform

Includes a rigorous estimate of truncation error due to missing out-of-band data

Key design requirements:

Works on finite bandwidth samples

Produces:

causality_violation_metric (scalar)

violation_time_support_ps (earliest negative time where violation exceeds bound)

optional per-element results

Configuration knobs:

Filter type: Chebyshev low-pass (default), order, cutoff, ripple

IFFT windowing policy

Threshold policy: strict vs permissive

Report both raw and filtered impulse response

Acceptance gate (default strict):

For each 
𝑆
𝑖
𝑗
S
ij
	​

, negative-time impulse response must lie within the truncation error bound, with a configurable margin.

### 7.3 Causality repair modes (optional but strongly recommended)

Causality “enforcement” in M4 is not magic; it is controlled repair with audit trails. The system supports:

(A) Delay alignment repair

Estimate a small additional delay 
Δ
𝜏
Δτ (e.g., from detected violation time extent)

Apply phase shift 
𝑆
𝑖
𝑗
(
𝑓
)
←
𝑆
𝑖
𝑗
(
𝑓
)
 
𝑒
−
𝑗
2
𝜋
𝑓
Δ
𝜏
S
ij
	​

(f)←S
ij
	​

(f)e
−j2πfΔτ

Re-run causality check; accept only if improvement is monotone and within bounds

This is consistent with Triverio’s example where adding a small delay removed causality violation that was breaking vector fitting quality.

(B) Time-gating repair

Compute filtered impulse response

Zero negative-time region (or apply smooth taper)

FFT back to frequency domain

Record gating window in metadata

Hard rule: repairs must never be silent. Every repair is an explicit artifact and provenance entry.

### 7.4 Fail policy

If causality still fails after allowed repair attempts:

Verdict = REJECTED_CAUSALITY

Store full diagnostic bundle for downstream investigation

## 8. Vector fitting subsystem
### 8.1 Core fitting options

We support two fit modes:

Mode 1: Fixed-order vector fit

User specifies n_poles_real, n_poles_cmplx, spacing policy (lin/log/custom), constant term, proportional term.

scikit-rf exposes these controls, including initial pole spacing and recommendation to fit on original S-parameters.

Mode 2: Automatic order selection (auto_fit)

Implements “vector fitting with adding and skimming”

Provides automatic model order optimization and improved convergence/noise robustness

This must be our default for large automated campaigns because we will encounter noisy or awkward responses.

### 8.2 Deterministic initial poles policy

To make results reproducible and improve mode tracking later, we enforce a deterministic initial pole policy:

Given 
𝑓
min
⁡
,
𝑓
max
⁡
f
min
	​

,f
max
	​

, define initial imaginary parts on lin/log grid

Real parts set to a negative damping proportional to spacing

Always include conjugate pairs

Optional real poles near DC for low-pass-like behavior

We must be careful not to over-order models, both for compute cost and for spurious resonances outside the fit band; scikit-rf warns about excessive poles introducing unwanted resonances outside the fit interval.

### 8.3 Fit weighting policy (required)

We implement frequency-dependent weighting because SI/PI work is rarely uniform:

Default: emphasize band of interest 
[
𝑓
𝑎
,
𝑓
𝑏
]
[f
a
	​

,f
b
	​

] (e.g., 0.1–6 GHz in tabletop VNA context)

Optional: emphasize return-loss band edges, resonance regions, or group-delay flatness

Weighting must be part of the recorded config hash.

### 8.4 Convergence and failure handling

We define convergence using:

RMS error reduction (per scikit-rf’s get_rms_error)

Pole movement norms

Maximum iteration limit and tolerance

If convergence fails:

Retry with:

modified pole spacing, then

increased order, then

auto_fit

If still fails: REJECTED_FIT_DID_NOT_CONVERGE

### 8.5 Spurious pole detection (recommended)

We support pole-residue pair classification as “spurious” based on band-limited energy norms of resonance curves; scikit-rf provides a method based on published work and includes a tunable sensitivity threshold gamma.

Policy:

In auto mode, “skim” spurious poles when safe

Always log what was removed and its impact on fit error/passivity

## 9. Passivity assessment and enforcement
### 9.1 Passivity assessment must be “guarantee-grade”

Sampling singular values at a finite grid is not enough. We require a method that identifies violation bands reliably for rational models.

scikit-rf’s passivity_test evaluates passivity for reciprocal fitted models using a half-size test matrix and returns frequency bands of violation.

The half-size test matrix approach reduces eigenvalue computation time significantly (the literature notes about an eightfold reduction due to cubic eigenvalue complexity).

We require:

Primary assessment: half-size test matrix bands (when reciprocity assumptions hold)

Secondary confirmation: dense sampling of singular values on a high-resolution grid

### 9.2 Reciprocity handling (important!)

Half-size assessment relies on symmetry/reciprocity assumptions (as stated by scikit-rf: “reciprocal vector fitted models”).

Therefore M4 must:

Detect reciprocity deviation in input data:

∥
𝑆
𝑖
𝑗
−
𝑆
𝑗
𝑖
∥
∥S
ij
	​

−S
ji
	​

∥ vs threshold

Either:

enforce reciprocity by symmetrization before fitting (optional), or

run a fallback passivity assessment path (grid sampling) if reciprocity too weak

### 9.3 Passivity enforcement algorithm (must implement)

We enforce passivity using singular value perturbation methods, as in scikit-rf’s passivity_enforce:

Passivity enforcement based on methods from the referenced literature

Can preserve DC by perturbing residues only (not constant term)

Parameter n_samples controls enforcement resolution; narrow violation bands require more samples

Implementation requirements:

Must support:

f_max override

preserve_dc logic:

only enabled if DC is already passive; otherwise disable and enforce DC too

Must track:

history_max_sigma per iteration

history_violation_bands per iteration

Must stop only when:

max singular value ≤ 1 + ε for all violating bands (ε default 1e-6–1e-4, configurable)

AND dense sampling check passes

### 9.4 Enforcement fail policy

If enforcement does not converge:

Attempt 1: increase n_samples

Attempt 2: increase model order (sometimes passivity requires more degrees)

Attempt 3: refit with different pole initialization

Else: REJECTED_PASSIVITY_ENFORCEMENT_FAILED

## 10. SPICE export subsystem
### 10.1 What we export and why

We must export a circuit model that:

Is stable and passive in time-domain simulation

Is compatible with mainstream simulators used in practice

Can be used in customer demos and later revenue workflows

scikit-rf provides write_spice_subcircuit_s, which generates an equivalent N-port subcircuit in SPICE netlist syntax compatible with LTspice, ngspice, Xyce (and others), based on a direct state-space implementation of the fitted model.

### 10.2 State-space realization approach

We standardize on state-space synthesis for the SPICE export, because:

It can represent general rational multiport behavior compactly

It is the approach explicitly described in scikit-rf’s vector fitting tutorial and implementation notes

scikit-rf notes that its implementation has evolved among equivalent admittances/impedances/state-space, and currently uses a direct state-space realization; it also notes simulator runtime differences depending on controlled source topology.

### 10.3 Export options and required variants

We export two variants:

Variant A: N pins, internal ground refs

Pins: p1 p2 ... pN

Reference nodes tied internally to node 0

This matches common SPICE usage

Variant B: N pin-pairs (explicit reference pins)

Pins: p1 p1_ref p2 p2_ref ... pN pN_ref

Enabled by create_reference_pins=True in scikit-rf

Required for advanced embedding and avoiding implicit ground assumptions

### 10.4 SPICE dialect compliance testing (mandatory)

M4 must include an automated validation step:

Export macromodel.sp

Run ngspice (or Xyce) in batch mode on a tiny test harness

Compute small-signal AC response (or equivalent extracted S via test fixture)

Compare to rational model evaluation at the same frequencies

The ngspice manual is available and up-to-date; we will target basic behavioral/controlled-source features used by scikit-rf exports.

Acceptance criterion: netlist reproduces fitted S within the same error thresholds as the rational model (within a small additional tolerance due to simulator numerical differences).

## 11. Numerical and computational design (hardware-aware)
### 11.1 Core numerical rules

All fitting and passivity operations default to float64/complex128 for robustness.

GPU acceleration is used where it improves throughput without destabilizing numerics (primarily evaluation, SVDs, batch error computation).

### 11.2 Frequency unit discipline

Internal canonical frequency is Hz (consistent with scikit-rf APIs for passivity enforcement and f_max).

Convert to 
𝜔
=
2
𝜋
𝑓
ω=2πf only inside algorithmic kernels that require rad/s.

### 11.3 Performance strategy on a single laptop

Vector fitting itself is often not the global bottleneck compared to EM solves—but at scale (thousands of coupons) it becomes material. The performance strategy:

P0: Parallelize over coupons

Each coupon fit is independent. Use process-level parallelism gated by M0 resource semaphores.

P1: Avoid re-fitting

Cache by input hash + config hash.

If only passivity enforcement config changes, reuse fit and re-run enforcement only.

P2: GPU-first evaluation

The repeated inner loops are:

evaluating 
𝑆
(
𝑗
𝜔
)
S(jω) over frequency grids

computing singular values (SVD) over many frequencies

computing error metrics over many frequencies

These are excellent GPU workloads.
M4 must provide an array-backend abstraction (xp = numpy|cupy) so that M5 can drop in custom CUDA kernels without rewriting M4.

### 11.4 Memory bounds

We must support up to (typical):

Nports: 2–8

Nfreq: 201–10,001

Poles: 10–100 (worst case)

We design all evaluation routines to be chunked by frequency to avoid VRAM spikes.

## 12. Quality metrics and acceptance gates
### 12.1 Fit accuracy metrics (required)

Compute:

RMS error across all S-parameters (scikit-rf provides RMS error calculation; we also compute ours)

Max magnitude error (dB)

Max phase error (deg)

Weighted error in band(s) of interest

Default gates (tunable per campaign):

RMS error ≤ 1e-2 (example-level); for strict campaigns ≤ 5e-3

Max |ΔS| ≤ 0.02 in-band (example target)

### 12.2 Passivity gates (hard)

No violation bands returned by passivity assessment

Dense sampling check confirms max singular value ≤ 1 + ε

Passivity assessment uses half-size test matrix for reciprocal models and returns explicit violation bands if present.

### 12.3 Causality gates (hard for “oracle accepted”)

Robust filtered-FFT causality check passes within truncation-error bounds

If repaired, must show:

violation metric reduced (monotone improvement)

repair magnitude within allowed limits (e.g., Δτ ≤ 100 ps unless explicitly allowed)

### 12.4 SPICE export gates (hard)

Netlist is generated

Netlist parses and runs in ngspice (and optionally Xyce)

AC response matches the rational evaluation within tolerance

### 12.5 Final verdict logic

ACCEPTED: passes all gates without repair

REPAIRED_AND_ACCEPTED: repairs applied and all gates passed; repairs recorded

REJECTED: any hard gate fails, or repairs exceed allowed limits

## 13. How we will know “for certain” the implementation meets engineering rigor

This section is the Definition of Done (DoD) for M4. M4 is complete only when all items below are true.

### 13.1 Correctness test suite (must pass in CI)

A. Analytic network tests

Generate known passive RLC networks (2-port ladders, coupled LC, etc.)

Compute “ground truth” S via analytic formulas

Fit model → enforce passivity → export SPICE

Verify:

fit error below strict threshold

passivity holds

SPICE reproduces response

B. Known dataset regression

Use scikit-rf example networks (e.g., ring slot examples) to ensure:

fit converges in expected order range

passivity enforcement works when violations exist

regression thresholds match expected behavior

(scikit-rf provides vector fitting examples; M4 uses them as reproducible regression fixtures.)

C. Causality detection tests

Construct a causal dataset, then inject non-causality:

apply a negative delay / phase warp

Verify robust causality check flags it (with negative-time violation)

Verify repair mode (delay alignment) can fix within allowed range

The robust check methodology and finite-bandwidth error accounting are described by Triverio.

D. Passivity stress tests

Create a fitted model and deliberately perturb residues to create passivity violation

Ensure:

passivity assessment returns correct violation bands

enforcement converges and eliminates bands

E. SPICE simulator round-trip

Export netlist

Simulate in ngspice

Compare to rational model evaluation on the same frequency grid

Fail CI if mismatch exceeds tolerance

### 13.2 Determinism and provenance (must be provable)

Same input + same config + same commit → identical macromodel.json (modulo floating rounding; we define deterministic serialization)

All outputs embed:

input hash

config hash

tool versions

commit hash

### 13.3 Performance gates (must be measurable)

We define a standard benchmark:

4-port, 1001 frequency points, target ~40 poles via auto_fit

Must finish within X seconds on the laptop CPU baseline

Must not regress >Y% across commits (tracked)

### 13.4 “Oracle audit bundle” (must exist)

Every run produces a compact audit bundle:

Causality report (plots + metrics)

Fit convergence plot + pole distribution

Passivity bands pre/post enforcement + max singular values

SPICE export + simulator logs (if enabled)

Final verdict explanation

## 14. Critical failure modes and mitigations
### 14.1 Non-causal raw data → “VF looks bad / unstable”

Mitigation:

Always run causality screen first

Attempt limited repairs; else reject with diagnostics

### 14.2 Passivity enforcement fails to converge

Mitigation:

Increase enforcement sampling density

Increase model order (more degrees of freedom)

Refit with different initial poles

If still fails: reject (do not ship unstable models)

### 14.3 Pole ordering instability (breaks downstream learning)

Mitigation in M4:

Canonical pole sorting and conjugate pairing

Export derived real second-order section parameters (stable ordering)

Provide a “mode signature” per pole pair (e.g., 
(
𝜔
0
,
𝛼
)
(ω
0
	​

,α)) for later matching

### 14.4 SPICE dialect incompatibilities

Mitigation:

Target scikit-rf’s tested export path (LTspice/ngspice/Xyce compatible)

Maintain simulator-based regression tests

Keep two netlist variants (with/without explicit reference pins)

## 15. Milestone completion checklist (single-page)

M4 is complete when:

✅ process_network() produces ACCEPTED/REPAIRED/REJECTED deterministically

✅ Robust causality check implemented (filtered FFT + truncation bound) and tested

✅ Vector fitting supports fixed-order and auto_fit (“adding & skimming”)

✅ Passivity assessment returns violation bands via half-size test matrix

✅ Passivity enforcement converges using singular value perturbation + optional DC preservation

✅ SPICE export produces a subcircuit compatible with LTspice/ngspice/Xyce and simulator round-trip tests pass

✅ All artifacts include provenance hashes and a full audit bundle

✅ CI includes analytic-network tests, causality injection tests, passivity perturbation tests, SPICE round-trip tests

✅ Performance benchmark exists and is regression-gated

✅ Documentation: one tutorial notebook + one CLI guide + one “debugging playbook”

## 16. Strategic note: why this M4 design maximizes odds of discovering a novel equation

Your “novel equation” is most likely to emerge as a discovered parametric law for the macromodel parameters (poles/residues/sections) as functions of geometry.

M4 ensures:

the learned targets are physically legal (passive/causal/stable)

the targets are compressive (low-dimensional, interpretable)

the targets are auditable (no hidden artifacts)

the targets are deployable (SPICE export is a direct commercialization path)

That combination is what turns symbolic regression from “curve fitting” into “engineering discovery”.

# M5 Design Document — GPU Acceleration Layer

Milestone goal: Make the inner loop of discovery (batched macromodel evaluation + constraint checks + objective scoring across large datasets) GPU-dominant, deterministic-enough, and auditable—so that symbolic discovery + adversarial falsification (M6/M7) becomes feasible on a single laptop GPU.

This document is written to be “repo-complete”: if you implement exactly what’s specified here, you will know (with high confidence) you have built the correct, performant, and rigorous M5 layer—and you will also know exactly when it is done.

## 0) Mission context and why M5 is existential to the project

Your overall pipeline is:

M1: generate parametric PCB coupons (geometry DSL → Gerbers/KiCad).

M2: full-wave oracle (openEMS) produces ground-truth frequency responses.

M4: vector-fitting + passivity/causality → compact rational macromodels for each design point.

M6: symbolic regression discovers formulas for macromodel parameters (poles/residues/etc) as functions of geometry.

M7: falsification/active learning breaks formulas and forces robustness.

The bottleneck in M6/M7 is not “training a model”—it’s scoring: for each candidate formula, you must evaluate predicted macromodel parameters over thousands of designs and hundreds of frequency points, compute objective metrics, and repeatedly do this in tournaments/falsification loops.

M5’s job: turn “evaluate candidate → compute metrics → compute constraints” into a GPU pipeline with fused kernels and minimal CPU overhead.

## 1) Hardware + platform assumptions
### 1.1 Target device class

NVIDIA GeForce RTX 5090 Laptop GPU with 24 GB VRAM (your stated machine; consistent with published RTX 5090 laptop specs in the press).

Compute capability for RTX 5090 = 12.0 (Blackwell consumer).

### 1.2 Critical compatibility risk (must be engineered around)

Blackwell consumer (sm_120 / compute capability 12.0) has had real-world “not compatible with current PyTorch install” failures depending on wheel/CUDA build. Treat this as expected unless you pin a known-good stack.

M5 requirement: the repo must provide:

A validated environment lock (container or reproducible env) where:

PyTorch GPU works on sm_120, or

the M5 layer can operate using CuPy RawKernel paths even if PyTorch is temporarily behind for that architecture.

### 1.3 CUDA graphs are a first-class optimization

CUDA Graphs reduce CPU overhead by recording GPU work once and replaying it repeatedly (with fixed memory addresses/arguments), and PyTorch supports CUDA graph construction via stream capture.

M5 must be graph-friendly for repeated objective evaluations at fixed shapes.

## 2) Scope, goals, and non-goals
### 2.1 Goals (what M5 must deliver)

M5 provides a GPU-first compute substrate used by M4/M6/M7:

Batched rational macromodel evaluation
Efficient evaluation of the canonical form used in this project:

𝐻
(
𝑠
)
=
𝐷
+
𝑠
𝐸
+
∑
𝑘
=
1
𝐾
𝑅
𝑘
𝑠
−
𝑎
𝑘
H(s)=D+sE+
k=1
∑
K
	​

s−a
k
	​

R
k
	​

	​


where 
𝐻
(
𝑠
)
H(s) is multiport complex (typically 2×2 or 4×4), 
𝑎
𝑘
a
k
	​

 poles, 
𝑅
𝑘
R
k
	​

 residue matrices, 
𝐷
,
𝐸
D,E constant matrices.
Must support batch dimensions over:

many designs 
𝐵
B

many frequencies 
𝐹
F

small port counts 
𝑃
∈
{
2
,
4
}
P∈{2,4}

moderate poles 
𝐾
∈
[
4
,
40
]
K∈[4,40] (configurable)

Fast objective evaluation (GPU reductions)
Compute, at minimum:

weighted complex error norms vs ground truth

max error metrics (per-design and global)

optional band-limited metrics (e.g., 0.5–6 GHz, etc.)

penalty terms for constraint violations

Constraint check kernels
At minimum:

stability (Re(poles)<0)

passivity proxy (largest singular value ≤ 1 for S-parameters)

reciprocity/symmetry checks where applicable
Must include both:

fast approximate checks for inner-loop search

strict checks for promotion/acceptance gates

Backend abstraction + no-silent-fallback
Everything must run on GPU by default and hard-fail if it silently falls back to CPU.

Custom CUDA kernels for the true bottlenecks
Use CuPy’s user-defined kernels (RawKernel/RawModule, ElementwiseKernel, ReductionKernel) as the primary vehicle.
(Rationale: CuPy kernels can be JIT compiled/cached per device and reused across processes.)

Optional torch.compile acceleration path for any PyTorch-based components
torch.compile uses TorchInductor as default compiler and leverages Triton for GPU codegen on major GPU backends.
Triton is a Python-based GPU kernel language/compiler.

Zero-copy interchange where needed
Support moving tensors between frameworks via DLPack when required. CuPy explicitly supports DLPack import/export.

Performance gates + correctness harness
M5 is not “done” until benchmarks and correctness tests pass with strict tolerances and reproducible profiles.

### 2.2 Non-goals (explicitly out of M5)

Running openEMS (M2)

Performing vector fitting itself (M4), except optional GPU helpers

Running symbolic regression search (M6)

Orchestrating agent workflows (M8)
M5 exists to make those milestones fast and repeatable, not to implement them.

## 3) External dependencies (pinned + why)
### 3.1 Primary GPU numeric substrate: CuPy

CuPy provides GPU arrays with a NumPy-like interface and supports custom kernel creation via ElementwiseKernel, ReductionKernel, and RawKernel/RawModule.

### 3.2 Optional compiler substrate: PyTorch + torch.compile

torch.compile compiles PyTorch code to optimized kernels; TorchInductor is the default backend and uses Triton for GPU codegen in many cases.

### 3.3 CUDA Graph support (optional but prioritized)

CUDA Graph concept & benefits:

PyTorch CUDA graph support:

Limitations for CUDA graphs (fixed args, no control flow, no sync triggers):

### 3.4 BLAS/solver libraries (only when needed)

cuBLAS supports batched GEMM/strided batched GEMM.

cuSOLVER provides GPU-accelerated decompositions/linear solves.

Design bias: for 
𝑃
∈
{
2
,
4
}
P∈{2,4}, prefer hand-coded small-matrix kernels over calling cuBLAS/cuSOLVER repeatedly.

## 4) High-level architecture
### 4.1 Package layout (repo contract)

Create a dedicated module (example naming):

repo/
  m5_gpu/
    __init__.py
    backend/
      device.py
      array.py
      streams.py
      memory.py
      dlpack.py
    kernels/
      build.py
      cache.py
      rational_eval.cu
      smallmat.cu
      reductions.cu
      passivity.cu
    ops/
      rational.py
      smallmat.py
      metrics.py
      constraints.py
      objectives.py
    profiling/
      nvtx.py
      timers.py
      nsight.md
    benchmarks/
      bench_rational_eval.py
      bench_objective.py
      bench_passivity.py
      suite.py
    tests/
      test_rational_eval_correctness.py
      test_objective_correctness.py
      test_constraints_correctness.py
      test_no_cpu_fallback.py
      test_memory_no_explosions.py
      test_determinism_policy.py

### 4.2 Integration points (upstream/downstream)

Inputs from M4: fitted macromodel parameters (poles/residues/D/E), plus target responses (Touchstone data or converted arrays).

Inputs from M6: candidate formula predicted parameters OR direct predicted responses.

Inputs from M7: large candidate sets of 
𝑥
x requiring fast evaluation + “worst-case” extraction.

Outputs:

scalar objectives

per-design metrics for active learning

pass/fail gates for promotion

## 5) Data model and tensor layout (this is make-or-break)
### 5.1 Canonical symbols

𝐵
B: number of designs in the evaluation batch (dataset batch)

𝐹
F: number of frequency samples

𝑃
P: number of ports (2 or 4 initially)

𝐾
K: number of poles

### 5.2 Canonical GPU representation (S-domain preferred)

You want to avoid repeated Z↔S conversions if possible; therefore M4 should export (or you should convert once) into a canonical domain. For constraint checks and corporate use, S-parameters are common.

Canonical arrays (GPU resident, complex64 by default):

s: shape (F,) complex64 containing 
𝑠
=
𝑗
2
𝜋
𝑓
s=j2πf

A: shape (B, K) complex64 poles 
𝑎
𝑘
a
k
	​


R: shape (B, K, P, P) complex64 residue matrices

D: shape (B, P, P) complex64

E: shape (B, P, P) complex64

S_target: shape (B, F, P, P) complex64 (or chunk-streamed)

### 5.3 Memory layout choice: AoS vs SoA

For 
𝑃
≤
4
P≤4, the performance bottleneck is reading residues and accumulating.

You will support two residue layouts, because it’s cheap to convert once and critical for performance tuning:

Layout L0 (AoS): R[B,K,P,P]

Pros: natural, simple.
Cons: may cause less-coalesced loads for per-element threads.

Layout L1 (SoA-flattened): R_flat[B,K,P2] where P2=P*P

and element index e = p*P + q.
Pros: contiguous loads per element index.
Cons: requires consistent flattening.

Contract: The GPU kernel API accepts either layout via a flag and uses the corresponding indexing. Provide a conversion function to_residue_layout(R, layout).

### 5.4 Chunking policy (VRAM constraints)

Even with 24 GB VRAM, (B,F,P,P) can blow up.

M5 must support:

chunk_B: process designs in chunks

chunk_F: process frequency in chunks

combined chunking if needed

Hard requirement: chunking must preserve exactly identical outputs (up to floating tolerance) vs non-chunked evaluation.

## 6) Core operators and kernel designs
### 6.1 Operator: Rational macromodel evaluation

API (public, stable):

S_pred = rational_eval(
    s: cupy.ndarray[F] complex64,
    A: cupy.ndarray[B,K] complex64,
    R: cupy.ndarray[B,K,P,P] complex64  (or R_flat[B,K,P2]),
    D: cupy.ndarray[B,P,P] complex64,
    E: cupy.ndarray[B,P,P] complex64,
    out: optional cupy.ndarray[B,F,P,P] complex64,
    *,
    layout: {"aos","soa"},
    chunk_B: int | None,
    chunk_F: int | None,
    stream: cupy.cuda.Stream | None,
) -> cupy.ndarray[B,F,P,P]


Correctness definition:
For each (b,f):

𝐻
𝑏
(
𝑠
𝑓
)
=
𝐷
𝑏
+
𝑠
𝑓
𝐸
𝑏
+
∑
𝑘
=
1
𝐾
𝑅
𝑏
,
𝑘
𝑠
𝑓
−
𝐴
𝑏
,
𝑘
H
b
	​

(s
f
	​

)=D
b
	​

+s
f
	​

E
b
	​

+
k=1
∑
K
	​

s
f
	​

−A
b,k
	​

R
b,k
	​

	​


Elementwise complex float error within tolerance vs CPU float64 reference (see §10).

#### 6.1.1 Why a custom kernel is required

The naive GPU implementation would broadcast 1/(s - A) into a tensor of shape (B,F,K), then broadcast residues to (B,F,K,P,P), causing:

huge intermediates: 
𝑂
(
𝐵
𝐹
𝐾
𝑃
2
)
O(BFKP
2
) memory

catastrophic VRAM thrash

Custom kernel requirement: must be streaming in K: never materialize (B,F,K,...).

#### 6.1.2 CUDA kernel mapping (warp-per-(b,f))

Primary design: one warp computes one (b,f) S-matrix.

Each warp handles one design-frequency pair (b,f)

Warp lanes map to matrix elements:

For P=4: lanes 0–15 map to the 16 elements

For P=2: lanes 0–3 map to 4 elements

Denominator for each pole is shared:

lane 0 computes den = 1/(s[f] - A[b,k])

broadcast den within warp via warp shuffle

Each active lane loads the corresponding residue element R[b,k,e] and accumulates.

Launch configuration:

WARPS_PER_BLOCK = 4 or 8 (tuned)

block threads = 32 * WARPS_PER_BLOCK

grid = ceil((B*F) / WARPS_PER_BLOCK)

Pseudocode (conceptual):

warp_id = threadIdx.x / 32
lane    = threadIdx.x % 32
idx     = blockIdx.x * WARPS_PER_BLOCK + warp_id
if idx >= B*F: return
b = idx / F
f = idx % F

if lane < P2:
  acc = D[b,lane] + s[f] * E[b,lane]
  for k in 0..K-1:
    if lane == 0:
      den = 1 / (s[f] - A[b,k])
    den = shfl_broadcast(den)
    acc += R[b,k,lane] * den
  out[b,f,lane] = acc


This design:

avoids B×F×K intermediates

uses warp shuffles instead of shared memory

keeps accumulation in registers

#### 6.1.3 Numerical stability considerations

Division near pole: if |s - a| is tiny, values can spike. For scoring, this is real physics (near resonance), but you must avoid NaNs.

Implement:

safe reciprocal: if |s-a| < eps, clamp denominator magnitude or add tiny complex epsilon

track NaN/Inf counters in a debug mode kernel and fail-fast when detected

#### 6.1.4 Dtypes

Default:

complex64 compute + accumulate

Optional:

accumulate in complex128 for strict-validation mode (slower, used only in gating tests).

Contract: dtype is selectable via dtype_policy object.

### 6.2 Operator: Objective evaluation (fused eval+reduce)

Storing S_pred[B,F,P,P] is often unnecessary; for inner-loop candidate scoring you typically need:

a scalar objective

some per-design summary metrics (for active learning)

possibly an argmax location for falsification

Therefore implement two modes:

#### 6.2.1 Mode A: eval_only

Returns S_pred (for debugging, plotting, or best-candidate export).

#### 6.2.2 Mode B: eval_reduce (primary hot path)

Computes metrics without materializing S_pred.

API:

metrics = rational_eval_reduce(
    s, A, R, D, E,
    S_target,
    weights_f: cupy.ndarray[F] float32 | None,
    *,
    return_per_design: bool,
    return_argmax: bool,
    layout, chunk_B, chunk_F, stream
) -> ObjectiveMetrics


ObjectiveMetrics contains:

loss_total: float32

optional loss_per_design[B]

optional max_err_per_design[B]

optional global_max_err: float32

optional argmax: (b*, f*, p*, q*)

#### 6.2.3 Canonical error metric definitions

You need metrics that are:

differentiable-ish (not required, but good)

robust to scale

stable

Implement at minimum:

Weighted relative Frobenius error over S-matrix:

e
r
r
𝑏
=
∑
𝑓
𝑤
𝑓
∥
𝑆
𝑝
𝑟
𝑒
𝑑
(
𝑏
,
𝑓
)
−
𝑆
𝑡
𝑔
𝑡
(
𝑏
,
𝑓
)
∥
𝐹
2
∑
𝑓
𝑤
𝑓
∥
𝑆
𝑡
𝑔
𝑡
(
𝑏
,
𝑓
)
∥
𝐹
2
+
𝜖
err
b
	​

=
∑
f
	​

w
f
	​

∥S
tgt
	​

(b,f)∥
F
2
	​

+ϵ
∑
f
	​

w
f
	​

∥S
pred
	​

(b,f)−S
tgt
	​

(b,f)∥
F
2
	​

	​

	​


Then overall objective:

l
o
s
s
=
m
e
a
n
𝑏
(
e
r
r
𝑏
)
loss=mean
b
	​

(err
b
	​

)

Max absolute element error (per design, across f,p,q):

m
a
x
e
r
r
𝑏
=
max
⁡
𝑓
,
𝑝
,
𝑞
∣
𝑆
𝑝
𝑟
𝑒
𝑑
−
𝑆
𝑡
𝑔
𝑡
∣
maxerr
b
	​

=
f,p,q
max
	​

∣S
pred
	​

−S
tgt
	​

∣

Optional: band-limited versions (user provides frequency masks).

#### 6.2.4 Reduction strategy

To avoid atomic contention:

Each warp computes partial sums for its (b,f) matrix

Accumulate into per-(b) accumulators using block-level reduction, then atomic add once per block per design
OR

Use a two-pass reduction:

pass 1: write partial sums into temporary buffer [B, F_chunk]

pass 2: reduce over frequency

Preference: two-pass is more deterministic; one-pass is faster.

Hard requirement: provide both modes:

reduce_mode="fast" (atomics allowed)

reduce_mode="deterministic" (two-pass)

### 6.3 Operator: Passivity check (S-parameter domain)

For an N-port S-matrix, passivity implies singular values ≤ 1 for all frequencies. In practice you check:

𝜎
𝑚
𝑎
𝑥
(
𝑆
(
𝑗
𝜔
)
)
≤
1
σ
max
	​

(S(jω))≤1

Compute 
𝜎
𝑚
𝑎
𝑥
σ
max
	​

 by:

𝐴
=
𝑆
𝐻
𝑆
A=S
H
S (Hermitian PSD)

𝜆
𝑚
𝑎
𝑥
(
𝐴
)
=
𝜎
𝑚
𝑎
𝑥
2
λ
max
	​

(A)=σ
max
2
	​


#### 6.3.1 Fast exact check for P=2 (analytic)

For 2×2 Hermitian 
𝐴
A, eigenvalues are analytic via trace/determinant:

tr = A11 + A22

det = A11*A22 - |A12|^2

lambda_max = (tr + sqrt(tr^2 - 4*det))/2

sigma_max = sqrt(lambda_max)

Implement as fused kernel:

compute S elements (from rational eval)

compute A11, A22, A12

compute sigma_max

track maximum over frequency

#### 6.3.2 Fast approximate check for P=4 (power iteration)

For 4×4:

compute A = S^H S

run small fixed-iteration power method (e.g., 5–8 iterations)

use fixed initial vector (e.g., [1,1,1,1]/2) for determinism

compute Rayleigh quotient

This is extremely fast and good for inner-loop screening.

#### 6.3.3 Strict check for P=4 (promotion gate)

Provide a strict passivity routine using a “more exact” method:

Option 1 (preferred): implement a small-matrix Jacobi eigen solver in CUDA for 4×4 Hermitian; deterministic iterations, no library overhead.

Option 2: call cuSOLVER batched eigen routines where applicable. cuSOLVER is NVIDIA’s GPU-accelerated decomposition/solve library.
(If you choose cuSOLVER, do it in batched form; do not loop per matrix on CPU.)

Contract: M5 exposes:

passivity = check_passivity(
  A,R,D,E,s, *,
  mode={"fast","strict"},
  ports=P,
  return_sigma_max=True,
  return_violation_mask=True,
)


Output includes:

sigma_max_per_design[B] (max over frequency)

violation_mask[B] = sigma_max>1+tol

optional sigma_max_overall

### 6.4 Operator: Stability check (pole locations)

This is trivial but must be GPU-accelerated for batch screening.

Definition: stable if real(A[b,k]) < -margin for all k.
Return:

stable_mask[B]

max_real_pole[B]

Implement as elementwise+reduction kernel (CuPy ReductionKernel is explicitly for custom reductions).

### 6.5 Operator: Reciprocity/symmetry checks

For reciprocal networks, S is symmetric: S_ij = S_ji.

Provide a kernel that computes:

max symmetry error per design:

max
⁡
𝑓
,
𝑖
<
𝑗
∣
𝑆
𝑖
𝑗
−
𝑆
𝑗
𝑖
∣
f,i<j
max
	​

∣S
ij
	​

−S
ji
	​

∣

Can be fused into eval_reduce if desired.

### 6.6 Operator: Domain conversions (optional but recommended)

If M4 exports Z or Y models, you may need:

𝑆
=
(
𝑍
−
𝑍
0
𝐼
)
(
𝑍
+
𝑍
0
𝐼
)
−
1
S=(Z−Z
0
	​

I)(Z+Z
0
	​

I)
−1

or the Y equivalent

For P=2 and P=4, implement explicit small-matrix inverse/solve kernels rather than generic batched LU.

## 7) Kernel implementation strategy (CuPy-first, then Triton/PyTorch optional)
### 7.1 CuPy kernel types and when to use them

CuPy supports:

Elementwise kernels

Reduction kernels

Raw kernels / raw modules

Policy:

Use RawKernel/RawModule for fused rational-eval and multi-output reductions.

Use ReductionKernel for custom reductions that don’t justify a full raw kernel.

Use ElementwiseKernel for simple transformations (dtype casts, feature scaling, masks).

Why RawKernel: it’s compiled on first call and cached per device, and the compiled binary is also cached to disk under ~/.cupy/kernel_cache/.

### 7.2 Compilation, caching, and versioning

M5 must implement a “kernel build cache key” that includes:

CUDA source hash

compile flags

compute capability target

CuPy version

CUDA runtime version

Even if CuPy caches, you need repo-level tracking so results are reproducible and debuggable.

### 7.3 Optional Triton path

Triton is a language/compiler for writing efficient GPU kernels in Python.
PyTorch’s compiler stack commonly uses Triton via TorchInductor.

When to use Triton:

If you need kernels that integrate seamlessly with PyTorch tensors and torch.compile.

If you want autotuning across block sizes (Triton can do this idiomatically).

But: for complex arithmetic + warp-shuffle patterns, Raw CUDA may still win. Therefore Triton is optional, not mandatory, for M5.

## 8) CUDA Graph integration (shape-stable hot loops)
### 8.1 Why

In M6/M7 you will call “eval_reduce” thousands of times with identical shapes. CUDA graphs reduce CPU launch overhead by capturing and replaying the same GPU work.

### 8.2 Constraints

CUDA graph capture requires fixed kernel arguments, dependencies, and (critically) stable memory addresses; control flow and sync-triggering ops are not allowed.

### 8.3 M5 design for graphs

Provide:

GraphableWorkspace object that preallocates all buffers needed for eval_reduce at fixed (B,F,P,K) and stores them in a stable pool.

A function capture_eval_reduce_graph(workspace, ...) that returns a callable replay().

Rule: If the caller changes B/F/K/P, they must request a new workspace and re-capture.

## 9) Interop: CuPy ↔ PyTorch via DLPack
### 9.1 Why this is needed

Some parts of the repo may be PyTorch-first (neural surrogates, etc.)

M5 kernels may be CuPy-first

You need a safe, explicit way to exchange buffers without copies

DLPack is the common in-memory tensor structure for sharing tensors among frameworks, and CuPy supports importing/exporting DLPack.

### 9.2 Required utilities

Implement:

torch_tensor_to_cupy(t) using DLPack

cupy_to_torch_tensor(a) using DLPack

Important engineering note: In most frameworks, DLPack exchange does not preserve autograd graphs automatically; treat this as a raw buffer interchange unless you explicitly wrap it.

## 10) Correctness strategy (how you will know it’s right)

This section is one of your “engineering rigor” pillars. M5 is not complete unless every item below is implemented and passes in CI.

### 10.1 CPU reference implementation (gold)

Create a pure NumPy reference that is:

float64 + complex128

uses the exact formula

optionally uses Kahan summation in K loop for tighter reference

Reference function:

rational_eval_np(s, A, R, D, E) -> S_pred_np

### 10.2 Property-based correctness tests

Use randomized tests over:

B ∈ {1, 2, 8, 64}

F ∈ {8, 64, 512}

P ∈ {2, 4}

K ∈ {4, 8, 16, 32}

dtypes: complex64/complex128

Test invariants:

GPU eval matches CPU reference within tolerances:

absolute tol and relative tol defined per dtype

Chunked evaluation equals non-chunked evaluation (within tighter tol)

eval_reduce metrics match metrics computed from eval_only outputs

### 10.3 Special case tests

Poles very close to frequency points (stress denominator)

Purely real poles/residues

Conjugate-pair poles (if used)

Zero residues for some poles

D/E only, no poles

### 10.4 Passivity test correctness

For P=2:

compare analytic sigma_max kernel to a CPU SVD reference (numpy.linalg.svd) for random complex matrices.

For P=4:

compare fast power-iteration estimate to CPU SVD, ensure:

it is a lower bound or near value (depending on method)

strict mode matches within tolerances.

## 11) Performance strategy (how you will know it’s fast enough)
### 11.1 Benchmark suite (must exist)

M5 must ship benchmarks/suite.py that runs:

bench_rational_eval

bench_eval_reduce

bench_passivity_fast

bench_passivity_strict

bench_chunking_overhead

bench_cuda_graph_replay (if graphs enabled)

Each benchmark prints:

shapes (B,F,P,K)

dtype

wall time

achieved throughput:

evaluations/sec (B*F per second)

“term-updates/sec” (BFK*P2 per second)

VRAM peak usage (from allocator stats)

kernel launch counts (where possible)

### 11.2 Baselines that must be compared

For each benchmark, compare:

CPU NumPy baseline

naive CuPy broadcast baseline (if feasible)

fused kernel (your implementation)

### 11.3 Hard performance gates (Definition-of-Done)

You must define concrete, measurable acceptance criteria. Here is the recommended set:

Gate G1 — No VRAM explosion
For a representative “real” workload (you will pin it):

B=2048, F=512, P=2, K=16, complex64
The fused eval_reduce must not allocate any tensor larger than O(BFP2) and must not allocate an O(BFK*P2) intermediate.
Verification method:

allocator peak memory check + code inspection + optional debug instrumentation.

Gate G2 — Fused kernel dominates
On the same workload:

end-to-end eval_reduce must be GPU-time dominated; CPU time per call must be “small” compared to GPU time.
Verification:

Nsight Systems trace (documented workflow in profiling/nsight.md)

kernel launch count should be minimal (ideally 1–3 kernels per call, depending on reductions)

Gate G3 — CUDA graph replay speedup
For repeated fixed-shape calls (e.g., 1000 replays), CUDA graph replay must reduce per-call overhead vs non-graph by a measurable factor when kernels are small. (Graphs are known to help when CPU launch overhead is significant.)

Gate G4 — Regression guard
You must add a CI performance test that fails if throughput regresses by >X% versus a stored baseline (where the runner GPU is stable).
If CI doesn’t have a stable GPU, then require a local “release gate” script that records baselines and requires manual sign-off.

## 12) Determinism + reproducibility policy

You cannot get perfect bitwise determinism on GPUs for all reductions without major cost, but you can enforce:

deterministic seeds for any randomized operations

deterministic “reduce_mode” option (two-pass)

strict promotion checks use deterministic mode

Repo contract:

any run that promotes a formula must include:

M5 version hash

kernel cache key

dtype policy

reduction mode

chunking parameters

passivity mode

## 13) Debuggability + observability requirements
### 13.1 NVTX ranges

Add NVTX ranges around:

H(s) evaluation kernel

metric reduction kernel

passivity kernel

host-device transfers

graph capture / replay

### 13.2 Failure modes must be explicit

If any of the following occurs, M5 must raise a structured error with dumpable context:

NaNs/Infs produced

CPU fallback detected (array not on GPU)

shape mismatch

dtype mismatch

out-of-memory without chunking fallback available

### 13.3 “Explain the objective” mode

Provide a debug mode that returns:

per-frequency error contributions for a chosen design b*

per-term (per pole) partial sums for a chosen (b*, f*, element)
This is invaluable when a discovered formula fails and you need to see why.

## 14) Security and correctness boundaries

Even though this is local, treat kernel compilation as a controlled surface:

The kernel source comes only from the repo (not arbitrary strings from an agent) unless explicitly allowed.

If you later add expression-to-kernel compilation, it must be sandboxed by:

allowlisted functions/operators

code generation that cannot inject arbitrary code beyond expression

## 15) M5 Definition of Done (DoD) — the “we know it’s complete” checklist

M5 is only complete when all items below are true:

### 15.1 Functional completeness

 rational_eval implemented for P=2 and P=4

 rational_eval_reduce implemented with at least:

weighted relative Frobenius error

max error per design

global max + argmax (optional but recommended)

 check_stability implemented (GPU)

 check_passivity implemented:

fast mode for P=2 (analytic)

fast mode for P=4 (power iteration)

strict mode for P=4 (Jacobi or batched solver)

 Chunking works for both B and F

 No-silent-fallback enforcement

### 15.2 Correctness rigor

 CPU float64 reference exists and is used in tests

 Property-based randomized tests cover multiple shapes and dtypes

 Chunked == unchunked within tolerance

 eval_reduce metrics == eval_only-derived metrics within tolerance

 Passivity check validated vs CPU SVD for random matrices (P=2 and P=4 strict)

### 15.3 Performance rigor

 Benchmark suite exists and produces stable reports

 Gate G1 (no VRAM explosion) passes on the representative workload

 Gate G2 (GPU-dominant) verified with at least one profiler trace saved as artifact

 Gate G3 (CUDA graphs) implemented + benchmarked (if graphs enabled)

### 15.4 Reproducibility + integration

 Kernel cache key recorded in run metadata

 All APIs are documented with exact shapes/dtypes

 Interop utilities via DLPack exist (CuPy↔Torch)

 Works on compute capability 12.0 GPU (RTX 5090 class)

 Environment pinned such that GPU stack is actually usable on sm_120 (or CuPy-only fallback works).

## 16) How M5 maximizes odds of discovering a novel equation

This milestone is not “just speed.” It directly enables discovery success by letting you:

Search a larger hypothesis space
Symbolic regression + falsification is compute-hungry. Faster scoring means more candidates and deeper tournaments.

Use stronger oracles in the objective
Passivity, stability, reciprocity—these are expensive checks. If they’re cheap, you can enforce them every generation rather than as an afterthought.

Do adversarial robustness at scale
M7’s falsification depends on evaluating formulas on huge candidate sets to find counterexamples. Without M5, you will under-sample and accept brittle formulas.

Produce corporate-grade artifacts
Your “secret formula” must be not only accurate but stable/passive and fast to evaluate. M5 is the evaluation engine that will eventually ship as the proprietary runtime.

## 17) Recommended implementation order (so you don’t get stuck)

Implement CPU reference + tests first (even before GPU kernels).

Implement a naive CuPy version (broadcasting) to establish correctness baseline.

Implement fused rational_eval kernel for P=2.

Extend fused kernel to P=4.

Implement eval_reduce fused path.

Add chunking (B then F).

Add passivity P=2 analytic kernel; validate vs CPU.

Add passivity P=4 fast (power iter).

Add strict P=4 method.

Add CUDA graph workspace/capture/replay.

## 18) Appendix — citations used for tooling facts

CuPy user-defined kernels (ElementwiseKernel/ReductionKernel/RawKernel/RawModule):

RawKernel compilation caching:

torch.compile / TorchInductor / Triton:

Triton overview:

CUDA Graphs concept and PyTorch support/limitations:

DLPack and CuPy interoperability:

cuBLAS batched GEMM/strided batched:

cuSOLVER purpose:

RTX 5090 compute capability:

sm_120 / PyTorch compatibility issue evidence:

RTX 5090 laptop 24GB context:

M6 — Equation Discovery Engine

Symbolic regression + physics‑guided search + unit/dimension constraints (GPU‑first, audit‑grade)

This design document specifies exactly what must exist in the repository for Milestone M6 to be considered complete, how it must behave, and the objective evidence (tests, benchmarks, reproducibility artifacts, constraint certificates) that proves we’ve reached the engineering rigor required for a high‑stakes “equation foundry” intended to discover commercially valuable, physically correct, novel formulas.

## 0. Mission directive and success definition
### 0.1 Mission directive (what M6 is for)

M6 exists to convert a dataset of the form:

inputs: a manufacturable geometry/stackup design vector x (with units), and

targets: physically meaningful response parameters θ(x) (e.g., rational macromodel poles/residues, equivalent circuit elements, or “optimal geometry” values),

into one or more interpretable symbolic formulas that:

Generalize across the domain of interest (not just interpolate),

Obey physics (dimensional consistency, stability, passivity/reciprocity constraints, expected monotonicity/asymptotics where applicable),

Are computationally cheap to evaluate (microseconds–milliseconds, batched on GPU),

Are auditable and reproducible from a commit hash + dataset version,

Provide competitive advantage by replacing expensive parametric sweeps / solver loops.

M6 is not a “cool SR demo.” It is the core IP foundry for discovering a drop‑in formula that big companies would embed into EDA optimization loops.

### 0.2 M6 completion = evidence, not vibes

M6 is “done” only when the repo can run a full discovery campaign and produce:

A ranked Pareto frontier of candidate formulas (accuracy vs complexity),

Constraint certificates (dimensional correctness + physics gates) across declared domains,

End‑to‑end validation against the golden oracle outputs (S‑parameters / Z/Y, not just θ),

Reproducibility artifacts (run manifests, seeds, exact dataset hashes, deterministic replays),

Performance reports proving GPU‑dominant scoring and no accidental CPU fallback.

## 1. External SOTA components we build on (and why)

M6 must integrate multiple SR paradigms because no single SR engine wins universally. We build a unified discovery framework that can accept candidates from multiple engines, then applies our physics constraints, end‑to‑end scoring, and GPU acceleration consistently.

### 1.1 PySR / SymbolicRegression.jl (workhorse)

PySR is engineered as a configurable, high‑performance symbolic regression tool and uses SymbolicRegression.jl as the search engine. It also explicitly supports workflows like “symbolic distillation” (distilling a neural net into an analytic equation) for higher‑dimensional problems.

SymbolicRegression.jl provides critical capabilities M6 will exploit:

Dimensional analysis with units (via DynamicQuantities) and options like dimensional_constraint_penalty and dimensionless_constants_only.

Template expressions to impose structure and variable access constraints (physics‑guided structural priors).

Examples for complex numbers, multiple outputs, and dimensional constraints.

### 1.2 AI Feynman 2.0 (physics‑oriented decomposition + robustness)

AI Feynman 2.0 targets Pareto‑optimal formulas and reports improvements in robustness to noise and “bad data,” plus methods for discovering generalized symmetries/modularity from gradient properties of a neural network fit, and hypothesis‑testing‑accelerated search.

This is valuable for our mission because our data (full‑wave EM sims + vector fitting) can contain:

imperfect fits,

numerical noise,

regime changes (mode transitions).

AI Feynman’s decomposition/symmetry detection is a strong complement to evolutionary SR.

### 1.3 SINDy / PySINDy (sparse identification when we can cast structure)

PySINDy is built around SINDy (Sparse Identification of Nonlinear Dynamical Systems), originally introduced by Brunton et al. (2016), and offers multiple optimizers (including sparse regression variants).

Even if our primary mapping is static x → θ, SINDy becomes essential if we:

discover state‑space surrogate dynamics,

enforce differential constraints,

use time‑domain responses as targets.

### 1.4 Benchmarking culture: SRBench

SRBench provides a “living benchmark” framework for symbolic regression, with a scikit‑learn style API requirement and a large set of benchmark datasets/methods (including many SR methods and datasets sourced from PMLB).

M6 will include a benchmark harness aligned with this culture to prevent “we broke SR quality and didn’t notice.”

### 1.5 Unit/Dimension infra: Pint + SymPy units (repo‑side)

We require unit metadata and dimensional reasoning at multiple layers (typed grammar generation, validation, export). We will use:

Pint for practical unit definitions, registry, and quantity handling.

SymPy physics.units for symbolic unit expressions, simplification, and codegen‑adjacent dimension reasoning.

Additionally, we will leverage SymbolicRegression.jl’s built‑in dimensional support for the PySR engine path.

## 2. M6 scope boundaries
### 2.1 In scope

M6 must deliver:

A unified discovery pipeline that:

ingests datasets from M3/M4 (design vectors + fitted macromodel parameters + raw S if available),

enforces dimensional and physics constraints,

runs multiple SR engines,

scores candidates end‑to‑end,

outputs a Pareto frontier and a selected “champion set.”

A GPU‑first candidate scoring layer:

batched evaluation of expressions over large datasets,

batched frequency response reconstruction for end‑to‑end error,

batched constraint checks (stability/passivity/reciprocity).

A hard dimensional consistency guarantee for accepted formulas:

“accepted” means dimensionally correct by proof (typed inference), not probabilistic.

A physics constraint framework that supports our EM macromodel mission:

stability gates,

passivity gates (for S or Z/Y),

reciprocity / symmetry gates,

optional monotonicity/asymptotics constraints.

Auditable artifacts and run manifests for deterministic reruns.

### 2.2 Explicitly out of scope for M6 (handled elsewhere)

Generating designs/Gerbers (M1)

Running openEMS oracle sims (M2)

Vector fitting and passivity enforcement of fitted models (M4)

Active learning / falsification loop (M7)

Multi‑agent orchestration (M8)

Physical measurement ingestion (M9)

M6, however, must expose stable tool interfaces so M7/M8 can call into it.

## 3. Data contracts M6 requires (no ambiguity)

M6 cannot be “over‑engineered” without strict contracts. These are mandatory.

### 3.1 Core dataset objects

We define a canonical dataset package EquationDiscoveryDataset with:

X: shape (N, d) numeric feature matrix (float32 preferred for GPU scoring).

X_schema: list of FeatureSpec (name, units, dimension vector, bounds, type).

Y_targets: dictionary of target groups:

theta_raw: raw fitted parameters from M4 (poles/residues/etc).

theta_canonical: canonicalized/aligned/parameterized version used for learning.

Y_schema: list of TargetSpec (name, units, dimension vector, bounds, constraints).

aux:

optional freq_grid: (F,)

optional S_measured_oracle: (N, F, nports, nports) complex64/complex128

optional S_fitted_oracle: same

optional weights: (N,) sample weights

splits:

train/val/test indices

plus domain splits (see 7.2)

### 3.2 FeatureSpec / TargetSpec required fields

Each FeatureSpec must include:

name: stable string key

unit: Pint unit string (e.g., mm, dimensionless)

dimension: 7‑vector (SI base dims) or reduced vector (L, M, T, I) with explicit mapping

bounds: [min, max] in SI (float)

dtype: float | int | bool

role: continuous | discrete_count | categorical_encoded

normalization: none | standardize | log | scale_to_unit_interval

Each TargetSpec must include:

name

unit

dimension

bounds

constraint_set: list of constraints (see section 6)

evaluation_transform: identity | log | log1p | signed_log | etc (must be dimensionally legal)

### 3.3 Canonical target representations (critical for macromodel work)

Vector fitting outputs poles/residues that are not automatically aligned across samples. M6 must define canonicalization pipelines per target family:

#### 3.3.1 Canonical form for stable complex poles

Represent each pole as:

a = -σ ± j ω, with σ > 0, ω >= 0

Learn symbolic formulas for:

σ(x) (units 1/s)

ω(x) (units 1/s)

This makes stability constraints easy and avoids “Re(a)<0” in raw complex form.

#### 3.3.2 Conjugate pairing and realness constraints

If the time‑domain response must be real, then poles and residues must occur in complex conjugate pairs. The canonicalizer must:

detect pairs,

store only one member per pair (or store (σ, ω)).

#### 3.3.3 Residue parameterization options

We define two supported residue models (choose per task; both implemented):

Option A — elementwise residues
Learn expressions for each independent residue matrix entry (e.g., upper triangle) and enforce symmetry/reciprocity by reconstruction.
Pros: straightforward.
Cons: passivity harder; many targets.

Option B — low‑rank PSD factorization (preferred for passivity)
Parameterize residue matrices as:

R_k(x) = B_k(x) B_k(x)^H (or sum of a few rank‑1 terms),
ensuring PSD by construction (if appropriate for the representation).
Pros: passivity constraints become more tractable.
Cons: requires careful derivation for the chosen network representation.

M6 must support both, but Option A is the minimum viable for first implementation; Option B is the “ideal” path for stronger physical guarantees.

## 4. System architecture (repo modules and responsibilities)
### 4.1 High‑level block diagram
            +---------------------------+
            | EquationDiscoveryRunner   |
            | (orchestrates pipeline)   |
            +-------------+-------------+
                          |
                          v
 +----------------+   +---+-------------------+   +----------------------+
 | Preprocessing  |-->| Candidate Generation  |-->| Unified GPU Scoring   |
 | & Canonicalize |   | (multiple engines)    |   | + Constraint Engine   |
 +----------------+   +---+-------------------+   +----------+-----------+
                          |                                 |
                          v                                 v
                 +------------------+             +----------------------+
                 | Pareto Frontier  |<------------| Postfit + Simplify   |
                 | + Model Selection|             | + Export             |
                 +------------------+             +----------------------+
                          |
                          v
                 +------------------+
                 | Artifacts +      |
                 | Certificates     |
                 +------------------+

### 4.2 Mandatory repository packages (M6 namespace)

A suggested (but strict) module map:

m6/

dataset/

schemas.py (Pydantic models: FeatureSpec/TargetSpec/etc)

io.py (parquet/arrow read/write, validation)

splits.py (domain splits, group CV)

canonicalization/

pole_tracking.py

residue_models.py

target_transforms.py

units/

dimensions.py (dimension vectors, operator rules)

pint_registry.py

sympy_units.py

expressions/

ir.py (ExpressionIR, Node, Operator)

parsers.py (from SymPy/PySR/AI‑Feynman strings)

simplify.py (SymPy rewrite pipeline, normalization)

hashing.py (stable expression hash)

constraints/

base.py (Constraint interface)

dimension.py (hard dimension typing + proof)

domain.py (finite, no singularities, bounded eval)

physics.py (stability/passivity/reciprocity/monotonicity)

repair.py (constraint‑preserving transforms)

engines/

base.py (EngineAdapter interface)

pysr_adapter.py

aifeynman_adapter.py

pysindy_adapter.py

grammar_search.py (our typed grammar sampler/enumerator)

optional/ (SISSO/Operon etc as future adapters)

scoring/

metrics.py (param loss, end‑to‑end loss)

gpu_eval.py (CuPy/Torch compiled evaluation)

pareto.py (Pareto maintenance, domination checks)

robustness.py (bootstrap, noise injection, OOD tests)

export/

codegen.py (C/CUDA/Python export)

latex.py

model_package.py (artifact bundling)

cli/

discover.py

score.py

export.py

reports/

certificates.py

html_report.py

## 5. Candidate generation engines (design and contracts)
### 5.1 EngineAdapter interface (strict)

All engines implement:

fit(dataset: EquationDiscoveryDataset, task: TaskSpec) -> EngineRunResult

yield_candidates() -> Iterator[CandidateExpression] (streaming)

stop() / cleanup()

get_run_manifest() -> dict (engine versioning, seeds, config)

CandidateExpression must contain:

expr_ir: ExpressionIR

expr_source: original string / engine object

engine_name, engine_version

target_name(s)

raw_metrics (engine’s own loss/complexity if available)

time_created, seed, config_hash

### 5.2 PySR / SymbolicRegression.jl adapter

PySR is used for:

fast baseline discovery on low‑dimensional problems,

generating an initial Pareto set,

enabling template structures and unit constraints (via backend).

Key implementation requirements:

Dimensional constraints integration
Use SymbolicRegression.jl’s unit/dimensional support: it can accept units (via DynamicQuantities) and includes hyperparameters like dimensional_constraint_penalty and dimensionless_constants_only.

Template / structure priors
Use TemplateExpressionSpec (or equivalent) to enforce known structure and variable access constraints. TemplateExpression is designed for domain‑specific structure constraints.

Performance knobs
Expose in config:

population size, iterations, procs/threads

turbo / LoopVectorization acceleration (when supported)

optional Bumper acceleration hooks when available

Export formats
PySR can export expressions to SymPy and other formats (including JAX and PyTorch), which we will use for GPU‑side re‑scoring and codegen.

### 5.3 AI Feynman 2.0 adapter

AI Feynman is used to:

exploit modularity/symmetry discovery,

handle noisy datasets robustly,

produce complementary candidates that PySR may miss.

We integrate it as:

a batch candidate generator + decomposition suggestions,

plus its discovered substructure used to generate templates for PySR/grammar search.

AI Feynman 2.0 seeks Pareto‑optimal formulas and includes techniques leveraging gradient properties of neural nets, normalizing flows, and hypothesis testing to improve robustness and search speed.

### 5.4 PySINDy adapter

Used when the task is naturally expressed as sparse regression over a library (especially dynamic/state‑space). PySINDy is explicitly a system identification package built around SINDy (Brunton et al. 2016) and includes additional optimizers.

### 5.5 Typed Grammar Search engine (must exist)

This is our engine for enforcing hard dimensional legality at generation time and exploiting GPU scoring.

It is required because:

dimensional constraints reduce the search space dramatically,

we cannot rely on “soft penalties” for correctness.

We will implement grammar search using attribute grammar ideas: dimensional constraints can be represented via attribute grammars and transformed into probabilistic grammars, improving benchmark equation discovery results (reported improvements on Feynman benchmarks).

This engine is the key to “physics‑guided symbolic search” rather than generic GP.

## 6. Constraint system (hard gates + penalties + repairs)

Constraints are not optional. M6 must treat constraints as first‑class citizens in both candidate generation and candidate acceptance.

### 6.1 Constraint types

We define constraints in four tiers:

Dimensional constraints (hard)

Domain safety constraints (hard)

Physical sign/stability constraints (hard)

System/network constraints (hard, but may be “probabilistically certified” over grids)

### 6.2 Dimensional constraints (hard proof)
#### 6.2.1 Dimension representation

Represent dimension as integer/rational exponent vector over base dimensions:

Minimum for EM/macromodel work:
[L, M, T, I] (length, mass, time, current)

Full SI (optional):
[L, M, T, I, Θ, N, J]

Every feature/target must specify its dimension vector.

#### 6.2.2 Operator legality rules

Operators must carry dimension typing rules, e.g.:

+/-: same dim required

*: dims add

/: dims subtract

pow(x, p):

p must be dimensionless

if p is rational, result dims scaled by p

sqrt(x): dims * 1/2

log/exp/sin/cos: argument must be dimensionless (output dimensionless)

abs: preserves dim

max/min: same dim

sign: dimensionless

#### 6.2.3 Enforcement modes

M6 must support three modes:

Strict typed generation: candidates are generated only if dimensionally valid (grammar search; preferred).

Strict post‑check: candidates from external engines are rejected unless dimensionally valid.

Soft penalty: only for exploration; never for final acceptance.

Note: SymbolicRegression.jl already supports dimensional analysis with hyperparameters like dimensional_constraint_penalty and can restrict constants to be dimensionless (dimensionless_constants_only).
We will still apply our own post‑check to guarantee correctness.

#### 6.2.4 Unit systems integration

Pint provides practical unit registry and references Buckingham Pi theorem in its advanced topics, indicating it’s used for dimensional analysis workflows.

SymPy provides physics.units capabilities for symbolic unit handling.

M6 will:

store units in Pint for I/O and conversions,

store symbolic dimensions in a lightweight internal structure for speed,

optionally use SymPy for final expression unit annotation and export checks.

### 6.3 Domain safety constraints (hard)

These prevent “fits” that exploit undefined behavior.

Required checks:

no NaN/Inf on train/val/test

denominator bounded away from zero over declared domain (with margin)

log/sqrt domain legality across domain

no catastrophic overflow under float32 evaluation across domain

bounded gradients (optional; used for robust physical behavior)

Implementation:

A GPU batched evaluator computes:

isfinite masks,

denominator minima,

domain‑guard booleans.

Any violation → candidate rejected or repaired (if repairable).

### 6.4 Physical constraints (hard)

These are target‑dependent and declared in TargetSpec.constraint_set.

Minimum required constraints for our macromodel mission:

#### 6.4.1 Positivity constraints

For parameters like R, L, C, σ, ω:

enforce > 0 or >= 0 with margin

Implementation:

post‑fit check + optional “repair transform” (see 6.6).

#### 6.4.2 Stability constraints for poles

If target is a pole parameterization:

require σ(x) > 0 (since pole real part is -σ), thus Re(a)<0.

#### 6.4.3 Reciprocity / symmetry

For reciprocal N‑port networks, enforce matrix symmetry constraints, e.g. S = S^T (or on Z/Y depending representation).
This can be implemented by:

learning only independent entries,

reconstructing full matrix deterministically,

verifying residual symmetry error.

### 6.5 System/network constraints (hard, but certified on grids)

These constraints validate the resulting network behavior from predicted θ.

For each candidate formula set producing a macromodel:

reconstruct frequency response on a grid,

verify:

passivity,

stability,

(optional) causality consistency checks.

Because exact symbolic passivity proof is difficult, we require grid‑certification with adversarial augmentation:

deterministic frequency grid over the band,

plus random frequency samples,

plus worst‑case “stress grid” near resonances.

The certificate must record:

grid definition,

max violation margin,

GPU numeric precision used,

commit + dataset hash.

### 6.6 Repair operators (constraint‑preserving transforms)

Repairs are allowed only if they preserve interpretability and are explicitly recorded.

Example repairs:

positivity: y = g(x)^2 or y = (g(x))^2 + ε

negativity: y = -g(x)^2

boundedness: y = y_max * tanh(g(x)) (used only if physically justified; usually avoid)

stability: σ = g(x)^2 then a = -σ + j ω

Repair policy:

repairs are applied only as:

outermost wrapper, or

within template‑declared slots,
so the “core” formula remains readable.

## 7. Physics‑guided search (how we bias toward true, novel laws)

Physics guidance is not “add a penalty.” It must shape the hypothesis space.

### 7.1 Dimensional analysis + dimensionless groups

We will implement automated Buckingham‑Pi feature construction:

Given dimensional matrix D ∈ ℚ^{k × d} for d variables in k base dims:

compute nullspace basis Π = {π_1, …, π_{d-k}}

each π is a product of variables with exponents yielding dimensionless.

We then offer three discovery modes:

Pure π‑space SR: discover ŷ = f(π_1,…,π_m) where y has been nondimensionalized.

Hybrid: allow both raw vars and π‑groups with strong priors for π usage.

Template scaling: enforce y = y0(x) * f(π) where y0 provides dimensional scaling.

This dramatically improves generalization across scale changes and reduces search complexity.

### 7.2 Domain splits (anti‑overfitting by construction)

Beyond train/val/test, we require domain splits that reflect our real use:

Interpolation split: random holdout in the interior of parameter ranges.

Extrapolation split: hold out corners and edges of the parameter hypercube.

Regime split: hold out regions known to cause behavior shifts (e.g., stub resonance entering the band).

A candidate is not “accepted” unless it meets thresholds on all declared splits.

### 7.3 Structural priors via templates

Use SymbolicRegression.jl template expressions to encode known physical structure and variable access constraints. TemplateExpression is explicitly designed to impose structured combinations of subexpressions and constrain variable usage.

Examples relevant to our mission:

rational forms: y = (a0 + a1*π1 + …) / (1 + b1*π2 + …)

log‑like inductance behavior templates: L ~ α * log(β * ratio + 1) + γ

separable structure: f(x1,x2,x3) = g(x1,x2) + h(x3)

### 7.4 Decomposition priors from AI Feynman

AI Feynman 2.0 can discover modularity/symmetry structures and is designed for robust Pareto‑optimal discovery.
We will:

use AI Feynman’s discovered modular forms to propose templates,

reduce dimensionality before PySR/grammar search.

### 7.5 Sparse library priors (SINDy‑style)

When we can propose a physically motivated library (polynomials in π‑groups, rational basis, log terms), we use sparse regression:

find minimal active terms,

then “lift” into symbolic expression form.

PySINDy provides this sparse system identification machinery.

## 8. Unified scoring: end‑to‑end, GPU‑dominant, multi‑objective
### 8.1 Why end‑to‑end scoring is mandatory

If we learn formulas for intermediate θ (e.g., poles), small parameter error can yield large response error. Therefore every candidate formula must be scored on:

parameter loss: error in θ space (fast)

response loss: error in reconstructed S/Z/Y over frequency (ground truth)

constraint loss: violations (hard rejection or heavy penalty)

complexity cost: expression size and operator costs

### 8.2 Metrics definitions (must be implemented)

For each candidate f:

Parameter loss (per target):

Lθ = median_i ( |θ_i - f(x_i)| / (|θ_i| + ε) )
(median for robustness; also compute max and 95th percentile)

Response loss (band‑weighted):

for each sample i, compute response error across freq:

magnitude + phase or complex error

require:

worst‑case (max) error ≤ threshold

and band‑weighted RMS error ≤ threshold

Constraint gates:

dimensional validity: pass/fail

domain safety: pass/fail

physics constraints: pass/fail

network constraints: pass/fail (grid certificate)

Complexity
We use two complexity measures:

Engine complexity (node count or engine’s complexity definition)

Operator‑weighted complexity:

assign weights: {+, -, *} = 1, / = 2, log/sqrt = 3, exp = 5, etc.

complexity = sum of node weights

We keep a Pareto frontier of (loss, complexity). AI Feynman explicitly targets Pareto optimality; we will maintain our own cross‑engine Pareto set.

SymbolicRegression.jl also uses a “score” notion based on change in log‑loss vs change in complexity between neighboring frontier points, and we will store this as a diagnostic metric.

### 8.3 GPU evaluation architecture (must be built)

M6 must reuse the GPU acceleration concepts from M5, but it must also function independently with a defined API.

#### 8.3.1 Expression evaluation backends

We implement ExpressionEvaluator with backends:

cupy_fused: preferred, highest throughput

torch_compile: optional (torch.compile + CUDA)

numpy: correctness baseline only

Key requirement: candidates are scored in large batches, and evaluation is GPU‑dominant.

#### 8.3.2 Kernel fusion strategy (essential)

Naively evaluating an expression tree as many small GPU ops causes kernel launch overhead. M6 must implement at least one of:

#### A) Codegen to a fused CUDA kernel (ideal)

Convert ExpressionIR to CUDA source (single kernel doing full expression)

Compile via CuPy RawKernel / NVRTC

Cache compiled kernels by expression hash

#### B) Torch graph fusion (acceptable if stable)

Convert ExpressionIR to Torch ops

Use torch.compile to fuse where possible

Cache compiled graphs by expression hash

The repo must include benchmark evidence showing fusion effectiveness.

#### 8.3.3 Batched end‑to‑end response reconstruction

Given predicted θ(x), reconstruct S(f;x) on GPU:

batched over N samples,

batched over F frequencies.

Then compute response loss and passivity constraints using GPU linear algebra:

compute SᴴS eigenvalues/singular values per frequency (for passivity in S‑domain),

compute symmetry errors.

Chunking is mandatory to avoid VRAM thrash (24 GB cap).

### 8.4 Asynchronous scoring pipeline (CPU generates, GPU scores)

To saturate hardware:

CPU threads/processes generate candidates (PySR/AI Feynman/grammar)

candidates are queued to GPU scorer

GPU scorer evaluates in micro‑batches and returns metrics

Pareto set is updated online

This prevents GPU idle time during heavy symbolic search.

## 9. Constant handling and post‑fit refinement (must be present)

Symbolic regression engines often optimize constants internally, but for engineering rigor we require post‑fit refinement under our end‑to‑end objective.

### 9.1 Constant types

We distinguish:

dimensionless constants (preferred default)

units‑carrying constants (allowed only if explicitly enabled and recorded)

SymbolicRegression.jl supports settings around dimensionless constants vs dimensional penalties.

### 9.2 Post‑fit refinement algorithm

For each candidate expression, we optionally run:

Quick linear solve if the expression is linear in parameters (detectable via SymPy).

Else bounded nonlinear optimization on GPU:

objective = response loss + regularization

constraints enforced (e.g., positivity via squaring)

optimizer = LBFGS/Adam (torch) or custom (cupy)

Refinement output must be recorded:

refined constants,

improvement metrics,

time cost.

### 9.3 Symbolic distillation path (optional but designed in)

If the mapping is higher dimensional than SR can handle directly, we support the PySR‑documented “symbolic distillation” workflow: train a neural net surrogate then distill it into a symbolic equation.

This is not the primary path for our low‑dimensional via family, but it must be supported because it can unlock future expansions (more geometry parameters, more complex families).

## 10. Output artifacts (what the repo must emit)

Every discovery run must emit a deterministic artifact bundle:

### 10.1 Required artifacts

run_manifest.json

git commit hash

dataset DVC hash

engine configs + versions

seeds

hardware info (GPU model, CUDA version)

numeric precision settings (float32/float64)

pareto_frontier.json

all non‑dominated candidates

metrics (loss, complexity, constraints pass/fail)

engine provenance per candidate

champions/

top K formulas chosen by model selection policy

certificates/

dimensional_certificate.json: proof traces / dimension inference results

physics_certificate.json: stability/positivity checks

network_certificate.json: passivity/reciprocity checks on declared grids

reports/

human‑readable HTML/PDF summary (plots, tables)

exports/

formula.sympy

formula.tex

formula.py

optionally formula.c / formula.cu

### 10.2 Deterministic hashing

Expression canonical form must produce a stable hash:

commutative normalization (a+b sorted)

constant normalization (float rounding policy)

operator canonicalization

This enables caching compiled GPU kernels and avoids duplicate evaluations.

## 11. CLI + API specification (must be stable)
### 11.1 CLI commands

m6 discover --config task.yaml --dataset <path> --out <run_dir>

m6 score --formula <formula.json> --dataset <path> --report <out>

m6 export --formula <formula.json> --format {py,c,cu,tex,sympy} --out <dir>

m6 certify --formula <formula.json> --dataset <path> --domain-grid <grid.yaml>

### 11.2 Python API

Must expose:

discover(task: TaskSpec, dataset: EquationDiscoveryDataset) -> DiscoveryResult

score(formula: FormulaBundle, dataset: EquationDiscoveryDataset) -> ScoreReport

certify(formula: FormulaBundle, dataset: EquationDiscoveryDataset, domain: DomainSpec) -> CertificateBundle

compile(formula: FormulaBundle, backend: str) -> CompiledEvaluator

All APIs must be:

type‑hinted

deterministic under fixed seeds

fail‑fast on contract violation

## 12. Testing and verification plan (how we know it’s correct)

This is the “we will know for certain” section: objective pass/fail gates.

### 12.1 Unit tests (must exist)

Dimension inference correctness

generate random expression trees + random dimension assignments

verify operator legality

verify inference matches expected dim arithmetic

Parser/IR roundtrip

SymPy ↔ IR ↔ SymPy must be stable (modulo canonical formatting)

GPU evaluator correctness

compare GPU evaluation vs NumPy baseline across random inputs and random expressions

require max relative error ≤ 1e‑6 (float32) for stable ops

Constraint checks correctness

craft expressions that violate each constraint type and ensure rejection

Kernel cache determinism

same expression hash → same compiled kernel signature

### 12.2 Integration tests (must exist)
#### 12.2.1 “Known law recovery” suite

A set of synthetic datasets with ground truth formulas:

dimensionless laws,

log‑like laws,

rational laws,

multi‑output laws,

stable pole parameterizations.

Each test requires the system to:

rediscover a formula within a complexity bound,

achieve error ≤ threshold on extrapolation split,

pass all dimensional constraints.

#### 12.2.2 Noise robustness suite

For each synthetic dataset:

inject noise (Gaussian + heavy‑tailed)

require the engine still recovers either:

the true formula, or

a mathematically equivalent formula within tolerance.

AI Feynman 2.0 is explicitly designed to improve robustness to noise/bad data; this test ensures our integration and scoring doesn’t destroy that advantage.

#### 12.2.3 “Macromodel consistency” suite

Using a small set of known passive rational models:

generate synthetic S‑parameter responses,

“fit” θ,

run M6 to rediscover θ(x) laws,

confirm reconstructed S passes passivity checks.

### 12.3 Benchmark tests (must exist)

We will include a lightweight SR benchmark harness referencing SRBench’s style:

scikit‑learn compatible runner

run on a fixed subset of benchmark problems

assert that our engine meets baseline success rates and doesn’t regress.

SRBench exists explicitly to provide reproducible comparisons across SR methods and datasets.

### 12.4 Performance tests (must exist)

We define two benchmark tiers:

Tier 1 — microbench
Evaluate a set of ~100 expressions across:

N=100k samples, d~20

require throughput target (see 13)

Tier 2 — end‑to‑end scorebench
Reconstruct and score responses for:

N=2048 samples, F=512 frequencies, 2‑port

require total scoring time per candidate ≤ T_target

Each benchmark must print:

GPU utilization evidence,

kernel launch counts (if feasible),

VRAM usage,

CPU fallback warnings (must be zero).

## 13. Performance targets (laptop‑feasible but aggressive)

These are not “nice to have.” They are acceptance gates.

### 13.1 GPU scoring throughput targets

On the single‑laptop GPU (24 GB VRAM class):

Expression evaluation:

Achieve ≥10× speedup vs NumPy CPU baseline for N≥100k, complexity≥30 nodes.

End‑to‑end response scoring (2‑port):

For one candidate formula set (producing θ, reconstructing S on F=512):

≤ 50 ms for N=2048 (chunked if needed).

These numbers are adjustable after the first profiling pass, but M6 must include benchmarks and enforce non‑regression.

### 13.2 Memory management requirements

All GPU operations must support chunking in N and/or F dimensions.

No intermediate tensor > 60% VRAM without explicit override.

Kernel compilation cache must have eviction policy.

### 13.3 Determinism requirements

All “final selection” runs must be reproducible:

same seed + same dataset hash + same commit = same champion formulas (within allowed tie‑break nondeterminism rules that are explicitly defined).

## 14. Definition of Done (DoD): the exact checklist

M6 is complete only when all items below are true.

### 14.1 Functional completeness

 m6 discover runs end‑to‑end on a provided dataset and emits the full artifact bundle (section 10).

 PySR adapter integrated and passing integration tests.

 AI Feynman 2.0 adapter integrated and passing integration tests.

 PySINDy adapter integrated (at least for one dynamic identification test).

 Typed grammar engine exists and can generate dimensionally valid candidates (section 5.5).

### 14.2 Correctness + physics rigor

 Dimensional correctness is guaranteed: every accepted formula has a dimensional certificate.

 Domain safety gates prevent NaN/Inf/singularities across declared domains.

 Stability constraints enforced for pole parameterizations.

 Passivity/reciprocity grid‑certificates generated for network outputs.

### 14.3 Robustness

 Known‑law recovery suite passes on:

interpolation split,

extrapolation split,

noise‑injected split.

 Candidate selection is not based solely on train loss; validation and OOD are enforced.

### 14.4 Performance

 GPU scoring benchmarks meet targets (section 13) and are enforced in CI (or local gated workflow if CI GPU unavailable).

 No silent CPU fallback: if GPU backend requested and unavailable, run fails loudly.

### 14.5 Reproducibility and auditing

 Every run emits run_manifest.json with dataset hash and commit.

 Every candidate in pareto set stores provenance (engine + config + seed).

 Re‑running with same inputs reproduces the same frontier and champion set.

## 15. Engineering rigor standards (repo‑level expectations for M6 code)

Even though M0 establishes repo substrate, M6 must adhere to these local standards:

Type‑driven interfaces (Pydantic models for configs and artifacts)

Pure functions where possible (to aid determinism)

No implicit global state (units registry must be explicit)

Structured logging for every candidate scoring event

Property‑based tests for dimension rules and IR evaluation

Version pinning in manifests (engine versions recorded)

## 16. Why this design maximizes the odds of discovering a novel equation

This M6 design increases discovery probability by combining:

Multiple complementary SR paradigms (evolutionary SR + modularity‑based SR + sparse regression)

Hard dimensional legality, which prevents huge volumes of nonsense candidates and forces the search into physically meaningful spaces (supported by both SymbolicRegression.jl unit tooling and attribute‑grammar approaches).

Physics‑guided structural priors (templates and variable constraints) to drastically reduce the hypothesis space and direct search toward plausible macromodel laws.

GPU‑dominant scoring, enabling far larger candidate throughput and more aggressive validation/certification than CPU‑only SR (critical on single‑laptop constraints).

End‑to‑end response‑level validation (not just parameter fitting), which is the difference between a publishable “fit” and a deployable “formula.”

## 17. Minimal “first implementation” vs “absolute ideal” (so you can stage without compromising rigor)

You asked for the absolute ideal. Here’s the minimum subset that still meets strict standards, and what remains “ideal‑plus.”

### 17.1 Minimum viable but rigorous (still passes DoD)

PySR adapter + typed dimension post‑checker

AI Feynman adapter (candidate generator)

Grammar engine for dimensionally consistent candidate sampling (even if simple)

GPU evaluator for:

expression evaluation,

end‑to‑end S reconstruction and error

Dimensional + stability + reciprocity constraints

Passivity certification on grids for 2‑port S

### 17.2 Ideal‑plus (recommended for maximum discovery odds)

Low‑rank PSD residue parameterization (passivity by construction)

Post‑fit GPU constant refinement for top candidates

Distillation path (NN → SR) to scale feature count

Full SRBench regression harness across a curated subset.

## 18. What you should expect M6 to produce for our specific mission (via/launch formula)

Once M6 is implemented, the repo must be capable of producing, for example:

A symbolic law for σ_k(x) and ω_k(x) (stable pole trajectories) with correct units (1/s),

A symbolic law for equivalent circuit values (R/L/C) with positivity constraints,

A symbolic design law for “optimal antipad radius / return‑via pitch” expressed as dimensionless ratios and scaling.

And crucially: the formula must ship with certificates proving:

unit correctness,

stability,

passivity/reciprocity (on declared bands/domains),

robust error bounds on held‑out corners.

That is the point where you can say, with engineering certainty, “we have properly developed M6.”

## Normative Requirements

The following requirements are normative for M4 and must be satisfied before the milestone is considered complete:

- [REQ-M4-001] Design document must pass the spec lint contract (milestone line, requirements, Definition of Done, Test Matrix)
- [REQ-M4-002] Vector fitting implementation must produce rational macromodels from S-parameter data
- [REQ-M4-003] Passivity enforcement must guarantee all poles have non-positive real parts
- [REQ-M4-004] SPICE export must produce valid subcircuit netlists from macromodels

## Definition of Done

M4 is complete when all of the following criteria are met:

1. All REQ-M4-XXX requirements pass their mapped tests
2. Vector fitting produces stable, passive macromodels from openEMS S-parameter outputs
3. SPICE subcircuits generated from macromodels simulate correctly in ngspice
4. End-to-end pipeline from coupon → S-parameters → macromodel → SPICE is demonstrated
5. Documentation and test coverage meet project standards

## Test Matrix

| Requirement | pytest node id(s) |
|-------------|-------------------|
| REQ-M4-001 | tests/test_design_document_contract.py::test_design_document_contract |
| REQ-M4-002 | tests/test_m2_sparam_extract.py::test_extract_sparams_from_touchstone |
| REQ-M4-003 | tests/test_m2_touchstone_validation.py::TestPassivityValidation::test_passive_network_passes |
| REQ-M4-004 | tests/test_m2_touchstone_validation.py::TestReciprocityValidation::test_reciprocal_network_passes |
