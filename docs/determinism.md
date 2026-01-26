# Determinism Guarantees and Limitations

**Last Updated:** 2026-01-25
**Applies to:** M0 and M2 pipelines (coupon generation and oracle simulation)

This document describes the determinism guarantees, limitations, and known sources of nondeterminism in the Formula Foundry pipeline. It sets correct expectations for reproducibility and auditability.

---

## 1. Overview

The pipeline provides determinism at two levels:

1. **Coupon Generation (M0/M1)**: PCB geometry generation via KiCad
2. **Oracle Simulation (M2)**: EM simulation via openEMS with postprocessing

Both require explicit toolchain pinning and seed management to achieve reproducibility.

---

## 2. Determinism Guarantees

### 2.1 What Is Guaranteed (Strict Mode)

When running in `--mode strict`, the pipeline guarantees:

| Guarantee | Mechanism |
|-----------|-----------|
| Same inputs → same outputs | Canonical JSON, sorted keys, normalized whitespace |
| Reproducible random operations | `PYTHONHASHSEED` pinned and recorded |
| Stable NumPy/CuPy operations | Seeds applied via `np.random.seed()` and `cp.random.seed()` |
| Stable PyTorch operations | `torch.manual_seed()` + deterministic algorithms enabled |
| Stable cuDNN behavior | `torch.backends.cudnn.deterministic = True` |
| No cuDNN benchmark variability | `torch.backends.cudnn.benchmark = False` |
| Stable cuBLAS operations | `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Toolchain identity | OCI images pinned by SHA256 digest |
| Mesh reproducibility | Mesh summary hash recorded and verified |
| S-parameter reproducibility | Per-port raw waves stored for offline recomputation |

### 2.2 Toolchain Pinning

Toolchain components are pinned to ensure reproducibility:

| Component | Pinning Method | Record Location |
|-----------|----------------|-----------------|
| KiCad | Docker image by SHA256 digest | `toolchain/kicad.lock.json` |
| openEMS | OCI container by SHA256 digest | `meta.json` provenance |
| gerber2ems | OCI container by SHA256 digest | `meta.json` provenance |
| Python packages | `requirements.lock` or `poetry.lock` | Repository root |

**Example pinned reference:**
```
kicad/kicad:9.0.7@sha256:abc123def456...
```

### 2.3 Hashing and Provenance

All outputs include cryptographic provenance:

- **`design_hash`**: SHA256 of canonical resolved design (coupon generation)
- **`toolchain_hash`**: SHA256 of toolchain metadata
- **`mesh_hash`**: SHA256 of mesh summary (simulation)
- **`case_hash`**: SHA256 of normalized oracle case config

See `docs/DETERMINISM_POLICY.md` for hash computation algorithms.

---

## 3. Limitations and Known Nondeterminism

### 3.1 GPU/CPU Differences

**Cross-device nondeterminism is expected.** The same code may produce different results on:

| Factor | Impact |
|--------|--------|
| GPU model | Different kernel implementations |
| CUDA version | Numerical library changes |
| cuDNN version | Convolution algorithm differences |
| Driver version | Low-level kernel behavior |
| CPU architecture | SIMD instruction differences |

**Mitigation:** Strict mode is designed for repeatability on a *single machine and driver stack*. Cross-machine reproducibility requires identical hardware and software environments.

### 3.2 GPU-Specific Nondeterminism

Even in strict mode, some GPU operations are inherently nondeterministic:

| Source | Reason | Mitigation |
|--------|--------|------------|
| Atomic operations | Race conditions in parallel reductions | Use deterministic algorithms (slower) |
| cuDNN autotuning | Selects different kernels per run | Disable benchmarking |
| Mixed precision | Reduced precision amplifies rounding | Use full precision in strict mode |
| Parallel FFT | Order-dependent accumulation | Pin thread count |

### 3.3 GPU/CPU Fallback Behavior

The postprocessing pipeline uses GPU acceleration when available:

```
Primary: CuPy (GPU FFT and batch transforms)
Fallback: NumPy (CPU)
```

**Fallback rules:**
1. GPU backend is attempted first if `cupy` is importable
2. Fallback to CPU is **explicit and recorded** in `meta.json`
3. Silent fallback is never permitted
4. `meta.json` includes `gpu_backend.used` and `gpu_backend.fallback_reason`

**Example meta.json entry:**
```json
{
  "gpu_backend": {
    "requested": "cupy",
    "used": "numpy",
    "fallback_reason": "CuPy import failed: No CUDA device available"
  }
}
```

### 3.4 Floating-Point Nondeterminism

IEEE 754 floating-point arithmetic can introduce variation:

| Source | Impact |
|--------|--------|
| Operation order | `(a + b) + c ≠ a + (b + c)` |
| Compiler optimizations | FMA instructions, reassociation |
| Library versions | NumPy, SciPy, CuPy internals |
| Thread scheduling | Parallel reduction order |

**Mitigation:** Use canonical comparison tolerances (not bit-exact equality) for S-parameter verification.

### 3.5 Simulation-Specific Nondeterminism

openEMS simulations have additional nondeterminism sources:

| Source | Impact | Mitigation |
|--------|--------|------------|
| Thread count | Order of parallel operations | Pin `threads` in solver_policy |
| Time step | Accumulated numerical error | Record `time_step_factor` |
| Termination | End criteria vs max_steps | Record termination cause |
| Field export | Large data may vary | Disable by default |

---

## 4. Nondeterministic Steps (Explicitly Marked)

Some pipeline steps are intentionally nondeterministic:

### 4.1 Lineage Fields

These fields vary per run and are excluded from design/case hashes:

| Field | Location | Reason |
|-------|----------|--------|
| `timestamp_utc` | `lineage` / `meta.json` | Build time varies |
| `git_sha` | `lineage` / `provenance` | May change between runs |
| `runtime_seconds` | `meta.json` | Execution time varies |

### 4.2 Intermediate Artifacts

These are not guaranteed stable:

| Artifact | Reason |
|----------|--------|
| Raw KiCad UUIDs | KiCad generates unique IDs |
| Log file timestamps | Wall-clock dependent |
| Cache keys | Include timestamp components |

### 4.3 Stub Mode Outputs

When running with `--allow-stub`, outputs are labeled **NON-GOLD** and are explicitly not subject to determinism guarantees.

---

## 5. Verification and Auditing

### 5.1 Determinism Gates (CI)

| Gate | Description | Verification |
|------|-------------|--------------|
| G1 | Schema + resolver determinism | Same spec → same design_hash |
| G2 | Constraint proof completeness | All tiers evaluated |
| G3 | DRC cleanliness | Exit code 0 |
| G4 | Export completeness | All layers present |
| G5 | Hash stability | 3 runs → identical hashes |

### 5.2 Mesh Invariance Gate

For simulations, the mesh invariance gate verifies:
- Baseline mesh vs refined mesh
- ΔS thresholds across frequency band
- Threshold enforcement in CI (nightly)

### 5.3 Calibration Regression

Calibration structures (CAL-0 through CAL-6) are metric-based (not bitwise) to accommodate acceptable numerical variation while detecting genuine regressions.

---

## 6. Recommendations for Users

### 6.1 For Maximum Reproducibility

1. **Use pinned Docker images** — Never use floating tags (`:latest`)
2. **Run in strict mode** — `--mode strict`
3. **Pin Python dependencies** — Use lockfiles
4. **Use consistent hardware** — Same GPU model and driver
5. **Record full provenance** — Verify `meta.json` completeness

### 6.2 For Cross-Environment Comparison

1. **Use tolerance-based comparison** — Not bit-exact equality
2. **Compare canonical hashes** — Not raw file hashes
3. **Check fallback status** — Verify GPU/CPU backend match
4. **Review termination cause** — End criteria vs max_steps

### 6.3 Debugging Nondeterminism

If you observe nondeterminism:

1. Check `meta.json` for fallback reasons
2. Verify toolchain digest matches
3. Compare mesh summary hashes
4. Review thread count configuration
5. Check for unpinned dependencies

---

## 7. References

- `docs/DETERMINISM_POLICY.md` — Hash computation and canonicalization
- `docs/m1_determinism_policy.md` — M1-specific determinism envelope
- `docs/m1_toolchain.md` — Toolchain pinning details
- `DESIGN_DOCUMENT.md` — Pipeline architecture and provenance requirements
