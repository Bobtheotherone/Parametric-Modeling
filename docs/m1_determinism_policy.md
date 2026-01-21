# M1 Determinism Policy

**Last Updated:** 2026-01-20
**Satisfies:** CP-5.1, CP-5.2, D5, D6 (ECO-M1-ALIGN-0002)

This document specifies the determinism requirements for M1 compliance in the parametric coupon generation pipeline. It ensures that identical inputs produce identical, verifiable outputs suitable for physics dataset generation and symbolic discovery.

---

## 1. Overview

The M1 determinism policy requires:
- Bit-for-bit reproducible outputs for the same inputs and toolchain
- Complete provenance tracking via cryptographic hashes
- Verifiable builds using pinned Docker images

---

## 2. Hash Inputs

### 2.1 Design Hash (`design_hash`)

The design hash uniquely identifies the resolved design parameters:

```
design_hash = SHA256(canonical_json(resolved_design))
```

**Included in design hash:**
- All resolved design parameters in integer nanometers (nm)
- Derived features (aspect ratios, impedances)
- Dimensionless groups
- Board dimensions and layer stackup
- Via, trace, and pad geometries
- Coupon family identifier (F0, F1, etc.)
- Fabrication profile constraints

**Explicitly excluded from design hash:**
- Timestamps (tracked in lineage)
- Git SHA (tracked in lineage)
- Toolchain version (tracked in toolchain_hash)
- Build environment variables
- Absolute file paths

### 2.2 Toolchain Hash (`toolchain_hash`)

The toolchain hash captures the complete build environment:

```
toolchain_hash = SHA256(canonical_json(toolchain_metadata))
```

**Included in toolchain hash:**
- `schema_version`: Lock file schema version
- `kicad_version`: Pinned KiCad version (e.g., "9.0.7")
- `docker_image`: Docker image reference
- `docker_digest`: SHA256 digest of pinned Docker image

---

## 3. Exclusions from Hashes

### 3.1 Timestamp Exclusions

The following timestamps are intentionally excluded from determinism hashes:

| Field | Location | Reason |
|-------|----------|--------|
| `timestamp_utc` | `lineage` object | Build time varies per run |
| KiCad file timestamps | `.kicad_pcb` | KiCad adds current time |
| Gerber date comments | `G04` lines | Non-functional metadata |
| DRC report timestamps | `date`, `time` keys | Report generation time |

### 3.2 Path Exclusions

Absolute file paths are excluded/normalized because they vary by environment:

- DRC report `source` paths are removed
- DRC report `filename` paths are reduced to basename
- Paths longer than 20 characters containing `/` are detected and removed

### 3.3 UUID Exclusions

KiCad object UUIDs are excluded from canonical representations:

- `(tstamp UUID)` patterns are stripped
- `(uuid UUID)` patterns are stripped
- UUIDs are retained in actual board files for KiCad compatibility

---

## 4. Digest Pinning

### 4.1 Docker Image Pinning

Docker images MUST be pinned by SHA256 digest for production builds:

```json
{
  "docker_image": "kicad/kicad:9.0.7",
  "docker_digest": "sha256:abc123def456..."
}
```

**Requirements:**
- `docker_digest` must NOT be `"sha256:PLACEHOLDER"` for production
- Use `tools/pin_kicad_image.py` to resolve actual digest
- CI/production builds MUST fail on placeholder digests

### 4.2 Pinned Image Reference Format

The pinned image reference combines tag and digest:

```
kicad/kicad:9.0.7@sha256:abc123def456...
```

This ensures:
- Same content even if upstream tag is moved
- Audit trail for exact image used
- Reproducible builds across environments

### 4.3 KiCad CLI Version Capture

For docker builds, the actual `kicad-cli --version` output MUST be captured:

```json
{
  "toolchain": {
    "kicad": {
      "version": "9.0.7",
      "cli_version_output": "KiCad 9.0.7 built on ..."
    }
  }
}
```

This provides:
- Verification that pinned image matches expected version
- Audit trail for debugging version mismatches
- Proof of actual toolchain used (not just configured)

---

## 5. Determinism Gates

M1 compliance is verified through these determinism gates:

| Gate | Name | Verification Method |
|------|------|---------------------|
| G1 | Schema + resolver determinism | Same spec → same `resolved_design` hash |
| G2 | Constraint proof completeness | All Tier 0-3 constraints evaluated |
| G3 | DRC cleanliness | `kicad-cli pcb drc` exit code 0 |
| G4 | Export completeness | All expected layers present |
| G5 | Hash stability | 3 consecutive runs produce identical hashes |

### 5.1 Gate Failure Handling

- **G1 Failure**: Investigate non-canonical JSON serialization or floating-point rounding
- **G2 Failure**: Review constraint engine for missing evaluations
- **G3 Failure**: Fix design rule violations in board layout
- **G4 Failure**: Check export layer configuration
- **G5 Failure**: Identify and eliminate nondeterminism source

---

## 6. Canonicalization Algorithms

### 6.1 Canonical JSON

All JSON hashing uses canonical representation:
- Keys sorted alphabetically (recursive)
- Minimal whitespace: no spaces after `:` or `,`
- UTF-8 encoding
- No trailing newlines

Implementation: `formula_foundry.substrate.canonical_json_dumps()`

### 6.2 KiCad Board Canonicalization

```
1. Remove (tstamp ...) patterns
2. Remove (uuid ...) patterns
3. Normalize CRLF → LF
4. Normalize CR → LF
```

### 6.3 Gerber Canonicalization

```
1. Remove G04 comment lines
2. Remove lines starting with ;
3. Normalize CRLF → LF
4. Trim trailing whitespace per line
```

### 6.4 DRC JSON Canonicalization

```
1. Remove date, time, timestamp keys
2. Remove source, filename absolute paths (retain basename)
3. Remove kicad_version key
4. Sort keys alphabetically
5. Stable array ordering where semantics allow
```

---

## 7. Implementation References

| Component | Location |
|-----------|----------|
| Canonical JSON | `formula_foundry.substrate.canonical_json_dumps()` |
| SHA256 hashing | `formula_foundry.substrate.sha256_bytes()` |
| Toolchain config | `formula_foundry.coupongen.toolchain.lock` |
| DRC canonicalization | `formula_foundry.coupongen.kicad.canonicalize` |
| Export hashing | `formula_foundry.coupongen.hashing` |

---

## 8. References

- ECO-M1-ALIGN-0002: Engineering Change Order for M1 compliance
- docs/DETERMINISM_POLICY.md: General determinism policy
- docs/m1_toolchain.md: Toolchain pinning and configuration
- KiCad CLI documentation: https://docs.kicad.org/9.0/en/cli/cli.html
