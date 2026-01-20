# Determinism Policy for Coupon Generation

**Last Updated:** 2026-01-20
**Satisfies:** CP-5.1, CP-5.2, D5 (ECO-M1-ALIGN-0001)

This document specifies the determinism guarantees and hashing policies for coupon generation. It ensures that identical inputs produce identical, verifiable outputs suitable for physics dataset generation and symbolic discovery.

---

## 1. Overview

The coupon generation pipeline guarantees that:
- Same `CouponSpec` + same toolchain version → same resolved geometry and canonical hashes
- All build artifacts are traceable via cryptographic hashes
- Docker builds have complete provenance (no `"unknown"` values allowed)

---

## 2. What Is Hashed

### 2.1 Design Hash (`design_hash`)

The design hash uniquely identifies the resolved design parameters. It is computed as:

```
design_hash = SHA256(canonical_json(resolved_design))
```

**Included in hash:**
- All resolved design parameters in integer nanometers (nm)
- Derived features (e.g., aspect ratios, impedances)
- Dimensionless groups
- Board dimensions and layer stackup
- Via, trace, and pad geometries

**Excluded from hash (intentionally):**
- Timestamps
- Git SHA (tracked in lineage, not design)
- Toolchain version (tracked separately in `toolchain_hash`)

### 2.2 Toolchain Hash (`toolchain_hash`)

The toolchain hash captures the complete build environment. It is computed as:

```
toolchain_hash = SHA256(canonical_json(toolchain_metadata))
```

**Included in hash:**
- `schema_version`: Lock file schema version
- `kicad_version`: Pinned KiCad version (e.g., "9.0.7")
- `docker_image`: Docker image reference (e.g., "kicad/kicad:9.0.7")
- `docker_digest`: SHA256 digest of the pinned Docker image

**Why toolchain_hash matters:**
- Ensures rebuilds use identical toolchain
- Cache invalidation when toolchain changes
- Audit trail for reproducibility

### 2.3 Export Hashes

Each fabrication export (Gerbers, drill files) is hashed after canonicalization:

```
export_hash = SHA256(canonical_text(file_content))
```

**Canonicalization removes:**
- Comment lines (starting with `G04` or `;`)
- CRLF → LF normalization
- Trailing whitespace

**Included in exports list:**
- Relative path to file
- Canonical hash

### 2.4 DRC Report Hash

The DRC report from KiCad is canonicalized and hashed:

```
drc_canonical_hash = SHA256(canonical_json(drc_report))
```

**Canonicalization removes/normalizes:**
- `date`, `time`, `timestamp` keys
- `source`, `filename` keys
- Absolute file paths (strings with `/` longer than 20 chars)
- Keys are sorted alphabetically

---

## 3. KiCad Board Canonicalization

KiCad board files (`.kicad_pcb`) contain non-deterministic elements that are stripped for hashing:

**Stripped elements:**
- `(tstamp UUID)` - Object timestamps/UUIDs
- `(uuid UUID)` - Object UUIDs

**Normalization:**
- CRLF → LF
- CR → LF

**Note:** Canonical board hashes are for verification, not for file storage. The actual board files retain UUIDs for KiCad compatibility.

---

## 4. Toolchain Lock File Schema

The `toolchain/kicad.lock.json` file pins the exact build environment:

```json
{
  "schema_version": "1.0",
  "kicad_version": "9.0.7",
  "docker_image": "kicad/kicad:9.0.7",
  "docker_digest": "sha256:<64-char-hex-digest>",
  "toolchain_hash": "<64-char-hex-hash>"
}
```

**Requirements:**
- `docker_digest` must be a valid SHA256 digest (not `"sha256:PLACEHOLDER"`)
- `toolchain_hash` is computed from the other fields (excluding itself)
- Use `tools/pin_kicad_image.py` to resolve and pin the digest

---

## 5. Manifest Completeness Requirements

### 5.1 Required Fields

Every manifest.json must contain:

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Manifest schema version |
| `coupon_family` | string | F0, F1, etc. |
| `design_hash` | string (64 hex) | SHA256 of resolved design |
| `coupon_id` | string (12 chars) | Base32 of design_hash prefix |
| `resolved_design` | object | All parameters in integer nm |
| `derived_features` | object | Computed geometric properties |
| `dimensionless_groups` | object | Physics ratios |
| `fab_profile` | object | Fabrication constraints |
| `stackup` | object | Layer stackup definition |
| `toolchain` | object | See below |
| `toolchain_hash` | string (64 hex) | SHA256 of toolchain metadata |
| `exports` | array | File paths and hashes |
| `verification` | object | Constraints + DRC results |
| `lineage` | object | git_sha + timestamp_utc |

### 5.2 Toolchain Object Structure

For docker mode, the toolchain object must contain:

```json
{
  "kicad": {
    "version": "9.0.7",
    "cli_version_output": "9.0.7"
  },
  "docker": {
    "image_ref": "kicad/kicad:9.0.7@sha256:..."
  },
  "mode": "docker",
  "generator_git_sha": "40-char-hex-sha"
}
```

### 5.3 No Unknown Values for Docker Builds

**CRITICAL:** Docker builds must never have `"unknown"` values for:
- `toolchain.kicad.version`
- `toolchain.kicad.cli_version_output`
- `toolchain.docker.image_ref`

The build pipeline must:
1. Run `kicad-cli --version` inside the Docker container
2. Capture the actual version output
3. Raise `ToolchainProvenanceError` if version cannot be determined

---

## 6. Determinism Gates (CI Verification)

The CI pipeline verifies determinism through these gates:

| Gate | Description | Verification |
|------|-------------|--------------|
| G1 | Schema + resolver determinism | Same spec → same resolved design |
| G2 | Constraint proof completeness | All tiers evaluated |
| G3 | DRC cleanliness | Exit code 0 from kicad-cli |
| G4 | Export completeness | All expected layers present |
| G5 | Hash stability | 3 runs produce identical hashes |

---

## 7. Implementation Notes

### 7.1 Canonical JSON

All JSON hashing uses canonical representation:
- Keys sorted alphabetically
- Minimal whitespace (no pretty-printing)
- UTF-8 encoding

Implementation: `formula_foundry.substrate.canonical_json_dumps()`

### 7.2 SHA256 Hashing

All cryptographic hashes use SHA256:
- Output: 64-character lowercase hexadecimal
- Implementation: `formula_foundry.substrate.sha256_bytes()`

### 7.3 Cache Key Composition

Build cache keys are composed of:
```
cache_key = f"{design_hash}_{toolchain_hash}"
```

Cache is invalidated when either component changes.

---

## 8. References

- ECO-M1-ALIGN-0001 Section 13.5: Manifest completeness requirements
- DESIGN_DOCUMENT.md Section 9: Canonicalization algorithms
- KiCad CLI docs: https://docs.kicad.org/8.0/en/cli/cli.html
- KiCad Docker images: https://www.kicad.org/download/docker/
