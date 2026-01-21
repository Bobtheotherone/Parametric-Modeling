# M1 Toolchain Documentation

**Last Updated:** 2026-01-20
**Satisfies:** CP-1.1, CP-1.2, CP-1.3 (ECO-M1-ALIGN-0002)

This document specifies the toolchain requirements, pinning strategy, and configuration for M1-compliant coupon generation.

---

## 1. Overview

The M1 toolchain consists of:
- KiCad 9.0.7 for PCB design, DRC, and export
- Docker containerization for reproducible builds
- Lock file mechanism for version pinning

---

## 2. Toolchain Lock File

### 2.1 Location

The toolchain lock file is located at:

```
toolchain/kicad.lock.json
```

### 2.2 Schema

```json
{
  "schema_version": "1.0",
  "kicad_version": "9.0.7",
  "docker_image": "kicad/kicad:9.0.7",
  "docker_digest": "sha256:<64-char-hex-digest>",
  "toolchain_hash": "<64-char-hex-hash>"
}
```

### 2.3 Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `kicad_version` | string | Pinned KiCad version (required) |
| `docker_image` | string | Docker image reference (required) |
| `schema_version` | string | Lock file schema version (default: "1.0") |
| `docker_digest` | string | SHA256 digest of Docker image (optional, but required for production) |
| `toolchain_hash` | string | Computed hash (auto-generated) |

---

## 3. Docker Image Pinning

### 3.1 Why Pin Docker Images?

Docker tags (like `kicad/kicad:9.0.7`) can be overwritten by upstream maintainers. To ensure reproducibility:

1. Resolve the tag to its current SHA256 digest
2. Store the digest in the lock file
3. Pull using `image@sha256:digest` format

### 3.2 Pinning Command

Use the provided pinning tool:

```bash
python tools/pin_kicad_image.py
```

This will:
1. Pull the configured image tag
2. Resolve to SHA256 digest
3. Update `toolchain/kicad.lock.json`
4. Recompute `toolchain_hash`

### 3.3 Placeholder Digest

A placeholder digest (`sha256:PLACEHOLDER`) indicates:
- Development/testing mode
- Digest has not been resolved
- NOT suitable for production builds

**Production builds MUST fail on placeholder digests.**

---

## 4. Toolchain Hash Computation

### 4.1 Algorithm

The `toolchain_hash` is computed as:

```python
# 1. Extract hashable fields (exclude toolchain_hash itself)
hashable = {k: v for k, v in lock_data.items() if k != "toolchain_hash"}

# 2. Serialize to canonical JSON (sorted keys, minimal whitespace)
canonical = canonical_json_dumps(hashable)

# 3. Compute SHA256
toolchain_hash = sha256(canonical.encode("utf-8")).hexdigest()
```

### 4.2 Hash Stability

The toolchain hash is deterministic:
- Same lock file content → same hash
- Key ordering does not affect hash (canonical JSON sorts keys)
- Hash changes when any field (except toolchain_hash) changes

---

## 5. KiCad CLI Integration

### 5.1 Supported Commands

The toolchain supports these KiCad CLI commands:

| Command | Purpose |
|---------|---------|
| `kicad-cli version` | Verify installed version |
| `kicad-cli pcb drc` | Run design rule checks |
| `kicad-cli pcb export gerbers` | Export Gerber files |
| `kicad-cli pcb export drill` | Export drill files |

### 5.2 Docker Runner

The `DockerKicadRunner` class executes KiCad commands in a container:

```python
from formula_foundry.coupongen.kicad.docker_runner import DockerKicadRunner

runner = DockerKicadRunner(toolchain_config)
result = runner.run_drc(board_path)
```

### 5.3 Version Capture

For manifest provenance, capture the actual CLI version:

```python
# Inside container
kicad-cli version

# Captured in manifest as:
{
  "toolchain": {
    "kicad": {
      "version": "9.0.7",
      "cli_version_output": "KiCad 9.0.7 built on 2025-..."
    }
  }
}
```

---

## 6. CI Integration

### 6.1 Workflow Location

The M1 CI workflow is at:

```
.github/workflows/ci_m1.yml
```

### 6.2 Jobs

| Job | Description |
|-----|-------------|
| `unit-tests` | Run pytest unit tests |
| `kicad-integration` | Run KiCad DRC/export in Docker |
| `determinism-gates` | Verify G1-G5 gates pass |

### 6.3 Docker Image in CI

CI uses the pinned image reference:

```yaml
services:
  kicad:
    image: kicad/kicad:9.0.7@sha256:...
```

---

## 7. API Reference

### 7.1 Loading Toolchain Config

```python
from formula_foundry.coupongen.toolchain import (
    load_toolchain_lock,
    ToolchainConfig,
)

# Load from default path
config = load_toolchain_lock()

# Load from custom path
config = load_toolchain_lock(lock_path=Path("path/to/lock.json"))

# Load with repo root
config = load_toolchain_lock(repo_root=Path("/path/to/repo"))
```

### 7.2 ToolchainConfig Properties

```python
config.schema_version    # "1.0"
config.kicad_version     # "9.0.7"
config.docker_image      # "kicad/kicad:9.0.7"
config.docker_digest     # "sha256:abc123..." or None
config.toolchain_hash    # 64-char hex string

# Computed property
config.pinned_image_ref  # "kicad/kicad:9.0.7@sha256:abc123..."
```

### 7.3 Computing Toolchain Hash

```python
from formula_foundry.coupongen.toolchain import compute_toolchain_hash

lock_data = {
    "schema_version": "1.0",
    "kicad_version": "9.0.7",
    "docker_image": "kicad/kicad:9.0.7",
    "docker_digest": "sha256:abc123...",
}

hash_value = compute_toolchain_hash(lock_data)  # 64-char hex
```

---

## 8. Error Handling

### 8.1 ToolchainLoadError

Raised when lock file cannot be loaded:

```python
from formula_foundry.coupongen.toolchain import ToolchainLoadError

try:
    config = load_toolchain_lock()
except ToolchainLoadError as e:
    print(f"Failed to load toolchain: {e}")
```

**Common causes:**
- Lock file not found
- Invalid JSON syntax
- Missing required fields (`kicad_version`, `docker_image`)

### 8.2 Provenance Validation

For production builds, validate provenance:

```python
def validate_toolchain_provenance(config: ToolchainConfig) -> None:
    if config.docker_digest is None:
        raise ValueError("Docker digest required for production")
    if "PLACEHOLDER" in config.docker_digest:
        raise ValueError("Placeholder digest not allowed in production")
```

---

## 9. Migration Guide

### 9.1 From Tag-Only to Digest-Pinned

1. Ensure lock file exists at `toolchain/kicad.lock.json`
2. Run `python tools/pin_kicad_image.py`
3. Verify `docker_digest` is populated (not PLACEHOLDER)
4. Commit updated lock file

### 9.2 Upgrading KiCad Version

1. Update `kicad_version` and `docker_image` in lock file
2. Run `python tools/pin_kicad_image.py` to resolve new digest
3. Run full test suite to verify compatibility
4. Update golden spec hashes if needed
5. Commit changes

---

## 10. References

- docs/m1_determinism_policy.md: Determinism requirements
- docs/DETERMINISM_POLICY.md: General determinism policy
- src/formula_foundry/coupongen/toolchain/lock.py: Implementation
- ECO-M1-ALIGN-0002: Engineering Change Order
