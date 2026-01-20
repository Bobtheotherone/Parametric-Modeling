# M3 Golden Demo

This directory contains the golden demo workflow for the M3 (Artifact Storage Backbone) subsystem. The tests demonstrate the complete M3 capability and serve as both integration tests and documentation.

## Overview

The M3 subsystem provides:
- **Content-addressed artifact storage** with SHA256 hashing
- **Lineage tracking** via SQLite graph database
- **Dataset snapshots** with versioning and Parquet indexing
- **Garbage collection** with configurable retention policies
- **Audit and verification** for provenance tracking

## Directory Structure

```
tests/golden_m3/
├── README.md                  # This file
├── __init__.py                # Package initialization
├── test_golden_m3_demo.py     # Comprehensive integration tests
└── golden_fixtures.json       # Reference data for testing
```

## Test Phases

The golden demo is organized into 7 phases, each demonstrating a key capability:

### Phase 1: Project Initialization

```python
# Initialize M3 infrastructure
from formula_foundry.m3.cli_main import cmd_init
cmd_init(root=project_path, force=False, quiet=True)
```

Creates:
- `data/objects/` - Content-addressed storage
- `data/manifests/` - Artifact metadata
- `data/datasets/` - Dataset snapshots
- `data/mlflow/` - MLflow integration
- `data/registry.db` - SQLite artifact index
- `data/lineage.sqlite` - Lineage graph

### Phase 2: Artifact Creation and Storage

```python
from formula_foundry.m3.artifact_store import ArtifactStore, LineageReference

store = ArtifactStore(root=data_dir, generator="demo", generator_version="1.0.0")

# Store artifact with content-addressed hashing
manifest = store.put(
    content=b'{"coupon_family": "F1"}',
    artifact_type="coupon_spec",
    roles=["config", "root_input"],
    run_id="run-001",
    artifact_id="art-spec-001",
)

# Retrieve by hash
content = store.get(manifest.content_hash.digest)

# Spec ID is deterministic 12-char base32 identifier
print(manifest.spec_id)  # e.g., "l5ua4w36g7wa"
```

### Phase 3: Lineage Tracking

```python
from formula_foundry.m3.lineage_graph import LineageGraph

lineage = LineageGraph(data_dir / "lineage.sqlite")
lineage.initialize()
lineage.add_manifest(manifest)

# Trace provenance
ancestors = lineage.get_ancestors("art-output-001")
descendants = lineage.get_descendants("art-spec-001")
roots = lineage.trace_to_roots("art-output-001")
```

### Phase 4: Dataset Snapshots

```python
from formula_foundry.m3.dataset_snapshot import (
    DatasetSnapshotWriter,
    DatasetSnapshotReader,
)

# Create dataset
writer = DatasetSnapshotWriter(
    dataset_id="em_dataset",
    version="v1.0",
    store=store,
    generator="demo",
    generator_version="1.0.0",
)
writer.add_member(manifest, role="oracle_output", features={"freq_ghz": 1.0})
snapshot = writer.finalize(output_dir=dataset_dir, write_parquet=True)

# Read and verify
reader = DatasetSnapshotReader(snapshot_path=manifest_path, store=store)
is_valid, errors = reader.verify_integrity()
```

### Phase 5: Garbage Collection

```python
from formula_foundry.m3.gc import GarbageCollector, RetentionPolicy, BUILTIN_POLICIES

gc = GarbageCollector(data_dir=data_dir, store=store, registry=registry, lineage=lineage)

# Pin important artifacts
gc.pin_artifact(artifact_id="art-important", reason="Production model")

# Dry run with retention policy
policy = BUILTIN_POLICIES["laptop_default"]
result = gc.run(policy=policy, dry_run=True, run_dvc_gc=False)

# Execute GC
result = gc.run(policy=policy, dry_run=False, run_dvc_gc=False)
print(f"Deleted {result.artifacts_deleted} artifacts, freed {result.bytes_freed} bytes")
```

### Phase 6: Audit and Verification

```python
from formula_foundry.m3.cli_main import cmd_audit

# Audit with full lineage trace and hash verification
result = cmd_audit(
    artifact_id="art-output-001",
    root=project_path,
    output_format="json",
    trace_roots=True,
    verify_hashes=True,
    required_roles="config,root_input",
)
```

### Phase 7: Full Workflow Integration

See `test_golden_m3_demo.py::TestPhase7FullWorkflow::test_complete_m3_workflow` for a complete end-to-end demonstration.

## Running the Tests

```bash
# Run all golden M3 tests
pytest tests/golden_m3/ -v

# Run specific phase
pytest tests/golden_m3/test_golden_m3_demo.py::TestPhase1Initialization -v

# Run full workflow integration test
pytest tests/golden_m3/test_golden_m3_demo.py::TestPhase7FullWorkflow -v
```

## CLI Commands

The M3 subsystem exposes these CLI commands:

| Command | Description |
|---------|-------------|
| `m3 init` | Initialize project infrastructure |
| `m3 run <stage>` | Execute DVC stage with metadata |
| `m3 artifact show <id>` | Display artifact details |
| `m3 artifact list` | List artifacts with filtering |
| `m3 dataset show <id>` | Display dataset snapshot |
| `m3 dataset diff <a> <b>` | Compare dataset versions |
| `m3 gc` | Garbage collect (dry-run by default) |
| `m3 gc-pin` | Pin artifact to protect from GC |
| `m3 gc-unpin` | Unpin artifact |
| `m3 gc-estimate` | Estimate space savings |
| `m3 audit` | Generate provenance report |

## Key Design Patterns

### Content-Addressed Storage

All artifacts are stored by their SHA256 hash:
```
data/objects/<first-2-chars>/<full-hash>
```

This provides:
- Automatic deduplication
- Tamper detection
- Deterministic references

### Atomic Writes

All file operations use atomic write patterns:
1. Write to temporary file
2. Sync to disk
3. Atomic rename to final path

This ensures crash safety.

### Deterministic Identifiers

- **artifact_id**: `art-YYYYMMDDTHHMMSS-xxxxxxxx`
- **run_id**: `run-YYYYMMDDTHHMMSS-xxxxxxxx`
- **spec_id**: 12-char base32 of SHA256

### Schema Versioning

All JSON manifests include `schema_version` for safe migrations.

## Retention Policies

Built-in policies:

| Policy | Age | Min Count | Pinned | Descendants | Budget |
|--------|-----|-----------|--------|-------------|--------|
| `laptop_default` | 14d | 5 | Yes | Yes | 50 GB |
| `ci_aggressive` | 7d | 2 | Yes | No | 10 GB |
| `archive` | 365d | 10 | Yes | Yes | 500 GB |
| `dev_minimal` | 3d | 1 | Yes | Yes | 5 GB |

## Golden Fixtures

The `golden_fixtures.json` file contains reference data including:
- Sample coupon spec (F1 family)
- Resolved design parameters
- Workflow stage definitions
- Retention policy configuration
- Expected artifact types and roles
- Lineage relation types

This data can be used to create consistent test scenarios.
