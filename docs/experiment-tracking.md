# Experiment Tracking Workflow

This document describes how to use the M3 experiment tracking infrastructure
to log runs, parameters, metrics, and artifacts in Formula Foundry.

## Overview

The tracking module integrates MLflow with the existing substrate infrastructure:

- **MLflow** handles experiment/run metadata, parameters, metrics, and tags
- **Substrate artifact store** handles large files via content-addressed storage
- **Artifact references** link MLflow runs to substrate artifacts without duplication

## Configuration

MLflow is configured via `config/mlflow.yaml`:

```yaml
tracking:
  # SQLite backend for experiment metadata
  backend_store_uri: "sqlite:///data/mlflow/mlruns.db"

  # Local artifact storage root
  artifact_root: "data/mlflow/artifacts"

  # Default experiment name
  default_experiment: "formula-foundry"
```

## Basic Usage

### Starting a Tracked Run

```python
from formula_foundry.tracking import get_tracker

tracker = get_tracker()

with tracker.start_run(run_name="my-coupon-build") as run:
    # Log parameters
    run.log_param("coupon_family", "F1_SINGLE_ENDED_VIA")
    run.log_param("fab_profile", "oshpark_4layer")

    # Log metrics
    run.log_metric("build_time_seconds", 12.5)
    run.log_metric("constraint_violations", 0)

    # Set tags
    run.set_tag("status", "success")
```

### Integrating with Substrate Manifest

```python
from formula_foundry.substrate import Manifest, DeterminismConfig
from formula_foundry.tracking import get_tracker

# Create manifest from environment
manifest = Manifest.from_environment(
    determinism=DeterminismConfig(mode="strict", seed=42, cublas_workspace_config=":4096:8")
)

tracker = get_tracker()

with tracker.start_run(run_name="tracked-build", manifest=manifest) as run:
    # Manifest data is automatically logged as tags:
    # - git_sha
    # - design_doc_sha256
    # - environment_fingerprint
    # - determinism_mode
    # - seed_* (for each PRNG)

    run.log_metric("some_metric", 42.0)
```

### Linking to Substrate RunArtifacts

```python
from formula_foundry.substrate import init_run_dir, create_run, Manifest
from formula_foundry.tracking import get_tracker

# Create substrate run directory
run_artifacts = init_run_dir(run_root=Path("data/runs"), run_id="run-001")

tracker = get_tracker()

with tracker.start_run(
    run_name="linked-run",
    run_artifacts=run_artifacts,
) as run:
    # MLflow run is linked to substrate run via tags:
    # - substrate_run_id
    # - substrate_run_dir

    run.log_metric("pipeline_stage", 1)
```

### Logging Artifact References

Instead of duplicating large artifacts, log references to the substrate store:

```python
from formula_foundry.substrate import ArtifactStore
from formula_foundry.tracking import get_tracker

store = ArtifactStore(root=Path("data/artifacts"))

# Add file to substrate store
entry = store.add_file(Path("output/coupon.kicad_pcb"), logical_path="coupon.kicad_pcb")

tracker = get_tracker()

with tracker.start_run(run_name="artifact-run") as run:
    # Log reference (creates a small JSON file, not the full artifact)
    run.log_artifact_reference(
        logical_path=entry.path,
        digest=entry.digest,
        size_bytes=entry.size_bytes,
    )
```

## Experiment Organization

### Standard Experiment Names

Use consistent experiment names per milestone:

| Milestone | Experiment Name |
|-----------|-----------------|
| M1 Coupongen | `ff-m1-coupongen` |
| M2 Simulation | `ff-m2-simulation` |
| M3 Pipeline | `ff-m3-pipeline` |
| M4 Macromodel | `ff-m4-macromodel` |
| M5 Symbolic | `ff-m5-symbolic` |
| M6 Loop | `ff-m6-loop` |

```python
tracker = get_tracker()

with tracker.start_run(
    run_name="sim-001",
    experiment_name="ff-m2-simulation",
) as run:
    run.log_metric("insertion_loss_db", -0.5)
```

### Nested Runs

Use nested runs for hierarchical tracking:

```python
tracker = get_tracker()

with tracker.start_run(run_name="pipeline-run") as parent:
    parent.log_param("pipeline_version", "1.0")

    with tracker.start_run(run_name="coupongen", nested=True) as child1:
        child1.log_metric("build_time", 5.0)

    with tracker.start_run(run_name="simulation", nested=True) as child2:
        child2.log_metric("sim_time", 120.0)
```

## Viewing Results

### Using MLflow UI

Start the MLflow UI to view experiments:

```bash
cd /path/to/project
mlflow ui --backend-store-uri sqlite:///data/mlflow/mlruns.db
```

Then open http://localhost:5000 in your browser.

### Programmatic Access

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///data/mlflow/mlruns.db")

# List experiments
experiments = mlflow.search_experiments()

# Search runs
runs = mlflow.search_runs(experiment_names=["formula-foundry"])

# Get specific run
run = mlflow.get_run(run_id="abc123")
```

## Best Practices

1. **Always use manifest integration** for reproducibility tracking
2. **Log artifact references** instead of duplicating large files
3. **Use consistent experiment names** per milestone
4. **Set meaningful run names** for easy identification
5. **Log required tags** (git_sha, design_doc_sha256, etc.)
6. **Use nested runs** for complex pipelines

## Troubleshooting

### MLflow Not Installed

The tracker operates in no-op mode when MLflow is not installed:

```bash
pip install formula-foundry[tracking]
```

### Database Locked

If you see "database is locked" errors, ensure only one process writes at a time.
For concurrent access, consider using PostgreSQL instead of SQLite.

### Missing Directories

The tracker automatically creates required directories on initialization.
If you see permission errors, ensure write access to `data/mlflow/`.
