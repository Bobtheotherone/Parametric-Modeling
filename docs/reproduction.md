# Reproduction Procedure (M0)

This document describes the canonical, deterministic reproduction path for M0 milestone verification. Follow these steps exactly to reproduce any M0-compliant run.

## Prerequisites

Before starting, ensure you have:

- **Python 3.10+** (required by `pyproject.toml`)
- **uv** package manager (`pip install uv`)
- **Git** for repository checkout
- **(Optional) CUDA-capable GPU** with appropriate drivers for GPU smoke tests

## Step 1: Clone and Checkout

Clone the repository and check out the exact commit you want to reproduce:

```bash
git clone <repo-url>
cd <repo-dir>
git checkout <commit-sha>
```

Replace `<commit-sha>` with the specific commit hash from the run you are reproducing. The commit hash is recorded in every run's `manifest.json` under `environment.git_commit`.

## Step 2: Verify Toolchain Lock File

The repository uses `uv.lock` for deterministic dependency pinning. Verify the lock file exists:

```bash
ls uv.lock
```

If the lock file is missing, the reproduction is invalid. Do **not** regenerate it; instead, obtain the correct lock file from the original run or a known-good state.

## Step 3: Bootstrap Environment

Run the bootstrap script to create a locked virtual environment:

```bash
./scripts/bootstrap_venv.sh
```

This script:
1. Validates that `uv.lock` exists
2. Creates a virtual environment at `.venv/`
3. Installs dependencies from the lock file with `--frozen` (no re-resolution)
4. Installs the project in editable mode

**Expected output:** `Environment ready at .venv`

If the script fails, check:
- `uv` is installed: `which uv`
- `uv.lock` is present and valid

## Step 4: Activate Environment

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Alternatively, use explicit interpreter paths (`./.venv/bin/python`) for all commands.

## Step 5: Run M0 Smoke Tests

The smoke test validates GPU availability and basic substrate operations:

```bash
./.venv/bin/python -m tools.m0 smoke
```

**Exit codes:**
- `0`: All checks passed
- `2`: One or more checks failed

For GPU-mandatory runs, add `--require-gpu`:

```bash
./.venv/bin/python -m tools.m0 smoke --require-gpu
```

**Output artifacts:** The smoke test writes results to `runs/<run-id>/`:
- `artifacts/smoke_report.json`: Detailed check results
- `manifest.json`: Full provenance including environment hash
- `logs.jsonl`: Event log

## Step 6: Run Reproducibility Check

The reproducibility check validates deterministic artifact generation:

```bash
./.venv/bin/python -m tools.m0 repro-check
```

This command:
1. Runs two identical passes with the same seed
2. Compares artifact hashes between passes
3. Fails if any hash differs

**Exit codes:**
- `0`: Artifacts are reproducible (hashes match)
- `2`: Non-determinism detected (hashes differ)

Optional parameters:
- `--seed <N>`: Specify random seed (default: 1234)
- `--payload-bytes <N>`: Size of test payload (default: 256)

## Step 7: Run Verification Gates

The verification tool validates the run against M0 gate requirements:

```bash
./.venv/bin/python -m tools.verify --strict-git
```

The `--strict-git` flag enforces:
- Clean working directory (no uncommitted changes)
- Recorded git commit in manifest matches HEAD

**Exit codes:**
- `0`: All gates passed
- Non-zero: One or more gates failed

## M0 Gates Summary

The M0 milestone enforces these gates:

| Gate | Description | Validation Command |
|------|-------------|-------------------|
| G0-ENV | Environment bootstrap succeeds | `./scripts/bootstrap_venv.sh` |
| G0-SMOKE | Smoke tests pass | `python -m tools.m0 smoke` |
| G0-REPRO | Reproducibility check passes | `python -m tools.m0 repro-check` |
| G0-VERIFY | Verification gates pass | `python -m tools.verify --strict-git` |

All gates must pass for M0 compliance.

## Toolchain Pinning

### Python Dependencies

Dependencies are pinned via `uv.lock`. Key packages:

| Package | Purpose |
|---------|---------|
| `cupy-cuda12x` | GPU array operations |
| `numpy` | CPU array fallback |
| `torch` | Deep learning and CUDA interop |
| `pytest` | Testing framework |

### KiCad Toolchain (M1+)

For milestones requiring KiCad integration, the toolchain is pinned in `toolchain/kicad.lock.json`:

```json
{
  "kicad_version": "9.0.7",
  "docker_image": "kicad/kicad:9.0.7",
  "docker_digest": "sha256:<digest>"
}
```

Pin the Docker image digest before production runs:

```bash
python tools/pin_kicad_image.py
```

## Manifest Provenance

Every run produces a `manifest.json` containing:

```json
{
  "schema_version": "1.0",
  "determinism": {
    "mode": "strict",
    "seed": 1234
  },
  "environment": {
    "python_version": "3.10.x",
    "git_commit": "<sha>",
    "platform": "<os-info>"
  },
  "artifacts": {
    "<filename>": "<sha256-hash>"
  }
}
```

To verify a reproduction, compare:
1. `environment.git_commit` matches your checkout
2. `artifacts` hashes match between original and reproduction

## Troubleshooting

### "uv.lock not found"

The lock file is missing. Obtain it from:
- The original commit
- A CI artifact
- Do **not** regenerate with `uv lock` as this may resolve different versions

### "GPU required but unavailable"

For `--require-gpu` runs without a GPU:
- Remove `--require-gpu` flag for CPU-only validation
- Or run on a GPU-enabled machine

### "Non-determinism detected"

The `repro-check` found hash mismatches. Common causes:
- Non-deterministic code paths (timestamps, unordered dicts)
- Different CUDA/cuDNN versions
- Floating-point non-determinism

Run with `--mode strict` and check `manifest.json` environment sections match.

### "Verification failed"

The `verify` tool detected issues. Check:
- Working directory is clean: `git status`
- HEAD matches expected commit
- All required artifacts exist

## Full Reproduction Sequence

For a complete M0 reproduction, run all steps in sequence:

```bash
# 1. Clone and checkout
git clone <repo-url>
cd <repo-dir>
git checkout <commit-sha>

# 2. Bootstrap
./scripts/bootstrap_venv.sh

# 3. Activate
source .venv/bin/activate

# 4. Run M0 gates
python -m tools.m0 smoke
python -m tools.m0 repro-check
python -m tools.verify --strict-git
```

If any step fails, do not proceed. Fix the reported issue and restart from step 1.

## CI Integration

For CI pipelines, use this minimal gate check:

```yaml
steps:
  - run: ./scripts/bootstrap_venv.sh
  - run: ./.venv/bin/python -m tools.m0 smoke --require-gpu
  - run: ./.venv/bin/python -m tools.m0 repro-check
  - run: ./.venv/bin/python -m tools.verify --strict-git
```

All commands must exit with code 0 for the CI job to pass.
