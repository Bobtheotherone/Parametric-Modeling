# Milestone Design Document

**Milestone:** M0 — Repo substrate: deterministic builds, GPU-first runtime, and reproducibility

## Scope
M0 turns this repository from an *agentic coding harness* into a **deterministic, GPU-first scientific compute substrate**.

After M0, we must be able to:
- reproduce an experiment from a commit hash (environment + inputs + outputs),
- prove we are actually using the GPU (and detect silent CPU fallbacks),
- run thousands of small jobs safely on a single laptop without corrupting state,
- generate auditable run artifacts (manifests, logs, hashes) suitable for later scientific claims.

M0 is not about EM solvers or symbolic regression yet. It is about **engineering rigor** so that later milestones can confidently discover a real, defensible, novel equation.

## Design Principles
1) **Determinism is a feature, not an aspiration.** If a run is nondeterministic, it must be explicit and opt-in.
2) **GPU-first by default.** CPU execution is allowed only when explicitly requested or when GPU is unavailable.
3) **No silent fallbacks.** If an operation falls back to CPU in “require-gpu” mode, fail loudly.
4) **Reproducible builds are not optional.** We must be able to rebuild the exact environment.
5) **Observability is part of correctness.** Every run must record enough metadata to explain and reproduce results.

## Architecture Overview
This milestone introduces four substrate layers.

### 1) Environment + build reproducibility
**Goal:** the same commit produces the same environment (within defined bounds) and the same build artifacts.

**Canonical approach (M0):**
- **Python deps locked** with `uv.lock` and installed via `uv sync --frozen`.
- **Container dev/runtime option** with an NVIDIA CUDA base image pinned by **digest**, so “same Dockerfile” really means “same bytes.” Docker digests identify immutable image content and support `image@sha256:<digest>` references.

### 2) Determinism controls
**Goal:** provide a single, correct place to turn determinism on/off and to fingerprint what mode was used.

We implement a deterministic policy layer with explicit modes:
- `strict`: force deterministic algorithms and stable numerics (may be slower).
- `fast`: allow nondeterminism for performance (must be labeled in manifests).

For PyTorch, strict determinism must include:
- `torch.use_deterministic_algorithms(True)`
- cuDNN deterministic settings and disabled benchmarking
- required CUDA env settings for cuBLAS determinism (`CUBLAS_WORKSPACE_CONFIG=:4096:8` or `:16:8`), per PyTorch deterministic-algorithms guidance.

### 3) GPU-first numeric runtime (CuPy + PyTorch)
**Goal:** array operations and objective evaluation are GPU-dominant by default.

Key pieces:
- **Backend selector** (`numpy` vs `cupy`) with an explicit “require GPU” mode.
- **DLPack interop** to move tensors between CuPy and PyTorch without host copies.

### 4) Run system: manifests, artifacts, logs, and local scheduling
**Goal:** every experiment is a typed, auditable run with deterministic I/O boundaries.

- `runs/<run_id>/manifest.json`: canonical JSON describing the run (git SHA, design-doc hash, environment fingerprint, determinism mode, seeds, command line, artifact hashes).
- `runs/<run_id>/logs.jsonl`: structured logs.
- Local job runner with GPU/CPU semaphores to safely overlap CPU-heavy tasks and GPU tasks.

### Required repo layout additions (post-M0)
This is a *target* layout (agents may adjust, but contracts must remain).

```
config/
  hardware.yaml
  determinism.yaml
docker/
  Dockerfile
  compose.yaml
scripts/
  bootstrap_venv.sh
  lock.sh
src/formula_foundry/substrate/
  __init__.py
  backends.py
  determinism.py
  fingerprint.py
  manifest.py
  runner.py
tools/
  m0.py               # CLI entrypoint: doctor/smoke/bench/repro-check
```

## Normative Requirements (must)

### A) Protocol + gating (must remain future-compatible)
- [REQ-M0-001] The repo must provide a `tools/verify.py` command (`python -m tools.verify`) that runs milestone gates and exits nonzero on failure.
- [REQ-M0-002] The repo must enforce a documented DESIGN_DOCUMENT format: (a) Milestone ID, (b) Normative Requirements with stable IDs, (c) Test Matrix mapping requirements to pytest node ids, (d) Definition of Done.
- [REQ-M0-003] The repo must include a `tools/spec_lint.py` checker that validates the DESIGN_DOCUMENT format and the requirement↔test mapping.
- [REQ-M0-004] `pytest` must include at least one test that fails if DESIGN_DOCUMENT.md violates the required format.
- [REQ-M0-005] The orchestrator must refuse to stop unless: `tools/verify.py` passes, `git status --porcelain` is clean, and a commit exists for the changes.

### B) Deterministic environment + build substrate
- [REQ-M0-006] The repo must adopt **locked dependency installation** as the default developer path using `uv.lock` + `uv sync --frozen` (or an equivalent hash-locked mode). “Frozen” installs must fail if the lock does not match.
- [REQ-M0-007] The repo must provide `scripts/bootstrap_venv.sh` that creates/updates a local `.venv` deterministically from the lock, without editing the lock.
- [REQ-M0-008] The repo must provide a container option (`docker/Dockerfile`) that can run M0 smoke/bench/repro-check. The base image must be pinned by digest (`@sha256:`) to ensure immutability.
- [REQ-M0-009] If the repo includes GitHub Actions workflows, all third-party actions must be pinned by full commit SHA (not floating tags).

### C) Determinism policy (strict vs fast)
- [REQ-M0-010] The repo must provide a determinism policy module that supports modes `{strict, fast}` and records the active mode in run manifests.
- [REQ-M0-011] In `strict` mode, the repo must set and/or validate:
  - `PYTHONHASHSEED` (documented expectation),
  - seeds for `random`, `numpy`, `cupy`, and `torch`,
  - `torch.use_deterministic_algorithms(True)`,
  - deterministic cuDNN settings (`benchmark=False`, deterministic enabled),
  - a valid `CUBLAS_WORKSPACE_CONFIG` setting required for deterministic cuBLAS behavior.
- [REQ-M0-012] The determinism module must expose a context manager (or equivalent) that applies strict settings for the duration of an experiment and restores prior settings where feasible.

### D) GPU-first runtime (no silent CPU fallbacks)
- [REQ-M0-013] The substrate must provide a backend selector that defaults to GPU arrays (CuPy) when CUDA is available, and to NumPy otherwise.
- [REQ-M0-014] The substrate must support an explicit “require GPU” mode that raises a clear error if GPU is not available or if configured to run CPU-only.
- [REQ-M0-015] The substrate must provide utilities for DLPack-based zero-copy conversion between CuPy arrays and PyTorch CUDA tensors.
- [REQ-M0-016] The substrate must provide a guardrail that detects (and can fail on) accidental host transfers for critical benchmark/smoke paths.

### E) Run manifests + artifact hashing
- [REQ-M0-017] Every execution of M0 commands must create a `runs/<run_id>/` directory containing at least:
  - `manifest.json` (canonical JSON),
  - `logs.jsonl` (structured logs),
  - `artifacts/` directory (hashed outputs).
- [REQ-M0-018] `manifest.json` must include: git commit SHA, `DESIGN_DOCUMENT.md` SHA256, environment fingerprint hash, determinism mode, RNG seeds, command line, and a map of artifact SHA256 hashes.
- [REQ-M0-019] Manifests must be **canonical**: stable key order, stable float formatting policy (documented), and explicit encoding (UTF-8).

### F) Local job runner (single-machine, resource-aware)
- [REQ-M0-020] The repo must provide a local job runner API that schedules tasks with declared resource requirements: CPU threads, RAM estimate, and GPU VRAM estimate.
- [REQ-M0-021] The runner must enforce concurrency limits via semaphores (GPU and CPU), and scheduling must be deterministic given the same task list and config.
- [REQ-M0-022] The repo must provide `config/hardware.yaml` to declare the machine limits (VRAM, RAM, CPU cores) and the runner must use it.

### G) M0 command surface (the “step 16” substrate contracts)
- [REQ-M0-023] The repo must provide `python -m tools.m0 doctor` which checks CUDA availability, prints a JSON report option, and exits nonzero on hard failures.
- [REQ-M0-024] The repo must provide `python -m tools.m0 smoke` which runs fast smoke checks:
  - CuPy import + simple GPU op,
  - PyTorch CUDA availability + `torch.compile` sanity (if supported),
  - DLPack roundtrip check,
  - manifest/log generation.
- [REQ-M0-025] The repo must provide `python -m tools.m0 bench` which runs microbenchmarks and writes a JSON report into the run directory.
- [REQ-M0-026] The repo must provide `python -m tools.m0 repro-check` which runs a deterministic mini-pipeline twice and asserts selected artifact hashes are identical (bit-for-bit) within the same environment.

### H) Verification integration
- [REQ-M0-027] `tools.verify` must include M0 substrate gates (at minimum: `tools.m0 smoke` + `tools.m0 repro-check`) when the active milestone is M0. For non-M0 milestones, these gates must be skipped unless explicitly requested.
- [REQ-M0-028] M0 verification gates must be designed to run in <= 90 seconds on the target laptop. Heavy benchmarks must be optional.

### I) Documentation
- [REQ-M0-029] The repo must document the exact reproduction procedure: “clone → bootstrap env → run smoke/repro-check → verify.”
- [REQ-M0-030] The repo must document determinism limitations (GPU nondeterminism edge cases) and explicitly list what is guaranteed by `strict` mode.

## Non-Functional Requirements
- **Deterministic:** In `strict` mode, `m0 repro-check` must be bit-for-bit reproducible for its declared artifact set.
- **GPU-first:** In environments with CUDA available, M0 smoke/bench must use GPU arrays/tensors by default.
- **Fail-loud:** Misconfiguration (CPU fallback, missing CUDA, missing lock) must fail with actionable errors.
- **Fast feedback:** `python -m tools.verify` must be fast enough to run frequently during development.

## Step 16 — What success looks like after M0
This section is *normative*. The implementation is not “done” unless every item below can be executed and demonstrated.

1. **Fresh clone reproducibility:** On a clean machine state (no cached venv), `git clone` + `git checkout <sha>` works.
2. **Frozen install:** `./scripts/bootstrap_venv.sh` installs deps via the lock in frozen mode and fails if the lock is out of date.
3. **Doctor report:** `python -m tools.m0 doctor --json` prints a machine-readable JSON report with GPU/driver/runtime info.
4. **GPU requirement switch:** `python -m tools.m0 doctor --require-gpu` exits nonzero if CUDA is unavailable.
5. **Smoke run:** `python -m tools.m0 smoke` completes and writes `runs/<run_id>/manifest.json` and `logs.jsonl`.
6. **GPU proof:** Smoke output (or manifest fields) proves GPU arrays/tensors were used (e.g., CuPy device, torch.cuda device).
7. **Torch compile sanity:** Smoke includes a `torch.compile` path check (skipped with a clear reason if unsupported).
8. **DLPack zero-copy:** Smoke proves CuPy↔Torch conversion without host copy (or fails with an explanation).
9. **Bench run:** `python -m tools.m0 bench --json` writes a benchmark report into the run dir.
10. **CPU fallback detection:** Bench includes at least one guard that would fail if data silently moved to CPU.
11. **Repro-check:** `python -m tools.m0 repro-check` produces two runs whose selected artifacts have identical SHA256 hashes.
12. **Manifest completeness:** Each run’s manifest contains git SHA, design-doc hash, env fingerprint hash, determinism mode, seeds, and artifact hashes.
13. **Canonical manifests:** Re-running the *same* command with the *same* seeds produces a manifest that is identical except for fields explicitly declared volatile (e.g., timestamps, run_id).
14. **Runner determinism:** A small demo workload run through the job runner produces a deterministic execution trace given the same inputs.
15. **tools.verify enforcement:** `python -m tools.verify` includes M0 substrate gates when milestone is M0.
16. **Completion gates:** `python -m tools.verify --strict-git` passes, `git status --porcelain` is empty, and a commit exists.

## Definition of Done
- All requirements REQ-M0-001..REQ-M0-030 are implemented.
- Every requirement is mapped to at least one pytest node id in the Test Matrix.
- `python -m tools.verify --strict-git` exits 0.
- The full Step 16 checklist above can be executed successfully on the target laptop.
- The repo has no uncommitted changes (`git status --porcelain` empty).

## Test Matrix
| Requirement | Pytest(s) |
|---|---|
| REQ-M0-001 | tests/test_verify_smoke.py::test_verify_script_exists_and_runs |
| REQ-M0-002 | tests/test_design_document_contract.py::test_design_document_contract |
| REQ-M0-003 | tests/test_spec_lint.py::test_spec_lint_detects_missing_sections |
| REQ-M0-004 | tests/test_design_document_contract.py::test_design_document_contract |
| REQ-M0-005 | tests/test_orchestrator_gates.py::test_orchestrator_completion_gates |
| REQ-M0-006 | tests/test_m0_env_locking.py::test_uv_lock_and_frozen_install |
| REQ-M0-007 | tests/test_m0_env_locking.py::test_bootstrap_venv_script_contract |
| REQ-M0-008 | tests/test_m0_container_contract.py::test_dockerfile_pins_base_image_by_digest |
| REQ-M0-009 | tests/test_m0_supply_chain.py::test_actions_pinned_by_sha_if_present |
| REQ-M0-010 | tests/test_m0_determinism.py::test_determinism_modes_recorded |
| REQ-M0-011 | tests/test_m0_determinism.py::test_strict_mode_sets_required_controls |
| REQ-M0-012 | tests/test_m0_determinism.py::test_determinism_context_manager_restores |
| REQ-M0-013 | tests/test_m0_backend_policy.py::test_backend_defaults_to_gpu_if_available |
| REQ-M0-014 | tests/test_m0_backend_policy.py::test_require_gpu_mode_fails_cleanly |
| REQ-M0-015 | tests/test_m0_dlpack.py::test_dlpack_roundtrip_zero_copy |
| REQ-M0-016 | tests/test_m0_dlpack.py::test_host_transfer_guard |
| REQ-M0-017 | tests/test_m0_run_artifacts.py::test_run_dir_contract |
| REQ-M0-018 | tests/test_m0_run_artifacts.py::test_manifest_required_fields |
| REQ-M0-019 | tests/test_m0_run_artifacts.py::test_manifest_canonical_json |
| REQ-M0-020 | tests/test_m0_runner.py::test_runner_task_resource_schema |
| REQ-M0-021 | tests/test_m0_runner.py::test_runner_deterministic_schedule |
| REQ-M0-022 | tests/test_m0_runner.py::test_hardware_yaml_contract |
| REQ-M0-023 | tests/test_m0_cli.py::test_doctor_command_contract |
| REQ-M0-024 | tests/test_m0_cli.py::test_smoke_command_contract |
| REQ-M0-025 | tests/test_m0_cli.py::test_bench_command_contract |
| REQ-M0-026 | tests/test_m0_cli.py::test_repro_check_contract |
| REQ-M0-027 | tests/test_verify_m0_integration.py::test_verify_runs_m0_gates_for_m0 |
| REQ-M0-028 | tests/test_verify_m0_integration.py::test_verify_time_budget |
| REQ-M0-029 | tests/test_m0_docs.py::test_reproduction_docs_present |
| REQ-M0-030 | tests/test_m0_docs.py::test_determinism_limitations_documented |

## References (implementation guidance)
- Reproducible Builds definition: https://reproducible-builds.org/docs/definition/
- PyTorch deterministic algorithms: https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
- cuBLAS reproducibility note (referenced by PyTorch docs): https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
- Docker image digests (immutable references): https://docs.docker.com/dhi/core-concepts/digests/
- pip hash-checking mode (alternative to uv): https://pip.pypa.io/en/stable/topics/secure-installs/#hash-checking-mode
- uv CLI docs: https://docs.astral.sh/uv/
- CuPy DLPack interop: https://docs.cupy.dev/en/stable/user_guide/interoperability.html#dlpack
