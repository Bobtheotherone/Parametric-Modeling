# Verify/Gate Wiring Analysis: Suspected Root Causes

**Analysis Date:** 2026-01-20
**Branch:** task/20260121T040631Z/analysis-hypotheses
**Verify Exit Code:** 2

---

## Executive Summary

Static review of the verify and gate orchestration code identified **7 primary hypotheses** for likely failure points. These are ranked by confidence and cross-referenced against the codebase structure.

---

## H1: Toolchain Lock Digest is Placeholder (HIGH CONFIDENCE)

**Location:** `toolchain/kicad.lock.json`

**Evidence:**
```json
"docker_digest": "sha256:0000000000000000000000000000000000000000000000000000000000000001"
```

**Issue:**
The docker digest is a zeroed-out placeholder (`sha256:0...01`), not a real image digest. While `lock.py:160-161` validates against explicit `"PLACEHOLDER"` strings, this numeric placeholder passes validation but is not a real pinned image.

**Impact:**
- Toolchain lock loader (`lock.py`) accepts this as valid (passes regex at line 163)
- CI may pull different content behind the tag
- Docker mode tests relying on pinned digest semantics will get inconsistent results
- Per ECO CP-1.1 and CP-6, placeholder digests are not allowed in strict mode

**Validation:**
Check if `tools.verify` or gate tests explicitly reject zeroed digests beyond the `PLACEHOLDER` string check.

---

## H2: Golden Specs Reference Non-Existent .json Variants (MEDIUM CONFIDENCE)

**Location:** `golden_hashes/design_hashes.json` + `tests/golden_specs/`

**Evidence:**
- `design_hashes.json` contains entries for both `.json` and `.yaml` variants (e.g., `f0_cal_001.json` and `f0_cal_001.yaml`)
- Directory listing shows only `.yaml` files exist: `f0_cal_001.yaml`, `f1_via_001.yaml`, etc.
- No `.json` spec files found in `tests/golden_specs/`

**Impact:**
- Gate G1 tests (`test_g1_determinism.py:152-176`) iterate over `golden_hashes` keys
- Tests will `pytest.skip` for missing specs, but this reduces coverage
- If tests expect minimum spec counts, missing files cause failures

**Validation:**
Confirm whether G1 tests require both JSON and YAML variants or just one format per spec.

---

## H3: M0 Gate Tests Require GPU but Environment May Lack It (MEDIUM CONFIDENCE)

**Location:** `tools/m0.py`, `tools/verify.py:290-294`

**Evidence:**
- `verify.py:290` runs M0 gates when `milestone_id == "M0"` or `--include-m0` is set
- `m0.py` smoke and repro-check commands probe CUDA availability
- If GPU is unavailable and `--require-gpu` is set, exit code 2 is returned

**Impact:**
- CI environments without GPU will fail M0 smoke checks
- `_run_smoke` returns 2 if `args.require_gpu and not backend.gpu_available`
- `_run_repro_check` depends on deterministic GPU operations

**Validation:**
Check if M0 gates are being run in CI without GPU and whether they're configured with `--require-gpu`.

---

## H4: pytest Marker Configuration May Not Match Gate Runner Expectations (MEDIUM CONFIDENCE)

**Location:** `pyproject.toml:45-52`, `gate_runner.py:32-38`

**Evidence:**
- `pyproject.toml` defines markers: `gate_g1`, `gate_g2`, `gate_g3`, `gate_g4`, `gate_g5`
- `gate_runner.py` maps gate IDs to markers: `"G1": "gate_g1"`, etc.
- Tests use `@pytest.mark.gate_g1` decorator

**Potential Issue:**
- If `audit-m1` runner is invoked but tests aren't collected properly, JUnit XML parsing returns empty results
- `categorize_tests_by_gate` uses classname/name heuristics that may not match all tests
- `compute_gate_status` returns `"no_tests"` for gates with no matched tests

**Impact:**
- Gate runner may report `"partial"` status even when tests exist but aren't categorized correctly
- `build_audit_report` line 350-354 determines overall status based on gate results

**Validation:**
Run `pytest --collect-only -m gate_g1` to verify test discovery matches expectations.

---

## H5: spec_lint May Fail Due to Missing Test Matrix Entries (MEDIUM CONFIDENCE)

**Location:** `tools/spec_lint.py:136-143`, `DESIGN_DOCUMENT.md`

**Evidence:**
- `lint_design_document` checks that all requirement IDs in the document have corresponding test matrix entries
- Lines 137-143 report errors for missing requirements or empty test lists
- `verify_test_matrix_exists_in_pytest` at line 154-161 checks that referenced pytest node IDs actually exist

**Impact:**
- If DESIGN_DOCUMENT.md references tests that don't exist, spec_lint fails
- `verify.py:287` runs spec_lint as first gate
- Exit code 2 cascades from any spec_lint failure

**Validation:**
Run `python -m tools.spec_lint DESIGN_DOCUMENT.md --collect` to see specific failures.

---

## H6: DRC/Export Integration Tests Require Docker (LOW-MEDIUM CONFIDENCE)

**Location:** `tests/gates/test_g3_drc.py`, `tests/gates/test_g4_export_completeness.py`, `tests/gates/test_g5_hash_stability.py`

**Evidence:**
- Gate G3, G4, G5 tests use `_FakeDrcRunner` and `_FakeExportRunner` for unit tests
- Real integration tests are in `tests/integration/` and require Docker
- Comments indicate: "Real KiCad DRC integration tests are in tests/integration/test_kicad_drc_integration.py and require Docker"

**Impact:**
- If CI runs integration tests without Docker, they fail
- Fake runners mock KiCad behavior but may not catch real pipeline issues
- `pytest.mark.kicad_integration` marker may not be properly skipped

**Validation:**
Check if CI configures `--ignore=tests/integration` or uses marker deselection for non-Docker environments.

---

## H7: Constraint Proof Schema File May Be Missing or Invalid (LOW CONFIDENCE)

**Location:** `schemas/constraint_proof.schema.json`, `tests/gates/test_g2_constraints.py:56`

**Evidence:**
- G2 tests at line 343-357 validate proofs against `CONSTRAINT_PROOF_SCHEMA_PATH`
- Tests skip if schema file doesn't exist: `if constraint_proof_schema is None: pytest.skip`
- Tests use `jsonschema.Draft202012Validator.check_schema()` to validate schema itself

**Impact:**
- If schema file is missing or malformed, G2 schema validation tests are skipped
- Skipped tests may cause "partial" status or reduced gate coverage

**Validation:**
Verify `schemas/constraint_proof.schema.json` exists and is valid JSON Schema.

---

## Prioritized Investigation Order

1. **H1 (Placeholder Digest)** - Most likely root cause; easy to verify by inspecting lock file
2. **H5 (spec_lint)** - First gate in verify pipeline; failures cascade
3. **H4 (Marker Config)** - May cause silent test non-collection
4. **H2 (Golden Spec Variants)** - Missing .json files may reduce G1 coverage
5. **H3 (GPU Requirement)** - Environment-dependent; check CI logs
6. **H6 (Docker Integration)** - Only relevant if integration tests are being run
7. **H7 (Schema File)** - Low impact if tests skip gracefully

---

## Cross-Reference with ECO Design Doc

| Hypothesis | Related ECO Section | Change Package |
|------------|---------------------|----------------|
| H1 | CP-1.1, CP-6 | Toolchain lock + digest pinning |
| H2 | CP-5.1 | Manifest + canonicalization |
| H3 | - | (M0 substrate, not M1) |
| H4 | CP-6.2 | Gate runner + pytest markers |
| H5 | - | spec_lint (pre-existing) |
| H6 | CP-1.3 | Docker runner + CI workflow |
| H7 | CP-3.4 | Constraint proof schema |

---

## Recommended Next Steps

1. **Immediate:** Resolve placeholder digest in `toolchain/kicad.lock.json` with real sha256 digest
2. **Short-term:** Run `python -m tools.spec_lint DESIGN_DOCUMENT.md --collect` to identify test matrix gaps
3. **Short-term:** Verify golden spec file coverage (JSON vs YAML variants)
4. **Medium-term:** Ensure CI workflow properly handles Docker-dependent vs unit-only test splits
