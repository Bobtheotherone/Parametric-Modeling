# KiCad Integration Fix - Root Cause Analysis

**Date:** 2026-01-22
**Author:** Claude (claude-opus-4-5-20251101)
**Branch:** agent-run/20260122T070400Z

## Summary

The KiCad integration tests were failing with `RuntimeError: KiCad DRC failed with returncode 3` (FILE_LOAD_ERROR). After investigation, multiple root causes were identified and fixed.

## Root Causes and Fixes

### 1. S-Expression Formatting Issues (sexpr.py)

**Problem:** The S-expression writer was generating multiline format for elements that KiCad 9 expects inline, causing parse errors.

**Example of broken format:**
```sexpr
(at
  1
  0
  0)
```

**Expected format:**
```sexpr
(at 1 0 0)
```

**Fix:** Added `always_inline` set to `SExprWriter` class containing elements that should never be multiline:
- Coordinate elements: `at`, `start`, `end`, `size`, `drill`
- Property elements: `width`, `thickness`, `layer`, `net`, `tstamp`, `uuid`
- Board elements: `pad`, `segment`, `via`

**Files changed:** `src/formula_foundry/coupongen/kicad/sexpr.py`

### 2. WSL2 Docker Path Mounting (cli.py)

**Problem:** In WSL2 environments with Docker Desktop, paths under `/tmp` are not accessible to Docker containers because Docker Desktop runs in a separate namespace.

**Impact:** Tests using pytest's `tmp_path` fixture (which creates directories under `/tmp`) would fail because Docker couldn't mount those paths.

**Fix:** Added `_docker_accessible_workdir` context manager that:
1. Detects if running in WSL2 environment
2. Checks if the workdir path is Docker-accessible
3. If not, copies files to a Docker-accessible temp directory under `$HOME`
4. Runs Docker with the accessible path
5. Copies results back to the original directory

**Files changed:** `src/formula_foundry/coupongen/kicad/cli.py`

### 3. Gerber File Extension Mapping (layer_sets.json)

**Problem:** The layer validation configuration expected `.gbr` extensions for all Gerber files, but KiCad exports with specific extensions based on layer type.

**Expected:**
- F.Cu: `-F_Cu.gbr`
- B.Cu: `-B_Cu.gbr`
- Edge.Cuts: `-Edge_Cuts.gbr`

**Actual (KiCad output):**
- F.Cu: `-F_Cu.gtl` (Top layer)
- B.Cu: `-B_Cu.gbl` (Bottom layer)
- In1.Cu: `-In1_Cu.g1` (Inner layer 1)
- F.Mask: `-F_Mask.gts` (Top solder mask)
- Edge.Cuts: `-Edge_Cuts.gm1` (Edge cuts)

**Fix:** Updated `gerber_extension_map` in `layer_sets.json` with correct KiCad extensions.

**Files changed:** `coupongen/layer_sets.json`

## Test Results

After fixes:
- **45 tests passed**
- **2 tests failed** (both for `f1_via_002`)

The failing tests are due to a pre-existing spec issue: `f1_via_002` has a via diameter of 457,000nm which is slightly below the fab profile minimum of 457,200nm. This is a spec configuration issue, not a code bug.

## Verification

```bash
# Run kicad_integration tests
python3 -m pytest tests/integration/test_export_determinism_integration.py -v -m "kicad_integration"

# Manual verification of board generation
coupongen build tests/golden_specs/f0_cal_001.yaml --out /tmp/test --mode docker
# Expected: {"cache_hit":false,"coupon_id":"...","design_hash":"..."}
```

## Files Modified

1. `src/formula_foundry/coupongen/kicad/sexpr.py`
   - Added `always_inline` frozenset to force inline formatting for specific elements
   - Updated `_write_list` to check for always_inline elements

2. `src/formula_foundry/coupongen/kicad/cli.py`
   - Added `_is_wsl()` function to detect WSL environment
   - Added `_is_path_docker_accessible()` to check path accessibility
   - Added `_docker_accessible_workdir()` context manager for path translation
   - Updated `run()` method to use context manager for Docker mode

3. `coupongen/layer_sets.json`
   - Updated `gerber_extension_map` with correct KiCad Gerber extensions

## Related Issues

- The `f1_via_002` spec should be updated to use a via diameter of at least 458um (458,000nm) to meet the oshpark_4layer fab profile minimum of 457,200nm.
