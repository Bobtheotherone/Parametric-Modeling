# Golden Exports Test Fixtures

This directory contains expected export metadata for G1-G5 determinism gates.

## Purpose

These fixtures define the expected layer lists and export structure for each coupon
family. They enable CI tests to verify:

1. **G4 Export Completeness**: All expected layers are present in exports
2. **G5 Export Hash Stability**: Canonical hashes remain stable across runs

## Structure

- `layer_sets.json`: Expected layer set definitions per coupon family
- `export_hashes.json`: Expected canonical export hashes per golden spec

## Usage in Tests

```python
# In test_export_determinism_integration.py
from tests.golden_exports import layer_sets, export_hashes

# Verify all expected layers exist
assert set(exported_files) >= set(layer_sets["F0_CAL_THRU_LINE"]["gerbers"])
assert set(exported_files) >= set(layer_sets["F0_CAL_THRU_LINE"]["drills"])

# Verify hash stability (after actual builds generate these)
# expected = export_hashes["f0_cal_001"]["canonical_gerber_hashes"]
```

## Notes

- Export hashes are placeholders until actual KiCad builds generate them
- Layer sets are locked per family to ensure export completeness
- All paths assume 4-layer stackup (default for oshpark_4layer fab profile)
