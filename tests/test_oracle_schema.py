"""Tests for oracle case schema and normalization.

Tests validate:
- Schema covers all fields from the design doc
- Explicit defaults are applied during normalization
- normalize_case produces canonical configs consistent with schema
- Tests fail on missing or extra keys
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from formula_foundry.oracle import (
    FrequencyConfig,
    GridPolicy,
    MixedModeConfig,
    OracleCase,
    OracleCaseExtraKeyError,
    OracleCaseMissingKeyError,
    OracleCaseValidationError,
    PortDefinition,
    PortsConfig,
    Position3D,
    PostprocessConfig,
    ProvenanceConfig,
    ROIMargins,
    SolverPolicy,
    StructuresConfig,
    VerificationConfig,
    VerificationThresholds,
    normalize_case,
    to_canonical_dict,
    validate_oracle_case,
)

# Path to the schema file
SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "oracle_case.schema.json"


@pytest.fixture
def minimal_valid_case() -> dict:
    """Minimal valid oracle case configuration."""
    return {
        "format_version": "1.0",
        "case_id": "test_case_001",
        "frequency": {
            "start_hz": 1e8,
            "stop_hz": 20e9,
        },
        "solver_policy": {},
        "grid_policy": {},
        "ports": {
            "port_definitions": [
                {
                    "port_id": 1,
                    "position": {"x_um": 0.0, "y_um": 0.0, "z_um": 0.0},
                    "orientation": "+x",
                },
                {
                    "port_id": 2,
                    "position": {"x_um": 1000.0, "y_um": 0.0, "z_um": 0.0},
                    "orientation": "-x",
                },
            ],
        },
        "structures": {
            "type": "thru",
        },
        "postprocess": {},
        "verification": {},
    }


@pytest.fixture
def full_case() -> dict:
    """Fully specified oracle case configuration."""
    return {
        "format_version": "1.0",
        "case_id": "full_case_001",
        "frequency": {
            "start_hz": 1e8,
            "stop_hz": 20e9,
            "npoints": 401,
            "spacing": "log",
        },
        "solver_policy": {
            "boundary": "PML_16",
            "pml_cells": 16,
            "end_criteria": 1e-6,
            "max_steps": 2000000,
            "threads": 8,
            "time_step_factor": 0.9,
        },
        "grid_policy": {
            "lambda_divisor_max_cell": 30,
            "thirds_rule": False,
            "max_ratio": 1.3,
            "roi_margins_um": {
                "x_min_um": 200.0,
                "x_max_um": 200.0,
                "y_min_um": 150.0,
                "y_max_um": 150.0,
                "z_min_um": 75.0,
                "z_max_um": 75.0,
            },
            "pml_clearance_policy": "fixed_um",
        },
        "ports": {
            "port_definitions": [
                {
                    "port_id": 1,
                    "position": {"x_um": 0.0, "y_um": 0.0, "z_um": 0.0},
                    "orientation": "+x",
                    "impedance_ohm": 50.0,
                    "excitation_amplitude": 1.0,
                    "deembed_um": 100.0,
                },
                {
                    "port_id": 2,
                    "position": {"x_um": 1000.0, "y_um": 0.0, "z_um": 0.0},
                    "orientation": "-x",
                    "impedance_ohm": 50.0,
                    "excitation_amplitude": 1.0,
                    "deembed_um": 100.0,
                },
            ],
            "reference_impedance_ohm": 50.0,
            "backend": "waveguide",
        },
        "structures": {
            "type": "thru",
            "expected_behavior_profile": "cal_thru",
            "port_map": {"input": 1, "output": 2},
        },
        "postprocess": {
            "export_touchstone": True,
            "ri_format": True,
            "renormalize_to_ohms": None,
            "mixed_mode": {
                "enabled": False,
                "port_pairs": [],
            },
        },
        "verification": {
            "enabled_checks": ["passivity", "reciprocity", "causality", "energy_decay"],
            "thresholds": {
                "passivity_margin": 0.005,
                "reciprocity_db": 0.05,
                "causality_samples": 20,
                "energy_decay_db": 60.0,
            },
            "strict": True,
        },
        "provenance": {
            "git_commit": "a" * 40,
            "toolchain_digest": "sha256:" + "b" * 64,
            "openems_version": "v0.0.36",
            "gerber2ems_version": "1.2.3",
        },
    }


class TestSchemaExists:
    """Test that schema file exists and is valid JSON."""

    def test_schema_file_exists(self) -> None:
        """Schema file must exist at expected path."""
        assert SCHEMA_PATH.exists(), f"Schema file not found: {SCHEMA_PATH}"

    def test_schema_is_valid_json(self) -> None:
        """Schema file must be valid JSON."""
        content = SCHEMA_PATH.read_text(encoding="utf-8")
        schema = json.loads(content)
        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


class TestSchemaCoversDesignDocFields:
    """Test that schema covers all fields from the design doc."""

    def test_schema_has_format_version(self) -> None:
        """Schema must have format_version field."""
        schema = json.loads(SCHEMA_PATH.read_text())
        assert "format_version" in schema["properties"]

    def test_schema_has_case_id(self) -> None:
        """Schema must have case_id field."""
        schema = json.loads(SCHEMA_PATH.read_text())
        assert "case_id" in schema["properties"]

    def test_schema_has_frequency_section(self) -> None:
        """Schema must have frequency section with all subfields."""
        schema = json.loads(SCHEMA_PATH.read_text())
        assert "frequency" in schema["properties"]
        freq_ref = schema["$defs"]["FrequencyConfig"]["properties"]
        assert "start_hz" in freq_ref
        assert "stop_hz" in freq_ref
        assert "npoints" in freq_ref
        assert "spacing" in freq_ref

    def test_schema_has_solver_policy_section(self) -> None:
        """Schema must have solver_policy section with all subfields."""
        schema = json.loads(SCHEMA_PATH.read_text())
        assert "solver_policy" in schema["properties"]
        solver_ref = schema["$defs"]["SolverPolicy"]["properties"]
        assert "boundary" in solver_ref
        assert "pml_cells" in solver_ref
        assert "end_criteria" in solver_ref
        assert "max_steps" in solver_ref
        assert "threads" in solver_ref
        assert "time_step_factor" in solver_ref

    def test_schema_has_grid_policy_section(self) -> None:
        """Schema must have grid_policy section with all subfields."""
        schema = json.loads(SCHEMA_PATH.read_text())
        assert "grid_policy" in schema["properties"]
        grid_ref = schema["$defs"]["GridPolicy"]["properties"]
        assert "lambda_divisor_max_cell" in grid_ref
        assert "thirds_rule" in grid_ref
        assert "max_ratio" in grid_ref
        assert "roi_margins_um" in grid_ref
        assert "pml_clearance_policy" in grid_ref

    def test_schema_has_ports_section(self) -> None:
        """Schema must have ports section with all subfields."""
        schema = json.loads(SCHEMA_PATH.read_text())
        assert "ports" in schema["properties"]
        ports_ref = schema["$defs"]["PortsConfig"]["properties"]
        assert "port_definitions" in ports_ref
        assert "reference_impedance_ohm" in ports_ref
        assert "backend" in ports_ref

    def test_schema_has_structures_section(self) -> None:
        """Schema must have structures section with all subfields."""
        schema = json.loads(SCHEMA_PATH.read_text())
        assert "structures" in schema["properties"]
        struct_ref = schema["$defs"]["StructuresConfig"]["properties"]
        assert "type" in struct_ref
        assert "expected_behavior_profile" in struct_ref
        assert "port_map" in struct_ref

    def test_schema_has_postprocess_section(self) -> None:
        """Schema must have postprocess section with all subfields."""
        schema = json.loads(SCHEMA_PATH.read_text())
        assert "postprocess" in schema["properties"]
        post_ref = schema["$defs"]["PostprocessConfig"]["properties"]
        assert "export_touchstone" in post_ref
        assert "ri_format" in post_ref
        assert "renormalize_to_ohms" in post_ref
        assert "mixed_mode" in post_ref

    def test_schema_has_verification_section(self) -> None:
        """Schema must have verification section with all subfields."""
        schema = json.loads(SCHEMA_PATH.read_text())
        assert "verification" in schema["properties"]
        verif_ref = schema["$defs"]["VerificationConfig"]["properties"]
        assert "enabled_checks" in verif_ref
        assert "thresholds" in verif_ref
        assert "strict" in verif_ref

    def test_schema_has_provenance_section(self) -> None:
        """Schema must have provenance section with all subfields."""
        schema = json.loads(SCHEMA_PATH.read_text())
        assert "provenance" in schema["properties"]
        prov_ref = schema["$defs"]["ProvenanceConfig"]["properties"]
        assert "git_commit" in prov_ref
        assert "toolchain_digest" in prov_ref
        assert "openems_version" in prov_ref
        assert "gerber2ems_version" in prov_ref


class TestSchemaDefaults:
    """Test that schema has explicit defaults for all optional fields."""

    def test_frequency_defaults(self) -> None:
        """FrequencyConfig defaults must be explicit."""
        schema = json.loads(SCHEMA_PATH.read_text())
        freq_def = schema["$defs"]["FrequencyConfig"]["properties"]
        assert freq_def["npoints"]["default"] == 201
        assert freq_def["spacing"]["default"] == "linear"

    def test_solver_policy_defaults(self) -> None:
        """SolverPolicy defaults must be explicit."""
        schema = json.loads(SCHEMA_PATH.read_text())
        solver_def = schema["$defs"]["SolverPolicy"]["properties"]
        assert solver_def["boundary"]["default"] == "PML_8"
        assert solver_def["pml_cells"]["default"] == 8
        assert solver_def["end_criteria"]["default"] == 1e-5
        assert solver_def["max_steps"]["default"] == 1000000
        assert solver_def["threads"]["default"] == 4
        assert solver_def["time_step_factor"]["default"] == 0.95

    def test_grid_policy_defaults(self) -> None:
        """GridPolicy defaults must be explicit."""
        schema = json.loads(SCHEMA_PATH.read_text())
        grid_def = schema["$defs"]["GridPolicy"]["properties"]
        assert grid_def["lambda_divisor_max_cell"]["default"] == 20
        assert grid_def["thirds_rule"]["default"] is True
        assert grid_def["max_ratio"]["default"] == 1.5
        assert grid_def["pml_clearance_policy"]["default"] == "wavelength_based"

    def test_ports_config_defaults(self) -> None:
        """PortsConfig defaults must be explicit."""
        schema = json.loads(SCHEMA_PATH.read_text())
        ports_def = schema["$defs"]["PortsConfig"]["properties"]
        assert ports_def["reference_impedance_ohm"]["default"] == 50.0
        assert ports_def["backend"]["default"] == "waveguide"

    def test_port_definition_defaults(self) -> None:
        """PortDefinition defaults must be explicit."""
        schema = json.loads(SCHEMA_PATH.read_text())
        port_def = schema["$defs"]["PortDefinition"]["properties"]
        assert port_def["impedance_ohm"]["default"] == 50.0
        assert port_def["excitation_amplitude"]["default"] == 1.0
        assert port_def["deembed_um"]["default"] == 0.0

    def test_postprocess_config_defaults(self) -> None:
        """PostprocessConfig defaults must be explicit."""
        schema = json.loads(SCHEMA_PATH.read_text())
        post_def = schema["$defs"]["PostprocessConfig"]["properties"]
        assert post_def["export_touchstone"]["default"] is True
        assert post_def["ri_format"]["default"] is True
        assert post_def["renormalize_to_ohms"]["default"] is None

    def test_verification_config_defaults(self) -> None:
        """VerificationConfig defaults must be explicit."""
        schema = json.loads(SCHEMA_PATH.read_text())
        verif_def = schema["$defs"]["VerificationConfig"]["properties"]
        assert verif_def["enabled_checks"]["default"] == ["passivity", "reciprocity", "energy_decay"]
        assert verif_def["strict"]["default"] is True

    def test_verification_thresholds_defaults(self) -> None:
        """VerificationThresholds defaults must be explicit."""
        schema = json.loads(SCHEMA_PATH.read_text())
        thresh_def = schema["$defs"]["VerificationThresholds"]["properties"]
        assert thresh_def["passivity_margin"]["default"] == 0.01
        assert thresh_def["reciprocity_db"]["default"] == 0.1
        assert thresh_def["causality_samples"]["default"] == 10
        assert thresh_def["energy_decay_db"]["default"] == 50.0


class TestValidationFailsOnMissingKeys:
    """Test that validation fails on missing required keys."""

    def test_missing_format_version(self, minimal_valid_case: dict) -> None:
        """Validation must fail when format_version is missing."""
        del minimal_valid_case["format_version"]
        with pytest.raises(OracleCaseMissingKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "format_version" in str(exc_info.value)

    def test_missing_case_id(self, minimal_valid_case: dict) -> None:
        """Validation must fail when case_id is missing."""
        del minimal_valid_case["case_id"]
        with pytest.raises(OracleCaseMissingKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "case_id" in str(exc_info.value)

    def test_missing_frequency(self, minimal_valid_case: dict) -> None:
        """Validation must fail when frequency is missing."""
        del minimal_valid_case["frequency"]
        with pytest.raises(OracleCaseMissingKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "frequency" in str(exc_info.value)

    def test_missing_frequency_start_hz(self, minimal_valid_case: dict) -> None:
        """Validation must fail when frequency.start_hz is missing."""
        del minimal_valid_case["frequency"]["start_hz"]
        with pytest.raises(OracleCaseMissingKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "start_hz" in str(exc_info.value)

    def test_missing_ports(self, minimal_valid_case: dict) -> None:
        """Validation must fail when ports is missing."""
        del minimal_valid_case["ports"]
        with pytest.raises(OracleCaseMissingKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "ports" in str(exc_info.value)

    def test_missing_port_definitions(self, minimal_valid_case: dict) -> None:
        """Validation must fail when port_definitions is missing."""
        del minimal_valid_case["ports"]["port_definitions"]
        with pytest.raises(OracleCaseMissingKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "port_definitions" in str(exc_info.value)

    def test_missing_port_id(self, minimal_valid_case: dict) -> None:
        """Validation must fail when port_id is missing from a port definition."""
        del minimal_valid_case["ports"]["port_definitions"][0]["port_id"]
        with pytest.raises(OracleCaseMissingKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "port_id" in str(exc_info.value)

    def test_missing_structures(self, minimal_valid_case: dict) -> None:
        """Validation must fail when structures is missing."""
        del minimal_valid_case["structures"]
        with pytest.raises(OracleCaseMissingKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "structures" in str(exc_info.value)

    def test_missing_structures_type(self, minimal_valid_case: dict) -> None:
        """Validation must fail when structures.type is missing."""
        del minimal_valid_case["structures"]["type"]
        with pytest.raises(OracleCaseMissingKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "type" in str(exc_info.value)


class TestValidationFailsOnExtraKeys:
    """Test that validation fails on unexpected extra keys."""

    def test_extra_top_level_key(self, minimal_valid_case: dict) -> None:
        """Validation must fail when an extra top-level key is present."""
        minimal_valid_case["unexpected_field"] = "value"
        with pytest.raises(OracleCaseExtraKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "unexpected_field" in str(exc_info.value)

    def test_extra_frequency_key(self, minimal_valid_case: dict) -> None:
        """Validation must fail when an extra key is in frequency."""
        minimal_valid_case["frequency"]["extra_freq_field"] = 123
        with pytest.raises(OracleCaseExtraKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "extra_freq_field" in str(exc_info.value)

    def test_extra_solver_policy_key(self, minimal_valid_case: dict) -> None:
        """Validation must fail when an extra key is in solver_policy."""
        minimal_valid_case["solver_policy"]["unknown_option"] = True
        with pytest.raises(OracleCaseExtraKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "unknown_option" in str(exc_info.value)

    def test_extra_port_definition_key(self, minimal_valid_case: dict) -> None:
        """Validation must fail when an extra key is in a port definition."""
        minimal_valid_case["ports"]["port_definitions"][0]["extra_port_field"] = 42
        with pytest.raises(OracleCaseExtraKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "extra_port_field" in str(exc_info.value)

    def test_extra_structures_key(self, minimal_valid_case: dict) -> None:
        """Validation must fail when an extra key is in structures."""
        minimal_valid_case["structures"]["unknown_struct_field"] = "value"
        with pytest.raises(OracleCaseExtraKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "unknown_struct_field" in str(exc_info.value)

    def test_extra_postprocess_key(self, minimal_valid_case: dict) -> None:
        """Validation must fail when an extra key is in postprocess."""
        minimal_valid_case["postprocess"]["extra_post_field"] = False
        with pytest.raises(OracleCaseExtraKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "extra_post_field" in str(exc_info.value)

    def test_extra_verification_key(self, minimal_valid_case: dict) -> None:
        """Validation must fail when an extra key is in verification."""
        minimal_valid_case["verification"]["extra_verify_field"] = []
        with pytest.raises(OracleCaseExtraKeyError) as exc_info:
            validate_oracle_case(minimal_valid_case)
        assert "extra_verify_field" in str(exc_info.value)


class TestNormalizeCaseAppliesDefaults:
    """Test that normalize_case applies all defaults correctly."""

    def test_normalize_applies_frequency_defaults(self, minimal_valid_case: dict) -> None:
        """Normalization must apply frequency defaults."""
        case = normalize_case(minimal_valid_case)
        assert case.frequency.npoints == 201
        assert case.frequency.spacing == "linear"

    def test_normalize_applies_solver_policy_defaults(self, minimal_valid_case: dict) -> None:
        """Normalization must apply solver_policy defaults."""
        case = normalize_case(minimal_valid_case)
        assert case.solver_policy.boundary == "PML_8"
        assert case.solver_policy.pml_cells == 8
        assert case.solver_policy.end_criteria == 1e-5
        assert case.solver_policy.max_steps == 1000000
        assert case.solver_policy.threads == 4
        assert case.solver_policy.time_step_factor == 0.95

    def test_normalize_applies_grid_policy_defaults(self, minimal_valid_case: dict) -> None:
        """Normalization must apply grid_policy defaults."""
        case = normalize_case(minimal_valid_case)
        assert case.grid_policy.lambda_divisor_max_cell == 20
        assert case.grid_policy.thirds_rule is True
        assert case.grid_policy.max_ratio == 1.5
        assert case.grid_policy.pml_clearance_policy == "wavelength_based"

    def test_normalize_applies_roi_margins_defaults(self, minimal_valid_case: dict) -> None:
        """Normalization must apply ROI margins defaults."""
        case = normalize_case(minimal_valid_case)
        roi = case.grid_policy.roi_margins_um
        assert roi.x_min_um == 100.0
        assert roi.x_max_um == 100.0
        assert roi.y_min_um == 100.0
        assert roi.y_max_um == 100.0
        assert roi.z_min_um == 50.0
        assert roi.z_max_um == 50.0

    def test_normalize_applies_ports_defaults(self, minimal_valid_case: dict) -> None:
        """Normalization must apply ports defaults."""
        case = normalize_case(minimal_valid_case)
        assert case.ports.reference_impedance_ohm == 50.0
        assert case.ports.backend == "waveguide"

    def test_normalize_applies_port_definition_defaults(self, minimal_valid_case: dict) -> None:
        """Normalization must apply port definition defaults."""
        case = normalize_case(minimal_valid_case)
        port = case.ports.port_definitions[0]
        assert port.impedance_ohm == 50.0
        assert port.excitation_amplitude == 1.0
        assert port.deembed_um == 0.0

    def test_normalize_applies_structures_defaults(self, minimal_valid_case: dict) -> None:
        """Normalization must apply structures defaults."""
        case = normalize_case(minimal_valid_case)
        assert case.structures.expected_behavior_profile == "custom"
        assert case.structures.port_map == {}

    def test_normalize_applies_postprocess_defaults(self, minimal_valid_case: dict) -> None:
        """Normalization must apply postprocess defaults."""
        case = normalize_case(minimal_valid_case)
        assert case.postprocess.export_touchstone is True
        assert case.postprocess.ri_format is True
        assert case.postprocess.renormalize_to_ohms is None

    def test_normalize_applies_mixed_mode_defaults(self, minimal_valid_case: dict) -> None:
        """Normalization must apply mixed_mode defaults."""
        case = normalize_case(minimal_valid_case)
        assert case.postprocess.mixed_mode.enabled is False
        assert case.postprocess.mixed_mode.port_pairs == ()

    def test_normalize_applies_verification_defaults(self, minimal_valid_case: dict) -> None:
        """Normalization must apply verification defaults."""
        case = normalize_case(minimal_valid_case)
        assert case.verification.enabled_checks == ("passivity", "reciprocity", "energy_decay")
        assert case.verification.strict is True

    def test_normalize_applies_verification_thresholds_defaults(self, minimal_valid_case: dict) -> None:
        """Normalization must apply verification thresholds defaults."""
        case = normalize_case(minimal_valid_case)
        thresh = case.verification.thresholds
        assert thresh.passivity_margin == 0.01
        assert thresh.reciprocity_db == 0.1
        assert thresh.causality_samples == 10
        assert thresh.energy_decay_db == 50.0


class TestNormalizeCasePreservesExplicitValues:
    """Test that normalize_case preserves explicitly set values."""

    def test_normalize_preserves_explicit_frequency_values(self, full_case: dict) -> None:
        """Normalization must preserve explicit frequency values."""
        case = normalize_case(full_case)
        assert case.frequency.start_hz == 1e8
        assert case.frequency.stop_hz == 20e9
        assert case.frequency.npoints == 401
        assert case.frequency.spacing == "log"

    def test_normalize_preserves_explicit_solver_values(self, full_case: dict) -> None:
        """Normalization must preserve explicit solver_policy values."""
        case = normalize_case(full_case)
        assert case.solver_policy.boundary == "PML_16"
        assert case.solver_policy.pml_cells == 16
        assert case.solver_policy.end_criteria == 1e-6
        assert case.solver_policy.max_steps == 2000000
        assert case.solver_policy.threads == 8
        assert case.solver_policy.time_step_factor == 0.9

    def test_normalize_preserves_explicit_grid_values(self, full_case: dict) -> None:
        """Normalization must preserve explicit grid_policy values."""
        case = normalize_case(full_case)
        assert case.grid_policy.lambda_divisor_max_cell == 30
        assert case.grid_policy.thirds_rule is False
        assert case.grid_policy.max_ratio == 1.3
        assert case.grid_policy.pml_clearance_policy == "fixed_um"

    def test_normalize_preserves_explicit_roi_margins(self, full_case: dict) -> None:
        """Normalization must preserve explicit ROI margins."""
        case = normalize_case(full_case)
        roi = case.grid_policy.roi_margins_um
        assert roi.x_min_um == 200.0
        assert roi.x_max_um == 200.0
        assert roi.y_min_um == 150.0
        assert roi.y_max_um == 150.0
        assert roi.z_min_um == 75.0
        assert roi.z_max_um == 75.0

    def test_normalize_preserves_explicit_verification_thresholds(self, full_case: dict) -> None:
        """Normalization must preserve explicit verification thresholds."""
        case = normalize_case(full_case)
        thresh = case.verification.thresholds
        assert thresh.passivity_margin == 0.005
        assert thresh.reciprocity_db == 0.05
        assert thresh.causality_samples == 20
        assert thresh.energy_decay_db == 60.0

    def test_normalize_preserves_provenance(self, full_case: dict) -> None:
        """Normalization must preserve provenance data."""
        case = normalize_case(full_case)
        assert case.provenance is not None
        assert case.provenance.git_commit == "a" * 40
        assert case.provenance.toolchain_digest == "sha256:" + "b" * 64
        assert case.provenance.openems_version == "v0.0.36"
        assert case.provenance.gerber2ems_version == "1.2.3"


class TestToCanonicalDict:
    """Test that to_canonical_dict produces schema-compliant output."""

    def test_canonical_dict_passes_validation(self, minimal_valid_case: dict) -> None:
        """Canonical dict from normalized case must pass schema validation."""
        case = normalize_case(minimal_valid_case)
        canonical = to_canonical_dict(case)
        # Should not raise
        validate_oracle_case(canonical)

    def test_canonical_dict_has_all_fields_explicit(self, minimal_valid_case: dict) -> None:
        """Canonical dict must have all fields explicitly set."""
        case = normalize_case(minimal_valid_case)
        canonical = to_canonical_dict(case)

        # Check top-level required fields
        assert "format_version" in canonical
        assert "case_id" in canonical
        assert "frequency" in canonical
        assert "solver_policy" in canonical
        assert "grid_policy" in canonical
        assert "ports" in canonical
        assert "structures" in canonical
        assert "postprocess" in canonical
        assert "verification" in canonical

        # Check nested fields have explicit values
        assert "npoints" in canonical["frequency"]
        assert "spacing" in canonical["frequency"]
        assert "boundary" in canonical["solver_policy"]
        assert "pml_cells" in canonical["solver_policy"]

    def test_canonical_dict_roundtrip(self, minimal_valid_case: dict) -> None:
        """Canonical dict should produce identical result when re-normalized."""
        case1 = normalize_case(minimal_valid_case)
        canonical1 = to_canonical_dict(case1)

        case2 = normalize_case(canonical1)
        canonical2 = to_canonical_dict(case2)

        # Normalize and convert again - should be identical
        assert canonical1 == canonical2


class TestOracleCaseDataclass:
    """Test OracleCase dataclass properties."""

    def test_oracle_case_is_frozen(self, minimal_valid_case: dict) -> None:
        """OracleCase must be immutable (frozen)."""
        case = normalize_case(minimal_valid_case)
        with pytest.raises(AttributeError):
            case.case_id = "new_id"  # type: ignore

    def test_frequency_config_is_frozen(self, minimal_valid_case: dict) -> None:
        """FrequencyConfig must be immutable (frozen)."""
        case = normalize_case(minimal_valid_case)
        with pytest.raises(AttributeError):
            case.frequency.npoints = 500  # type: ignore

    def test_solver_policy_is_frozen(self, minimal_valid_case: dict) -> None:
        """SolverPolicy must be immutable (frozen)."""
        case = normalize_case(minimal_valid_case)
        with pytest.raises(AttributeError):
            case.solver_policy.threads = 16  # type: ignore


class TestSchemaValidationDetails:
    """Test specific schema validation rules."""

    def test_case_id_pattern(self, minimal_valid_case: dict) -> None:
        """case_id must match allowed pattern."""
        minimal_valid_case["case_id"] = "invalid case id!"
        with pytest.raises(OracleCaseValidationError):
            validate_oracle_case(minimal_valid_case)

    def test_format_version_must_be_1_0(self, minimal_valid_case: dict) -> None:
        """format_version must be '1.0'."""
        minimal_valid_case["format_version"] = "2.0"
        with pytest.raises(OracleCaseValidationError):
            validate_oracle_case(minimal_valid_case)

    def test_frequency_start_must_be_positive(self, minimal_valid_case: dict) -> None:
        """frequency.start_hz must be positive."""
        minimal_valid_case["frequency"]["start_hz"] = 0
        with pytest.raises(OracleCaseValidationError):
            validate_oracle_case(minimal_valid_case)

    def test_port_orientation_must_be_cardinal(self, minimal_valid_case: dict) -> None:
        """Port orientation must be a cardinal direction."""
        minimal_valid_case["ports"]["port_definitions"][0]["orientation"] = "diagonal"
        with pytest.raises(OracleCaseValidationError):
            validate_oracle_case(minimal_valid_case)

    def test_structure_type_must_be_valid(self, minimal_valid_case: dict) -> None:
        """structures.type must be from allowed enum."""
        minimal_valid_case["structures"]["type"] = "invalid_type"
        with pytest.raises(OracleCaseValidationError):
            validate_oracle_case(minimal_valid_case)

    def test_pml_cells_minimum(self, minimal_valid_case: dict) -> None:
        """solver_policy.pml_cells must be at least 4."""
        minimal_valid_case["solver_policy"]["pml_cells"] = 2
        with pytest.raises(OracleCaseValidationError):
            validate_oracle_case(minimal_valid_case)

    def test_max_ratio_bounds(self, minimal_valid_case: dict) -> None:
        """grid_policy.max_ratio must be between 1.0 and 2.0."""
        minimal_valid_case["grid_policy"]["max_ratio"] = 2.5
        with pytest.raises(OracleCaseValidationError):
            validate_oracle_case(minimal_valid_case)

    def test_git_commit_pattern(self, full_case: dict) -> None:
        """provenance.git_commit must be 40 hex characters."""
        full_case["provenance"]["git_commit"] = "short"
        with pytest.raises(OracleCaseValidationError):
            validate_oracle_case(full_case)

    def test_toolchain_digest_pattern(self, full_case: dict) -> None:
        """provenance.toolchain_digest must match sha256:... pattern."""
        full_case["provenance"]["toolchain_digest"] = "invalid_digest"
        with pytest.raises(OracleCaseValidationError):
            validate_oracle_case(full_case)

    def test_port_definitions_minimum(self, minimal_valid_case: dict) -> None:
        """port_definitions must have at least 1 port."""
        minimal_valid_case["ports"]["port_definitions"] = []
        with pytest.raises(OracleCaseValidationError):
            validate_oracle_case(minimal_valid_case)

    def test_enabled_checks_unique_items(self, minimal_valid_case: dict) -> None:
        """verification.enabled_checks must have unique items."""
        minimal_valid_case["verification"]["enabled_checks"] = ["passivity", "passivity"]
        with pytest.raises(OracleCaseValidationError):
            validate_oracle_case(minimal_valid_case)
