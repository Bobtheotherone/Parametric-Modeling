from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .units import LengthNM

_SCHEMA_DIR = Path(__file__).parent.parent.parent.parent / "coupongen" / "schemas"
COUPONSPEC_SCHEMA_PATH = _SCHEMA_DIR / "coupon_spec.schema.json"


class _SpecBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KicadToolchain(_SpecBase):
    version: str = Field(..., min_length=1)
    docker_image: str = Field(..., min_length=1)


class Toolchain(_SpecBase):
    kicad: KicadToolchain


class FabProfile(_SpecBase):
    id: str = Field(..., min_length=1)
    overrides: dict[str, Any] = Field(default_factory=dict)


class StackupMaterials(_SpecBase):
    er: float
    loss_tangent: float


class Stackup(_SpecBase):
    copper_layers: int = Field(..., ge=1)
    thicknesses_nm: dict[str, LengthNM]
    materials: StackupMaterials


class BoardOutline(_SpecBase):
    width_nm: LengthNM
    length_nm: LengthNM
    corner_radius_nm: LengthNM


class BoardOrigin(_SpecBase):
    mode: str = Field(..., min_length=1)


class BoardText(_SpecBase):
    coupon_id: str = Field(..., min_length=1)
    include_manifest_hash: bool


class Board(_SpecBase):
    outline: BoardOutline
    origin: BoardOrigin
    text: BoardText


class Connector(_SpecBase):
    footprint: str = Field(..., min_length=1)
    position_nm: tuple[LengthNM, LengthNM]
    rotation_deg: int


class Connectors(_SpecBase):
    left: Connector
    right: Connector


class ViaSpec(_SpecBase):
    drill_nm: LengthNM
    diameter_nm: LengthNM


class SignalViaSpec(_SpecBase):
    drill_nm: LengthNM
    diameter_nm: LengthNM
    pad_diameter_nm: LengthNM


class GroundViaFence(_SpecBase):
    enabled: bool
    pitch_nm: LengthNM
    offset_from_gap_nm: LengthNM
    via: ViaSpec


class TransmissionLine(_SpecBase):
    """Transmission line parameters.

    For F1 coupons, length_right_nm is deprecated and should not be specified.
    The right length is derived from the continuity formula:
        length_right = x_right_connector - x_discontinuity_center

    If length_right_nm is provided for F1 coupons, it will be validated against
    the derived value (must match exactly, i.e., continuity_length_error_nm == 0).

    CP-2.2: Make right length derived for F1 coupons (ECO-M1-ALIGN-0001).
    """

    type: str = Field(..., min_length=1)
    layer: str = Field(..., min_length=1)
    w_nm: LengthNM
    gap_nm: LengthNM
    length_left_nm: LengthNM
    # DEPRECATED for F1 coupons: length_right_nm is derived from continuity formula.
    # If specified, it must match the derived value exactly (continuity_length_error_nm == 0).
    # For F0 coupons, this field is still required (both lengths define the through-line).
    length_right_nm: LengthNM | None = None
    ground_via_fence: GroundViaFence | None = None


class Antipad(_SpecBase):
    shape: str = Field(..., min_length=1)
    rx_nm: LengthNM | None = None
    ry_nm: LengthNM | None = None
    corner_nm: LengthNM | None = None
    r_nm: LengthNM | None = None


class PlaneCutout(_SpecBase):
    shape: str = Field(..., min_length=1)
    length_nm: LengthNM
    width_nm: LengthNM
    rotation_deg: int


class ReturnVias(_SpecBase):
    pattern: str = Field(..., min_length=1)
    count: int = Field(..., ge=1)
    radius_nm: LengthNM
    via: ViaSpec


class Discontinuity(_SpecBase):
    type: str = Field(..., min_length=1)
    signal_via: SignalViaSpec
    antipads: dict[str, Antipad] = Field(default_factory=dict)
    return_vias: ReturnVias | None = None
    plane_cutouts: dict[str, PlaneCutout] = Field(default_factory=dict)


class DrcSpec(_SpecBase):
    must_pass: bool
    severity: str = Field(..., min_length=1)


class SymmetrySpec(_SpecBase):
    enforce: bool


class ConstraintsSpec(_SpecBase):
    mode: Literal["REJECT", "REPAIR"]
    drc: DrcSpec
    symmetry: SymmetrySpec
    allow_unconnected_copper: bool


class ExportGerbers(_SpecBase):
    enabled: bool
    format: str = Field(..., min_length=1)


class ExportDrill(_SpecBase):
    enabled: bool
    format: str = Field(..., min_length=1)


class ExportSpec(_SpecBase):
    gerbers: ExportGerbers
    drill: ExportDrill
    outputs_dir: str = Field(..., min_length=1)


class CouponSpec(_SpecBase):
    schema_version: int = Field(..., ge=1)
    coupon_family: str = Field(..., min_length=1)
    units: Literal["nm"] = "nm"
    toolchain: Toolchain
    fab_profile: FabProfile
    stackup: Stackup
    board: Board
    connectors: Connectors
    transmission_line: TransmissionLine
    discontinuity: Discontinuity | None = None
    constraints: ConstraintsSpec
    export: ExportSpec


COUPONSPEC_SCHEMA = CouponSpec.model_json_schema()


def load_couponspec(data: dict[str, Any]) -> CouponSpec:
    """Validate and load a CouponSpec from a dictionary.

    Args:
        data: Dictionary containing the CouponSpec data.

    Returns:
        Validated CouponSpec instance with all lengths normalized to integer nm.

    Raises:
        pydantic.ValidationError: If the data fails validation.
    """
    return CouponSpec.model_validate(data)


def load_couponspec_from_file(path: Path | str) -> CouponSpec:
    """Load and validate a CouponSpec from a YAML or JSON file.

    Args:
        path: Path to the YAML (.yaml, .yml) or JSON (.json) file.

    Returns:
        Validated CouponSpec instance with all lengths normalized to integer nm.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
        pydantic.ValidationError: If the data fails validation.
        yaml.YAMLError: If YAML parsing fails.
        json.JSONDecodeError: If JSON parsing fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CouponSpec file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("pyyaml is required for YAML support: pip install pyyaml") from exc
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Use .yaml, .yml, or .json")

    if not isinstance(data, dict):
        raise ValueError(f"CouponSpec file must contain a mapping, got {type(data).__name__}")

    return load_couponspec(data)


def get_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for CouponSpec.

    Loads from the canonical schema file if available, otherwise
    returns the Pydantic-generated schema.

    Returns:
        JSON Schema dictionary.
    """
    if COUPONSPEC_SCHEMA_PATH.exists():
        return json.loads(COUPONSPEC_SCHEMA_PATH.read_text(encoding="utf-8"))
    return COUPONSPEC_SCHEMA


def validate_against_json_schema(data: dict[str, Any]) -> list[str]:
    """Validate data against the CouponSpec JSON Schema.

    Uses the jsonschema library (Draft 2020-12) to validate the input
    data against the canonical JSON Schema file. This provides schema
    validation independent of Pydantic for interoperability.

    Args:
        data: Dictionary containing the CouponSpec data.

    Returns:
        List of validation error messages (empty if valid).

    Raises:
        ImportError: If jsonschema package is not installed.
    """
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ImportError("jsonschema is required for JSON Schema validation: pip install jsonschema") from exc

    schema = get_json_schema()
    validator = Draft202012Validator(schema)
    errors = []
    for error in validator.iter_errors(data):
        path = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"{path}: {error.message}")
    return errors


class StrictValidationError(Exception):
    """Raised when strict validation fails.

    This exception is raised when a CouponSpec fails strict mode validation,
    which includes JSON schema validation, extra field rejection, and
    family-specific constraint enforcement.

    Attributes:
        errors: List of validation error messages.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Strict validation failed with {len(errors)} error(s): {errors}")


def validate_strict(
    data: dict[str, Any],
    *,
    check_family_constraints: bool = True,
) -> CouponSpec:
    """Validate a CouponSpec in strict mode.

    This function performs comprehensive validation that:
    1. Validates against the JSON Schema (rejects unknown/extra fields via
       additionalProperties: false)
    2. Validates via Pydantic (extra="forbid" enforced on all models)
    3. Optionally validates family-specific constraints (F0 vs F1 rules)

    Strict mode validation enforces REQ-M1-002:
    - Unknown/extra fields are rejected (no silent accept)
    - Family-specific correctness is enforced (F0 cannot include F1-only
      blocks like discontinuity, F1 requires discontinuity)
    - Cross-family fields cause validation failures

    Args:
        data: Dictionary containing the CouponSpec data.
        check_family_constraints: If True (default), validate family-specific
            constraints. Set to False to skip family validation (useful for
            schema-only validation).

    Returns:
        Validated CouponSpec instance.

    Raises:
        StrictValidationError: If validation fails. The exception contains
            a list of all validation errors.

    Example:
        >>> data = {"schema_version": 1, "coupon_family": "F0_CAL_THRU_LINE", ...}
        >>> try:
        ...     spec = validate_strict(data)
        ... except StrictValidationError as e:
        ...     print(f"Validation failed: {e.errors}")
    """
    from .families import validate_family

    errors: list[str] = []

    # Step 1: JSON Schema validation (catches extra fields via additionalProperties: false)
    schema_errors = validate_against_json_schema(data)
    errors.extend(schema_errors)

    # Step 2: Pydantic validation (extra="forbid" enforced on all _SpecBase models)
    spec: CouponSpec | None = None
    try:
        spec = load_couponspec(data)
    except Exception as e:
        errors.append(f"Pydantic validation failed: {e}")

    # If we have fatal errors at this point, raise early
    if spec is None:
        raise StrictValidationError(errors)

    # Step 3: Family-specific constraint validation
    if check_family_constraints:
        try:
            validate_family(spec)
        except ValueError as e:
            errors.append(f"Family validation failed: {e}")

        # Additional cross-family field checks
        family_errors = _validate_cross_family_fields(spec)
        errors.extend(family_errors)

    if errors:
        raise StrictValidationError(errors)

    return spec


def _validate_cross_family_fields(spec: CouponSpec) -> list[str]:
    """Validate cross-family field constraints.

    This helper enforces family-specific field requirements:
    - F0 (thru-line) requires transmission_line.length_right_nm to be specified
      (both lengths define the symmetric through-line)
    - F1 (via transition) deprecates transmission_line.length_right_nm
      (it is derived from the continuity formula; if specified, it must be
      validated against the derived value during resolve)

    Args:
        spec: Validated CouponSpec instance.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    # Import here to avoid circular imports
    from .families import FAMILY_F0, FAMILY_F1

    if spec.coupon_family == FAMILY_F0:
        # F0 requires both length_left_nm and length_right_nm
        if spec.transmission_line.length_right_nm is None:
            errors.append(
                "F0_CAL_THRU_LINE requires transmission_line.length_right_nm to be specified "
                "(both lengths define the symmetric through-line)"
            )

    elif spec.coupon_family == FAMILY_F1:
        # F1 coupons: length_right_nm is deprecated (derived from continuity formula)
        # We don't error here, but the resolve step will validate if it's specified
        # For strict mode, we warn if it's specified (the user should remove it)
        if spec.transmission_line.length_right_nm is not None:
            # This is a warning/info for strict mode - length_right_nm is deprecated for F1
            # The resolve step will validate that specified value matches derived value
            # For now, we allow it but could optionally reject in stricter modes
            pass

    return errors
