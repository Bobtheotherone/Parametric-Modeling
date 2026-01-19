from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .units import LengthNM


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
    type: str = Field(..., min_length=1)
    layer: str = Field(..., min_length=1)
    w_nm: LengthNM
    gap_nm: LengthNM
    length_left_nm: LengthNM
    length_right_nm: LengthNM
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
    return CouponSpec.model_validate(data)
