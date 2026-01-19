from __future__ import annotations

import abc
import uuid
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path

from ..resolve import ResolvedDesign
from ..spec import CouponSpec

_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "coupongen")


class IKiCadBackend(abc.ABC):
    name: str

    @abc.abstractmethod
    def write_board(self, spec: CouponSpec, resolved: ResolvedDesign, out_dir: Path) -> Path:
        raise NotImplementedError


class BackendA(IKiCadBackend):
    name = "sexpr"

    def write_board(self, spec: CouponSpec, resolved: ResolvedDesign, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        board_path = out_dir / "coupon.kicad_pcb"
        board_text = build_board_text(spec, resolved)
        board_path.write_text(board_text, encoding="utf-8")
        return board_path


def deterministic_uuid(schema_version: int, path: str) -> str:
    namespace = uuid.uuid5(_UUID_NAMESPACE, f"schema:{schema_version}")
    return str(uuid.uuid5(namespace, path))


def build_board_text(spec: CouponSpec, resolved: ResolvedDesign) -> str:
    width_nm = int(spec.board.outline.width_nm)
    length_nm = int(spec.board.outline.length_nm)
    half_width = width_nm // 2
    start = _format_point(0, -half_width)
    end = _format_point(length_nm, half_width)
    outline_uuid = deterministic_uuid(spec.schema_version, "board.outline")
    lines = [
        "(kicad_pcb (version 20240101) (generator coupongen)",
        '  (general (thickness 1.6))',
        '  (paper "A4")',
        '  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))',
        '  (net 0 "")',
        '  (net 1 "SIG")',
        '  (net 2 "GND")',
        f'  (gr_rect (start {start}) (end {end}) (layer "Edge.Cuts") (width 0.1) (tstamp {outline_uuid}))',
    ]
    lines.extend(_footprint_blocks(spec))
    lines.append(")")
    return "\n".join(lines)


def _footprint_blocks(spec: CouponSpec) -> Iterable[str]:
    blocks: list[str] = []
    for side in ("left", "right"):
        connector = getattr(spec.connectors, side)
        uuid_value = deterministic_uuid(spec.schema_version, f"connector.{side}")
        position = _format_point(int(connector.position_nm[0]), int(connector.position_nm[1]))
        blocks.append(
            f'  (footprint "{connector.footprint}" (layer "F.Cu") '
            f'(at {position} {connector.rotation_deg}) (tstamp {uuid_value}))'
        )
    return blocks


def _format_point(x_nm: int, y_nm: int) -> str:
    return f"{_nm_to_mm(x_nm)} {_nm_to_mm(y_nm)}"


def _nm_to_mm(value_nm: int) -> str:
    mm = Decimal(value_nm) / Decimal(1_000_000)
    text = format(mm, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"
