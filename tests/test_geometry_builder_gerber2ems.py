"""Tests for deterministic Gerber2EMS geometry builder output.

REQ-M2-004: geometry.xml must be deterministic for fixed inputs and settings.
"""

from __future__ import annotations

from pathlib import Path

from formula_foundry.geometry.gerber2ems import Gerber2EmsSettings, build_geometry_xml


def _write_gerbers(root: Path) -> dict[str, Path]:
    gerber_dir = root / "gerbers"
    gerber_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "F.Cu": gerber_dir / "F.Cu.gbr",
        "B.Cu": gerber_dir / "B.Cu.gbr",
        "Edge.Cuts": gerber_dir / "Edge.Cuts.gbr",
    }
    for path in paths.values():
        path.write_text("G04 Test*\nX0Y0D02*\n", encoding="utf-8")
    return paths


def _extract_layers(xml_text: str) -> list[str]:
    layers: list[str] = []
    for line in xml_text.splitlines():
        line = line.strip()
        if not line.startswith("<gerber "):
            continue
        start = line.index('layer="') + len('layer="')
        end = line.index('"', start)
        layers.append(line[start:end])
    return layers


def test_gerber2ems_geometry_is_deterministic_given_fixed_settings(tmp_path: Path) -> None:
    inputs = _write_gerbers(tmp_path)
    settings = Gerber2EmsSettings(
        rasterization={
            "dpi": 1200,
            "oversample": 2,
            "filter": "gaussian",
            "invert": True,
        },
        contour={
            "tolerance_um": 2.5,
            "min_area_um2": 10,
            "preserve_holes": False,
        },
    )

    output_a = build_geometry_xml(inputs, settings)
    output_b = build_geometry_xml(
        {
            "Edge.Cuts": inputs["Edge.Cuts"],
            "B.Cu": inputs["B.Cu"],
            "F.Cu": inputs["F.Cu"],
        },
        settings,
    )
    output_c = build_geometry_xml(inputs, settings)

    assert output_a == output_b == output_c
    assert _extract_layers(output_a) == sorted(inputs.keys())

    expected_raster = {
        "dpi": "1200",
        "oversample": "2",
        "filter": "gaussian",
        "invert": "true",
    }
    expected_contour = {
        "tolerance_um": "2.5",
        "min_area_um2": "10",
        "preserve_holes": "false",
    }
    for name, value in expected_raster.items():
        assert f'<param name="{name}" value="{value}" />' in output_a
    for name, value in expected_contour.items():
        assert f'<param name="{name}" value="{value}" />' in output_a
