"""Unit tests for coupongen paths module.

Tests the footprint path parsing and path resolution functions
used for locating .kicad_mod footprint files in the repository.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Direct import to avoid broken import chain in formula_foundry.__init__
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
_paths_spec = importlib.util.spec_from_file_location("paths", _SRC_DIR / "formula_foundry" / "coupongen" / "paths.py")
_paths = importlib.util.module_from_spec(_paths_spec)  # type: ignore[arg-type]
_paths_spec.loader.exec_module(_paths)  # type: ignore[union-attr]

FOOTPRINT_LIB_DIR = _paths.FOOTPRINT_LIB_DIR
FOOTPRINT_META_DIR = _paths.FOOTPRINT_META_DIR
get_footprint_lib_dir = _paths.get_footprint_lib_dir
get_footprint_module_path = _paths.get_footprint_module_path
parse_footprint_path = _paths.parse_footprint_path


class TestParseFootprintPath:
    """Tests for parse_footprint_path function."""

    def test_basic_lib_colon_name_format(self) -> None:
        """Parse 'Lib:Name' format correctly."""
        lib, name = parse_footprint_path("Coupongen_Connectors:SMA_EndLaunch_Generic")
        assert lib == "Coupongen_Connectors"
        assert name == "SMA_EndLaunch_Generic"

    def test_simple_lib_name(self) -> None:
        """Parse simple library and name."""
        lib, name = parse_footprint_path("MyLib:MyFootprint")
        assert lib == "MyLib"
        assert name == "MyFootprint"

    def test_name_with_underscores(self) -> None:
        """Parse name containing underscores."""
        lib, name = parse_footprint_path("Lib:Footprint_With_Many_Underscores")
        assert lib == "Lib"
        assert name == "Footprint_With_Many_Underscores"

    def test_lib_with_underscores(self) -> None:
        """Parse library name containing underscores."""
        lib, name = parse_footprint_path("My_Custom_Lib:FP")
        assert lib == "My_Custom_Lib"
        assert name == "FP"

    def test_name_with_numbers(self) -> None:
        """Parse name containing numbers."""
        lib, name = parse_footprint_path("Connectors:SMA_50ohm_V1")
        assert lib == "Connectors"
        assert name == "SMA_50ohm_V1"

    def test_name_with_multiple_colons(self) -> None:
        """Second colon and beyond is part of the name (edge case)."""
        lib, name = parse_footprint_path("Lib:Name:With:Colons")
        assert lib == "Lib"
        assert name == "Name:With:Colons"

    def test_missing_colon_raises_valueerror(self) -> None:
        """Path without colon separator raises ValueError."""
        with pytest.raises(ValueError, match="Lib:Name"):
            parse_footprint_path("NoColonHere")

    def test_empty_library_raises_valueerror(self) -> None:
        """Empty library name raises ValueError."""
        with pytest.raises(ValueError, match="Lib:Name"):
            parse_footprint_path(":NameOnly")

    def test_empty_name_raises_valueerror(self) -> None:
        """Empty footprint name raises ValueError."""
        with pytest.raises(ValueError, match="Lib:Name"):
            parse_footprint_path("LibOnly:")

    def test_only_colon_raises_valueerror(self) -> None:
        """Just a colon raises ValueError."""
        with pytest.raises(ValueError, match="Lib:Name"):
            parse_footprint_path(":")

    def test_empty_string_raises_valueerror(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Lib:Name"):
            parse_footprint_path("")


class TestGetFootprintLibDir:
    """Tests for get_footprint_lib_dir function."""

    def test_returns_path_with_pretty_suffix(self) -> None:
        """Library directory has .pretty suffix."""
        path = get_footprint_lib_dir("TestLib")
        assert path.name == "TestLib.pretty"

    def test_parent_is_footprint_lib_dir(self) -> None:
        """Library directory is under FOOTPRINT_LIB_DIR."""
        path = get_footprint_lib_dir("MyLib")
        assert path.parent == FOOTPRINT_LIB_DIR

    def test_different_libs_give_different_paths(self) -> None:
        """Different library names give different paths."""
        path_a = get_footprint_lib_dir("LibA")
        path_b = get_footprint_lib_dir("LibB")
        assert path_a != path_b

    def test_coupongen_connectors_lib(self) -> None:
        """Standard Coupongen_Connectors library path."""
        path = get_footprint_lib_dir("Coupongen_Connectors")
        assert path.name == "Coupongen_Connectors.pretty"


class TestGetFootprintModulePath:
    """Tests for get_footprint_module_path function."""

    def test_returns_kicad_mod_file(self) -> None:
        """Returns path to .kicad_mod file."""
        path = get_footprint_module_path("TestLib", "TestFP")
        assert path.suffix == ".kicad_mod"
        assert path.stem == "TestFP"

    def test_under_correct_library_dir(self) -> None:
        """Module file is under correct library .pretty directory."""
        path = get_footprint_module_path("MyLib", "MyFootprint")
        assert path.parent.name == "MyLib.pretty"

    def test_full_path_structure(self) -> None:
        """Full path structure is correct."""
        path = get_footprint_module_path("Coupongen_Connectors", "SMA_EndLaunch_Generic")
        assert path.name == "SMA_EndLaunch_Generic.kicad_mod"
        assert path.parent.name == "Coupongen_Connectors.pretty"
        assert path.parent.parent == FOOTPRINT_LIB_DIR

    def test_combined_with_parse_footprint_path(self) -> None:
        """Integration: parse path then get module path."""
        lib, name = parse_footprint_path("Coupongen_Connectors:SMA_EndLaunch_Generic")
        path = get_footprint_module_path(lib, name)
        assert path.name == "SMA_EndLaunch_Generic.kicad_mod"


class TestPathConstants:
    """Tests for path constants."""

    def test_footprint_lib_dir_is_absolute(self) -> None:
        """FOOTPRINT_LIB_DIR is an absolute path."""
        assert FOOTPRINT_LIB_DIR.is_absolute()

    def test_footprint_meta_dir_is_absolute(self) -> None:
        """FOOTPRINT_META_DIR is an absolute path."""
        assert FOOTPRINT_META_DIR.is_absolute()

    def test_footprint_dirs_are_under_coupongen(self) -> None:
        """Both footprint directories are under coupongen/libs."""
        assert "coupongen" in str(FOOTPRINT_LIB_DIR)
        assert "coupongen" in str(FOOTPRINT_META_DIR)

    def test_lib_and_meta_dirs_different(self) -> None:
        """Library and metadata directories are different."""
        assert FOOTPRINT_LIB_DIR != FOOTPRINT_META_DIR
        assert FOOTPRINT_LIB_DIR.name == "footprints"
        assert FOOTPRINT_META_DIR.name == "footprints_meta"
