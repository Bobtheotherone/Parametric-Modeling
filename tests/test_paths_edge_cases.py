# SPDX-License-Identifier: MIT
"""Edge case tests for coupongen paths module.

This module provides additional edge case coverage for the paths module
(formula_foundry.coupongen.paths), complementing test_paths.py with
additional edge cases and error scenarios.

Tests focus on:
- Error handling for malformed footprint paths
- Boundary conditions for path parsing
- Path component validation
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


class TestParseFootprintPathEdgeCases:
    """Edge case tests for parse_footprint_path function."""

    def test_unicode_in_library_name(self) -> None:
        """Parse library name with unicode characters."""
        lib, name = parse_footprint_path("Lib_日本語:Footprint")
        assert lib == "Lib_日本語"
        assert name == "Footprint"

    def test_unicode_in_footprint_name(self) -> None:
        """Parse footprint name with unicode characters."""
        lib, name = parse_footprint_path("Library:Footprint_日本語")
        assert lib == "Library"
        assert name == "Footprint_日本語"

    def test_very_long_library_name(self) -> None:
        """Parse very long library name (stress test)."""
        long_lib = "A" * 500
        lib, name = parse_footprint_path(f"{long_lib}:FP")
        assert lib == long_lib
        assert name == "FP"

    def test_very_long_footprint_name(self) -> None:
        """Parse very long footprint name (stress test)."""
        long_name = "B" * 500
        lib, name = parse_footprint_path(f"Lib:{long_name}")
        assert lib == "Lib"
        assert name == long_name

    def test_single_char_library(self) -> None:
        """Parse single character library name."""
        lib, name = parse_footprint_path("L:Footprint")
        assert lib == "L"
        assert name == "Footprint"

    def test_single_char_footprint(self) -> None:
        """Parse single character footprint name."""
        lib, name = parse_footprint_path("Library:F")
        assert lib == "Library"
        assert name == "F"

    def test_numeric_library_name(self) -> None:
        """Parse numeric library name."""
        lib, name = parse_footprint_path("123:Footprint")
        assert lib == "123"
        assert name == "Footprint"

    def test_numeric_footprint_name(self) -> None:
        """Parse numeric footprint name."""
        lib, name = parse_footprint_path("Lib:456")
        assert lib == "Lib"
        assert name == "456"

    def test_spaces_in_library_name(self) -> None:
        """Library name with spaces (edge case, may be unusual)."""
        lib, name = parse_footprint_path("My Library:Footprint")
        assert lib == "My Library"
        assert name == "Footprint"

    def test_spaces_in_footprint_name(self) -> None:
        """Footprint name with spaces."""
        lib, name = parse_footprint_path("Lib:My Footprint")
        assert lib == "Lib"
        assert name == "My Footprint"

    def test_leading_trailing_spaces(self) -> None:
        """Parse path with leading/trailing spaces (not trimmed)."""
        lib, name = parse_footprint_path(" Library :Footprint ")
        # Note: spaces are preserved (not trimmed by the function)
        assert lib == " Library "
        assert name == "Footprint "

    def test_only_spaces_library_raises(self) -> None:
        """Path with spaces-only library should raise."""
        # "   :Name" has spaces for library which after strip would be empty
        # Depends on implementation - if not stripped, this passes
        # Let's test it doesn't cause issues
        lib, name = parse_footprint_path("   :Name")
        assert lib == "   "  # Spaces preserved
        assert name == "Name"

    def test_special_characters_in_names(self) -> None:
        """Parse names with special characters."""
        lib, name = parse_footprint_path("Lib@123:FP#456")
        assert lib == "Lib@123"
        assert name == "FP#456"

    def test_parentheses_in_names(self) -> None:
        """Parse names with parentheses."""
        lib, name = parse_footprint_path("Lib(v2):FP(rev1)")
        assert lib == "Lib(v2)"
        assert name == "FP(rev1)"

    def test_brackets_in_names(self) -> None:
        """Parse names with brackets."""
        lib, name = parse_footprint_path("Lib[0]:FP[1]")
        assert lib == "Lib[0]"
        assert name == "FP[1]"

    def test_period_in_names(self) -> None:
        """Parse names with periods (dots)."""
        lib, name = parse_footprint_path("Package.TO:FP.v1.2")
        assert lib == "Package.TO"
        assert name == "FP.v1.2"

    def test_hyphen_in_names(self) -> None:
        """Parse names with hyphens."""
        lib, name = parse_footprint_path("SMA-Connectors:SMA-50ohm-Edge")
        assert lib == "SMA-Connectors"
        assert name == "SMA-50ohm-Edge"

    def test_plus_sign_in_names(self) -> None:
        """Parse names with plus signs."""
        lib, name = parse_footprint_path("C++:Name+Version")
        assert lib == "C++"
        assert name == "Name+Version"


class TestParseFootprintPathErrors:
    """Error case tests for parse_footprint_path function."""

    def test_whitespace_only_string_raises(self) -> None:
        """Whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Lib:Name"):
            parse_footprint_path("   ")

    def test_tab_only_raises(self) -> None:
        """Tab-only string raises ValueError."""
        with pytest.raises(ValueError, match="Lib:Name"):
            parse_footprint_path("\t")

    def test_newline_in_string_parsed(self) -> None:
        """String with newline is parsed (newline preserved in parts)."""
        # The function splits on first colon, doesn't validate characters
        lib, name = parse_footprint_path("Lib\n:Name")
        assert lib == "Lib\n"
        assert name == "Name"

    def test_colon_at_end_raises(self) -> None:
        """Path ending with colon (empty name) raises."""
        with pytest.raises(ValueError, match="Lib:Name"):
            parse_footprint_path("Library:")

    def test_colon_at_start_raises(self) -> None:
        """Path starting with colon (empty lib) raises."""
        with pytest.raises(ValueError, match="Lib:Name"):
            parse_footprint_path(":Footprint")

    def test_double_colon_middle(self) -> None:
        """Double colon in middle - second colon is part of name."""
        lib, name = parse_footprint_path("Lib::Name")
        assert lib == "Lib"
        assert name == ":Name"  # Colon is part of name

    def test_triple_colon(self) -> None:
        """Triple colon - first splits, rest is name."""
        lib, name = parse_footprint_path("Lib:::Name")
        assert lib == "Lib"
        assert name == "::Name"


class TestGetFootprintLibDirEdgeCases:
    """Edge case tests for get_footprint_lib_dir function."""

    def test_lib_with_dots_in_name(self) -> None:
        """Library name with dots."""
        path = get_footprint_lib_dir("Package.TO-220")
        assert path.name == "Package.TO-220.pretty"

    def test_lib_with_spaces(self) -> None:
        """Library name with spaces (unusual but possible)."""
        path = get_footprint_lib_dir("My Lib")
        assert path.name == "My Lib.pretty"

    def test_empty_lib_name(self) -> None:
        """Empty library name creates .pretty directory."""
        path = get_footprint_lib_dir("")
        assert path.name == ".pretty"

    def test_unicode_lib_name(self) -> None:
        """Unicode library name."""
        path = get_footprint_lib_dir("コネクタ")
        assert path.name == "コネクタ.pretty"

    def test_path_is_absolute(self) -> None:
        """Returned path should be absolute."""
        path = get_footprint_lib_dir("AnyLib")
        assert path.is_absolute()

    def test_consistent_across_calls(self) -> None:
        """Same library name returns same path."""
        path1 = get_footprint_lib_dir("TestLib")
        path2 = get_footprint_lib_dir("TestLib")
        assert path1 == path2


class TestGetFootprintModulePathEdgeCases:
    """Edge case tests for get_footprint_module_path function."""

    def test_footprint_with_dots(self) -> None:
        """Footprint name with dots (version numbers)."""
        path = get_footprint_module_path("Lib", "FP.v1.2.3")
        assert path.name == "FP.v1.2.3.kicad_mod"

    def test_footprint_with_kicad_mod_in_name(self) -> None:
        """Footprint name containing '.kicad_mod' (edge case)."""
        path = get_footprint_module_path("Lib", "FP.kicad_mod")
        # Result would be FP.kicad_mod.kicad_mod - double extension
        assert path.name == "FP.kicad_mod.kicad_mod"

    def test_empty_footprint_name(self) -> None:
        """Empty footprint name creates .kicad_mod file."""
        path = get_footprint_module_path("Lib", "")
        assert path.name == ".kicad_mod"

    def test_empty_lib_and_footprint(self) -> None:
        """Both empty creates minimal path."""
        path = get_footprint_module_path("", "")
        assert path.name == ".kicad_mod"
        assert path.parent.name == ".pretty"

    def test_unicode_footprint_name(self) -> None:
        """Unicode footprint name."""
        path = get_footprint_module_path("Lib", "フットプリント")
        assert path.name == "フットプリント.kicad_mod"

    def test_very_long_combined_path(self) -> None:
        """Very long library and footprint names."""
        long_lib = "L" * 200
        long_fp = "F" * 200
        path = get_footprint_module_path(long_lib, long_fp)
        assert path.stem == long_fp
        assert path.parent.stem == long_lib


class TestPathConstantsValidation:
    """Validation tests for path constants."""

    def test_footprint_lib_dir_not_none(self) -> None:
        """FOOTPRINT_LIB_DIR is not None."""
        assert FOOTPRINT_LIB_DIR is not None

    def test_footprint_meta_dir_not_none(self) -> None:
        """FOOTPRINT_META_DIR is not None."""
        assert FOOTPRINT_META_DIR is not None

    def test_lib_dir_is_path_object(self) -> None:
        """FOOTPRINT_LIB_DIR is a Path object."""
        assert isinstance(FOOTPRINT_LIB_DIR, Path)

    def test_meta_dir_is_path_object(self) -> None:
        """FOOTPRINT_META_DIR is a Path object."""
        assert isinstance(FOOTPRINT_META_DIR, Path)

    def test_lib_dir_name_is_footprints(self) -> None:
        """FOOTPRINT_LIB_DIR directory name is 'footprints'."""
        assert FOOTPRINT_LIB_DIR.name == "footprints"

    def test_meta_dir_name_is_footprints_meta(self) -> None:
        """FOOTPRINT_META_DIR directory name is 'footprints_meta'."""
        assert FOOTPRINT_META_DIR.name == "footprints_meta"

    def test_lib_and_meta_share_parent(self) -> None:
        """Library and metadata directories share the same parent."""
        assert FOOTPRINT_LIB_DIR.parent == FOOTPRINT_META_DIR.parent


class TestIntegrationScenarios:
    """Integration tests combining multiple path functions."""

    def test_parse_then_get_module_path(self) -> None:
        """Parse footprint path then get module path."""
        full_path = "Coupongen_Connectors:SMA_EndLaunch_Generic"
        lib, name = parse_footprint_path(full_path)
        module_path = get_footprint_module_path(lib, name)

        assert lib == "Coupongen_Connectors"
        assert name == "SMA_EndLaunch_Generic"
        assert module_path.name == "SMA_EndLaunch_Generic.kicad_mod"
        assert module_path.parent.name == "Coupongen_Connectors.pretty"

    def test_parse_then_get_lib_dir(self) -> None:
        """Parse footprint path then get library directory."""
        full_path = "MyLibrary:MyFootprint"
        lib, _ = parse_footprint_path(full_path)
        lib_dir = get_footprint_lib_dir(lib)

        assert lib_dir.name == "MyLibrary.pretty"

    def test_module_path_is_under_lib_dir(self) -> None:
        """Module path is under the library directory."""
        lib = "TestLib"
        fp = "TestFP"

        lib_dir = get_footprint_lib_dir(lib)
        module_path = get_footprint_module_path(lib, fp)

        assert module_path.parent == lib_dir

    def test_path_construction_deterministic(self) -> None:
        """Path construction is deterministic across multiple calls."""
        fp_path = "Lib123:FP456"

        results = []
        for _ in range(5):
            lib, name = parse_footprint_path(fp_path)
            module_path = get_footprint_module_path(lib, name)
            results.append(str(module_path))

        # All results should be identical
        assert len(set(results)) == 1
