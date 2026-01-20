# SPDX-License-Identifier: MIT
"""Integration tests for KiCad toolchain and Docker runner.

This package contains tests that require:
- Docker with KiCad image (kicad/kicad:<version>)
- Real KiCad CLI execution for DRC and export operations

Mark tests with @pytest.mark.kicad_integration to identify them.
These tests are separated from unit tests for faster CI feedback.
"""
