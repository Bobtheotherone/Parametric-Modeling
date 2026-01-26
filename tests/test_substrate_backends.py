# SPDX-License-Identifier: MIT
"""Unit tests for formula_foundry/substrate/backends.py.

Tests the numeric backend selection and host transfer guard functionality:
- Backend selection (NumPy/CuPy) based on GPU availability
- GPU requirement enforcement
- Host transfer guard for detecting unwanted GPU->CPU transfers
- Exception classes for backend selection errors
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from formula_foundry.substrate.backends import (
    Backend,
    BackendName,
    BackendSelectionError,
    GPUNotAvailableError,
    HostTransferError,
    HostTransferGuard,
    host_transfer_guard,
    select_backend,
)

# =============================================================================
# Exception Class Tests
# =============================================================================


class TestBackendSelectionError:
    """Tests for BackendSelectionError exception."""

    def test_is_runtime_error(self) -> None:
        """BackendSelectionError should inherit from RuntimeError."""
        error = BackendSelectionError("Test error")
        assert isinstance(error, RuntimeError)
        assert str(error) == "Test error"

    def test_can_be_raised_and_caught(self) -> None:
        """BackendSelectionError can be raised and caught."""
        with pytest.raises(BackendSelectionError, match="test message"):
            raise BackendSelectionError("test message")


class TestGPUNotAvailableError:
    """Tests for GPUNotAvailableError exception."""

    def test_is_backend_selection_error(self) -> None:
        """GPUNotAvailableError should inherit from BackendSelectionError."""
        error = GPUNotAvailableError("GPU unavailable")
        assert isinstance(error, BackendSelectionError)
        assert isinstance(error, RuntimeError)

    def test_can_be_caught_as_base_class(self) -> None:
        """GPUNotAvailableError can be caught as BackendSelectionError."""
        with pytest.raises(BackendSelectionError):
            raise GPUNotAvailableError("GPU required")


class TestHostTransferError:
    """Tests for HostTransferError exception."""

    def test_is_runtime_error(self) -> None:
        """HostTransferError should inherit from RuntimeError."""
        error = HostTransferError("Transfer detected")
        assert isinstance(error, RuntimeError)
        assert str(error) == "Transfer detected"


# =============================================================================
# Backend Dataclass Tests
# =============================================================================


class TestBackend:
    """Tests for Backend dataclass."""

    def test_creation(self) -> None:
        """Backend stores all attributes correctly."""
        mock_module = MagicMock()
        backend = Backend(
            name="numpy",
            module=mock_module,
            gpu_available=False,
            require_gpu=False,
        )
        assert backend.name == "numpy"
        assert backend.module is mock_module
        assert backend.gpu_available is False
        assert backend.require_gpu is False

    def test_frozen(self) -> None:
        """Backend should be immutable (frozen)."""
        mock_module = MagicMock()
        backend = Backend(
            name="numpy",
            module=mock_module,
            gpu_available=False,
            require_gpu=False,
        )
        with pytest.raises(FrozenInstanceError):
            backend.name = "cupy"  # type: ignore[misc]

    def test_name_literal_types(self) -> None:
        """Backend name should accept valid BackendName literals."""
        mock_module = MagicMock()
        numpy_backend = Backend(
            name="numpy",
            module=mock_module,
            gpu_available=False,
            require_gpu=False,
        )
        cupy_backend = Backend(
            name="cupy",
            module=mock_module,
            gpu_available=True,
            require_gpu=True,
        )
        assert numpy_backend.name == "numpy"
        assert cupy_backend.name == "cupy"


# =============================================================================
# Backend Selection Tests
# =============================================================================


class TestSelectBackend:
    """Tests for select_backend function."""

    def test_selects_numpy_when_no_gpu(self) -> None:
        """Selects NumPy backend when GPU is not available."""
        with patch("formula_foundry.substrate.backends._import_optional") as mock_import:
            import numpy

            mock_import.side_effect = lambda name: numpy if name == "numpy" else None

            backend = select_backend(require_gpu=False, prefer_gpu=False)

            assert backend.name == "numpy"
            assert backend.module is numpy
            assert backend.gpu_available is False
            assert backend.require_gpu is False

    def test_raises_when_gpu_required_but_unavailable(self) -> None:
        """Raises GPUNotAvailableError when GPU required but unavailable."""
        with patch("formula_foundry.substrate.backends._import_optional") as mock_import:
            import numpy

            mock_import.side_effect = lambda name: numpy if name == "numpy" else None

            with pytest.raises(GPUNotAvailableError, match="GPU required"):
                select_backend(require_gpu=True, prefer_gpu=True)

    def test_raises_when_require_gpu_but_prefer_gpu_false(self) -> None:
        """Raises GPUNotAvailableError when require_gpu=True but prefer_gpu=False."""
        with pytest.raises(GPUNotAvailableError, match="CPU-only backend requested"):
            select_backend(require_gpu=True, prefer_gpu=False)

    def test_raises_when_numpy_unavailable(self) -> None:
        """Raises BackendSelectionError when NumPy is unavailable."""
        with patch("formula_foundry.substrate.backends._import_optional") as mock_import:
            mock_import.return_value = None

            with pytest.raises(BackendSelectionError, match="NumPy backend unavailable"):
                select_backend(require_gpu=False, prefer_gpu=False)


# =============================================================================
# Host Transfer Guard Tests
# =============================================================================


class TestHostTransferGuard:
    """Tests for HostTransferGuard class."""

    def test_default_values(self) -> None:
        """HostTransferGuard has expected defaults."""
        guard = HostTransferGuard()
        assert guard.fail_on_transfer is True
        assert guard.transfers == []

    def test_custom_fail_on_transfer(self) -> None:
        """HostTransferGuard accepts custom fail_on_transfer."""
        guard = HostTransferGuard(fail_on_transfer=False)
        assert guard.fail_on_transfer is False

    def test_record_adds_transfer(self) -> None:
        """record() adds transfer to list."""
        guard = HostTransferGuard(fail_on_transfer=False)
        guard.record("test transfer")
        assert "test transfer" in guard.transfers

    def test_record_raises_when_fail_on_transfer_true(self) -> None:
        """record() raises HostTransferError when fail_on_transfer=True."""
        guard = HostTransferGuard(fail_on_transfer=True)
        with pytest.raises(HostTransferError, match="test transfer"):
            guard.record("test transfer")

    def test_record_no_raise_when_fail_on_transfer_false(self) -> None:
        """record() does not raise when fail_on_transfer=False."""
        guard = HostTransferGuard(fail_on_transfer=False)
        guard.record("transfer 1")
        guard.record("transfer 2")
        assert len(guard.transfers) == 2

    def test_install_and_uninstall(self) -> None:
        """install() and uninstall() work correctly."""
        guard = HostTransferGuard(fail_on_transfer=False)
        guard.install()
        guard.uninstall()
        # Should not raise, just verify the cycle completes
        assert guard._patches == []

    def test_multiple_installs(self) -> None:
        """Multiple installs don't cause issues."""
        guard = HostTransferGuard(fail_on_transfer=False)
        guard.install()
        guard.install()  # Second install
        guard.uninstall()
        guard.uninstall()  # Extra uninstall
        assert guard._patches == []


class TestHostTransferGuardContextManager:
    """Tests for host_transfer_guard context manager."""

    def test_context_manager_basic(self) -> None:
        """Context manager installs and uninstalls guard."""
        with host_transfer_guard(fail_on_transfer=False) as guard:
            assert isinstance(guard, HostTransferGuard)
            assert guard.fail_on_transfer is False

    def test_context_manager_uninstalls_on_exit(self) -> None:
        """Context manager uninstalls guard on exit."""
        guard_ref: HostTransferGuard | None = None
        with host_transfer_guard(fail_on_transfer=False) as guard:
            guard_ref = guard
        # After context exit, patches should be cleared
        assert guard_ref is not None
        assert guard_ref._patches == []

    def test_context_manager_uninstalls_on_exception(self) -> None:
        """Context manager uninstalls guard even on exception."""
        guard_ref: HostTransferGuard | None = None
        try:
            with host_transfer_guard(fail_on_transfer=False) as guard:
                guard_ref = guard
                raise ValueError("Test exception")
        except ValueError:
            pass
        assert guard_ref is not None
        assert guard_ref._patches == []

    def test_context_manager_default_fail_on_transfer(self) -> None:
        """Context manager defaults to fail_on_transfer=True."""
        with host_transfer_guard() as guard:
            assert guard.fail_on_transfer is True


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestBackendEdgeCases:
    """Edge case tests for backend functionality."""

    def test_backend_equality(self) -> None:
        """Two backends with same attributes are equal."""
        mock_module = MagicMock()
        backend1 = Backend(
            name="numpy",
            module=mock_module,
            gpu_available=False,
            require_gpu=False,
        )
        backend2 = Backend(
            name="numpy",
            module=mock_module,
            gpu_available=False,
            require_gpu=False,
        )
        assert backend1 == backend2

    def test_backend_inequality_different_name(self) -> None:
        """Backends with different names are not equal."""
        mock_module = MagicMock()
        backend1 = Backend(
            name="numpy",
            module=mock_module,
            gpu_available=False,
            require_gpu=False,
        )
        backend2 = Backend(
            name="cupy",
            module=mock_module,
            gpu_available=True,
            require_gpu=False,
        )
        assert backend1 != backend2

    def test_host_transfer_guard_empty_uninstall(self) -> None:
        """uninstall() on empty guard doesn't raise."""
        guard = HostTransferGuard()
        guard.uninstall()  # Should not raise
        assert guard._patches == []

    def test_error_messages_are_informative(self) -> None:
        """Error messages should be informative."""
        error1 = BackendSelectionError("Custom message about backend")
        error2 = GPUNotAvailableError("CUDA device count is zero")
        error3 = HostTransferError("cupy.asnumpy called unexpectedly")

        assert "Custom message" in str(error1)
        assert "CUDA" in str(error2)
        assert "cupy.asnumpy" in str(error3)
