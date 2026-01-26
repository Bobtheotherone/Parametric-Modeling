# SPDX-License-Identifier: MIT
"""Unit tests for M3 locking module.

Tests the file-based locking mechanisms for safe concurrent access to
the artifact store. Key functionality:
- FileLock class for cross-process file locking
- StoreLock class for store-wide and per-artifact locks
- Lock acquisition/release and timeout behavior
- Thread safety within process
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from formula_foundry.m3.locking import (
    FileLock,
    LockAcquisitionError,
    LockError,
    LockInfo,
    LockNotHeldError,
    StoreLock,
)


class TestFileLock:
    """Tests for FileLock class."""

    def test_basic_exclusive_lock(self, tmp_path: Path) -> None:
        """Basic exclusive lock acquisition and release."""
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path)

        assert not lock.is_locked
        lock.acquire_exclusive(blocking=False)
        assert lock.is_locked
        assert lock_path.exists()

        lock.release()
        assert not lock.is_locked

    def test_basic_shared_lock(self, tmp_path: Path) -> None:
        """Basic shared lock acquisition and release."""
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path)

        assert not lock.is_locked
        lock.acquire_shared(blocking=False)
        assert lock.is_locked

        lock.release()
        assert not lock.is_locked

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Lock creates parent directories if needed."""
        lock_path = tmp_path / "deep" / "nested" / "dir" / "test.lock"
        lock = FileLock(lock_path)

        lock.acquire_exclusive(blocking=False)
        assert lock_path.exists()
        assert lock_path.parent.exists()
        lock.release()

    def test_release_without_lock_raises(self, tmp_path: Path) -> None:
        """Releasing without holding lock raises LockNotHeldError."""
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path)

        with pytest.raises(LockNotHeldError):
            lock.release()

    def test_lock_info_when_held(self, tmp_path: Path) -> None:
        """lock_info returns LockInfo when lock is held."""
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path)

        lock.acquire_exclusive(blocking=False)
        info = lock.lock_info

        assert info is not None
        assert isinstance(info, LockInfo)
        assert info.path == lock_path
        assert info.lock_type == "exclusive"
        assert info.process_id > 0
        assert info.thread_id > 0

        lock.release()

    def test_lock_info_when_not_held(self, tmp_path: Path) -> None:
        """lock_info returns None when lock is not held."""
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path)

        assert lock.lock_info is None

    def test_context_manager_exclusive(self, tmp_path: Path) -> None:
        """Context manager with exclusive lock."""
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path)

        with lock.exclusive() as info:
            assert lock.is_locked
            assert info.lock_type == "exclusive"

        assert not lock.is_locked

    def test_context_manager_shared(self, tmp_path: Path) -> None:
        """Context manager with shared lock."""
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path)

        with lock.shared() as info:
            assert lock.is_locked
            assert info.lock_type == "shared"

        assert not lock.is_locked

    def test_default_context_manager_is_exclusive(self, tmp_path: Path) -> None:
        """Default __enter__/__exit__ acquires exclusive lock."""
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path)

        with lock as info:
            assert lock.is_locked
            assert info.lock_type == "exclusive"

        assert not lock.is_locked

    def test_context_manager_releases_on_exception(self, tmp_path: Path) -> None:
        """Context manager releases lock even on exception."""
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path)

        with pytest.raises(RuntimeError), lock.exclusive():
            assert lock.is_locked
            raise RuntimeError("Test error")

        assert not lock.is_locked

    def test_non_blocking_returns_false_when_locked(self, tmp_path: Path) -> None:
        """Non-blocking acquire returns False when lock is held."""
        lock_path = tmp_path / "test.lock"
        lock1 = FileLock(lock_path)
        lock2 = FileLock(lock_path)

        lock1.acquire_exclusive(blocking=False)

        # lock2 cannot acquire exclusive when lock1 holds it
        result = lock2.acquire_exclusive(blocking=False)
        assert result is False
        assert not lock2.is_locked

        lock1.release()

    def test_multiple_shared_locks(self, tmp_path: Path) -> None:
        """Multiple shared locks can be held simultaneously."""
        lock_path = tmp_path / "test.lock"
        lock1 = FileLock(lock_path)
        lock2 = FileLock(lock_path)

        lock1.acquire_shared(blocking=False)
        assert lock1.is_locked

        # lock2 can also acquire shared
        result = lock2.acquire_shared(blocking=False)
        assert result is True
        assert lock2.is_locked

        lock1.release()
        lock2.release()


class TestFileLockTimeout:
    """Tests for FileLock timeout behavior."""

    def test_timeout_raises_on_contention(self, tmp_path: Path) -> None:
        """Timeout raises LockAcquisitionError when lock is held."""
        lock_path = tmp_path / "test.lock"
        lock1 = FileLock(lock_path)
        lock2 = FileLock(lock_path)

        lock1.acquire_exclusive(blocking=False)

        # Very short timeout should fail
        with pytest.raises(LockAcquisitionError) as excinfo:
            lock2.acquire_exclusive(blocking=True, timeout=0.05)

        assert "Timeout" in str(excinfo.value)
        assert not lock2.is_locked

        lock1.release()

    def test_timeout_succeeds_when_lock_released(self, tmp_path: Path) -> None:
        """Lock acquisition succeeds if lock is released before timeout."""
        lock_path = tmp_path / "test.lock"
        lock1 = FileLock(lock_path)
        lock2 = FileLock(lock_path)

        def release_after_delay() -> None:
            time.sleep(0.1)
            lock1.release()

        lock1.acquire_exclusive(blocking=False)

        # Start thread to release lock1
        thread = threading.Thread(target=release_after_delay)
        thread.start()

        # lock2 should acquire before timeout
        result = lock2.acquire_exclusive(blocking=True, timeout=1.0)
        thread.join()

        assert result is True
        assert lock2.is_locked

        lock2.release()


class TestStoreLock:
    """Tests for StoreLock class."""

    def test_global_lock(self, tmp_path: Path) -> None:
        """Global lock acquisition and release."""
        store_lock = StoreLock(tmp_path)

        with store_lock.global_lock() as info:
            assert info is not None
            assert info.lock_type == "exclusive"

    def test_global_lock_shared(self, tmp_path: Path) -> None:
        """Global lock can be acquired as shared."""
        store_lock = StoreLock(tmp_path)

        with store_lock.global_lock(shared=True) as info:
            assert info is not None
            assert info.lock_type == "shared"

    def test_spec_id_lock(self, tmp_path: Path) -> None:
        """Per-artifact spec_id lock."""
        store_lock = StoreLock(tmp_path)

        with store_lock.spec_id_lock("abc123") as info:
            assert info is not None
            assert info.lock_type == "exclusive"

    def test_multiple_different_spec_id_locks(self, tmp_path: Path) -> None:
        """Different spec_id locks can be held simultaneously."""
        store_lock = StoreLock(tmp_path)

        with store_lock.spec_id_lock("spec1") as info1:
            assert info1 is not None

            with store_lock.spec_id_lock("spec2") as info2:
                assert info2 is not None

    def test_try_spec_id_lock_success(self, tmp_path: Path) -> None:
        """try_spec_id_lock returns lock when available."""
        store_lock = StoreLock(tmp_path)

        lock = store_lock.try_spec_id_lock("abc123")
        assert lock is not None
        assert lock.is_locked

        lock.release()

    def test_try_spec_id_lock_returns_none_when_held(self, tmp_path: Path) -> None:
        """try_spec_id_lock returns None when lock is held."""
        store_lock = StoreLock(tmp_path)

        # First, acquire the lock
        with store_lock.spec_id_lock("abc123"):
            # Now try to acquire non-blocking
            lock = store_lock.try_spec_id_lock("abc123")
            assert lock is None

    def test_locks_dir_structure(self, tmp_path: Path) -> None:
        """Verify lock directory structure is created."""
        store_lock = StoreLock(tmp_path)

        with store_lock.global_lock():
            pass

        locks_dir = tmp_path / ".locks"
        assert locks_dir.exists()

        global_lock = locks_dir / "artifact_store.lock"
        assert global_lock.exists()

    def test_spec_id_lock_dir_structure(self, tmp_path: Path) -> None:
        """Verify spec_id lock directory structure is created."""
        store_lock = StoreLock(tmp_path)

        with store_lock.spec_id_lock("test_spec"):
            pass

        spec_locks_dir = tmp_path / ".locks" / "spec_id"
        assert spec_locks_dir.exists()

        spec_lock = spec_locks_dir / "test_spec.lock"
        assert spec_lock.exists()

    def test_cleanup_stale_locks(self, tmp_path: Path) -> None:
        """cleanup_stale_locks removes old lock files."""
        store_lock = StoreLock(tmp_path)

        # Create the lock directory structure
        spec_locks_dir = tmp_path / ".locks" / "spec_id"
        spec_locks_dir.mkdir(parents=True, exist_ok=True)

        # Create some "old" lock files by setting mtime
        old_lock = spec_locks_dir / "old_spec.lock"
        old_lock.touch()

        # Set mtime to 2 hours ago
        import os

        old_time = time.time() - 7200
        os.utime(old_lock, (old_time, old_time))

        # Create a "new" lock file
        new_lock = spec_locks_dir / "new_spec.lock"
        new_lock.touch()

        # Cleanup with 1 hour max age
        removed = store_lock.cleanup_stale_locks(max_age_seconds=3600.0)

        assert old_lock in removed
        assert not old_lock.exists()
        assert new_lock.exists()


class TestLockErrorTypes:
    """Tests for lock error types."""

    def test_lock_error_is_exception(self) -> None:
        """LockError is an Exception."""
        error = LockError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_lock_acquisition_error_is_lock_error(self) -> None:
        """LockAcquisitionError is a LockError."""
        error = LockAcquisitionError("Timeout")
        assert isinstance(error, LockError)
        assert isinstance(error, Exception)

    def test_lock_not_held_error_is_lock_error(self) -> None:
        """LockNotHeldError is a LockError."""
        error = LockNotHeldError("Not held")
        assert isinstance(error, LockError)
        assert isinstance(error, Exception)


class TestLockInfo:
    """Tests for LockInfo dataclass."""

    def test_lock_info_creation(self, tmp_path: Path) -> None:
        """LockInfo can be created with all fields."""
        info = LockInfo(
            path=tmp_path / "test.lock",
            lock_type="exclusive",
            acquired_at=time.monotonic(),
            process_id=12345,
            thread_id=67890,
        )

        assert info.path == tmp_path / "test.lock"
        assert info.lock_type == "exclusive"
        assert info.process_id == 12345
        assert info.thread_id == 67890


class TestThreadSafety:
    """Tests for thread safety of locks."""

    def test_multiple_threads_serialize_on_exclusive(self, tmp_path: Path) -> None:
        """Multiple threads serialize properly on exclusive lock."""
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path)
        counter = [0]
        errors = []

        def increment_with_lock() -> None:
            try:
                with lock.exclusive(timeout=5.0):
                    # Read, sleep, increment (race condition without lock)
                    val = counter[0]
                    time.sleep(0.01)
                    counter[0] = val + 1
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=increment_with_lock) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert counter[0] == 5  # All increments should be serialized
