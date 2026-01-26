"""Tests for M3 concurrency control and locking.

This module tests the file-based locking mechanisms in the M3 artifact store:
- FileLock for cross-process file locking
- StoreLock for high-level store locking interface
- Concurrent artifact creation safety
- Lock timeout and error handling
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from formula_foundry.m3.artifact_store import (
    ArtifactExistsError,
    ArtifactStore,
    LockTimeoutError,
)
from formula_foundry.m3.locking import (
    FileLock,
    LockAcquisitionError,
    LockInfo,
    LockNotHeldError,
    StoreLock,
)


class TestFileLock:
    """Tests for FileLock class."""

    def test_exclusive_lock_basic(self, tmp_path: Path) -> None:
        """Test basic exclusive lock acquisition and release."""
        lock_file = tmp_path / "test.lock"
        lock = FileLock(lock_file)

        assert not lock.is_locked

        acquired = lock.acquire_exclusive()
        assert acquired
        assert lock.is_locked
        assert lock.lock_info is not None
        assert lock.lock_info.lock_type == "exclusive"
        assert lock.lock_info.process_id == os.getpid()

        lock.release()
        assert not lock.is_locked

    def test_shared_lock_basic(self, tmp_path: Path) -> None:
        """Test basic shared lock acquisition and release."""
        lock_file = tmp_path / "test.lock"
        lock = FileLock(lock_file)

        acquired = lock.acquire_shared()
        assert acquired
        assert lock.is_locked
        assert lock.lock_info is not None
        assert lock.lock_info.lock_type == "shared"

        lock.release()
        assert not lock.is_locked

    def test_context_manager_exclusive(self, tmp_path: Path) -> None:
        """Test context manager for exclusive lock."""
        lock_file = tmp_path / "test.lock"
        lock = FileLock(lock_file)

        with lock.exclusive() as info:
            assert lock.is_locked
            assert info.lock_type == "exclusive"

        assert not lock.is_locked

    def test_context_manager_shared(self, tmp_path: Path) -> None:
        """Test context manager for shared lock."""
        lock_file = tmp_path / "test.lock"
        lock = FileLock(lock_file)

        with lock.shared() as info:
            assert lock.is_locked
            assert info.lock_type == "shared"

        assert not lock.is_locked

    def test_release_without_lock_raises(self, tmp_path: Path) -> None:
        """Test that releasing an unheld lock raises an error."""
        lock_file = tmp_path / "test.lock"
        lock = FileLock(lock_file)

        with pytest.raises(LockNotHeldError):
            lock.release()

    def test_creates_lock_file(self, tmp_path: Path) -> None:
        """Test that acquiring a lock creates the lock file."""
        lock_file = tmp_path / "subdir" / "test.lock"
        assert not lock_file.exists()

        lock = FileLock(lock_file)
        with lock.exclusive():
            assert lock_file.exists()

    def test_non_blocking_lock(self, tmp_path: Path) -> None:
        """Test non-blocking lock acquisition."""
        lock_file = tmp_path / "test.lock"
        lock1 = FileLock(lock_file)
        lock2 = FileLock(lock_file)

        # First lock succeeds
        assert lock1.acquire_exclusive(blocking=False)

        # Second lock fails (non-blocking)
        assert not lock2.acquire_exclusive(blocking=False)

        lock1.release()

        # Now second lock succeeds
        assert lock2.acquire_exclusive(blocking=False)
        lock2.release()

    def test_lock_timeout(self, tmp_path: Path) -> None:
        """Test lock acquisition with timeout."""
        lock_file = tmp_path / "test.lock"
        lock1 = FileLock(lock_file)
        lock2 = FileLock(lock_file)

        lock1.acquire_exclusive()

        start = time.monotonic()
        with pytest.raises(LockAcquisitionError):
            lock2.acquire_exclusive(timeout=0.1)
        elapsed = time.monotonic() - start

        # Should have waited approximately the timeout
        assert elapsed >= 0.1
        assert elapsed < 0.5  # But not much longer

        lock1.release()

    def test_concurrent_threads_exclusive(self, tmp_path: Path) -> None:
        """Test that exclusive locks serialize thread access."""
        lock_file = tmp_path / "test.lock"
        counter = {"value": 0}
        errors: list[Exception] = []

        def increment() -> None:
            lock = FileLock(lock_file)
            try:
                with lock.exclusive():
                    # Read-modify-write without lock would race
                    current = counter["value"]
                    time.sleep(0.001)  # Simulate work
                    counter["value"] = current + 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert counter["value"] == 10

    def test_multiple_shared_locks(self, tmp_path: Path) -> None:
        """Test that multiple shared locks can be held simultaneously."""
        lock_file = tmp_path / "test.lock"
        acquired_count = {"value": 0}
        max_concurrent = {"value": 0}
        lock_obj = threading.Lock()
        errors: list[Exception] = []

        def reader() -> None:
            file_lock = FileLock(lock_file)
            try:
                with file_lock.shared():
                    with lock_obj:
                        acquired_count["value"] += 1
                        if acquired_count["value"] > max_concurrent["value"]:
                            max_concurrent["value"] = acquired_count["value"]
                    time.sleep(0.05)  # Hold lock briefly
                    with lock_obj:
                        acquired_count["value"] -= 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Multiple readers should have been able to hold the lock concurrently
        assert max_concurrent["value"] > 1


class TestStoreLock:
    """Tests for StoreLock class."""

    def test_global_lock(self, tmp_path: Path) -> None:
        """Test global store lock acquisition."""
        store_lock = StoreLock(tmp_path / "data")

        with store_lock.global_lock() as info:
            assert info.lock_type == "exclusive"
            # Lock file should exist
            lock_file = tmp_path / "data" / ".locks" / "artifact_store.lock"
            assert lock_file.exists()

    def test_global_lock_shared(self, tmp_path: Path) -> None:
        """Test global store lock in shared mode."""
        store_lock = StoreLock(tmp_path / "data")

        with store_lock.global_lock(shared=True) as info:
            assert info.lock_type == "shared"

    def test_spec_id_lock(self, tmp_path: Path) -> None:
        """Test per-artifact spec_id lock."""
        store_lock = StoreLock(tmp_path / "data")
        spec_id = "abc123defgh"

        with store_lock.spec_id_lock(spec_id) as info:
            assert info.lock_type == "exclusive"
            # Lock file should exist
            lock_file = tmp_path / "data" / ".locks" / "spec_id" / f"{spec_id}.lock"
            assert lock_file.exists()

    def test_try_spec_id_lock(self, tmp_path: Path) -> None:
        """Test non-blocking spec_id lock attempt."""
        store_lock = StoreLock(tmp_path / "data")
        spec_id = "test123"

        # First lock succeeds
        lock = store_lock.try_spec_id_lock(spec_id)
        assert lock is not None
        assert lock.is_locked

        # Second lock fails
        lock2 = store_lock.try_spec_id_lock(spec_id)
        assert lock2 is None

        lock.release()

    def test_cleanup_stale_locks(self, tmp_path: Path) -> None:
        """Test cleanup of stale lock files."""
        store_lock = StoreLock(tmp_path / "data")

        # Create some "stale" lock files
        locks_dir = tmp_path / "data" / ".locks" / "spec_id"
        locks_dir.mkdir(parents=True)

        stale_lock = locks_dir / "stale123.lock"
        stale_lock.touch()

        # Set modification time to old
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(stale_lock, (old_time, old_time))

        # Create a fresh lock
        fresh_lock = locks_dir / "fresh456.lock"
        fresh_lock.touch()

        # Cleanup with 1 hour max age
        removed = store_lock.cleanup_stale_locks(max_age_seconds=3600)

        assert len(removed) == 1
        assert stale_lock in removed
        assert not stale_lock.exists()
        assert fresh_lock.exists()


class TestArtifactStoreConcurrency:
    """Tests for concurrent artifact store operations."""

    def test_concurrent_puts_same_content(self, tmp_path: Path) -> None:
        """Test that concurrent puts of the same content are safe."""
        store = ArtifactStore(
            tmp_path / "data",
            enable_locking=True,
            spec_id_lock_timeout=5.0,
        )
        content = b"identical content for all threads"
        errors: list[Exception] = []
        manifests: list[Any] = []
        lock = threading.Lock()

        def put_artifact(thread_id: int) -> None:
            try:
                manifest = store.put(
                    content=content,
                    artifact_type="other",
                    roles=["intermediate"],
                    run_id=f"run-{thread_id}",
                )
                with lock:
                    manifests.append(manifest)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=put_artifact, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(manifests) == 10

        # All manifests should have the same content hash
        hashes = {m.content_hash.digest for m in manifests}
        assert len(hashes) == 1

        # Only one object file should exist
        object_path = store._object_path(manifests[0].content_hash.digest)
        assert object_path.exists()
        assert object_path.read_bytes() == content

    def test_concurrent_puts_different_content(self, tmp_path: Path) -> None:
        """Test that concurrent puts of different content are safe."""
        store = ArtifactStore(
            tmp_path / "data",
            enable_locking=True,
        )
        errors: list[Exception] = []
        manifests: list[Any] = []
        lock = threading.Lock()

        def put_artifact(thread_id: int) -> None:
            try:
                content = f"content for thread {thread_id}".encode()
                manifest = store.put(
                    content=content,
                    artifact_type="other",
                    roles=["intermediate"],
                    run_id=f"run-{thread_id}",
                )
                with lock:
                    manifests.append(manifest)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=put_artifact, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(manifests) == 10

        # All manifests should have different content hashes
        hashes = {m.content_hash.digest for m in manifests}
        assert len(hashes) == 10

    def test_store_locking_disabled(self, tmp_path: Path) -> None:
        """Test that store works correctly with locking disabled."""
        store = ArtifactStore(
            tmp_path / "data",
            enable_locking=False,
        )

        manifest = store.put(
            content=b"test content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        assert manifest.artifact_id is not None

        # No lock files should be created
        locks_dir = tmp_path / "data" / ".locks"
        assert not locks_dir.exists()

    def test_lock_timeout_error(self, tmp_path: Path) -> None:
        """Test that lock timeout raises appropriate error."""
        store = ArtifactStore(
            tmp_path / "data",
            enable_locking=True,
            spec_id_lock_timeout=0.1,  # Very short timeout
        )

        content = b"test content"

        # Pre-create a spec_id lock to simulate contention
        store_lock = StoreLock(tmp_path / "data")
        spec_id = store.compute_spec_id(content)

        with store_lock.spec_id_lock(spec_id):
            # Now try to put with the same content - should timeout
            with pytest.raises(LockTimeoutError):
                store.put(
                    content=content,
                    artifact_type="other",
                    roles=["intermediate"],
                    run_id="run-001",
                )

    def test_concurrent_reads_during_write(self, tmp_path: Path) -> None:
        """Test that reads can proceed while writes are in progress."""
        store = ArtifactStore(
            tmp_path / "data",
            enable_locking=True,
        )

        # First, create an artifact
        content = b"original content"
        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        errors: list[Exception] = []
        read_results: list[bytes] = []
        lock = threading.Lock()

        def read_artifact() -> None:
            try:
                result = store.get(manifest.content_hash.digest)
                with lock:
                    read_results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        def write_artifact() -> None:
            try:
                # Write a different artifact while reads happen
                store.put(
                    content=b"new content",
                    artifact_type="other",
                    roles=["intermediate"],
                    run_id="run-002",
                )
            except Exception as e:
                with lock:
                    errors.append(e)

        # Start readers and writers concurrently
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=read_artifact))
        for _ in range(3):
            threads.append(threading.Thread(target=write_artifact))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(read_results) == 5
        assert all(r == content for r in read_results)


class TestConcurrencyEdgeCases:
    """Edge case tests for concurrency handling."""

    def test_lock_release_on_exception(self, tmp_path: Path) -> None:
        """Test that locks are properly released when exceptions occur."""
        lock_file = tmp_path / "test.lock"
        lock = FileLock(lock_file)

        with pytest.raises(ValueError), lock.exclusive():
            raise ValueError("test error")

        # Lock should be released
        assert not lock.is_locked

        # Another lock should be able to acquire
        lock2 = FileLock(lock_file)
        assert lock2.acquire_exclusive(blocking=False)
        lock2.release()

    def test_store_lock_context_manager_exception(self, tmp_path: Path) -> None:
        """Test StoreLock context manager with exception."""
        store_lock = StoreLock(tmp_path / "data")

        with pytest.raises(ValueError), store_lock.global_lock():
            raise ValueError("test error")

        # Should be able to acquire lock again
        with store_lock.global_lock():
            pass  # Should not hang

    def test_artifact_store_exception_releases_locks(self, tmp_path: Path) -> None:
        """Test that artifact store operations release locks on failure."""
        store = ArtifactStore(
            tmp_path / "data",
            enable_locking=True,
            spec_id_lock_timeout=5.0,
        )

        # Create first artifact
        content = b"test content"
        manifest = store.put(
            content=content,
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-001",
        )

        # Try to create with same artifact_id (should fail)
        with pytest.raises(ArtifactExistsError):
            store.put(
                content=b"different content",
                artifact_type="other",
                roles=["intermediate"],
                run_id="run-002",
                artifact_id=manifest.artifact_id,
                allow_overwrite=False,
            )

        # Locks should be released, so subsequent operations should work
        new_manifest = store.put(
            content=b"yet another content",
            artifact_type="other",
            roles=["intermediate"],
            run_id="run-003",
        )
        assert new_manifest is not None

    def test_spec_id_lock_different_artifacts(self, tmp_path: Path) -> None:
        """Test that spec_id locks don't interfere with different artifacts."""
        store = ArtifactStore(
            tmp_path / "data",
            enable_locking=True,
        )

        content1 = b"content one"
        content2 = b"content two"

        # Both should succeed without blocking
        errors: list[Exception] = []
        manifests: list[Any] = []
        lock = threading.Lock()

        def put_content(content: bytes, run_id: str) -> None:
            try:
                manifest = store.put(
                    content=content,
                    artifact_type="other",
                    roles=["intermediate"],
                    run_id=run_id,
                )
                with lock:
                    manifests.append(manifest)
            except Exception as e:
                with lock:
                    errors.append(e)

        t1 = threading.Thread(target=put_content, args=(content1, "run-1"))
        t2 = threading.Thread(target=put_content, args=(content2, "run-2"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0
        assert len(manifests) == 2
