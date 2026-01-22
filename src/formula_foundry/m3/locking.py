"""Concurrency control and locking for M3 artifact store.

This module provides file-based locking mechanisms for safe concurrent access
to the artifact store. It implements:

- Global store lock for metadata updates
- Per-artifact spec_id locks during creation
- Cross-process and cross-thread safety using fcntl/flock

The locking strategy follows the design document (Section 15):
- data/.locks/artifact_store.lock for store-wide metadata updates
- data/.locks/spec_id/{spec_id}.lock for per-artifact creation locks
- Atomic rename commit pattern for artifact content

Thread Safety:
    Locks are process-safe via fcntl and thread-safe via threading.Lock wrappers.

Example usage:
    store_lock = StoreLock(data_dir)
    with store_lock.global_lock():
        # Protected metadata operations
        ...

    with store_lock.spec_id_lock("abc123defgh"):
        # Protected artifact creation for this spec_id
        ...
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal

if TYPE_CHECKING:
    from types import TracebackType


# Lock type for different operations
LockType = Literal["shared", "exclusive"]


class LockError(Exception):
    """Base exception for locking errors."""


class LockAcquisitionError(LockError):
    """Raised when a lock cannot be acquired within the timeout."""


class LockNotHeldError(LockError):
    """Raised when attempting to release a lock that is not held."""


@dataclass
class LockInfo:
    """Information about a held lock."""

    path: Path
    lock_type: LockType
    acquired_at: float
    process_id: int
    thread_id: int


class FileLock:
    """Cross-process file lock using fcntl.

    This class provides a file-based lock that works across processes using
    POSIX fcntl advisory locks. It supports both shared (read) and exclusive
    (write) locks, with optional timeout.

    The lock is also thread-safe within a single process using a threading.Lock
    wrapper, ensuring that threads don't interfere with each other while also
    providing cross-process safety.

    Example usage:
        lock = FileLock(Path("/tmp/mylock.lock"))
        with lock.acquire_exclusive():
            # Critical section
            ...
    """

    def __init__(self, lock_path: Path, create_dirs: bool = True) -> None:
        """Initialize a file lock.

        Args:
            lock_path: Path to the lock file. Will be created if it doesn't exist.
            create_dirs: If True, create parent directories if they don't exist.
        """
        self.lock_path = lock_path
        self._create_dirs = create_dirs
        self._fd: int | None = None
        self._lock_type: LockType | None = None
        self._thread_lock = threading.Lock()
        self._acquired_at: float | None = None

    def _ensure_lock_file(self) -> None:
        """Ensure the lock file and its parent directories exist."""
        if self._create_dirs:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Create the lock file if it doesn't exist
        if not self.lock_path.exists():
            try:
                self.lock_path.touch(exist_ok=True)
            except FileExistsError:
                pass  # Race condition, file was created by another process

    def _acquire(
        self,
        lock_type: LockType,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> bool:
        """Acquire the lock.

        Args:
            lock_type: "shared" for read lock, "exclusive" for write lock.
            blocking: If True, wait for the lock. If False, return immediately.
            timeout: Maximum time to wait for the lock (seconds). None means wait forever.

        Returns:
            True if the lock was acquired, False if non-blocking and lock unavailable.

        Raises:
            LockAcquisitionError: If timeout expires while waiting for the lock.
        """
        self._ensure_lock_file()

        # Get the thread lock first (non-blocking or timed)
        if not self._thread_lock.acquire(blocking=blocking, timeout=timeout or -1):
            return False

        try:
            # Open the lock file
            self._fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT)

            # Determine fcntl lock type
            if lock_type == "shared":
                fcntl_op = fcntl.LOCK_SH
            else:
                fcntl_op = fcntl.LOCK_EX

            if not blocking:
                fcntl_op |= fcntl.LOCK_NB

            if timeout is not None and blocking:
                # Implement timeout with polling
                start_time = time.monotonic()
                while True:
                    try:
                        fcntl.flock(self._fd, fcntl_op | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        elapsed = time.monotonic() - start_time
                        if elapsed >= timeout:
                            os.close(self._fd)
                            self._fd = None
                            self._thread_lock.release()
                            raise LockAcquisitionError(
                                f"Timeout acquiring {lock_type} lock on {self.lock_path} "
                                f"after {timeout:.2f}s"
                            )
                        # Sleep briefly before retrying
                        time.sleep(min(0.01, timeout - elapsed))
            else:
                try:
                    fcntl.flock(self._fd, fcntl_op)
                except BlockingIOError:
                    os.close(self._fd)
                    self._fd = None
                    self._thread_lock.release()
                    return False

            self._lock_type = lock_type
            self._acquired_at = time.monotonic()
            return True

        except Exception:
            # Clean up on any failure
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            if self._thread_lock.locked():
                self._thread_lock.release()
            raise

    def acquire_shared(
        self,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> bool:
        """Acquire a shared (read) lock.

        Multiple processes can hold shared locks simultaneously.

        Args:
            blocking: If True, wait for the lock. If False, return immediately.
            timeout: Maximum time to wait (seconds). None means wait forever.

        Returns:
            True if acquired, False if non-blocking and unavailable.
        """
        return self._acquire("shared", blocking, timeout)

    def acquire_exclusive(
        self,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> bool:
        """Acquire an exclusive (write) lock.

        Only one process can hold an exclusive lock, and it blocks all shared locks.

        Args:
            blocking: If True, wait for the lock. If False, return immediately.
            timeout: Maximum time to wait (seconds). None means wait forever.

        Returns:
            True if acquired, False if non-blocking and unavailable.
        """
        return self._acquire("exclusive", blocking, timeout)

    def release(self) -> None:
        """Release the lock.

        Raises:
            LockNotHeldError: If the lock is not currently held.
        """
        if self._fd is None:
            raise LockNotHeldError(f"Lock not held: {self.lock_path}")

        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
        finally:
            self._fd = None
            self._lock_type = None
            self._acquired_at = None
            self._thread_lock.release()

    @property
    def is_locked(self) -> bool:
        """Check if this lock instance currently holds the lock."""
        return self._fd is not None

    @property
    def lock_info(self) -> LockInfo | None:
        """Get information about the currently held lock."""
        if not self.is_locked or self._lock_type is None or self._acquired_at is None:
            return None
        return LockInfo(
            path=self.lock_path,
            lock_type=self._lock_type,
            acquired_at=self._acquired_at,
            process_id=os.getpid(),
            thread_id=threading.get_ident(),
        )

    @contextmanager
    def shared(
        self,
        timeout: float | None = None,
    ) -> Iterator[LockInfo]:
        """Context manager for acquiring a shared lock.

        Args:
            timeout: Maximum time to wait for the lock (seconds).

        Yields:
            LockInfo about the acquired lock.

        Raises:
            LockAcquisitionError: If the lock cannot be acquired.
        """
        self.acquire_shared(blocking=True, timeout=timeout)
        try:
            yield self.lock_info  # type: ignore[misc]
        finally:
            self.release()

    @contextmanager
    def exclusive(
        self,
        timeout: float | None = None,
    ) -> Iterator[LockInfo]:
        """Context manager for acquiring an exclusive lock.

        Args:
            timeout: Maximum time to wait for the lock (seconds).

        Yields:
            LockInfo about the acquired lock.

        Raises:
            LockAcquisitionError: If the lock cannot be acquired.
        """
        self.acquire_exclusive(blocking=True, timeout=timeout)
        try:
            yield self.lock_info  # type: ignore[misc]
        finally:
            self.release()

    def __enter__(self) -> LockInfo:
        """Default context manager acquires exclusive lock."""
        self.acquire_exclusive(blocking=True)
        return self.lock_info  # type: ignore[return-value]

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release the lock on context exit."""
        self.release()


class StoreLock:
    """High-level locking interface for the artifact store.

    This class provides a simplified interface for the two types of locks
    needed by the artifact store:

    1. Global store lock: Protects store-wide metadata operations
    2. Spec ID locks: Protects individual artifact creation by spec_id

    The lock files are stored in:
    - data/.locks/artifact_store.lock (global)
    - data/.locks/spec_id/{spec_id}.lock (per-artifact)

    Example usage:
        store_lock = StoreLock(Path("data"))

        # Protect metadata operations
        with store_lock.global_lock():
            # Safe to modify manifests
            ...

        # Protect artifact creation
        with store_lock.spec_id_lock("abc123defgh"):
            # Safe to create artifact with this spec_id
            ...
    """

    LOCKS_DIR = ".locks"
    GLOBAL_LOCK_NAME = "artifact_store.lock"
    SPEC_ID_LOCKS_DIR = "spec_id"

    # Default timeouts
    DEFAULT_GLOBAL_TIMEOUT = 30.0  # seconds
    DEFAULT_SPEC_ID_TIMEOUT = 60.0  # seconds (artifact creation can be slower)

    def __init__(self, root: Path | str) -> None:
        """Initialize the store lock.

        Args:
            root: Root directory of the data store.
        """
        self.root = Path(root)
        self.locks_dir = self.root / self.LOCKS_DIR
        self._global_lock_path = self.locks_dir / self.GLOBAL_LOCK_NAME
        self._spec_id_locks_dir = self.locks_dir / self.SPEC_ID_LOCKS_DIR

        # Cache of FileLock instances for spec_id locks
        self._spec_id_locks: dict[str, FileLock] = {}
        self._spec_id_locks_mutex = threading.Lock()

        # Global lock instance
        self._global_lock = FileLock(self._global_lock_path)

    def _ensure_dirs(self) -> None:
        """Ensure lock directories exist."""
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        self._spec_id_locks_dir.mkdir(parents=True, exist_ok=True)

    def _get_spec_id_lock(self, spec_id: str) -> FileLock:
        """Get or create a FileLock for a spec_id.

        Args:
            spec_id: The spec_id to lock.

        Returns:
            FileLock instance for this spec_id.
        """
        with self._spec_id_locks_mutex:
            if spec_id not in self._spec_id_locks:
                lock_path = self._spec_id_locks_dir / f"{spec_id}.lock"
                self._spec_id_locks[spec_id] = FileLock(lock_path)
            return self._spec_id_locks[spec_id]

    @contextmanager
    def global_lock(
        self,
        timeout: float | None = None,
        shared: bool = False,
    ) -> Iterator[LockInfo]:
        """Acquire the global store lock.

        This lock should be held when:
        - Writing/updating manifests
        - Performing store-wide operations (GC, rebuild, etc.)
        - Any operation that modifies store metadata

        Args:
            timeout: Maximum time to wait (defaults to DEFAULT_GLOBAL_TIMEOUT).
            shared: If True, acquire shared lock (for reads). Default is exclusive.

        Yields:
            LockInfo about the acquired lock.

        Raises:
            LockAcquisitionError: If the lock cannot be acquired.
        """
        self._ensure_dirs()
        effective_timeout = timeout if timeout is not None else self.DEFAULT_GLOBAL_TIMEOUT

        if shared:
            with self._global_lock.shared(timeout=effective_timeout) as info:
                yield info
        else:
            with self._global_lock.exclusive(timeout=effective_timeout) as info:
                yield info

    @contextmanager
    def spec_id_lock(
        self,
        spec_id: str,
        timeout: float | None = None,
    ) -> Iterator[LockInfo]:
        """Acquire a per-artifact spec_id lock.

        This lock should be held when:
        - Creating a new artifact with this spec_id
        - Computing and writing content for this spec_id

        The lock prevents duplicate work when multiple processes try to create
        the same artifact simultaneously.

        Args:
            spec_id: The spec_id to lock.
            timeout: Maximum time to wait (defaults to DEFAULT_SPEC_ID_TIMEOUT).

        Yields:
            LockInfo about the acquired lock.

        Raises:
            LockAcquisitionError: If the lock cannot be acquired.
        """
        self._ensure_dirs()
        effective_timeout = timeout if timeout is not None else self.DEFAULT_SPEC_ID_TIMEOUT

        lock = self._get_spec_id_lock(spec_id)
        with lock.exclusive(timeout=effective_timeout) as info:
            yield info

    def try_spec_id_lock(
        self,
        spec_id: str,
    ) -> FileLock | None:
        """Try to acquire a spec_id lock without blocking.

        This is useful for checking if another process is already creating
        an artifact with this spec_id.

        Args:
            spec_id: The spec_id to lock.

        Returns:
            FileLock instance if acquired, None if the lock is held by another.
        """
        self._ensure_dirs()
        lock = self._get_spec_id_lock(spec_id)
        if lock.acquire_exclusive(blocking=False):
            return lock
        return None

    def cleanup_stale_locks(self, max_age_seconds: float = 3600.0) -> list[Path]:
        """Clean up stale spec_id lock files.

        Lock files older than max_age_seconds are assumed to be from crashed
        processes and are removed.

        Args:
            max_age_seconds: Maximum age of lock files to keep (default 1 hour).

        Returns:
            List of paths to removed lock files.
        """
        removed: list[Path] = []
        if not self._spec_id_locks_dir.exists():
            return removed

        current_time = time.time()
        for lock_file in self._spec_id_locks_dir.glob("*.lock"):
            try:
                mtime = lock_file.stat().st_mtime
                if current_time - mtime > max_age_seconds:
                    lock_file.unlink()
                    removed.append(lock_file)
            except (OSError, FileNotFoundError):
                pass  # File was removed by another process

        return removed


# Convenience functions for common locking patterns


def with_store_lock(
    root: Path | str,
    timeout: float | None = None,
    shared: bool = False,
) -> contextmanager[Iterator[LockInfo]]:
    """Create a context manager for the global store lock.

    This is a convenience function for one-off lock acquisitions.
    For repeated use, prefer creating a StoreLock instance.

    Args:
        root: Root directory of the data store.
        timeout: Maximum time to wait.
        shared: If True, acquire shared lock.

    Returns:
        Context manager that yields LockInfo.
    """
    store_lock = StoreLock(root)
    return store_lock.global_lock(timeout=timeout, shared=shared)


def with_spec_id_lock(
    root: Path | str,
    spec_id: str,
    timeout: float | None = None,
) -> contextmanager[Iterator[LockInfo]]:
    """Create a context manager for a spec_id lock.

    This is a convenience function for one-off lock acquisitions.
    For repeated use, prefer creating a StoreLock instance.

    Args:
        root: Root directory of the data store.
        spec_id: The spec_id to lock.
        timeout: Maximum time to wait.

    Returns:
        Context manager that yields LockInfo.
    """
    store_lock = StoreLock(root)
    return store_lock.spec_id_lock(spec_id, timeout=timeout)
