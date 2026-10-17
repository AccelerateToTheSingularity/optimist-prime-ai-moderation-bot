"""Cross-process guard that prevents concurrent Optimist runtime cycles."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path


_registry_guard = threading.Lock()
_held_paths: set[str] = set()


def runtime_cycle_lock_path(state_path: str | os.PathLike[str]) -> Path:
    """Return the shared lock path for every spelling of a runtime state path."""
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(state_path)))
    canonical_state = Path(expanded).resolve()
    return canonical_state.with_name(canonical_state.name + ".cycle.lock")


class RuntimeCycleLock:
    """A non-blocking OS lock on a persistent, per-runtime lock file.

    The lock file is deliberately left in place after release. Deleting a file
    while another process has it open can split Windows file locks across two
    file objects and allow overlapping workers.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).expanduser().resolve()
        self._key = os.path.normcase(str(self.path))
        self._file = None
        self._acquired = False

    def acquire(self) -> bool:
        with _registry_guard:
            if self._key in _held_paths:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            stream = open(self.path, "a+b")
            try:
                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (ImportError, OSError):
                stream.close()
                return False
            self._file = stream
            self._acquired = True
            _held_paths.add(self._key)
            return True

    def release(self) -> None:
        with _registry_guard:
            if not self._acquired:
                return
            stream = self._file
            self._file = None
            self._acquired = False
            _held_paths.discard(self._key)
            if stream is None:
                return
            try:
                stream.seek(0)
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()


__all__ = ["RuntimeCycleLock", "runtime_cycle_lock_path"]
