"""
Cross-platform file locking utility for preventing concurrent access.

Uses fcntl on Unix and msvcrt on Windows.
"""

import os
import sys
import time
import json
import logging
import tempfile


def _lock_file(file_path: str):
    """Acquire an exclusive lock on a file."""
    lock_path = file_path + ".lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)

    if sys.platform == "win32":
        import msvcrt
        f = open(lock_path, "w")
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            f.close()
            return None
        return f
    else:
        import fcntl
        f = open(lock_path, "w")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            f.close()
            return None
        return f


def _unlock_file(lock_file):
    """Release the lock on a file."""
    if lock_file is None:
        return
    lock_path = lock_file.name
    lock_file.close()
    try:
        os.unlink(lock_path)
    except OSError:
        pass


def safe_json_load(file_path: str, default: dict | list | None = None):
    """Safely load JSON with retry on lock contention."""
    if default is None:
        default = {}

    for attempt in range(5):
        lock = _lock_file(file_path)
        if lock is None:
            time.sleep(0.1 * (attempt + 1))
            continue

        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default
        except Exception:
            return default
        finally:
            _unlock_file(lock)

    return default


def safe_json_save(file_path: str, data: dict | list) -> bool:
    """Atomically save JSON, preserving the old file if serialization fails."""
    try:
        serialized = json.dumps(data, indent=2)
    except Exception as exc:
        logging.getLogger("optimist_prime").error(
            "JSON state could not be serialized; existing file was preserved",
            extra={
                "event_name": "runtime_state.save_failed",
                "component": "state_store",
                "operation_stage": "serialize",
                "error_type": type(exc).__name__,
            },
        )
        return False

    directory = os.path.dirname(file_path) or "."
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as exc:
        logging.getLogger("optimist_prime").error(
            "JSON state directory could not be prepared",
            extra={
                "event_name": "runtime_state.save_failed",
                "component": "state_store",
                "operation_stage": "create_directory",
                "error_type": type(exc).__name__,
            },
        )
        return False

    for attempt in range(5):
        try:
            lock = _lock_file(file_path)
        except Exception as exc:
            logging.getLogger("optimist_prime").error(
                "JSON state lock could not be opened",
                extra={
                    "event_name": "runtime_state.save_failed",
                    "component": "state_store",
                    "operation_stage": "open_lock",
                    "error_type": type(exc).__name__,
                    "attempt": attempt + 1,
                },
            )
            return False
        if lock is None:
            time.sleep(0.1 * (attempt + 1))
            continue

        temporary_path = None
        retry_after_permission_error = False
        try:
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{os.path.basename(file_path)}.",
                suffix=".tmp",
                dir=directory,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as f:
                f.write(serialized)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, file_path)
            return True
        except Exception as exc:
            # On Windows, antivirus and other readers can briefly open the
            # destination without delete sharing. That makes os.replace raise
            # PermissionError even though the state is still safe to retry.
            retry_after_permission_error = (
                sys.platform == "win32"
                and isinstance(exc, PermissionError)
                and attempt < 4
            )
            logging.getLogger("optimist_prime").log(
                logging.WARNING if retry_after_permission_error else logging.ERROR,
                "Atomic JSON state replacement was blocked; retrying"
                if retry_after_permission_error
                else "Atomic JSON state replacement failed; existing file was preserved",
                extra={
                    "event_name": "runtime_state.save_failed",
                    "component": "state_store",
                    "operation_stage": "write_or_replace",
                    "error_type": type(exc).__name__,
                    "error_number": getattr(exc, "errno", None),
                    "win_error": getattr(exc, "winerror", None),
                    "attempt": attempt + 1,
                    "retrying": retry_after_permission_error,
                },
            )
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            _unlock_file(lock)

        if retry_after_permission_error:
            time.sleep(0.1 * (attempt + 1))
            continue
        return False

    logging.getLogger("optimist_prime").error(
        "JSON state lock remained unavailable after retries",
        extra={
            "event_name": "runtime_state.save_failed",
            "component": "state_store",
            "operation_stage": "lock_contention",
            "error_type": "LockUnavailable",
            "attempt": 5,
        },
    )
    return False
