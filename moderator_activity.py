"""Short-lived worker acceleration after a verified moderator interaction.

The activity record contains only a timestamp. It is deliberately separate
from the manual fast-test lease so either control can expire independently.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any


MODERATOR_ACTIVE_SECONDS = 3 * 60


def state_path() -> Path:
    configured = os.environ.get("BOT_RUNTIME_MODERATOR_ACTIVITY_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        root = Path(local_app_data)
    else:
        profile = os.environ.get("USERPROFILE", "").strip() or os.environ.get("HOME", "").strip()
        if profile:
            root = Path(profile) / "AppData" / "Local"
        else:
            # Minimal test/diagnostic environments can intentionally clear
            # profile variables; their runtime file remains gitignored.
            root = Path(__file__).resolve().parent / "data"
    return root / "OptimistPrime" / "moderator_activity.json"


def _read_last_activity() -> float | None:
    try:
        payload = json.loads(state_path().read_text(encoding="utf-8"))
        value = float(payload["last_interaction_at"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def record_moderator_activity(*, now: float | None = None) -> float:
    """Atomically refresh the global, PII-free three-minute activity window."""
    current = time.time() if now is None else float(now)
    if not math.isfinite(current) or current <= 0:
        raise ValueError("moderator activity time must be a positive finite value")
    # Concurrent trusted moderator turns should keep the newest timestamp.
    previous = _read_last_activity()
    if previous is not None:
        current = max(current, previous)

    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump({"version": 1, "last_interaction_at": current}, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return current


def get_moderator_activity_status(*, now: float | None = None) -> dict[str, Any]:
    """Return whether a verified moderator was active in the past three minutes."""
    current = time.time() if now is None else float(now)
    last_activity = _read_last_activity()
    if last_activity is None:
        return {"active": False, "remaining_seconds": 0, "expires_at": None}
    expires_at = last_activity + MODERATOR_ACTIVE_SECONDS
    remaining = expires_at - current
    if remaining <= 0:
        return {"active": False, "remaining_seconds": 0, "expires_at": expires_at}
    return {
        "active": True,
        "remaining_seconds": max(1, math.ceil(remaining)),
        "expires_at": expires_at,
        "last_interaction_at": last_activity,
    }


__all__ = [
    "MODERATOR_ACTIVE_SECONDS",
    "get_moderator_activity_status",
    "record_moderator_activity",
    "state_path",
]
