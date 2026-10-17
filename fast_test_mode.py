"""Short-lived local lease for faster bot-cycle testing.

The lease is shared by the GUI and the existing worker through one small
local JSON file. It changes only the wait between completed cycles.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

LEASE_SECONDS = 2 * 60
FAST_INTERVAL_SECONDS = 3


def state_path() -> Path:
    configured = os.environ.get("BOT_RUNTIME_FAST_TEST_MODE_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return root / "OptimistPrime" / "fast_test_mode.json"


def _write_state(payload: dict[str, Any]) -> None:
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
            json.dump(payload, temporary, separators=(",", ":"))
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


def get_fast_test_mode(*, now: float | None = None) -> dict[str, Any]:
    """Return the current lease status; expired or malformed files are inactive."""
    current = time.time() if now is None else float(now)
    try:
        payload = json.loads(state_path().read_text(encoding="utf-8"))
        started = float(payload["started_at"])
        expires = float(payload["expires_at"])
        interval = int(payload["interval_seconds"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return {"active": False, "remaining_seconds": 0, "expires_at": None}
    if (
        interval != FAST_INTERVAL_SECONDS
        or expires <= started
        or expires - started > LEASE_SECONDS + 1
        or expires <= current
    ):
        return {"active": False, "remaining_seconds": 0, "expires_at": None}
    return {
        "active": True,
        "remaining_seconds": max(0, int(expires - current + 0.999)),
        "expires_at": expires,
        "interval_seconds": interval,
    }


def start_fast_test_mode(*, now: float | None = None) -> dict[str, Any]:
    started = time.time() if now is None else float(now)
    _write_state(
        {
            "version": 1,
            "started_at": started,
            "expires_at": started + LEASE_SECONDS,
            "interval_seconds": FAST_INTERVAL_SECONDS,
        }
    )
    return get_fast_test_mode(now=started)


def stop_fast_test_mode() -> dict[str, Any]:
    path = state_path()
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # An unreadable stale lease is still bounded by its own expiry.
        pass
    return {"active": False, "remaining_seconds": 0, "expires_at": None}


def main(argv: list[str] | None = None) -> int:
    """Provide a small local switch for supervised test sessions."""
    parser = argparse.ArgumentParser(description="Control Optimist Prime's short test-cycle lease.")
    parser.add_argument("action", choices=("start", "stop", "status"))
    args = parser.parse_args(argv)

    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
    if args.action == "start":
        result = start_fast_test_mode()
    elif args.action == "stop":
        result = stop_fast_test_mode()
    else:
        result = get_fast_test_mode()
    visible_result = {
        key: result[key]
        for key in ("active", "remaining_seconds", "interval_seconds")
        if key in result
    }
    print(json.dumps(visible_result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
