"""Local runtime-state storage for the bot and its tests.

The bot needs durable state for idempotency, cooldowns, queues, and report
reviews. State may contain Reddit account names or moderation-related records,
so keep runtime files local and out of source control. The default locked JSON
backend is intended for local use. Encrypted storage is available when
explicitly requested or when running in GitHub Actions.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from file_lock import safe_json_load, safe_json_save


class StateStoreError(RuntimeError):
    """Raised when runtime state cannot be safely loaded or saved."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_github_actions() -> bool:
    return _env_bool("GITHUB_ACTIONS")


def encryption_required(*, dry_run: bool = False) -> bool:
    """Whether this invocation must use encrypted state."""
    requested = os.environ.get("BOT_STATE_BACKEND", "").strip().lower()
    if requested in {"encrypted", "fernet"}:
        return True
    if is_github_actions() and not dry_run:
        return True
    return False


def runtime_state_path(state_file: str | Path | None = None, *, dry_run: bool = False) -> Path:
    if state_file is not None:
        return Path(state_file)
    configured = os.environ.get("BOT_RUNTIME_STATE_FILE", "").strip()
    if configured:
        return Path(os.path.expandvars(os.path.expanduser(configured)))
    if encryption_required(dry_run=dry_run) or os.environ.get("BOT_STATE_ENCRYPTION_KEY"):
        return Path("data/bot_state.enc")
    return Path("data/bot_state.json")


def _fernet_from_environment(*, required: bool) -> Fernet | None:
    raw_key = os.environ.get("BOT_STATE_ENCRYPTION_KEY", "").strip()
    if not raw_key:
        if required:
            raise StateStoreError(
                "Live runtime state requires BOT_STATE_ENCRYPTION_KEY; refusing plaintext fallback."
            )
        return None
    try:
        # Fernet validates the URL-safe base64 key and its 32-byte payload.
        return Fernet(raw_key.encode("ascii"))
    except (ValueError, TypeError, UnicodeEncodeError) as exc:
        raise StateStoreError("BOT_STATE_ENCRYPTION_KEY is not a valid Fernet key") from exc


def load_state(
    state_file: str | Path | None = None,
    default: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Load plaintext local state or encrypted state.

    A missing state file is a clean first-run state. A malformed existing file
    raises instead of silently starting over, because losing idempotency state
    can cause duplicate Reddit actions.
    """
    fallback = dict(default or {})
    path = runtime_state_path(state_file, dry_run=dry_run)
    # A dry run may use a blank in-memory state when the encrypted
    # key is unavailable.  It must never need the key merely to preview work,
    # and it must never write the preview back to disk.
    required = not dry_run and (
        encryption_required(dry_run=dry_run) or path.suffix.lower() == ".enc"
    )
    fernet = _fernet_from_environment(required=required)

    if fernet is None:
        if path.suffix.lower() == ".enc":
            return fallback
        if not path.exists():
            return fallback
        missing = object()
        loaded = safe_json_load(str(path), missing)
        if loaded is missing:
            raise StateStoreError(
                f"Could not read existing plaintext runtime state; refusing to start with empty state: {path}"
            )
        if not isinstance(loaded, dict):
            raise StateStoreError("Plaintext runtime state must contain a JSON object")
        return loaded

    if not path.exists():
        # A live process must not silently start from an empty state when a
        # legacy plaintext state file is still present.  That could repeat
        # previously completed Reddit actions.  The explicit migration tool
        # below is the only path that converts it to encrypted storage.
        if required and state_file is None and not os.environ.get("BOT_RUNTIME_STATE_FILE", "").strip() and Path("data/bot_state.json").exists():
            raise StateStoreError(
                "Encrypted state is missing while legacy plaintext state exists; "
                "run migrate_state.py with BOT_STATE_ENCRYPTION_KEY before live execution."
            )
        return fallback
    envelope = safe_json_load(str(path), None)
    if not isinstance(envelope, dict) or envelope.get("version") != 1:
        raise StateStoreError(f"Encrypted state file has an invalid envelope: {path}")
    ciphertext = envelope.get("ciphertext")
    if not isinstance(ciphertext, str) or not ciphertext:
        raise StateStoreError(f"Encrypted state file has no ciphertext: {path}")
    try:
        plaintext = fernet.decrypt(ciphertext.encode("ascii"))
        loaded = json.loads(plaintext.decode("utf-8"))
    except (InvalidToken, ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise StateStoreError(
            "Could not decrypt runtime state; refusing to replace it with a blank state."
        ) from exc
    if not isinstance(loaded, dict):
        raise StateStoreError("Encrypted runtime state must contain a JSON object")
    return loaded


def save_state(
    state: dict[str, Any],
    state_file: str | Path | None = None,
    *,
    dry_run: bool = False,
) -> None:
    """Persist state, failing closed when live encryption is unavailable."""
    if dry_run:
        return
    path = runtime_state_path(state_file, dry_run=dry_run)
    required = not dry_run and (
        encryption_required(dry_run=dry_run) or path.suffix.lower() == ".enc"
    )
    fernet = _fernet_from_environment(required=required)
    if fernet is None:
        if not safe_json_save(str(path), state):
            raise StateStoreError(f"Failed to save runtime state: {path}")
        return

    plaintext = json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    envelope = {
        "version": 1,
        "algorithm": "fernet",
        "ciphertext": fernet.encrypt(plaintext).decode("ascii"),
    }
    if not safe_json_save(str(path), envelope):
        raise StateStoreError(f"Failed to save encrypted runtime state: {path}")


def generate_fernet_key() -> str:
    """Generate a key for maintainers; never writes or prints it automatically."""
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


def migrate_plaintext_state(
    source: str | Path = "data/bot_state.json",
    target: str | Path = "data/bot_state.enc",
) -> dict[str, Any]:
    """Convert an existing local JSON state file to encrypted state once.

    The target must not already exist.  This deliberate non-overwrite rule
    prevents an accidental migration from replacing a newer encrypted state.
    Unknown fields are carried through unchanged so newer callers
    do not lose state during the transition.
    """
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.exists():
        raise StateStoreError(f"Plaintext state file not found: {source_path}")
    if target_path.exists():
        raise StateStoreError(
            f"Encrypted state already exists: {target_path}; refusing to overwrite it."
        )
    try:
        loaded = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateStoreError(f"Could not read plaintext state: {source_path}") from exc
    if not isinstance(loaded, dict):
        raise StateStoreError("Plaintext runtime state must contain a JSON object")
    save_state(loaded, target_path)
    return loaded


def migrate_plaintext_state_to_local(
    source: str | Path,
    target: str | Path,
) -> dict[str, Any]:
    """Copy plaintext state to another local path once.

    This is an explicit operator migration. It never
    overwrites the target or deletes the source, and it preserves fields that
    newer versions may not know about.
    """
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.exists():
        raise StateStoreError(f"Plaintext state file not found: {source_path}")
    if target_path.exists():
        raise StateStoreError(
            f"Runtime state already exists: {target_path}; refusing to overwrite it."
        )
    try:
        loaded = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateStoreError(f"Could not read plaintext state: {source_path}") from exc
    if not isinstance(loaded, dict):
        raise StateStoreError("Plaintext runtime state must contain a JSON object")
    if not safe_json_save(str(target_path), loaded):
        raise StateStoreError(f"Failed to save runtime state: {target_path}")
    return loaded
