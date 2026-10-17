"""
File-based audit log for bot actions.
Tracks all actions with event type, target, author, and outcome.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

MAX_ENTRIES = 200


def audit_log_path() -> Path:
    """Return the current private audit-log path.

    ``BOT_RUNTIME_AUDIT_LOG_FILE`` is the operator-machine setting used by the
    local worker. ``AUDIT_LOG_PATH`` remains a compatibility alias for older
    installations and tests. The repository-relative default is retained only
    for contributor development and fixtures.
    """
    raw = (
        os.environ.get("BOT_RUNTIME_AUDIT_LOG_FILE")
        or os.environ.get("AUDIT_LOG_PATH")
        or "data/audit_log.json"
    ).strip()
    return Path(os.path.expandvars(os.path.expanduser(raw)))


# Compatibility export for diagnostics and older integrations. Read/write
# operations call ``audit_log_path`` dynamically so a test or executable that
# loads local settings after import still uses the effective path.
AUDIT_LOG_FILE = str(audit_log_path())


def _load_log() -> list[dict]:
    """Load audit log from file with file locking."""
    from file_lock import safe_json_load
    data = safe_json_load(str(audit_log_path()), [])
    if isinstance(data, list):
        return data
    return []


def _save_log(entries: list[dict]) -> None:
    """Save audit log to file, capping at MAX_ENTRIES, with file locking."""
    from file_lock import safe_json_save
    
    # Keep only the most recent entries
    trimmed = entries[-MAX_ENTRIES:]
    
    # Clean up entries older than 30 days
    now = datetime.now(timezone.utc)
    thirty_days_ago = now.timestamp() - (30 * 24 * 3600)
    
    cleaned = []
    for entry in trimmed:
        try:
            ts = datetime.fromisoformat(entry.get("timestamp", "")).timestamp()
            if ts >= thirty_days_ago:
                cleaned.append(entry)
        except (ValueError, TypeError):
            # Keep entries with unparseable timestamps
            cleaned.append(entry)
    
    safe_json_save(str(audit_log_path()), cleaned)


def log_audit_event(
    event_type: str,
    target_id: str,
    author: str,
    text_snippet: str,
    action: str,
    success: bool,
    message: str = "",
) -> None:
    """
    Log an audit event.

    Args:
        event_type: Type of event (e.g. 'tldr', 'reply', 'summon', 'moderation', 'ban')
        target_id: Reddit ID of the target (post/comment)
        author: Username of the content author
        text_snippet: First 200 chars of the content
        action: Action taken (e.g. 'posted', 'skipped', 'removed', 'reported')
        success: Whether the action succeeded
        message: Additional context
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "target_id": target_id,
        "author": author,
        "text_snippet": text_snippet[:200],
        "action": action,
        "success": success,
        "message": message,
    }

    log = _load_log()
    log.append(entry)
    _save_log(log)


def get_recent_audit_events(limit: int = 20) -> list[dict]:
    """Get the most recent audit events."""
    log = _load_log()
    return log[-limit:]
