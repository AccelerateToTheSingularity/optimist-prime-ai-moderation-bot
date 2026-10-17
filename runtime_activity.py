"""Local activity and cycle-count snapshots.

The headless worker already fetches Reddit items for its normal cycle.  This
module records a small, bounded copy for the native console and for adaptive
scheduling.  It never creates a Reddit client or performs a Reddit request.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from file_lock import safe_json_load, safe_json_save


REPO_ROOT = Path(__file__).resolve().parent
MAX_ACTIVITY_ITEMS = 500


def _runtime_path(env_name: str, default_name: str, path: str | Path | None = None) -> Path:
    raw = str(path or os.environ.get(env_name, "")).strip()
    if not raw:
        return REPO_ROOT / "data" / default_name
    candidate = Path(os.path.expandvars(os.path.expanduser(raw)))
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def activity_path(path: str | Path | None = None) -> Path:
    return _runtime_path("BOT_RUNTIME_ACTIVITY_FILE", "activity.json", path)


def cycle_summary_path(path: str | Path | None = None) -> Path:
    return _runtime_path("BOT_RUNTIME_CYCLE_SUMMARY_FILE", "cycle_summary.json", path)


def _author_name(item: Any) -> str:
    author = getattr(item, "author", None)
    if author is None:
        return "[deleted]"
    return str(getattr(author, "name", author) or "[deleted]")[:80]


def _created_utc(item: Any) -> float:
    try:
        return float(getattr(item, "created_utc", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _record(item: Any, content_type: str, subreddit: str, run_id: str | None) -> dict[str, Any]:
    is_post = content_type == "post"
    body = getattr(item, "selftext", "") if is_post else getattr(item, "body", "")
    title = getattr(item, "title", "") if is_post else getattr(item, "link_title", "")
    created = _created_utc(item)
    return {
        "id": str(getattr(item, "id", "")),
        "contentType": content_type,
        "author": _author_name(item),
        "subreddit": subreddit,
        "linkTitle": str(title or "")[:500],
        "body": str(body or "")[:4000],
        "bodyTruncated": len(str(body or "")) > 4000,
        "createdUtc": created,
        "createdAt": datetime.fromtimestamp(created, tz=timezone.utc).isoformat() if created else "",
        "score": getattr(item, "score", None),
        "permalink": str(getattr(item, "permalink", "") or "")[:1000],
        "processingState": "captured",
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "runId": str(run_id or "")[:80],
    }


def read_activity_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    candidate = activity_path(path)
    default = {"version": 1, "updatedAt": None, "subreddit": None, "items": []}
    loaded = safe_json_load(str(candidate), default)
    if not isinstance(loaded, dict):
        return default
    items = loaded.get("items")
    loaded["items"] = [dict(item) for item in items if isinstance(item, dict)] if isinstance(items, list) else []
    return loaded


def record_cycle_activity(
    posts: Iterable[Any],
    comments: Iterable[Any],
    *,
    subreddit: str,
    run_id: str | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge already-captured items into the bounded local snapshot."""
    snapshot = read_activity_snapshot(path)
    by_key = {
        (str(item.get("contentType", "")), str(item.get("id", ""))): item
        for item in snapshot.get("items", [])
        if item.get("id")
    }
    for item in list(posts or ()):
        record = _record(item, "post", subreddit, run_id)
        if record["id"]:
            by_key[("post", record["id"])] = record
    for item in list(comments or ()):
        record = _record(item, "comment", subreddit, run_id)
        if record["id"]:
            by_key[("comment", record["id"])] = record
    items = sorted(by_key.values(), key=lambda row: float(row.get("createdUtc") or 0), reverse=True)
    result = {
        "version": 1,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "subreddit": subreddit,
        "items": items[:MAX_ACTIVITY_ITEMS],
    }
    candidate = activity_path(path)
    if not safe_json_save(str(candidate), result):
        raise OSError(f"Could not save local activity snapshot: {candidate}")
    return result


def write_cycle_summary(summary: dict[str, Any], path: str | Path | None = None) -> None:
    """Write count-only cycle evidence for the local worker."""
    result = dict(summary)
    result.setdefault("version", 1)
    result["writtenAt"] = datetime.now(timezone.utc).isoformat()
    candidate = cycle_summary_path(path)
    if not safe_json_save(str(candidate), result):
        raise OSError(f"Could not save cycle summary: {candidate}")


def read_cycle_summary(path: str | Path | None = None) -> dict[str, Any]:
    candidate = cycle_summary_path(path)
    loaded = safe_json_load(str(candidate), {})
    return dict(loaded) if isinstance(loaded, dict) else {}


__all__ = [
    "MAX_ACTIVITY_ITEMS",
    "activity_path",
    "cycle_summary_path",
    "read_activity_snapshot",
    "read_cycle_summary",
    "record_cycle_activity",
    "write_cycle_summary",
]
