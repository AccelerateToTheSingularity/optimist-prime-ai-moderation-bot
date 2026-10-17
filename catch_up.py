"""Persistent, bounded catch-up scanning for subreddit posts and comments.

The local worker normally runs every few minutes, but the operator's PC can be
off for much longer.  A plain ``new(limit=...)`` scan silently loses work when
that happens.  This module keeps only listing frontiers and retry descriptors
in runtime state, then uses Reddit's listing pagination to walk older items on
later cycles.

No item body, title, URL, username, or model output is written to this state.
The normal runner still decides which enabled feature may act on each item.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import config


_KIND_PREFIXES = {"posts": "t3_", "comments": "t1_"}


@dataclass(frozen=True)
class CatchUpResult:
    """The bounded work returned to the runner for one content type."""

    kind: str
    items: tuple[Any, ...]
    fetched: int
    pages: int
    complete: bool
    oldest_seen_utc: float | None
    newest_seen_utc: float | None
    rehydrated: int = 0
    error: str | None = None
    incomplete_gap: dict | None = None


def _now_utc() -> float:
    return datetime.now(timezone.utc).timestamp()


def _ensure_bucket(state: dict, kind: str) -> dict:
    if kind not in _KIND_PREFIXES:
        raise ValueError(f"Unsupported catch-up kind: {kind}")
    buckets = state.setdefault("catch_up", {})
    if not isinstance(buckets, dict):
        buckets = {}
        state["catch_up"] = buckets
    bucket = buckets.setdefault(kind, {})
    if not isinstance(bucket, dict):
        bucket = {}
        buckets[kind] = bucket
    bucket.setdefault("cursor_fullname", None)
    bucket.setdefault("cursor_id", None)
    bucket.setdefault("cursor_created_utc", None)
    bucket.setdefault("newest_fullname", None)
    bucket.setdefault("newest_created_utc", None)
    bucket.setdefault("backfill_after_fullname", None)
    bucket.setdefault("backfill_target_fullname", None)
    bucket.setdefault("backfill_queue", [])
    bucket.setdefault("last_scan_at", None)
    bucket.setdefault("last_successful_scan_at", None)
    bucket.setdefault("last_scan_count", 0)
    bucket.setdefault("last_scan_pages", 0)
    bucket.setdefault("last_scan_complete", True)
    bucket.setdefault("last_error", None)
    bucket.setdefault("incomplete_gap", None)
    pending = bucket.setdefault("pending", [])
    if not isinstance(pending, list):
        bucket["pending"] = []
    return bucket


def ensure_catch_up_state(state: dict) -> dict:
    """Migrate an existing state object without removing unknown fields."""
    _ensure_bucket(state, "posts")
    _ensure_bucket(state, "comments")
    return state


def _item_id(item: Any) -> str | None:
    value = getattr(item, "id", None)
    return str(value).strip() if value is not None and str(value).strip() else None


def _item_fullname(item: Any, kind: str) -> str | None:
    fullname = getattr(item, "fullname", None)
    if fullname:
        return str(fullname).strip()
    item_id = _item_id(item)
    if not item_id:
        return None
    return f"{_KIND_PREFIXES[kind]}{item_id}"


def _item_created_utc(item: Any) -> float | None:
    try:
        value = float(getattr(item, "created_utc", 0) or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _item_sort_key(item: Any) -> tuple[float, str]:
    return (_item_created_utc(item) or 0.0, _item_id(item) or "")


def _listing_source(subreddit: Any, kind: str):
    return getattr(subreddit, "new" if kind == "posts" else "comments")


def _fetch_listing(subreddit: Any, kind: str, limit: int, params: dict | None = None) -> list[Any]:
    """Fetch one bounded listing request using PRAW's supported pagination."""
    source = _listing_source(subreddit, kind)
    if params:
        listing = source(limit=limit, params=params)
    else:
        listing = source(limit=limit)
    return list(listing)


def _rehydrate_pending(
    subreddit: Any,
    kind: str,
    bucket: dict,
    reddit: Any | None,
) -> tuple[list[Any], int, str | None]:
    """Re-fetch retryable IDs without storing their content in state."""
    pending = [
        entry for entry in bucket.get("pending", [])
        if isinstance(entry, dict) and entry.get("id")
    ]
    if not pending or reddit is None or not hasattr(reddit, "info"):
        return [], 0, None

    fullnames = [str(entry.get("fullname") or "").strip() for entry in pending]
    fullnames = [value for value in fullnames if value]
    if not fullnames:
        return [], 0, None
    try:
        objects = list(reddit.info(fullnames=fullnames[:200]))
    except Exception as exc:
        return [], 0, f"retry rehydration failed: {type(exc).__name__}"

    expected_prefix = _KIND_PREFIXES[kind]
    items: list[Any] = []
    for item in objects:
        fullname = _item_fullname(item, kind)
        if fullname and fullname.startswith(expected_prefix):
            items.append(item)
    return items, len(items), None


def _append_unique(items: list[Any], seen_ids: set[str], item: Any, kind: str) -> None:
    item_id = _item_id(item)
    if not item_id or item_id in seen_ids:
        return
    if not _item_fullname(item, kind):
        return
    seen_ids.add(item_id)
    items.append(item)


def _append_unique_bounded(
    items: list[Any], seen_ids: set[str], item: Any, kind: str, limit: int
) -> None:
    """Respect the one-cycle work cap while keeping retry items first."""
    if len(items) >= limit:
        return
    _append_unique(items, seen_ids, item, kind)


def _set_frontier(bucket: dict, item: Any, kind: str) -> None:
    fullname = _item_fullname(item, kind)
    item_id = _item_id(item)
    created = _item_created_utc(item)
    if fullname:
        bucket["cursor_fullname"] = fullname
    if item_id:
        bucket["cursor_id"] = item_id
    if created is not None:
        bucket["cursor_created_utc"] = created


def _oldest_item(items: Iterable[Any]) -> Any | None:
    values = [item for item in items if _item_created_utc(item) is not None]
    return min(values, key=_item_sort_key) if values else None


def _newest_item(items: Iterable[Any]) -> Any | None:
    values = [item for item in items if _item_created_utc(item) is not None]
    return max(values, key=_item_sort_key) if values else None


def _update_pending_descriptor(bucket: dict, item: Any, kind: str, reason: str | None) -> None:
    item_id = _item_id(item)
    if not item_id:
        return
    fullname = _item_fullname(item, kind)
    created = _item_created_utc(item)
    pending = [
        entry for entry in bucket.get("pending", [])
        if isinstance(entry, dict) and str(entry.get("id")) != item_id
    ]
    previous = next(
        (
            entry for entry in bucket.get("pending", [])
            if isinstance(entry, dict) and str(entry.get("id")) == item_id
        ),
        {},
    )
    descriptor = {
        "id": item_id,
        "fullname": fullname,
        "created_utc": created,
        "attempts": int(previous.get("attempts", 0) or 0) + 1,
        "last_error": str(reason or "retryable failure")[:240],
        "last_attempt_at": _now_utc(),
    }
    pending.append(descriptor)
    bucket["pending"] = pending[-500:]


def remember_retryable_item(state: dict, kind: str, item: Any, reason: str | None = None) -> None:
    """Remember only enough identity to retry a failed Reddit operation."""
    bucket = _ensure_bucket(state, kind)
    _update_pending_descriptor(bucket, item, kind, reason)


def clear_completed_retry_items(state: dict, kind: str, completed_ids: Iterable[str]) -> None:
    """Remove retry descriptors only after the caller confirms success."""
    bucket = _ensure_bucket(state, kind)
    completed = {str(value) for value in completed_ids if value is not None}
    if not completed:
        return
    bucket["pending"] = [
        entry for entry in bucket.get("pending", [])
        if not isinstance(entry, dict) or str(entry.get("id")) not in completed
    ]


def _normalise_backfill_queue(bucket: dict) -> list[dict]:
    """Migrate the first-generation single-backlog fields to a queue."""
    raw_queue = bucket.get("backfill_queue")
    queue = [
        entry for entry in (raw_queue if isinstance(raw_queue, list) else [])
        if isinstance(entry, dict) and entry.get("after_fullname")
    ]
    if not queue and bucket.get("backfill_after_fullname"):
        queue = [{
            "after_fullname": bucket.get("backfill_after_fullname"),
            "target_fullname": bucket.get("backfill_target_fullname"),
        }]
    bucket["backfill_queue"] = queue
    return queue


def collect_catch_up_items(
    subreddit: Any,
    kind: str,
    state: dict,
    limit: int,
    *,
    reddit: Any | None = None,
    now_utc: float | None = None,
) -> CatchUpResult:
    """Return bounded recent work and continue an older backlog safely.

    The current listing is checked for items newer than the newest cursor.  A
    bounded ``after`` page then advances one or more older backfill streams.
    This means new activity and an outage backlog can make progress in the
    same cycle without one starving the other.  A backlog is retained in state
    if this cycle runs out of its item budget.
    """
    bucket = _ensure_bucket(state, kind)
    limit = max(1, int(limit))
    now = float(now_utc if now_utc is not None else _now_utc())
    cutoff = now - (max(1, int(config.MAX_AGE_HOURS)) * 3600)

    pending_items, rehydrated, rehydrate_error = _rehydrate_pending(
        subreddit, kind, bucket, reddit
    )
    items: list[Any] = []
    seen_ids: set[str] = set()
    for item in pending_items:
        _append_unique_bounded(items, seen_ids, item, kind, limit)

    fetched = 0
    pages = 0
    listing_items: list[Any] = []
    queue = _normalise_backfill_queue(bucket)
    previous_cursor = bucket.get("cursor_fullname")
    previous_newest = bucket.get("newest_fullname")
    gap_detected: dict | None = None

    try:
        # ``before`` asks Reddit for entries newer than the last newest item;
        # with no cursor it establishes the initial bounded baseline.
        latest = _fetch_listing(
            subreddit,
            kind,
            limit,
            params={"before": previous_newest} if previous_newest else None,
        )
        pages += 1
        fetched += len(latest)
        listing_items.extend(latest)

        # Reddit can return a successful empty page when a saved fullname is
        # older than the listing positions it still exposes.  An empty page
        # also means a genuinely quiet subreddit, so verify it against one
        # bounded newest-page request.  If the saved anchor is still present,
        # this was simply a quiet interval.  Otherwise, recover what Reddit
        # still exposes and either page back toward the anchor or record the
        # missing interval explicitly.
        if previous_newest and not latest:
            newest_page = _fetch_listing(subreddit, kind, limit)
            pages += 1
            fetched += len(newest_page)
            listing_items.extend(newest_page)
            anchor_visible = any(
                _item_fullname(item, kind) == previous_newest
                for item in newest_page
            )
            if anchor_visible:
                latest = []
            else:
                latest = newest_page
                newest_oldest = _oldest_item(newest_page)
                oldest_created = _item_created_utc(newest_oldest) if newest_oldest else None
                saved_created = bucket.get("newest_created_utc") or bucket.get("cursor_created_utc")
                if newest_page and oldest_created is not None and saved_created is not None:
                    if float(saved_created) < cutoff:
                        gap_detected = {
                            "fromUtc": float(saved_created),
                            "toUtc": oldest_created,
                            "reason": "The saved position is older than the configured processing window.",
                        }
                    else:
                        candidate = {
                            "after_fullname": _item_fullname(newest_oldest, kind),
                            "target_fullname": previous_newest,
                        }
                        if candidate["after_fullname"] and not any(
                            entry.get("after_fullname") == candidate["after_fullname"]
                            and entry.get("target_fullname") == candidate["target_fullname"]
                            for entry in queue
                        ):
                            queue.insert(0, candidate)
                else:
                    gap_detected = {
                        "fromUtc": float(saved_created) if saved_created is not None else None,
                        "toUtc": oldest_created,
                        "reason": "Reddit no longer returned the saved position or any content around it.",
                    }

        latest_eligible = [
            item for item in latest
            if (_item_created_utc(item) or 0) >= cutoff
        ]
        for item in latest_eligible:
            _append_unique_bounded(items, seen_ids, item, kind, limit)

        latest_newest = _newest_item(latest)

        # A full initial page means there may be older in-window content.  A
        # full ``before`` page means more new content may have arrived between
        # cycles; that stream targets the previous newest cursor.
        latest_oldest = _oldest_item(latest)
        if latest_oldest is not None and (_item_created_utc(latest_oldest) or 0) >= cutoff:
            if not previous_newest and len(latest) >= limit:
                candidate = {
                    "after_fullname": _item_fullname(latest_oldest, kind),
                    "target_fullname": None,
                }
                if not any(entry["after_fullname"] == candidate["after_fullname"] for entry in queue):
                    queue.append(candidate)
            elif previous_newest and len(latest) >= limit and not gap_detected:
                candidate = {
                    "after_fullname": _item_fullname(latest_oldest, kind),
                    "target_fullname": previous_newest,
                }
                if not any(
                    entry["after_fullname"] == candidate["after_fullname"]
                    and entry.get("target_fullname") == candidate["target_fullname"]
                    for entry in queue
                ):
                    queue.insert(0, candidate)

        remaining = max(0, limit - len(items))
        older_seen: list[Any] = []
        while queue and remaining:
            stream = queue[0]
            older = _fetch_listing(
                subreddit,
                kind,
                remaining,
                params={"after": stream["after_fullname"]},
            )
            pages += 1
            fetched += len(older)
            listing_items.extend(older)
            older_seen.extend(older)
            for item in older:
                created = _item_created_utc(item)
                if created is not None and created >= cutoff:
                    _append_unique_bounded(items, seen_ids, item, kind, limit)

            target_frontier = stream.get("target_fullname")
            reached_target = bool(
                target_frontier
                and any(_item_fullname(item, kind) == target_frontier for item in older)
            )
            reached_cutoff = any(
                (_item_created_utc(item) or 0) < cutoff for item in older
            )
            older_oldest = _oldest_item(older)
            if stream.get("target_fullname") and not reached_target and (
                not older or (len(older) < remaining and not reached_cutoff)
            ):
                gap_detected = {
                    "fromUtc": bucket.get("newest_created_utc") or bucket.get("cursor_created_utc"),
                    "toUtc": _item_created_utc(older_oldest) if older_oldest else None,
                    "reason": "Reddit stopped returning pages before the saved position was reached.",
                }
            elif stream.get("target_fullname") and reached_cutoff and not reached_target:
                gap_detected = {
                    "fromUtc": bucket.get("newest_created_utc") or bucket.get("cursor_created_utc"),
                    "toUtc": _item_created_utc(older_oldest) if older_oldest else cutoff,
                    "reason": "The saved position could not be reached before the processing window expired.",
                }
            if reached_target or reached_cutoff or len(older) < remaining or older_oldest is None:
                queue.pop(0)
            else:
                next_after = _item_fullname(older_oldest, kind)
                if next_after == stream["after_fullname"]:
                    queue.pop(0)
                else:
                    stream["after_fullname"] = next_after
            remaining -= len(older)
            if not older:
                break

        bucket["backfill_queue"] = queue
        bucket["backfill_after_fullname"] = (
            queue[0].get("after_fullname") if queue else None
        )
        bucket["backfill_target_fullname"] = (
            queue[0].get("target_fullname") if queue else None
        )

        # ``cursor_fullname`` is the oldest contiguous item reached by the
        # scan.  New-only pages do not move it while an older stream is still
        # pending, so an outage gap cannot be skipped.
        frontier_candidates = [
            item for item in older_seen if (_item_created_utc(item) or 0) >= cutoff
        ]
        if not frontier_candidates and not queue:
            frontier_candidates = latest_eligible
        if not frontier_candidates and not previous_cursor and latest_eligible:
            frontier_candidates = latest_eligible
        frontier = _oldest_item(frontier_candidates)
        if frontier is not None:
            _set_frontier(bucket, frontier, kind)

        if latest_newest is not None:
            bucket["newest_fullname"] = _item_fullname(latest_newest, kind)
            bucket["newest_created_utc"] = _item_created_utc(latest_newest)

        bucket["last_scan_at"] = now
        bucket["last_successful_scan_at"] = now
        bucket["last_scan_count"] = fetched
        bucket["last_scan_pages"] = pages
        if gap_detected:
            gap_detected["detectedAtUtc"] = now
            bucket["incomplete_gap"] = gap_detected
        scan_complete = not bool(queue) and not bool(gap_detected)
        bucket["last_scan_complete"] = scan_complete
        # An inaccessible old interval is recorded separately from a failed
        # listing. Consumers can still process the recent items Reddit did
        # return; they must not mistake the gap for a failed current scan.
        bucket["last_error"] = rehydrate_error
        items.sort(key=_item_sort_key, reverse=True)
        return CatchUpResult(
            kind=kind,
            items=tuple(items),
            fetched=fetched,
            pages=pages,
            complete=scan_complete,
            oldest_seen_utc=min(
                (_item_created_utc(item) for item in listing_items if _item_created_utc(item) is not None),
                default=None,
            ),
            newest_seen_utc=max(
                (_item_created_utc(item) for item in listing_items if _item_created_utc(item) is not None),
                default=None,
            ),
            rehydrated=rehydrated,
            error=bucket["last_error"],
            incomplete_gap=gap_detected,
        )
    except Exception as exc:
        # Keep the old cursor/backlog untouched.  A failed listing is a
        # retryable operational failure, not proof that the backlog is empty.
        bucket["last_scan_at"] = now
        bucket["last_scan_pages"] = pages
        bucket["last_scan_count"] = fetched
        bucket["last_scan_complete"] = False
        bucket["last_error"] = f"listing failed: {type(exc).__name__}"
        items.sort(key=_item_sort_key, reverse=True)
        error = bucket["last_error"]
        if rehydrate_error:
            error = f"{error}; {rehydrate_error}"
        return CatchUpResult(
            kind=kind,
            items=tuple(items),
            fetched=fetched,
            pages=pages,
            complete=False,
            oldest_seen_utc=None,
            newest_seen_utc=None,
            rehydrated=rehydrated,
            error=error,
            incomplete_gap=gap_detected,
        )


def catch_up_summary(state: dict) -> dict:
    """Return diagnostics-safe catch-up status without content or usernames."""
    ensure_catch_up_state(state)
    result: dict[str, dict] = {}
    for kind in _KIND_PREFIXES:
        bucket = state["catch_up"][kind]
        result[kind] = {
            "cursorCreatedUtc": bucket.get("cursor_created_utc"),
            "newestCreatedUtc": bucket.get("newest_created_utc"),
            "lastScanAt": bucket.get("last_scan_at"),
            "lastSuccessfulScanAt": bucket.get("last_successful_scan_at"),
            "lastScanCount": bucket.get("last_scan_count", 0),
            "lastScanPages": bucket.get("last_scan_pages", 0),
            "lastScanComplete": bool(bucket.get("last_scan_complete", True)),
            "incompleteGap": bucket.get("incomplete_gap"),
            "pendingRetryCount": len(bucket.get("pending", [])),
            "lastError": bucket.get("last_error"),
        }
    return result
