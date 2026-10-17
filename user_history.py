"""Shared, bounded Reddit history collection.

The collector is deliberately independent of moderation decisions.  It only
retrieves public Reddit comments and submissions, applies the configured
retention and item limits, and returns small structured objects that callers
can evaluate without having to duplicate PRAW pagination logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable


@dataclass(frozen=True)
class HistoryItem:
    """A bounded representation of one public Reddit item."""

    item_id: str
    kind: str  # "comment" or "post"
    subreddit: str
    created_utc: float
    score: int | None
    permalink: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UserHistory:
    """Collected history plus enough metadata for diagnostics and messages."""

    username: str
    items: tuple[HistoryItem, ...]
    comments_seen: int
    posts_seen: int
    truncated: bool = False
    errors: tuple[str, ...] = ()
    comments_scan_limited: bool = False
    posts_scan_limited: bool = False
    comments_cap_reached: bool = False
    posts_cap_reached: bool = False
    duplicate_items_skipped: int = 0

    @property
    def comments(self) -> tuple[HistoryItem, ...]:
        return tuple(item for item in self.items if item.kind == "comment")

    @property
    def posts(self) -> tuple[HistoryItem, ...]:
        return tuple(item for item in self.items if item.kind == "post")

    def to_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "items": [item.to_dict() for item in self.items],
            "comments_seen": self.comments_seen,
            "posts_seen": self.posts_seen,
            "truncated": self.truncated,
            "errors": list(self.errors),
            "comments_scan_limited": self.comments_scan_limited,
            "posts_scan_limited": self.posts_scan_limited,
            "comments_cap_reached": self.comments_cap_reached,
            "posts_cap_reached": self.posts_cap_reached,
            "duplicate_items_skipped": self.duplicate_items_skipped,
        }


def _display_name(value: Any, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    return str(getattr(value, "display_name", getattr(value, "name", value)))


def _item_subreddit_name(value: Any) -> str:
    """Read the subreddit name from listing data without hydrating its object.

    PRAW can turn ``item.subreddit`` into a lazy Subreddit object.  Asking that
    object for ``display_name`` may issue an extra ``/about`` request for every
    history item, which is both slow and unnecessary because Reddit listings
    already include the name in ``subreddit_name_prefixed`` or
    ``subreddit_name``.
    """
    for attribute in ("subreddit_name_prefixed", "subreddit_name"):
        raw_name = getattr(value, attribute, None)
        if raw_name:
            name = str(raw_name).strip()
            return name[2:] if name.lower().startswith("r/") else name

    # Do not read ``value.subreddit`` directly here: PRAW may lazily fetch the
    # subreddit object when a listing item omits the explicit name fields.
    # The raw listing JSON or an already-cached plain object is sufficient.
    raw_fields = getattr(value, "_fetched", None)
    raw_subreddit = raw_fields.get("subreddit") if isinstance(raw_fields, dict) else None
    if raw_subreddit is None:
        try:
            raw_subreddit = vars(value).get("subreddit")
        except TypeError:
            raw_subreddit = None
    if raw_subreddit is None:
        return "unknown"
    if isinstance(raw_subreddit, dict):
        raw_subreddit = raw_subreddit.get("display_name") or raw_subreddit.get("name") or "unknown"
    return _display_name(raw_subreddit)


def _created_utc(value: Any) -> float | None:
    try:
        timestamp = float(getattr(value, "created_utc"))
    except (AttributeError, TypeError, ValueError):
        return None
    return timestamp


def _permalink(value: Any) -> str:
    link = getattr(value, "permalink", "") or ""
    link = str(link)
    if link.startswith("http://") or link.startswith("https://"):
        return link
    return f"https://www.reddit.com{link}" if link else ""


def _text(value: Any, kind: str) -> str:
    if kind == "comment":
        return str(getattr(value, "body", "") or "")
    title = str(getattr(value, "title", "") or "")
    body = str(getattr(value, "selftext", "") or "")
    return f"Title: {title}\n\n{body}".strip()


def _item_from_praw(value: Any, kind: str, cutoff: float) -> HistoryItem | None:
    text = _text(value, kind).strip()
    if not text or text == "[deleted]":
        return None
    created = _created_utc(value)
    if created is None or created < cutoff:
        return None
    item_id = str(getattr(value, "id", "") or "")
    if not item_id:
        return None
    score_value = getattr(value, "score", None)
    try:
        score = int(score_value) if score_value is not None else None
    except (TypeError, ValueError):
        score = None
    return HistoryItem(
        item_id=item_id,
        kind=kind,
        subreddit=_item_subreddit_name(value),
        created_utc=created,
        score=score,
        permalink=_permalink(value),
        text=text,
    )


def _safe_iter(source: Any) -> Iterable[Any]:
    try:
        return iter(source)
    except TypeError:
        return iter(())


def collect_user_history(
    reddit,
    username: str,
    *,
    lookback_days: int = 365,
    max_comments: int = 100,
    max_posts: int = 50,
    max_items: int | None = 90,
    now_utc: float | None = None,
    allowed_subreddits: frozenset[str] | None = None,
    max_scan_comments: int | None = None,
    max_scan_posts: int | None = None,
    prefer_posts: bool = False,
    excluded_item_ids: frozenset[str] | set[str] | None = None,
    exclude_through_utc: float | None = None,
    max_created_utc: float | None = None,
) -> UserHistory:
    """Collect bounded comments and posts across Reddit.

    Per-type caps count matching, usable items. Optional scan caps bound how
    far through a newest-first account listing the collector may look when
    filtering out communities or already-reviewed items. By default comments
    keep their historical priority. Review callers may prefer posts; then
    posts are collected first, and comments fill only the unused part of
    ``max_items``.
    """

    username = str(username or "").strip()
    now = now_utc if now_utc is not None else datetime.now(timezone.utc).timestamp()
    cutoff = now - max(0, int(lookback_days)) * 86400
    comment_limit = max(0, int(max_comments))
    post_limit = max(0, int(max_posts))
    total_limit = None if max_items is None else max(0, int(max_items))
    excluded = {str(value).strip() for value in (excluded_item_ids or ()) if str(value).strip()}
    older_than = float(exclude_through_utc) if exclude_through_utc is not None else None
    newest_than = float(max_created_utc) if max_created_utc is not None else None
    duplicate_items_skipped = 0

    def is_excluded(raw: Any, kind: str, created: float | None) -> bool:
        nonlocal duplicate_items_skipped
        if created is None:
            return False
        if newest_than is not None and created > newest_than:
            return True
        if older_than is not None and created <= older_than:
            return True
        item_id = str(getattr(raw, "id", "") or "").strip()
        prefix = "t3_" if kind == "post" else "t1_"
        if item_id and prefix + item_id in excluded:
            duplicate_items_skipped += 1
            return True
        return False

    comments: list[HistoryItem] = []
    posts: list[HistoryItem] = []
    comments_seen = 0
    posts_seen = 0
    errors: list[str] = []
    comments_scan_limited = False
    posts_scan_limited = False
    comments_cap_reached = False
    posts_cap_reached = False

    if not username:
        return UserHistory(username="", items=(), comments_seen=0, posts_seen=0)

    redditor = None

    def collect_comments(limit: int) -> None:
        nonlocal redditor, comments_seen, comments_scan_limited, comments_cap_reached
        scan_limit = (
            max(0, int(max_scan_comments))
            if max_scan_comments is not None
            else limit
        )
        if not scan_limit or not limit:
            return
        try:
            if redditor is None:
                redditor = reddit.redditor(username)
            for raw in _safe_iter(redditor.comments.new(limit=scan_limit)):
                comments_seen += 1
                created = _created_utc(raw)
                if created is not None and created < cutoff:
                    break
                if is_excluded(raw, "comment", created):
                    continue
                item = _item_from_praw(raw, "comment", cutoff)
                if item is not None and (
                    allowed_subreddits is None
                    or item.subreddit.lower() in allowed_subreddits
                ):
                    comments.append(item)
                    if len(comments) >= limit:
                        comments_cap_reached = True
                        break
            else:
                comments_scan_limited = bool(
                    comments_seen >= scan_limit and len(comments) < limit
                )
        except Exception as exc:
            comments_seen = max(comments_seen, len(comments))
            errors.append(f"comments: {exc}")

    def collect_posts(limit: int) -> None:
        nonlocal redditor, posts_seen, posts_scan_limited, posts_cap_reached
        scan_limit = (
            max(0, int(max_scan_posts))
            if max_scan_posts is not None
            else limit
        )
        if not scan_limit or not limit:
            return
        try:
            if redditor is None:
                redditor = reddit.redditor(username)
            for raw in _safe_iter(redditor.submissions.new(limit=scan_limit)):
                posts_seen += 1
                created = _created_utc(raw)
                if created is not None and created < cutoff:
                    break
                if is_excluded(raw, "post", created):
                    continue
                item = _item_from_praw(raw, "post", cutoff)
                if item is not None and (
                    allowed_subreddits is None
                    or item.subreddit.lower() in allowed_subreddits
                ):
                    posts.append(item)
                    if len(posts) >= limit:
                        posts_cap_reached = True
                        break
            else:
                posts_scan_limited = bool(
                    posts_seen >= scan_limit and len(posts) < limit
                )
        except Exception as exc:
            posts_seen = max(posts_seen, len(posts))
            errors.append(f"posts: {exc}")

    if prefer_posts:
        post_collection_limit = (
            min(post_limit, total_limit)
            if total_limit is not None
            else post_limit
        )
        collect_posts(post_collection_limit)
        remaining_items = (
            max(0, total_limit - len(posts))
            if total_limit is not None
            else comment_limit
        )
        collect_comments(min(comment_limit, remaining_items))
    else:
        collect_comments(comment_limit)
        collect_posts(post_limit)

    comments.sort(key=lambda item: (-item.created_utc, item.item_id))
    posts.sort(key=lambda item: (-item.created_utc, item.item_id))
    selected = posts + comments if prefer_posts else comments + posts
    truncated = False
    if total_limit is not None and len(selected) > total_limit:
        selected = posts[:total_limit] if prefer_posts else comments[:total_limit]
        if len(selected) < total_limit:
            remainder = total_limit - len(selected)
            selected.extend(
                comments[:remainder] if prefer_posts else posts[:remainder]
            )
        truncated = True
    elif prefer_posts and total_limit is not None and total_limit > 0 and len(selected) == total_limit:
        # Reaching the combined cap means the profile sample may omit other
        # eligible items; no second listing pass is made just to prove that.
        truncated = True

    return UserHistory(
        username=username,
        items=tuple(selected),
        comments_seen=comments_seen,
        posts_seen=posts_seen,
        truncated=truncated,
        errors=tuple(errors),
        comments_scan_limited=comments_scan_limited,
        posts_scan_limited=posts_scan_limited,
        comments_cap_reached=comments_cap_reached,
        posts_cap_reached=posts_cap_reached,
        duplicate_items_skipped=duplicate_items_skipped,
    )


def history_cutoff_timestamp(lookback_days: int, now_utc: float | None = None) -> float:
    """Expose the cutoff calculation for tests and other collectors."""

    now = now_utc if now_utc is not None else datetime.now(timezone.utc).timestamp()
    return (datetime.fromtimestamp(now, tz=timezone.utc) - timedelta(days=max(0, lookback_days))).timestamp()
