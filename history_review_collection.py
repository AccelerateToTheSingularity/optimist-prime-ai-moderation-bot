from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from history_review_ledger import HistoryReviewLedger, item_fullname
from user_history import UserHistory, collect_user_history


@dataclass(frozen=True)
class HistoryReviewCollection:
    history: UserHistory
    already_seen_count: int


def keep_claimed_history(history: UserHistory, claimed: frozenset[str]) -> UserHistory:
    """Limit the AI payload to item IDs atomically claimed by this review."""
    selected = tuple(item for item in history.items if item_fullname(item) in claimed)
    return replace(
        history,
        items=selected,
        truncated=history.truncated or len(selected) < len(history.items),
    )


def collect_unseen_user_history(
    reddit: Any,
    username: str,
    *,
    ledger: HistoryReviewLedger,
    collector=collect_user_history,
    **history_options: Any,
) -> HistoryReviewCollection:
    """Collect the newest history items that have not already been reviewed."""
    target = str(username or "").strip()
    seen = ledger.seen_fullnames(target)
    history = collector(
        reddit,
        target,
        excluded_item_ids=seen,
        **history_options,
    )
    return HistoryReviewCollection(
        history=history,
        already_seen_count=len(seen),
    )
