"""Pure combined-workload interval calculation for the local worker.

The scheduler aims to process a small, predictable number of distinct work
items each cycle. The shared Reddit request budget remains the hard API safety
boundary; this module only chooses when the next non-overlapping cycle starts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveSchedulerState:
    last_interval_seconds: int = 60
    samples: int = 0
    last_item_count: int = 0
    last_eligible_posts: int = 0
    last_eligible_comments: int = 0
    last_private_chat_messages: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return count-only state safe to include in the cycle summary."""
        return {
            "lastIntervalSeconds": int(self.last_interval_seconds),
            "samples": int(self.samples),
            "lastItemCount": int(self.last_item_count),
            "lastEligiblePosts": int(self.last_eligible_posts),
            "lastEligibleComments": int(self.last_eligible_comments),
            "lastPrivateChatMessages": int(self.last_private_chat_messages),
        }


def choose_next_interval(
    state: AdaptiveSchedulerState,
    *,
    eligible_posts: int = 0,
    eligible_comments: int = 0,
    private_chat_messages: int = 0,
    target_items: int = 3,
    minimum_seconds: int = 2,
    maximum_seconds: int = 60,
) -> tuple[int, AdaptiveSchedulerState]:
    """Adjust the interval by 15% around the target combined item count.

    More work than the target shortens the interval by 15%; less work lengthens
    it by 15%; an exact match preserves it. The result is clamped to the
    configured bounds. Inputs are counts of distinct items already selected by
    the normal Reddit and private-chat handlers, including catch-up items.
    """
    minimum_seconds = max(2, int(minimum_seconds))
    maximum_seconds = max(minimum_seconds, int(maximum_seconds))
    target_items = max(1, int(target_items))
    eligible_posts = max(0, int(eligible_posts))
    eligible_comments = max(0, int(eligible_comments))
    private_chat_messages = max(0, int(private_chat_messages))
    item_count = eligible_posts + eligible_comments + private_chat_messages

    previous_interval = min(
        maximum_seconds,
        max(minimum_seconds, int(state.last_interval_seconds or 60)),
    )
    if item_count > target_items:
        desired = min(
            int(round(previous_interval * 0.85)),
            previous_interval - 1,
        )
    elif item_count < target_items:
        desired = int(round(previous_interval * 1.15))
    else:
        desired = previous_interval
    desired = min(maximum_seconds, max(minimum_seconds, desired))

    return desired, AdaptiveSchedulerState(
        last_interval_seconds=desired,
        samples=max(0, int(state.samples)) + 1,
        last_item_count=item_count,
        last_eligible_posts=eligible_posts,
        last_eligible_comments=eligible_comments,
        last_private_chat_messages=private_chat_messages,
    )


__all__ = ["AdaptiveSchedulerState", "choose_next_interval"]
