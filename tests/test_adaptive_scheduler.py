"""Tests for the combined-workload local cycle interval calculation."""

from __future__ import annotations

import unittest

from adaptive_scheduler import AdaptiveSchedulerState, choose_next_interval


class TestAdaptiveScheduler(unittest.TestCase):
    def test_below_target_lengthens_by_fifteen_percent(self):
        interval, state = choose_next_interval(
            AdaptiveSchedulerState(last_interval_seconds=180),
            eligible_comments=2,
            maximum_seconds=900,
        )
        self.assertEqual(interval, 207)
        self.assertEqual(state.last_item_count, 2)
        self.assertEqual(state.samples, 1)

    def test_exact_target_keeps_the_current_interval(self):
        interval, state = choose_next_interval(
            AdaptiveSchedulerState(last_interval_seconds=180),
            eligible_posts=1,
            eligible_comments=1,
            private_chat_messages=1,
            maximum_seconds=900,
        )
        self.assertEqual(interval, 180)
        self.assertEqual(state.last_item_count, 3)

    def test_above_target_shortens_by_fifteen_percent(self):
        interval, state = choose_next_interval(
            AdaptiveSchedulerState(last_interval_seconds=180),
            eligible_posts=2,
            eligible_comments=1,
            private_chat_messages=1,
            maximum_seconds=900,
        )
        self.assertEqual(interval, 153)
        self.assertEqual(state.last_item_count, 4)

    def test_repeated_quiet_cycles_reach_configured_maximum(self):
        state = AdaptiveSchedulerState(last_interval_seconds=180)
        for _ in range(100):
            interval, state = choose_next_interval(state, maximum_seconds=900)
        self.assertEqual(interval, 900)

    def test_new_default_interval_is_one_minute(self):
        interval, state = choose_next_interval(AdaptiveSchedulerState())
        self.assertEqual(interval, 60)
        self.assertEqual(state.last_interval_seconds, 60)

    def test_repeated_busy_cycles_reach_two_second_minimum(self):
        state = AdaptiveSchedulerState(last_interval_seconds=180)
        for _ in range(100):
            interval, state = choose_next_interval(state, eligible_comments=4)
        self.assertEqual(interval, 2)

    def test_configured_minimum_cannot_drop_below_two_seconds(self):
        interval, _ = choose_next_interval(
            AdaptiveSchedulerState(last_interval_seconds=2),
            eligible_comments=4,
            minimum_seconds=1,
            maximum_seconds=60,
        )
        self.assertEqual(interval, 2)

    def test_three_second_interval_can_reduce_to_two_second_floor(self):
        interval, _ = choose_next_interval(
            AdaptiveSchedulerState(last_interval_seconds=3),
            eligible_comments=4,
            minimum_seconds=2,
            maximum_seconds=60,
        )
        self.assertEqual(interval, 2)

    def test_configured_bounds_are_enforced(self):
        low, _ = choose_next_interval(
            AdaptiveSchedulerState(last_interval_seconds=16),
            eligible_comments=4,
            minimum_seconds=15,
            maximum_seconds=900,
        )
        high, _ = choose_next_interval(
            AdaptiveSchedulerState(last_interval_seconds=899),
            minimum_seconds=15,
            maximum_seconds=900,
        )
        self.assertEqual(low, 15)
        self.assertEqual(high, 900)

    def test_state_summary_is_count_only(self):
        state = AdaptiveSchedulerState(
            last_interval_seconds=153,
            samples=8,
            last_item_count=4,
            last_eligible_posts=2,
            last_eligible_comments=1,
            last_private_chat_messages=1,
        )
        self.assertEqual(
            state.to_dict(),
            {
                "lastIntervalSeconds": 153,
                "samples": 8,
                "lastItemCount": 4,
                "lastEligiblePosts": 2,
                "lastEligibleComments": 1,
                "lastPrivateChatMessages": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
