"""Tests for crosspost_handler.py."""
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
from datetime import date

import config

from crosspost_handler import (
    check_and_crosspost,
    select_post_for_crossposting,
    is_already_crossposted,
    get_todays_schedule,
    fetch_candidate_posts,
    perform_crosspost,
)


class TestIsAlreadyCrossposted(unittest.TestCase):
    def test_detects_crosspost_by_url(self):
        submission = MagicMock()
        submission.id = "abc123"
        submission.permalink = "/r/example_community/comments/abc123/test/"
        submission.url = "https://example.com/article"

        existing_urls = {"https://example.com/article"}
        history = []

        self.assertTrue(is_already_crossposted(submission, existing_urls, history))

    def test_detects_crosspost_by_history(self):
        submission = MagicMock()
        submission.id = "abc123"
        submission.permalink = "/r/example_community/comments/abc123/test/"
        submission.url = "https://example.com/article"

        existing_urls = set()
        history = [{"source_post_id": "abc123"}]

        self.assertTrue(is_already_crossposted(submission, existing_urls, history))

    def test_not_crossposted(self):
        submission = MagicMock()
        submission.id = "abc123"
        submission.permalink = "/r/example_community/comments/abc123/test/"
        submission.url = "https://example.com/article"

        existing_urls = set()
        history = []

        self.assertFalse(is_already_crossposted(submission, existing_urls, history))


class TestGetTodaysSchedule(unittest.TestCase):
    def test_new_day_sets_schedule(self):
        state = {"crosspost": {}}
        with patch("crosspost_handler.CROSSPOST_TIME_WINDOW_UTC", (8, 16)), patch(
            "crosspost_handler.random.randint", return_value=12
        ):
            scheduled_hour, should_skip = get_todays_schedule(state)

        self.assertIsNotNone(state["crosspost"].get("scheduled_date"))
        self.assertFalse(should_skip)

    def test_returns_cached_schedule(self):
        today = date.today().isoformat()
        state = {
            "crosspost": {
                "scheduled_date": today,
                "scheduled_hour": 3,
                "skip_today": False,
            }
        }
        with patch("crosspost_handler.CROSSPOST_TIME_WINDOW_UTC", (3, 9)):
            scheduled_hour, should_skip = get_todays_schedule(state)

        self.assertEqual(scheduled_hour, 3)
        self.assertFalse(should_skip)


class TestPerformCrosspost(unittest.TestCase):
    def test_created_crosspost_is_success_even_if_followup_comment_fails(self):
        submission = MagicMock()
        crosspost = MagicMock()
        crosspost.permalink = "/r/example_target/comments/target/posted/"
        crosspost.reply.side_effect = RuntimeError("temporary reply failure")
        submission.crosspost.return_value = crosspost

        success, result = perform_crosspost(
            submission,
            MagicMock(),
            "Title",
            "Comment",
        )

        self.assertTrue(success)
        self.assertEqual(result, "https://reddit.com/r/example_target/comments/target/posted/")
        submission.crosspost.assert_called_once()


class TestCrosspostSelection(unittest.TestCase):
    def _client(self, answer):
        client = MagicMock()
        client.ready = True
        client.calls_used = 0
        client.max_calls = 1
        client.last_usage = {"input_tokens": 16, "cost": 0.0002}
        client.decide_many.return_value = answer
        return client

    def test_selection_uses_operator_guidance_and_configured_cutoff(self):
        client = self._client({"crosspost_operator_criteria": 0.8})
        submission = MagicMock(title="Community garden update", selftext="Details")
        with patch.object(config, "CROSSPOST_SELECTION_THRESHOLD_PERCENT", 70):
            selected, token_info = select_post_for_crossposting(
                submission,
                jev_client=client,
                guidance="Select posts announcing local community events.",
            )

        self.assertTrue(selected)
        self.assertEqual(
            token_info,
            {"input_tokens": 16, "output_tokens": 0, "total_tokens": 16, "cost": 0.0002},
        )
        client.decide_many.assert_called_once()
        args = client.decide_many.call_args.kwargs
        self.assertEqual(args["state"]["operator_crosspost_guidance"], "Select posts announcing local community events.")
        self.assertEqual(args["state"]["post_title"], "Community garden update")
        self.assertEqual(list(args["questions"]), ["crosspost_operator_criteria"])
        question = args["questions"]["crosspost_operator_criteria"]
        self.assertEqual(question["type"], "noul")
        self.assertEqual(set(question["criteria"]), {"true", "false"})

    def test_selection_uses_operator_cutoff_and_includes_boundary(self):
        client = self._client({"crosspost_operator_criteria": 0.5})
        with patch.object(config, "CROSSPOST_SELECTION_THRESHOLD_PERCENT", 50):
            selected, _ = select_post_for_crossposting(
                MagicMock(title="Example", selftext=""),
                jev_client=client,
                guidance="Choose posts about a topic supplied by the operator.",
            )
        self.assertTrue(selected)

    def test_below_cutoff_is_not_selected(self):
        client = self._client({"crosspost_operator_criteria": 0.2})
        with patch.object(config, "CROSSPOST_SELECTION_THRESHOLD_PERCENT", 70):
            selected, _ = select_post_for_crossposting(
                MagicMock(title="Example", selftext=""),
                jev_client=client,
                guidance="Choose posts about a topic supplied by the operator.",
            )
        self.assertFalse(selected)
        client.decide_many.assert_called_once()

    def test_missing_or_invalid_jev_answer_fails_closed_but_keeps_request_cost(self):
        for answer in (
            {},
            {"crosspost_operator_criteria": float("nan")},
            {"crosspost_operator_criteria": 1.1},
        ):
            with self.subTest(answer=answer):
                client = self._client(answer)
                with patch.object(config, "CROSSPOST_SELECTION_THRESHOLD_PERCENT", 70):
                    selected, token_info = select_post_for_crossposting(
                        MagicMock(title="Unclear", selftext=""),
                        jev_client=client,
                        guidance="Use only the operator's selection criteria.",
                    )
                self.assertFalse(selected)
                self.assertEqual(token_info["total_tokens"], 16)
                self.assertEqual(token_info["cost"], 0.0002)
                client.decide_many.assert_called_once()

    def test_missing_policy_values_skip_before_provider_request(self):
        client = self._client({"crosspost_operator_criteria": 0.99})
        with patch.object(config, "CROSSPOST_SELECTION_THRESHOLD_PERCENT", None):
            selected, token_info = select_post_for_crossposting(
                MagicMock(title="Example", selftext="Details"),
                jev_client=client,
                guidance="Choose suitable posts.",
            )
        self.assertFalse(selected)
        self.assertEqual(token_info["total_tokens"], 0)
        client.decide_many.assert_not_called()

        with patch.object(config, "CROSSPOST_SELECTION_THRESHOLD_PERCENT", 50):
            selected, _ = select_post_for_crossposting(
                MagicMock(title="Example", selftext="Details"),
                jev_client=client,
                guidance="",
            )
        self.assertFalse(selected)
        client.decide_many.assert_not_called()

    def test_exhausted_jev_budget_skips_without_a_request(self):
        client = self._client({"crosspost_operator_criteria": 0.99})
        client.calls_used = 1
        with patch.object(config, "CROSSPOST_SELECTION_THRESHOLD_PERCENT", 50):
            selected, token_info = select_post_for_crossposting(
                MagicMock(title="Example", selftext="Details"),
                jev_client=client,
                guidance="Choose suitable posts.",
            )
        self.assertFalse(selected)
        self.assertEqual(token_info["total_tokens"], 0)
        client.decide_many.assert_not_called()

    def test_missing_jev_client_skips(self):
        with patch.object(config, "CROSSPOST_SELECTION_THRESHOLD_PERCENT", 50):
            selected, token_info = select_post_for_crossposting(
                MagicMock(title="Example", selftext="Details"),
                guidance="Choose suitable posts.",
            )
        self.assertFalse(selected)
        self.assertEqual(token_info["total_tokens"], 0)

    def test_title_and_comment_generation_reserve_reasoning_output_budget(self):
        from crosspost_handler import enhance_title, generate_crosspost_comment

        model = MagicMock()
        model.generate_content.side_effect = [
            MagicMock(
                text="A clearer title",
                usage_metadata=MagicMock(prompt_token_count=10, candidates_token_count=6),
            ),
            MagicMock(
                text="A concise comment",
                usage_metadata=MagicMock(prompt_token_count=10, candidates_token_count=6),
            ),
        ]

        title, _ = enhance_title("Original title", model)
        comment, _ = generate_crosspost_comment("Title", "Content", model)

        self.assertEqual(title, "A clearer title")
        self.assertEqual(comment, "Optimist Prime bot: A concise comment")
        configs = [call.kwargs["generation_config"] for call in model.generate_content.call_args_list]
        self.assertEqual([item["max_output_tokens"] for item in configs], [512, 512])


class TestCrosspostPreview(unittest.TestCase):
    def test_preview_can_run_while_feature_switch_is_off(self):
        state = {}
        with patch.object(config, "CROSSPOST_ENABLED", False), \
             patch("crosspost_handler.is_time_to_crosspost", return_value=False):
            result = check_and_crosspost(
                MagicMock(),
                MagicMock(),
                state,
                dry_run=True,
                preview=True,
            )

        self.assertEqual(result[:3], (0, 0, 0.0))
        self.assertEqual(result[3], state)

    def test_normal_cycle_does_not_fetch_when_crosspost_is_off(self):
        reddit = MagicMock()
        with patch.object(config, "CROSSPOST_ENABLED", False):
            result = check_and_crosspost(reddit, MagicMock(), {}, dry_run=True)

        self.assertEqual(result[:3], (0, 0, 0.0))
        reddit.subreddit.assert_not_called()

    def test_enabled_crossposting_requires_operator_limits_before_reddit_requests(self):
        reddit = MagicMock()
        with patch.object(config, "CROSSPOST_ENABLED", True), \
             patch.object(config, "CROSSPOST_SELECTION_GUIDANCE", "Choose community updates."), \
             patch("crosspost_handler.CROSSPOST_SOURCE_SUB", "example_source"), \
             patch("crosspost_handler.CROSSPOST_TARGET_SUB", "example_target"), \
             patch.object(config, "CROSSPOST_SELECTION_THRESHOLD_PERCENT", None), \
             patch("crosspost_handler.CROSSPOST_MAX_PER_DAY", None), \
             patch("crosspost_handler.CROSSPOST_MIN_HOURS_OLD", None), \
             patch("crosspost_handler.CROSSPOST_MAX_HOURS_OLD", None), \
             patch("crosspost_handler.CROSSPOST_TIME_WINDOW_UTC", (None, None)), \
             patch("crosspost_handler.CROSSPOST_LOOKBACK_DAYS", None):
            result = check_and_crosspost(reddit, MagicMock(), {}, dry_run=True)

        self.assertEqual(result[:3], (0, 0, 0.0))
        reddit.subreddit.assert_not_called()

    def test_candidate_fetch_uses_a_time_filter_that_covers_the_configured_window(self):
        source = MagicMock()
        source.top.return_value = []
        with patch("crosspost_handler.CROSSPOST_MIN_HOURS_OLD", 1), \
             patch("crosspost_handler.CROSSPOST_MAX_HOURS_OLD", 120):
            candidates = fetch_candidate_posts(source)

        self.assertEqual(candidates, [])
        source.top.assert_called_once_with(time_filter="week", limit=50)


if __name__ == "__main__":
    unittest.main()
