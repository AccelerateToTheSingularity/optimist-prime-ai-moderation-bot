import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config

from history_scope import load_history_subreddits
from moderation_context import cached_comment_context, listing_index
from moderation_rules import ModerationRule
from ordered_gate import evaluate_first_match
from user_history import collect_user_history


class ContextTests(unittest.TestCase):
    def test_cached_context_never_hydrates(self):
        class Lazy:
            def __getattr__(self, key):
                raise AssertionError("Network hydration attempted")
        comment = Lazy()
        comment.id, comment.parent_id, comment.link_id = "c", "t1_p", "t3_s"
        parent = SimpleNamespace(id="p", body="x" * 2000)
        post = SimpleNamespace(id="s", title="title")
        context = cached_comment_context(comment, listing_index([parent], "t1_"), listing_index([post], "t3_"))
        self.assertEqual(len(context["preceding_comment_other_speaker"]), 1000)
        self.assertEqual(context["post_title"], "title")
        self.assertEqual(cached_comment_context(comment, {}, {}), {})

    def test_parent_quote_cannot_be_used_as_target_evidence(self):
        model = Mock(generate_content=Mock(return_value=SimpleNamespace(text=json.dumps({
            "rule": "r", "reason": "match", "evidence": "parent statement"}))))
        result = evaluate_first_match("I disagree", [ModerationRule("r", "q")], model,
                                      context={"preceding_comment_other_speaker": "parent statement"})
        self.assertIsNone(result)

    def test_filter_before_item_cap_and_no_extra_subreddit_calls(self):
        def item(i, sub):
            return SimpleNamespace(id=i, body="text", subreddit_name_prefixed="r/" + sub,
                                   created_utc=100, score=1, permalink="/example")
        listing = Mock(new=Mock(return_value=[item("1", "example_community"), item("2", "example_other"), item("3", "example_community")]))
        reddit = Mock(redditor=Mock(return_value=SimpleNamespace(comments=listing, submissions=Mock())))
        with patch.object(config, "SUBREDDIT", "example_community"):
            result = collect_user_history(
                reddit,
                "example",
                now_utc=101,
                max_posts=0,
                max_items=2,
                allowed_subreddits=load_history_subreddits(),
            )
        self.assertEqual([i.item_id for i in result.items], ["1", "3"])
        self.assertEqual(result.comments_seen, 3)
        reddit.subreddit.assert_not_called()
        reddit.redditor.return_value.submissions.new.assert_not_called()

    def test_all_local_fixtures_have_two_positive_and_one_negative(self):
        from pathlib import Path
        cases = json.loads((Path(__file__).parent / "fixtures/local_detector_comments.json").read_text())
        gates = {c["gate"] for c in cases}
        self.assertEqual(len(gates), 3)
        for gate in gates:
            self.assertEqual(sorted(c["expected"] for c in cases if c["gate"] == gate), [False, True, True])
