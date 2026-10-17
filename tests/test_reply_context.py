"""Tests for avoiding extra Reddit reads when resolving short flair follow-ups."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from reply_handler import _parent_flair_context


class TestParentRequestContext(unittest.TestCase):
    def test_uses_same_cycle_parent_without_reddit_fetch(self):
        parent = SimpleNamespace(
            id="parent-comment",
            body="Your custom flair is ready. Ask me to set it for you.",
            author=SimpleNamespace(name="example_bot"),
        )
        reply = SimpleNamespace(
            id="follow-up",
            parent_id="t1_parent-comment",
            body="Yes please, I'd like that too",
            parent=Mock(side_effect=AssertionError("must use captured parent")),
        )

        context = _parent_flair_context(
            reply,
            "example_bot",
            recent_comments=[parent],
        )

        self.assertIn("custom flair", context)
        reply.parent.assert_not_called()

    def test_context_is_not_limited_to_a_known_follow_up_phrase(self):
        parent = SimpleNamespace(
            id="parent-comment",
            body="Your custom flair is available if you want it set.",
            author=SimpleNamespace(name="example_bot"),
        )
        reply = SimpleNamespace(
            parent_id="t1_parent-comment",
            body="That sounds great for me, thanks!",
            parent=Mock(side_effect=AssertionError("must use captured parent")),
        )

        context = _parent_flair_context(
            reply,
            "example_bot",
            recent_comments=[parent],
        )

        self.assertIn("Your custom flair is available if you want it set", context)
        reply.parent.assert_not_called()

    def test_non_bot_parent_is_not_used_as_bot_flair_context(self):
        parent = SimpleNamespace(
            id="parent-comment",
            body="Do me next",
            author=SimpleNamespace(name="ordinary_user"),
        )
        reply = SimpleNamespace(
            parent_id="t1_test000004-comment",
            body="Do me next",
            parent=Mock(side_effect=AssertionError("captured parent is sufficient")),
        )

        self.assertIsNone(_parent_flair_context(
            reply,
            "example_bot",
            recent_comments=[parent],
        ))
        reply.parent.assert_not_called()

    def test_uncaptured_parent_never_causes_an_extra_reddit_fetch(self):
        reply = SimpleNamespace(
            parent_id="t1_test000001-parent",
            body="Could you set mine up too?",
            parent=Mock(side_effect=AssertionError("must not hydrate parent")),
        )

        self.assertIsNone(_parent_flair_context(
            reply,
            "example_bot",
            recent_comments=[],
        ))
        reply.parent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
