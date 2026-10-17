"""Tests for bot_runner.find_bot_comment owned-alias matching."""

import unittest
from unittest.mock import MagicMock, patch

import config
from bot_runner import find_bot_comment, find_bot_discussion_summary_comment


def _submission_with_comments(comments):
    submission = MagicMock()
    comment_forest = MagicMock()
    comment_forest.replace_more = MagicMock()
    comment_forest.__iter__ = lambda self: iter(comments)
    submission.comments = comment_forest
    return submission


class TestFindBotComment(unittest.TestCase):
    def setUp(self):
        alias_patch = patch.object(config, "BOT_OWNED_USERNAMES", ["example_bot"])
        alias_patch.start()
        self.addCleanup(alias_patch.stop)

    def test_finds_distinguished_comment_from_migrated_alias(self):
        legacy = MagicMock()
        legacy.author = MagicMock()
        legacy.author.name = "example_bot"
        legacy.distinguished = True

        other = MagicMock()
        other.author = MagicMock()
        other.author.name = "ordinary-user"
        other.distinguished = True

        found = find_bot_comment(
            _submission_with_comments([other, legacy]),
            "future-runtime-name",
        )
        self.assertIs(found, legacy)

    def test_ignores_undistinguished_owned_comment(self):
        owned = MagicMock()
        owned.author = MagicMock()
        owned.author.name = "another_bot_account"
        owned.distinguished = False

        self.assertIsNone(
            find_bot_comment(
                _submission_with_comments([owned]),
                "future-runtime-name",
            )
        )

    def test_discussion_summary_lookup_does_not_select_a_tldr(self):
        tldr = MagicMock()
        tldr.author = MagicMock()
        tldr.author.name = "example_bot"
        tldr.distinguished = True
        tldr.body = "**Post TLDR:** the short version"

        summary = MagicMock()
        summary.author = MagicMock()
        summary.author.name = "example_bot"
        summary.distinguished = True
        summary.body = "**💬 Discussion Summary (20+ comments):** the main views"

        found = find_bot_discussion_summary_comment(
            _submission_with_comments([tldr, summary]),
            "future-runtime-name",
        )

        self.assertIs(found, summary)


if __name__ == "__main__":
    unittest.main()
