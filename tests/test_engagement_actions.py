"""Tests for configurable summon and inbox engagement actions."""

import unittest
from unittest.mock import patch

from engagement_actions import execute_engagement_action, normalize_engagement_action


class _Mod:
    def __init__(self):
        self.distinguish_calls = []

    def distinguish(self, **kwargs):
        self.distinguish_calls.append(kwargs)


class _Reply:
    id = "t1_reply"
    permalink = "/r/example_community/comments/reply"

    def __init__(self):
        self.mod = _Mod()


class _Content:
    id = "t1_test03"
    permalink = "/r/example_community/comments/source"
    parent_id = "t1_test04"

    def __init__(self):
        self.replies = []
        self.reports = []
        self.mod = _Mod()

    def reply(self, text):
        self.replies.append(text)
        return _Reply()

    def refresh(self):
        return None

    def report(self, reason):
        self.reports.append(reason)


class _Subreddit:
    def __init__(self):
        self.modmail = _Modmail()


class _InternalConversation:
    is_internal = True

    def __init__(self, subject, body):
        self.id = f"mod-{len(subject)}-{len(body)}"
        self.subject = subject
        self.body = body


class _Modmail:
    def __init__(self):
        self.conversations_created = []
        self.fail_after_create = False

    def conversations(self, *, state, sort, limit):
        self.assert_listing_args = (state, sort, limit)
        return list(reversed(self.conversations_created))

    def create(self, *, subject, body, recipient):
        self.assert_recipient = recipient
        conversation = _InternalConversation(subject, body)
        self.conversations_created.append(conversation)
        if self.fail_after_create:
            self.fail_after_create = False
            raise RuntimeError("Reddit accepted the write but the acknowledgement was lost")
        return conversation


class TestEngagementActions(unittest.TestCase):
    def test_unknown_action_defaults_to_public_comment(self):
        self.assertEqual(normalize_engagement_action("not-supported"), "comment")
        self.assertEqual(normalize_engagement_action(None), "comment")

    def test_comment_mode_posts_and_validates_reply(self):
        content = _Content()
        with patch("audit_log.log_audit_event") as audit:
            self.assertTrue(
                execute_engagement_action(
                    content,
                    _Subreddit(),
                    "example_user",
                    "source text",
                    "generated answer",
                    event_type="summon",
                    action_mode="comment",
                    response_state={},
                    persist_state=lambda _: None,
                    bot_username="example_bot",
                )
            )
        self.assertEqual(len(content.replies), 1)
        self.assertIn("generated answer", content.replies[0])
        self.assertEqual(len(audit.call_args_list), 1)

    def test_report_and_modmail_modes_do_not_post_public_replies(self):
        content = _Content()
        subreddit = _Subreddit()
        with patch("audit_log.log_audit_event"):
            self.assertTrue(
                execute_engagement_action(
                    content,
                    subreddit,
                    "example_user",
                    "source text",
                    "generated answer",
                    event_type="reply",
                    action_mode="report",
                )
            )
            self.assertTrue(
                execute_engagement_action(
                    content,
                    subreddit,
                    "example_user",
                    "source text",
                    "generated answer",
                    event_type="reply",
                    action_mode="modmail",
                )
            )
        self.assertEqual(len(content.replies), 0)
        self.assertEqual(len(content.reports), 1)
        self.assertEqual(len(subreddit.modmail.conversations_created), 1)
        self.assertIn("t1_test03", subreddit.modmail.conversations_created[0].subject)
        self.assertEqual(subreddit.modmail.assert_recipient, None)

    def test_modmail_retry_reuses_conversation_after_lost_create_acknowledgement(self):
        content = _Content()
        subreddit = _Subreddit()
        subreddit.modmail.fail_after_create = True

        with patch("audit_log.log_audit_event") as audit:
            self.assertFalse(
                execute_engagement_action(
                    content,
                    subreddit,
                    "example_user",
                    "source text",
                    "generated answer",
                    event_type="reply",
                    action_mode="modmail",
                )
            )
            self.assertTrue(
                execute_engagement_action(
                    content,
                    subreddit,
                    "example_user",
                    "source text",
                    "generated answer",
                    event_type="reply",
                    action_mode="modmail",
                )
            )

        self.assertEqual(len(subreddit.modmail.conversations_created), 1)
        audit.assert_called_once()

    def test_log_mode_only_records_locally(self):
        content = _Content()
        with patch("audit_log.log_audit_event") as audit:
            self.assertTrue(
                execute_engagement_action(
                    content,
                    _Subreddit(),
                    "example_user",
                    "source text",
                    "generated answer",
                    event_type="reply",
                    action_mode="log",
                )
            )
        self.assertEqual(content.replies, [])
        self.assertEqual(content.reports, [])
        self.assertEqual(audit.call_args.args[4], "logged")

    def test_dry_run_never_calls_reddit_or_audit(self):
        content = _Content()
        subreddit = _Subreddit()
        with patch("audit_log.log_audit_event") as audit:
            self.assertTrue(
                execute_engagement_action(
                    content,
                    subreddit,
                    "example_user",
                    "source text",
                    "generated answer",
                    event_type="summon",
                    action_mode="modmail",
                    dry_run=True,
                )
            )
        self.assertEqual(content.replies, [])
        self.assertEqual(content.reports, [])
        self.assertEqual(subreddit.modmail.conversations_created, [])
        audit.assert_not_called()

    def test_failed_action_returns_false_so_callers_can_retry(self):
        content = _Content()

        def fail_reply(_text):
            raise RuntimeError("temporary Reddit failure")

        content.reply = fail_reply
        with patch("audit_log.log_audit_event"):
            self.assertFalse(
                execute_engagement_action(
                    content,
                    _Subreddit(),
                    "example_user",
                    "source text",
                    "generated answer",
                    event_type="summon",
                    action_mode="comment",
                    response_state={},
                    persist_state=lambda _: None,
                    bot_username="example_bot",
                )
            )

    def test_distinguish_failure_does_not_retry_a_posted_reply(self):
        content = _Content()
        reply = _Reply()
        reply.mod.distinguish = lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("temporary distinguish failure")
            )
        content.reply = lambda text: (content.replies.append(text) or reply)
        with patch("audit_log.log_audit_event"):
            self.assertTrue(
                execute_engagement_action(
                    content,
                    _Subreddit(),
                    "example_user",
                    "source text",
                    "generated answer",
                    event_type="summon",
                    action_mode="comment",
                    response_state={},
                    persist_state=lambda _: None,
                    bot_username="example_bot",
                )
            )
        self.assertEqual(len(content.replies), 1)


if __name__ == "__main__":
    unittest.main()
