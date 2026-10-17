"""Regression tests for durable public-response idempotency and limits."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from public_response_safety import (
    public_response_already_handled,
    post_public_comment,
    record_public_abstention,
    response_capacity_available,
)


BOT = "example_bot"


class _SourceComment:
    id = "source-1"
    parent_id = "t1_test04"
    link_id = "t3_post456-1"
    subreddit = SimpleNamespace(display_name="example_community")
    submission = SimpleNamespace(id="thread-1")

    def __init__(self, *, accept_then_raise=False):
        self.replies = []
        self.refresh_calls = 0
        self.accept_then_raise = accept_then_raise

    def reply(self, text):
        self.replies.append(
            SimpleNamespace(
                id=f"reply-{len(self.replies) + 1}",
                permalink="/r/example_community/comments/reply",
                author=SimpleNamespace(name=BOT),
                body=text,
            )
        )
        if self.accept_then_raise:
            raise TimeoutError("response lost after Reddit accepted the comment")
        return self.replies[-1]

    def refresh(self):
        self.refresh_calls += 1


class MoreComments:
    """Stand-in for PRAW's unexpanded reply-page marker."""


def _persisted(state):
    saves = []

    def save(value):
        saves.append(copy.deepcopy(value))

    return saves, save


class PublicResponseSafetyTests(unittest.TestCase):
    def test_terminal_abstention_clears_catch_up_retry_item(self):
        state = {
            "catch_up": {
                "comments": {
                    "pending": [
                        {"id": "source-1", "reason": "Jev unavailable"}
                    ]
                }
            }
        }
        saves, persist = _persisted(state)

        record_public_abstention(
            state,
            source_id="source-1",
            thread_id="thread-1",
            author_name="example_user",
            persist_state=persist,
        )

        self.assertEqual(state["catch_up"]["comments"]["pending"], [])
        self.assertEqual(
            state["public_response_ledger"]["comment:source-1"]["status"],
            "abstained",
        )
        self.assertEqual(len(saves), 1)

    def test_retry_item_is_cleared_after_confirmed_public_reply(self):
        source = _SourceComment()
        state = {
            "catch_up": {
                "comments": {
                    "pending": [
                        {"id": "source-1", "reason": "Jev unavailable"}
                    ]
                }
            }
        }
        _, persist = _persisted(state)

        reply = post_public_comment(
            source,
            "A response",
            state,
            source_id="source-1",
            author_name="example_user",
            bot_username=BOT,
            persist_state=persist,
            validate=lambda response, _: None,
        )

        self.assertIsNotNone(reply)
        self.assertEqual(state["catch_up"]["comments"]["pending"], [])

    def test_durable_pending_reservation_is_saved_before_public_write(self):
        source = _SourceComment()
        state = {}
        observed = []

        def persist(current):
            observed.append((len(source.replies), current["public_response_ledger"]["comment:source-1"]["status"]))

        reply = post_public_comment(
            source,
            "A response",
            state,
            source_id="source-1",
            author_name="example_user",
            bot_username=BOT,
            persist_state=persist,
            validate=lambda response, _: None,
        )

        self.assertIsNotNone(reply)
        self.assertEqual(observed, [(0, "pending"), (1, "completed")])
        self.assertEqual(len(source.replies), 1)

    def test_failed_reservation_persistence_prevents_reddit_write(self):
        source = _SourceComment()

        def fail(_state):
            raise OSError("state disk unavailable")

        with self.assertRaises(OSError):
            post_public_comment(
                source,
                "A response",
                {},
                source_id="source-1",
                author_name="example_user",
                bot_username=BOT,
                persist_state=fail,
            )
        self.assertEqual(source.replies, [])

    def test_accepted_write_with_lost_response_is_reconciled_without_duplicate(self):
        source = _SourceComment(accept_then_raise=True)
        state = {}
        saves, persist = _persisted(state)

        first = post_public_comment(
            source,
            "A response",
            state,
            source_id="source-1",
            author_name="example_user",
            bot_username=BOT,
            persist_state=persist,
        )
        self.assertIsNone(first)
        self.assertEqual(state["public_response_ledger"]["comment:source-1"]["status"], "pending")
        # A restarted process receives only the already-persisted reservation.
        state = copy.deepcopy(saves[-1])

        already_handled = public_response_already_handled(
            source,
            state,
            source_id="source-1",
            bot_username=BOT,
            persist_state=persist,
        )
        self.assertTrue(already_handled)
        self.assertEqual(len(source.replies), 1)
        self.assertEqual(state["public_response_ledger"]["comment:source-1"]["status"], "completed")

    def test_existing_bot_reply_is_recovered_when_local_ledger_is_missing(self):
        source = _SourceComment()
        source.replies.append(
            SimpleNamespace(
                id="old-reply",
                author=SimpleNamespace(name=BOT),
                body="An earlier reply whose local marker was lost",
            )
        )
        state = {}
        _, persist = _persisted(state)

        result = post_public_comment(
            source,
            "A duplicate response",
            state,
            source_id="source-1",
            author_name="example_user",
            bot_username=BOT,
            persist_state=persist,
        )

        self.assertIsNone(result)
        self.assertEqual(len(source.replies), 1)
        self.assertEqual(state["public_response_ledger"]["comment:source-1"]["status"], "completed")
        self.assertEqual(state["public_response_ledger"]["comment:source-1"]["reply_id"], "reconciled")

    def test_unavailable_readback_holds_response_without_publishing(self):
        source = _SourceComment()

        def fail_refresh():
            raise TimeoutError("Reddit readback unavailable")

        source.refresh = fail_refresh
        state = {}
        _, persist = _persisted(state)

        result = post_public_comment(
            source,
            "A response",
            state,
            source_id="source-1",
            author_name="example_user",
            bot_username=BOT,
            persist_state=persist,
        )

        self.assertIsNone(result)
        self.assertEqual(source.replies, [])
        self.assertEqual(state["public_response_ledger"], {})

    def test_unexpanded_reply_page_holds_response_until_existing_reply_can_be_checked(self):
        source = _SourceComment()
        source.replies.append(MoreComments())
        state = {}
        _, persist = _persisted(state)

        result = post_public_comment(
            source,
            "A response",
            state,
            source_id="source-1",
            author_name="example_user",
            bot_username=BOT,
            persist_state=persist,
        )

        self.assertIsNone(result)
        self.assertEqual(source.refresh_calls, 1)
        self.assertEqual(len(source.replies), 1)
        self.assertEqual(state["public_response_ledger"], {})

    def test_pending_write_is_not_retried_when_comment_listing_looks_empty(self):
        source = _SourceComment()
        source.refresh = lambda: setattr(source, "refresh_calls", source.refresh_calls + 1)
        state = {
            "public_response_ledger": {
                "comment:source-1": {
                    "status": "pending",
                    "created_at": 1000,
                    "thread_id": "thread-1",
                    "tier": "ordinary",
                }
            }
        }

        self.assertTrue(
            public_response_already_handled(
                source,
                state,
                source_id="source-1",
                bot_username=BOT,
                persist_state=lambda _: None,
            )
        )
        self.assertEqual(source.refresh_calls, 1)
        self.assertEqual(
            state["public_response_ledger"]["comment:source-1"]["status"],
            "uncertain",
        )
        self.assertTrue(
            public_response_already_handled(
                source,
                state,
                source_id="source-1",
                bot_username=BOT,
                persist_state=lambda _: None,
            )
        )
        self.assertEqual(source.refresh_calls, 1)
        self.assertEqual(source.replies, [])

    def test_corrupt_idempotency_ledger_fails_closed(self):
        source = _SourceComment()
        with self.assertRaisesRegex(RuntimeError, "idempotency record"):
            post_public_comment(
                source,
                "A response",
                {"public_response_ledger": []},
                source_id="source-1",
                author_name="example_user",
                bot_username=BOT,
                persist_state=lambda _: None,
            )
        self.assertEqual(source.replies, [])

    def test_corrupt_ledger_entry_fails_closed_instead_of_being_pruned(self):
        source = _SourceComment()
        corrupt_states = (
            {"public_response_ledger": {"comment:old": "completed"}},
            {"public_response_ledger": {"comment:old": {"status": "unknown"}}},
            {"public_response_ledger": {"comment:old": {
                "status": "completed", "created_at": 1000, "tier": "ordinary"
            }}},
            {"public_response_ledger": None},
        )
        for state in corrupt_states:
            with self.subTest(state=state):
                with self.assertRaisesRegex(RuntimeError, "unreadable"):
                    post_public_comment(
                        source,
                        "A response",
                        state,
                        source_id="source-1",
                        author_name="example_user",
                        bot_username=BOT,
                        persist_state=lambda _: None,
                    )
        self.assertEqual(source.replies, [])

    def test_ordinary_and_moderator_ceilings_use_independent_pools(self):
        now = 1_800_000_000.0
        state = {
            "_current_moderators_verified": True,
            "_current_moderators": ["mod"],
            "public_response_ledger": {
                f"comment:mixed-{index}": {
                    "status": "completed",
                    "created_at": now - 10,
                    "thread_id": (
                        "ordinary-thread" if index < 15 else
                        "moderator-thread" if index < 75 else f"thread-{index}"
                    ),
                    "tier": "ordinary" if index < 150 else "moderator",
                }
                for index in range(750)
            }
        }
        self.assertEqual(
            response_capacity_available(
                state, "new-thread", author_name="example_user", now=now
            ),
            (False, "ordinary_global"),
        )
        self.assertEqual(
            response_capacity_available(
                state, "new-thread", author_name="mod", now=now
            ),
            (False, "moderator_global"),
        )

        ordinary_only = {
            "_current_moderators_verified": True,
            "_current_moderators": ["mod"],
            "public_response_ledger": {
                f"comment:o-{index}": {
                    "status": "completed",
                    "created_at": now - 10,
                    "thread_id": "thread-1",
                    "tier": "ordinary",
                }
                for index in range(15)
            },
        }
        self.assertEqual(
            response_capacity_available(
                ordinary_only, "next", author_name="mod", now=now
            ),
            (True, None),
        )
        self.assertEqual(
            response_capacity_available(
                ordinary_only, "thread-1", author_name="example_user", now=now
            ),
            (False, "ordinary_thread"),
        )

        moderator_only = {
            "_current_moderators_verified": True,
            "_current_moderators": ["mod"],
            "public_response_ledger": {
                f"comment:m-{index}": {
                    "status": "completed",
                    "created_at": now - 10,
                    "thread_id": "thread-1" if index < 60 else f"m-{index}",
                    "tier": "moderator",
                }
                for index in range(600)
            },
        }
        self.assertEqual(
            response_capacity_available(
                moderator_only, "next", author_name="example_user", now=now
            ),
            (True, None),
        )
        self.assertEqual(
            response_capacity_available(
                moderator_only, "next", author_name="mod", now=now
            ),
            (False, "moderator_global"),
        )
        moderator_thread_only = {
            "_current_moderators_verified": True,
            "_current_moderators": ["mod"],
            "public_response_ledger": {
                f"comment:m-thread-{index}": {
                    "status": "completed",
                    "created_at": now - 10,
                    "thread_id": "thread-1",
                    "tier": "moderator",
                }
                for index in range(60)
            },
        }
        self.assertEqual(
            response_capacity_available(
                moderator_thread_only, "thread-1", author_name="mod", now=now
            ),
            (False, "moderator_thread"),
        )

    def test_writer_enforces_each_tier_against_only_its_own_history(self):
        now = 1_800_000_000.0
        moderator_state = {
            "_current_moderators_verified": True,
            "_current_moderators": ["mod"],
            "public_response_ledger": {
                f"comment:m-{index}": {
                    "status": "completed",
                    "created_at": now - 10,
                    "thread_id": f"prior-{index}",
                    "tier": "moderator",
                }
                for index in range(600)
            },
        }
        ordinary_state = {
            "_current_moderators_verified": True,
            "_current_moderators": ["mod"],
            "public_response_ledger": {
                f"comment:o-{index}": {
                    "status": "completed",
                    "created_at": now - 10,
                    "thread_id": f"prior-{index}",
                    "tier": "ordinary",
                }
                for index in range(150)
            },
        }

        ordinary_source = _SourceComment()
        ordinary_source.id = "ordinary-cap-source"
        ordinary_source.submission = SimpleNamespace(id="new-ordinary")
        _, persist_ordinary = _persisted(moderator_state)
        with patch("public_response_safety._now_timestamp", return_value=now):
            ordinary_reply = post_public_comment(
                ordinary_source,
                "Ordinary request",
                moderator_state,
                source_id=ordinary_source.id,
                author_name="example_user",
                bot_username=BOT,
                persist_state=persist_ordinary,
                validate=lambda response, _: None,
            )
        self.assertIsNotNone(ordinary_reply)

        moderator_source = _SourceComment()
        moderator_source.id = "moderator-cap-source"
        moderator_source.submission = SimpleNamespace(id="new-moderator")
        _, persist_moderator = _persisted(ordinary_state)
        with patch("public_response_safety._now_timestamp", return_value=now):
            moderator_reply = post_public_comment(
                moderator_source,
                "Verified moderator request",
                ordinary_state,
                source_id=moderator_source.id,
                author_name="mod",
                bot_username=BOT,
                persist_state=persist_moderator,
                validate=lambda response, _: None,
            )
        self.assertIsNotNone(moderator_reply)

    def test_group_membership_or_unverified_username_cannot_use_moderator_pool(self):
        now = 1_800_000_000.0
        state = {
            "public_response_ledger": {
                f"comment:o-{index}": {
                    "status": "completed",
                    "created_at": now - 10,
                    "thread_id": f"thread-{index}",
                    "tier": "ordinary",
                }
                for index in range(150)
            }
        }
        self.assertEqual(
            response_capacity_available(state, "new", author_name="chat-admin", now=now),
            (False, "ordinary_global"),
        )

    def test_same_source_comment_id_cannot_be_replied_to_twice(self):
        source = _SourceComment()
        state = {}
        _, persist = _persisted(state)
        for text, action in (("First", "summon"), ("Edited request", "reply")):
            post_public_comment(
                source,
                text,
                state,
                source_id="source-1",
                author_name="example_user",
                bot_username=BOT,
                persist_state=persist,
                action=action,
                validate=lambda response, _: None,
            )
        self.assertEqual(len(source.replies), 1)


if __name__ == "__main__":
    unittest.main()
