"""Tests for the private, moderator-only Modmail assistant."""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import config
from moderator_modmail_assistant import (
    SUBJECT,
    process_modmail_assistant,
)


class _Conversation:
    id = "thread-1"
    subject = "Optimist Prime moderator assistant"
    is_internal = True

    def __init__(
        self, messages, *, conversation_id="thread-1", subject=SUBJECT,
        is_internal=True, last_updated=None, participant=None,
    ):
        self.id = conversation_id
        self.subject = subject
        self.is_internal = is_internal
        self.messages = list(messages)
        self.sent = []
        self.participant = participant
        self.last_updated = (last_updated or datetime.now(timezone.utc)).isoformat()
        self.num_messages = len(self.messages)

    def reply(self, *, body, internal=False):
        self.sent.append((body, internal))
        message = SimpleNamespace(
            id=f"bot-{len(self.sent)}",
            is_internal=internal,
            author=SimpleNamespace(name="example_bot"),
            body_markdown=body,
        )
        self.messages.append(message)
        self.num_messages = len(self.messages)
        self.last_updated = datetime.now(timezone.utc).isoformat()
        return message


class _Modmail:
    def __init__(self, conversation, *, states=None):
        self.conversation = conversation
        self.by_id = {conversation.id: conversation}
        self.states = states or {
            "all": [conversation], "appeals": [], "mod": [conversation]
        }
        self.created = []

    def __call__(self, conversation_id):
        if conversation_id not in self.by_id:
            raise AssertionError("unexpected Modmail conversation id")
        return self.by_id[conversation_id]

    def conversations(self, **kwargs):
        return iter(self.states.get(kwargs.get("state"), []))

    def create(self, *, subject, body, recipient):
        self.created.append((subject, body, recipient))
        return self.conversation


class ModeratorModmailAssistantTests(unittest.TestCase):
    @staticmethod
    def _turn(response, *, route="history_review", tokens=0, cost=0.0, handled=True):
        return {
            "kind": "moderator_turn", "route": route, "handled": handled,
            "response": response, "conversation_response": response,
            "token_info": {"total_tokens": tokens, "cost": cost},
        }

    def _setup(self, message):
        conversation = _Conversation([message])
        modmail = _Modmail(conversation)
        subreddit = SimpleNamespace(modmail=modmail)
        state = self._state(conversation.id)
        return subreddit, conversation, state

    @staticmethod
    def _state(conversation_id):
        return {
            "_current_moderators_verified": True,
            "_current_moderators": ["example_mod"],
            "moderator_modmail_assistant": {
                "conversation_id": conversation_id,
                "processed_message_ids": [],
                "pending_reply": None,
            },
        }

    @staticmethod
    def _message(message_id, author, body, *, internal=False):
        return SimpleNamespace(
            id=message_id,
            is_internal=internal,
            author=SimpleNamespace(name=author),
            body_markdown=body,
        )

    def test_disabled_assistant_makes_no_modmail_request(self):
        class NoModmailAllowed:
            @property
            def modmail(self):
                raise AssertionError("disabled assistant must not contact Reddit Modmail")

        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", False):
            result = process_modmail_assistant(
                NoModmailAllowed(), object(), {}, object(),
                bot_username="example_bot",
            )
        self.assertEqual(result["commands_seen"], 0)
        self.assertEqual(result["replies_sent"], 0)

    def test_moderator_scan_replies_only_as_internal_note_and_remembers_result(self):
        request = SimpleNamespace(
            id="mod-message-1",
            is_internal=True,
            author=SimpleNamespace(name="example_mod"),
            body_markdown="Review u/example_user",
        )
        subreddit, conversation, state = self._setup(request)
        saves = []
        model = object()
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn", return_value=self._turn("History for Alice", tokens=12, cost=0.01)) as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, model,
                bot_username="example_bot",
                persist_state=lambda changed: saves.append(changed.copy()),
            )

        self.assertEqual(result["commands_seen"], 1)
        self.assertEqual(result["replies_sent"], 1)
        self.assertEqual(result["tokens"], 12)
        self.assertEqual(conversation.sent, [("History for Alice", True)])
        self.assertIsNone(state["moderator_modmail_assistant"]["pending_reply"])
        self.assertIn("mod-message-1", state["moderator_modmail_assistant"]["processed_message_ids"])
        self.assertGreaterEqual(len(saves), 2)
        turn.assert_called_once()
        self.assertEqual(turn.call_args.kwargs["text"], "Review u/example_user")
        self.assertTrue(turn.call_args.kwargs["verified_moderator"])

    def test_bare_username_scan_is_processed_in_the_assistant_thread(self):
        request = SimpleNamespace(
            id="mod-message-bare-target",
            is_internal=True,
            author=SimpleNamespace(name="example_mod"),
            body_markdown="Scan example_user",
        )
        subreddit, conversation, state = self._setup(request)
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn", return_value=self._turn("Report")) as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, object(),
                bot_username="example_bot",
            )
        self.assertEqual(result["commands_seen"], 1)
        self.assertEqual(conversation.sent, [("Report", True)])
        self.assertEqual(turn.call_args.kwargs["text"], "Scan example_user")

    def test_incomplete_scan_gets_internal_usage_guidance_without_model_call(self):
        request = SimpleNamespace(
            id="mod-message-missing-target",
            is_internal=True,
            author=SimpleNamespace(name="example_mod"),
            body_markdown="Scan",
        )
        subreddit, conversation, state = self._setup(request)
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn", return_value=self._turn(
                 "Please include the username and what you'd like me to review.",
                 route="clarify",
             )) as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, object(),
                bot_username="example_bot",
            )
        self.assertEqual(result["replies_sent"], 1)
        self.assertTrue(conversation.sent[0][1])
        self.assertIn("username", conversation.sent[0][0].casefold())
        self.assertIn("mod-message-missing-target", state["moderator_modmail_assistant"]["processed_message_ids"])
        turn.assert_called_once()

    def test_ordinary_chat_in_dedicated_assistant_gets_a_conversational_note(self):
        request = SimpleNamespace(
            id="mod-message-thanks",
            is_internal=True,
            author=SimpleNamespace(name="example_mod"),
            body_markdown="Thanks, that helps.",
        )
        subreddit, conversation, state = self._setup(request)
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn", return_value=self._turn(
                 "A brief reply", route="conversation", handled=False,
             )) as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, object(),
                bot_username="example_bot",
            )
        self.assertEqual(result["replies_sent"], 1)
        self.assertEqual(conversation.sent, [("A brief reply", True)])
        turn.assert_called_once()

    def test_non_moderator_message_is_never_sent_to_the_model(self):
        request = SimpleNamespace(
            id="member-message",
            is_internal=True,
            author=SimpleNamespace(name="member"),
            body_markdown="scan u/example_user",
        )
        subreddit, conversation, state = self._setup(request)
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn") as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, object(),
                bot_username="example_bot",
            )
        self.assertEqual(result["commands_seen"], 0)
        self.assertEqual(conversation.sent, [])
        turn.assert_not_called()

    def test_assistant_fails_closed_without_current_cycle_roster(self):
        request = self._message(
            "stale-roster-request", "example_mod", "review u/example_user"
        )
        subreddit, conversation, state = self._setup(request)
        state.pop("_current_moderators_verified")
        state.pop("_current_moderators")
        state["moderator_cache"] = {"moderators": ["example_mod"]}
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn") as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, object(), bot_username="example_bot"
            )
        self.assertEqual(result["failures"], 1)
        self.assertEqual(result["commands_seen"], 0)
        self.assertEqual(conversation.sent, [])
        turn.assert_not_called()

    def test_removed_moderator_in_stale_cache_cannot_trigger_provider_or_note(self):
        request = self._message(
            "former-mod-request", "example_mod", "review u/example_user"
        )
        subreddit, conversation, state = self._setup(request)
        state["_current_moderators"] = ["current_mod"]
        state["moderator_cache"] = {"moderators": ["example_mod"]}
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn") as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, object(), bot_username="example_bot"
            )
        self.assertEqual(result["commands_seen"], 0)
        self.assertEqual(conversation.sent, [])
        self.assertIn(
            "former-mod-request",
            state["moderator_modmail_assistant"]["processed_message_ids"],
        )
        turn.assert_not_called()

    def test_pending_reply_from_removed_moderator_is_discarded(self):
        request = self._message(
            "revoked-pending-request", "example_mod", "review u/example_user"
        )
        subreddit, conversation, state = self._setup(request)
        state["_current_moderators"] = ["current_mod"]
        state["moderator_cache"] = {"moderators": ["example_mod"]}
        state["moderator_modmail_assistant"]["pending_reply"] = {
            "conversation_id": conversation.id,
            "source_message_id": "revoked-pending-request",
            "body": "Saved report",
        }
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn") as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, object(), bot_username="example_bot"
            )
        self.assertEqual(result["replies_sent"], 0)
        self.assertEqual(conversation.sent, [])
        self.assertIsNone(state["moderator_modmail_assistant"]["pending_reply"])
        turn.assert_not_called()

    def test_dry_run_does_not_create_or_send_a_thread(self):
        request = SimpleNamespace(
            id="mod-message-1",
            is_internal=True,
            author=SimpleNamespace(name="example_mod"),
            body_markdown="scan u/example_user",
        )
        conversation = _Conversation([request])
        subreddit = SimpleNamespace(modmail=_Modmail(conversation))
        state = self._state(conversation.id)
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn") as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, object(),
                bot_username="example_bot",
                dry_run=True,
            )
        self.assertEqual(result["commands_seen"], 1)
        self.assertEqual(conversation.sent, [])
        self.assertEqual(subreddit.modmail.created, [])
        turn.assert_not_called()

    def test_pending_reply_reuses_saved_text_without_a_second_analysis(self):
        request = SimpleNamespace(
            id="mod-message-1",
            is_internal=True,
            author=SimpleNamespace(name="example_mod"),
            body_markdown="scan u/example_user",
        )
        subreddit, conversation, state = self._setup(request)
        state["moderator_modmail_assistant"]["pending_reply"] = {
            "source_message_id": "mod-message-1",
            "body": "Saved report",
        }
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn") as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, object(),
                bot_username="example_bot",
            )
        self.assertEqual(result["replies_sent"], 1)
        self.assertEqual(conversation.sent, [("Saved report", True)])
        turn.assert_not_called()

    def test_failed_analysis_leaves_the_moderator_request_retryable(self):
        request = SimpleNamespace(
            id="mod-message-1",
            is_internal=True,
            author=SimpleNamespace(name="example_mod"),
            body_markdown="scan u/example_user",
        )
        subreddit, _, state = self._setup(request)
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn", side_effect=RuntimeError("temporary provider failure")):
            with self.assertRaisesRegex(RuntimeError, "temporary provider failure"):
                process_modmail_assistant(
                    subreddit, object(), state, object(),
                    bot_username="example_bot",
                )

        self.assertNotIn(
            "mod-message-1",
            state["moderator_modmail_assistant"]["processed_message_ids"],
        )

    def test_inactive_thread_over_seven_days_is_not_hydrated_or_processed(self):
        now = datetime(2026, 10, 9, tzinfo=timezone.utc)
        request = self._message("old-request", "example_mod", "scan u/example_user")
        stale = _Conversation(
            [request], last_updated=now - timedelta(days=7, seconds=1)
        )
        modmail = _Modmail(stale)
        subreddit = SimpleNamespace(modmail=modmail)
        state = self._state(stale.id)
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn") as handler:
            result = process_modmail_assistant(
                subreddit, object(), state, object(),
                bot_username="example_bot", now=now,
            )
        self.assertEqual(result["stale_conversations_skipped"], 1)
        self.assertEqual(result["conversations_checked"], 0)
        self.assertEqual(stale.sent, [])
        handler.assert_not_called()

    def test_activity_exactly_seven_days_old_is_still_recent(self):
        now = datetime(2026, 10, 9, tzinfo=timezone.utc)
        request = self._message(
            "boundary-request", "example_mod", "Hey Optimist, review u/example_user's history"
        )
        thread = _Conversation(
            [request], conversation_id="boundary-thread", is_internal=False,
            last_updated=now - timedelta(days=7),
        )
        assistant = _Conversation([], conversation_id="assistant", last_updated=now)
        modmail = _Modmail(
            assistant,
            states={"all": [thread], "appeals": [], "mod": [assistant]},
        )
        modmail.by_id[thread.id] = thread
        subreddit = SimpleNamespace(modmail=modmail)
        state = self._state(assistant.id)
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn", return_value=self._turn("Report")) as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, object(),
                bot_username="example_bot", now=now,
            )
        self.assertEqual(result["stale_conversations_skipped"], 0)
        self.assertEqual(result["replies_sent"], 1)
        self.assertEqual(thread.sent, [("Report", True)])
        turn.assert_called_once()

    def test_unaddressed_modmail_near_miss_reaches_jev_and_is_silently_skipped(self):
        now = datetime(2026, 10, 9, tzinfo=timezone.utc)
        message = self._message(
            "near-miss", "example_mod", "The team should review u/example_user's history later."
        )
        thread = _Conversation(
            [message], conversation_id="mod-discussion", is_internal=False,
            last_updated=now,
        )
        assistant = _Conversation([], conversation_id="assistant", last_updated=now)
        modmail = _Modmail(
            assistant,
            states={"all": [thread], "appeals": [], "mod": [assistant]},
        )
        modmail.by_id[thread.id] = thread
        subreddit = SimpleNamespace(modmail=modmail)
        state = self._state(assistant.id)
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn") as turn:
            turn.return_value = self._turn(
                "[[NO_REPLY]]", route="not_addressed", handled=False,
            )
            result = process_modmail_assistant(
                subreddit, object(), state, object(),
                bot_username="example_bot", now=now,
            )
        self.assertEqual(result["commands_seen"], 1)
        self.assertEqual(result["replies_sent"], 0)
        self.assertEqual(thread.sent, [])
        turn.assert_called_once()
        self.assertEqual(turn.call_args.kwargs["text"], message.body_markdown)
        self.assertFalse(turn.call_args.kwargs["is_summon"])
        self.assertIn("near-miss", state["moderator_modmail_assistant"]["processed_message_ids"])

    def test_dedicated_modmail_acknowledgement_does_not_send_no_reply_marker(self):
        request = self._message("mod-message-thanks", "example_mod", "Thanks, that helps.")
        subreddit, conversation, state = self._setup(request)
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn", return_value=self._turn(
                 "[[NO_REPLY]]", route="no_reply", handled=False,
             )) as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, object(),
                bot_username="example_bot",
            )
        self.assertEqual(result["replies_sent"], 0)
        self.assertEqual(conversation.sent, [])
        self.assertIn("mod-message-thanks", state["moderator_modmail_assistant"]["processed_message_ids"])
        turn.assert_called_once()

    def test_recent_activity_by_member_reactivates_old_unprocessed_mod_request(self):
        now = datetime(2026, 10, 9, tzinfo=timezone.utc)
        request = self._message(
            "old-request", "example_mod", "Hey Optimist, scan example_user"
        )
        request.created_utc = (now - timedelta(days=8)).timestamp()
        member_reply = self._message("member-reply", "appeal_user", "I have added more context.")
        member_reply.created_utc = now.timestamp()
        appeal = _Conversation(
            [request, member_reply], conversation_id="appeal-1",
            is_internal=False, last_updated=now,
            participant=SimpleNamespace(name="appeal_user"),
        )
        assistant = _Conversation([], conversation_id="assistant", last_updated=now)
        modmail = _Modmail(
            assistant,
            states={"all": [], "appeals": [appeal], "mod": [assistant]},
        )
        modmail.by_id[appeal.id] = appeal
        subreddit = SimpleNamespace(modmail=modmail)
        state = self._state(assistant.id)
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn", return_value=self._turn("Private report")) as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, object(),
                bot_username="example_bot", now=now,
            )
        self.assertEqual(result["replies_sent"], 1)
        self.assertEqual(appeal.sent, [("Private report", True)])
        self.assertEqual(turn.call_args.kwargs["parent_author"], "appeal_user")

    def test_recent_moderator_summon_in_appeal_uses_private_note_and_participant_target(self):
        now = datetime(2026, 10, 9, tzinfo=timezone.utc)
        request = self._message(
            "appeal-request", "example_mod",
            "u/example_bot, check whether this person repeatedly targets another user with insults",
        )
        appeal = _Conversation(
            [request], conversation_id="appeal-2", is_internal=False,
            last_updated=now, participant=SimpleNamespace(name="appeal_user"),
        )
        assistant = _Conversation([], conversation_id="assistant", last_updated=now)
        modmail = _Modmail(
            assistant,
            states={"all": [], "appeals": [appeal], "mod": [assistant]},
        )
        modmail.by_id[appeal.id] = appeal
        subreddit = SimpleNamespace(modmail=modmail)
        state = self._state(assistant.id)
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn", return_value=self._turn("Private report")) as turn:
            result = process_modmail_assistant(
                subreddit, object(), state, object(),
                bot_username="example_bot", now=now,
            )
        self.assertEqual(result["replies_sent"], 1)
        self.assertEqual(appeal.sent, [("Private report", True)])
        self.assertEqual(turn.call_args.kwargs["parent_author"], "appeal_user")


    def test_new_unprocessed_message_is_handled_once_without_replaying_old_request(self):
        first = self._message("first", "example_mod", "scan u/example_user")
        conversation = _Conversation([first])
        subreddit = SimpleNamespace(modmail=_Modmail(conversation))
        state = self._state(conversation.id)
        with patch.object(config, "MODMAIL_ASSISTANT_ENABLED", True), \
             patch("moderator_modmail_assistant.process_moderator_turn", side_effect=[
                 self._turn("First report"), self._turn("Second report")
             ]) as turn:
            process_modmail_assistant(
                subreddit, object(), state, object(), bot_username="example_bot"
            )
            second = self._message("second", "example_mod", "review u/example_user_b")
            conversation.messages.append(second)
            conversation.num_messages += 1
            conversation.last_updated = datetime.now(timezone.utc).isoformat()
            process_modmail_assistant(
                subreddit, object(), state, object(), bot_username="example_bot"
            )
        self.assertEqual([body for body, _internal in conversation.sent], ["First report", "Second report"])
        self.assertEqual(turn.call_count, 2)
        self.assertEqual(
            [call.kwargs["text"] for call in turn.call_args_list],
            ["scan u/example_user", "review u/example_user_b"],
        )


if __name__ == "__main__":
    unittest.main()
