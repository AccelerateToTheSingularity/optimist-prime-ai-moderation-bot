from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bot_utils import get_daily_non_moderator_reply_count, record_reply_success
from bot_logging import JsonLineFormatter
from llm_client import RateLimitedLLMModel
from private_chat_handler import (
    _conversation_key,
    process_private_chat,
    reply_to_latest_private_chat,
)
from summon_handler import is_summon
from reddit_conversation_transport import (
    NATIVE_GROUP_CHAT_SUPPORTED,
    send_message_reply,
)


class FakeSubreddit:
    def __init__(self, moderators=("mod",)):
        self.mods = [SimpleNamespace(name=name) for name in moderators]
        self.fail_moderator_list = False

    def moderator(self):
        if self.fail_moderator_list:
            raise RuntimeError("roster unavailable")
        return list(self.mods)


class FakeReddit:
    def __init__(self, *, moderators=("mod",)):
        self.sub = FakeSubreddit(moderators)
        self.inbox = None

    def subreddit(self, _name):
        return self.sub


class JsonModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, contents, generation_config=None):
        self.calls.append((contents, generation_config))
        value = self.responses.pop(0) if self.responses else "A concise reply."
        return SimpleNamespace(text=value, usage_metadata=SimpleNamespace(
            prompt_token_count=12, candidates_token_count=4
        ))


class RouteChoiceJev:
    """Fake typed Jev client for one-call private-chat routing tests."""

    ready = True
    max_calls = 100

    def __init__(self, choice: str | None):
        self.choice = choice
        self.calls_used = 0
        self.last_usage = {"input_tokens": 12, "cost": 0.0001}
        self.requests = []

    def choose_many_detailed(self, *, state, questions):
        self.calls_used += 1
        self.requests.append((state, questions))
        if self.choice is None:
            return {}
        answers = {}
        for name, question in questions.items():
            if name == "public_interaction":
                selected = (
                    "history_review.review"
                    if self.choice == "history_review"
                    else self.choice
                )
            elif name == "moderator_action":
                selected = self.choice
            else:
                continue
            if selected not in question.get("criteria", {}):
                continue
            answers[name] = {
                "choice": selected,
                "confidence": 0.95,
                "probabilities": {selected: 0.95},
            }
        return answers


class FakeMessage:
    _reddit_private_message = True

    def __init__(self, message_id, author, body, *, parent_id=None, dest="OptimistPrime", created_utc=None):
        self.id = message_id
        self.author = SimpleNamespace(name=author)
        self.body = body
        self.parent_id = parent_id
        self.dest = dest
        # Private-chat processing applies the configured message-age limit.
        # Keep default fake messages current for that configured limit.
        self.created_utc = (
            created_utc
            if created_utc is not None
            else datetime.now(timezone.utc).timestamp()
        )
        self.subreddit = None
        self.read = False
        self.replies = []

    def reply(self, body):
        response = SimpleNamespace(id=f"answer-{self.id}")
        self.replies.append((body, response))
        return response

    def mark_read(self):
        self.read = True


class FakeInbox:
    def __init__(self, messages):
        self.messages = messages

    def unread(self, **_kwargs):
        return [message for message in self.messages if not message.read]


class PrivateConversationTests(unittest.TestCase):
    def setUp(self):
        activity_writer = patch("moderator_action_router.record_moderator_activity")
        self.record_activity = activity_writer.start()
        self.addCleanup(activity_writer.stop)

    def test_public_history_result_replies_to_the_newest_saved_one_to_one_chat(self):
        state = {
            "private_message_conversation_ids": {
                "old-message": "private:mod:old-room",
                "new-message-1": "private:mod:new-room",
                "new-message-2": "private:mod:new-room",
            },
            "private_conversations": {
                "private:mod:old-room": {"history": [], "updated_at": 100.0},
                "private:mod:new-room": {"history": [], "updated_at": 200.0},
            },
        }
        anchor = SimpleNamespace(
            author=SimpleNamespace(name="mod"),
            dest="example_bot",
        )
        reddit = SimpleNamespace(inbox=SimpleNamespace(message=Mock(return_value=anchor)))
        sent = SimpleNamespace(id="new-reply")
        with patch("private_chat_handler.send_message_reply", return_value=sent) as send:
            delivered = reply_to_latest_private_chat(
                reddit,
                state,
                username="mod",
                bot_username="example_bot",
                text="The report is ready.",
                request_text="Review u/example_user.",
            )

        self.assertTrue(delivered)
        reddit.inbox.message.assert_called_once_with("new-message-2")
        send.assert_called_once_with(anchor, "The report is ready.")
        self.assertEqual(
            state["private_message_conversation_ids"]["new-reply"],
            "private:mod:new-room",
        )
        self.assertEqual(
            [row["role"] for row in state["private_conversations"]["private:mod:new-room"]["history"]],
            ["user", "assistant"],
        )

    def test_public_history_result_drops_when_no_saved_chat_anchor_exists(self):
        reddit = SimpleNamespace(inbox=SimpleNamespace(message=Mock()))
        with patch("private_chat_handler.send_message_reply") as send:
            delivered = reply_to_latest_private_chat(
                reddit,
                {"private_message_conversation_ids": {}, "private_conversations": {}},
                username="mod",
                bot_username="example_bot",
                text="The report is ready.",
            )

        self.assertFalse(delivered)
        reddit.inbox.message.assert_not_called()
        send.assert_not_called()

    def test_public_history_result_drops_rejected_chat_reply_without_retry(self):
        state = {
            "private_message_conversation_ids": {"known-message": "private:mod"},
            "private_conversations": {"private:mod": {"history": [], "updated_at": 100.0}},
        }
        anchor = SimpleNamespace(
            author=SimpleNamespace(name="mod"),
            dest="example_bot",
        )
        reddit = SimpleNamespace(inbox=SimpleNamespace(message=Mock(return_value=anchor)))
        with patch("private_chat_handler.send_message_reply", side_effect=OSError("rejected")) as send:
            delivered = reply_to_latest_private_chat(
                reddit,
                state,
                username="mod",
                bot_username="example_bot",
                text="The report is ready.",
            )

        self.assertFalse(delivered)
        reddit.inbox.message.assert_called_once_with("known-message")
        send.assert_called_once_with(anchor, "The report is ready.")
        self.assertEqual(state["private_conversations"]["private:mod"]["history"], [])

    def test_public_history_result_will_not_reply_to_a_different_chat(self):
        state = {
            "private_message_conversation_ids": {"known-message": "private:mod"},
            "private_conversations": {"private:mod": {"history": [], "updated_at": 100.0}},
        }
        anchor = SimpleNamespace(
            author=SimpleNamespace(name="another_user"),
            dest="example_bot",
        )
        reddit = SimpleNamespace(inbox=SimpleNamespace(message=Mock(return_value=anchor)))
        with patch("private_chat_handler.send_message_reply") as send:
            delivered = reply_to_latest_private_chat(
                reddit,
                state,
                username="mod",
                bot_username="example_bot",
                text="The report is ready.",
            )

        self.assertFalse(delivered)
        reddit.inbox.message.assert_called_once_with("known-message")
        send.assert_not_called()

    def test_unlinked_private_messages_share_recent_context_without_cross_user_leakage(self):
        first = FakeMessage("dm-a-1", "example_user", "My project is called the Redwood study.")
        inbox = FakeInbox([first])
        reddit = FakeReddit(moderators=())
        reddit.inbox = inbox
        model = JsonModel("I will remember that.", "You called it the Redwood study.", "Hello, Bob.")
        jev = RouteChoiceJev("conversation")
        state = {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0}

        process_private_chat(reddit, model, state, "OptimistPrime", jev_client=jev)
        second = FakeMessage("dm-a-2", "example_user", "What did I call it?")
        inbox.messages.append(second)
        process_private_chat(reddit, model, state, "OptimistPrime", jev_client=jev)

        second_prompt = str(model.calls[1][0])
        self.assertIn("My project is called the Redwood study", second_prompt)
        self.assertIn("I will remember that", second_prompt)
        self.assertEqual(
            list(state["private_conversations"]),
            ["private:example_user"],
        )

        bob = FakeMessage("dm-b-1", "bob", "Hello Optimist")
        inbox.messages.append(bob)
        state["_cycle_non_moderator_replies"] = 0
        process_private_chat(reddit, model, state, "OptimistPrime", jev_client=jev)
        self.assertNotIn("Redwood study", str(model.calls[2][0]))
        self.assertIn("private:bob", state["private_conversations"])

    def test_old_message_keyed_private_histories_merge_into_stable_user_conversation(self):
        state = {
            "private_message_conversation_ids": {
                "message-1": "private:example_user:message-1",
                "answer-message-1": "private:example_user:message-1",
                "message-2": "private:example_user:message-2",
                "answer-message-2": "private:example_user:message-2",
            },
            "private_conversations": {
                "private:example_user:message-1": {
                    "history": [
                        {"role": "user", "text": "Review this profile."},
                        {"role": "assistant", "text": "Which profile?"},
                    ],
                    "updated_at": 100.0,
                },
                "private:example_user:message-2": {
                    "history": [
                        {"role": "user", "text": "example_candidate"},
                        {"role": "assistant", "text": "What kind of review?"},
                    ],
                    "updated_at": 200.0,
                },
            },
        }

        key = _conversation_key(FakeMessage("message-3", "example_user", "Review."), "example_user", state)

        self.assertEqual(key, "private:example_user")
        self.assertEqual(
            [row["text"] for row in state["private_conversations"][key]["history"]],
            ["Review this profile.", "Which profile?", "example_candidate", "What kind of review?"],
        )
        self.assertEqual(state["private_message_conversation_ids"]["message-1"], key)
        self.assertNotIn("private:example_user:message-1", state["private_conversations"])

    def test_private_history_migration_does_not_merge_an_explicit_room_by_id_collision(self):
        explicit_key = "private:example_user:room-1"
        state = {
            "private_message_conversation_ids": {
                # A message ID can coincidentally equal another conversation's
                # ID. Only self-mapped message-keyed rows are legacy rows.
                "room-1": "private:example_user:another-room",
            },
            "private_conversations": {
                explicit_key: {
                    "history": [{"role": "user", "text": "Keep this room private."}],
                    "updated_at": 100.0,
                },
            },
        }
        message = FakeMessage("next-message", "example_user", "Follow-up")
        message.conversation_id = "room-1"

        key = _conversation_key(message, "example_user", state)

        self.assertEqual(key, explicit_key)
        self.assertEqual(
            [row["text"] for row in state["private_conversations"][explicit_key]["history"]],
            ["Keep this room private."],
        )
        self.assertNotIn("private:example_user", state["private_conversations"])

    def test_verified_moderator_general_chat_uses_jev_then_replies_normally(self):
        message = FakeMessage(
            "mod-ordinary-chat", "mod", "What do you think of today's AI news?",
            created_utc=datetime.now(timezone.utc).timestamp(),
        )
        reddit = FakeReddit(moderators=("mod",))
        reddit.inbox = FakeInbox([message])
        model = JsonModel("A normal conversational answer.")
        state = {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0}

        jev = RouteChoiceJev("conversation")
        with patch("private_chat_handler.route_public_interaction") as public_route:
            replies, _tokens, _cost, _state, metrics = process_private_chat(
                reddit, model, state, "OptimistPrime", jev_client=jev
            )

        self.assertEqual(replies, 1)
        self.assertEqual(metrics.messages_failed, 0)
        self.assertEqual(message.replies[0][0], "A normal conversational answer.")
        self.assertEqual(len(model.calls), 1)
        public_route.assert_not_called()
        self.assertEqual(jev.calls_used, 1)

    def test_private_chat_does_not_filter_human_username_ending_in_bot(self):
        message = FakeMessage("example_human_bot-chat", "example_human_bot", "Hello Optimist")
        reddit = FakeReddit(moderators=())
        reddit.inbox = FakeInbox([message])
        model = JsonModel("Hello, example_human_bot!")
        jev = RouteChoiceJev("conversation")

        replies, _tokens, _cost, _state, metrics = process_private_chat(
            reddit,
            model,
            {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0},
            "OptimistPrime",
            jev_client=jev,
        )

        self.assertEqual(replies, 1)
        self.assertEqual(metrics.messages_handled, 1)
        self.assertEqual(message.replies[0][0], "Hello, example_human_bot!")

    def test_moderator_private_no_reply_decision_is_not_sent_as_chat_text(self):
        message = FakeMessage("mod-no-reply", "mod", "Good guess!")
        reddit = FakeReddit(moderators=("mod",))
        reddit.inbox = FakeInbox([message])
        model = JsonModel("The no-reply route must not generate prose.")
        jev = RouteChoiceJev("no_reply")
        state = {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0}
        persist_state = Mock()

        replies, _tokens, _cost, state, metrics = process_private_chat(
            reddit, model, state, "OptimistPrime", persist_state=persist_state,
            jev_client=jev,
        )

        self.assertEqual(replies, 0)
        self.assertEqual(message.replies, [])
        self.assertTrue(message.read)
        self.assertEqual(metrics.messages_handled, 1)
        self.assertEqual(metrics.messages_failed, 0)
        self.assertIn("mod-no-reply", state["processed_private_messages"])
        history = next(iter(state["private_conversations"].values()))["history"]
        self.assertEqual(history, [{
            "role": "user",
            "text": "Good guess!",
            "source_message_id": "mod-no-reply",
        }])
        self.assertNotIn(
            "[[NO_REPLY]]",
            json.dumps(state["private_conversations"]),
        )
        self.assertEqual(persist_state.call_args.args[0], state)
        self.assertEqual(model.calls, [])
        self.assertEqual(jev.calls_used, 1)

    def test_private_transport_refuses_internal_no_reply_marker(self):
        message = FakeMessage("marker-boundary", "mod", "No reply needed")

        with self.assertRaisesRegex(ValueError, "Internal no-reply marker"):
            send_message_reply(message, " [[NO_REPLY]] ")

        self.assertEqual(message.replies, [])

    def test_ordinary_private_no_reply_marker_is_not_sent_as_chat_text(self):
        message = FakeMessage("user-no-reply", "user", "A reaction that needs no answer")
        reddit = FakeReddit(moderators=())
        reddit.inbox = FakeInbox([message])
        model = JsonModel("This prose model must not be called")
        jev = RouteChoiceJev("no_reply")
        state = {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0}

        replies, _tokens, _cost, state, metrics = process_private_chat(
            reddit, model, state, "OptimistPrime", jev_client=jev
        )

        self.assertEqual(replies, 0)
        self.assertEqual(message.replies, [])
        self.assertTrue(message.read)
        self.assertEqual(metrics.messages_handled, 1)
        self.assertEqual(metrics.messages_failed, 0)
        self.assertEqual(get_daily_non_moderator_reply_count(state), 0)
        self.assertIn("user-no-reply", state["processed_private_messages"])
        history = next(iter(state["private_conversations"].values()))["history"]
        self.assertEqual(history, [{
            "role": "user",
            "text": "A reaction that needs no answer",
            "source_message_id": "user-no-reply",
        }])
        self.assertNotIn("[[NO_REPLY]]", json.dumps(state))
        self.assertEqual(model.calls, [])
        self.assertEqual(jev.calls_used, 1)

    def test_private_chat_uses_one_jev_choice_then_generates_only_selected_conversation(self):
        message = FakeMessage("user-conversation", "user", "What do you think about this idea?")
        reddit = FakeReddit(moderators=())
        reddit.inbox = FakeInbox([message])
        model = JsonModel("A useful answer.")
        jev = RouteChoiceJev("conversation")

        replies, tokens, cost, _state, metrics = process_private_chat(
            reddit,
            model,
            {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0},
            "OptimistPrime",
            jev_client=jev,
        )

        self.assertEqual(replies, 1)
        self.assertEqual(message.replies[0][0], "A useful answer.")
        self.assertEqual(jev.calls_used, 1)
        self.assertEqual(len(jev.requests), 1)
        _state, questions = jev.requests[0]
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions["public_interaction"]["type"], "choice")
        self.assertEqual(tokens, 28)
        self.assertGreater(cost, 0)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(metrics.messages_failed, 0)

    def test_private_chat_uses_jev_for_clarification_and_sends_no_generated_prose(self):
        message = FakeMessage("user-clarify", "user", "Can you do that thing for me?")
        reddit = FakeReddit(moderators=())
        reddit.inbox = FakeInbox([message])
        model = JsonModel("This must not be used.")
        jev = RouteChoiceJev("clarify")

        process_private_chat(
            reddit,
            model,
            {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0},
            "OptimistPrime",
            jev_client=jev,
        )

        self.assertEqual(message.replies[0][0], "I’m not sure what you’d like me to do. Could you say a bit more?")
        self.assertEqual(jev.calls_used, 1)
        self.assertEqual(model.calls, [])

    def test_private_chat_does_not_process_non_moderator_moderation_requests(self):
        message = FakeMessage("user-ban-request", "user", "Optimist, ban that user")
        reddit = FakeReddit(moderators=())
        reddit.inbox = FakeInbox([message])
        model = JsonModel("This must not be used.")
        jev = RouteChoiceJev("unsupported_mod_action")

        process_private_chat(
            reddit,
            model,
            {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0},
            "OptimistPrime",
            jev_client=jev,
        )

        self.assertIn("moderator-only", message.replies[0][0])
        self.assertEqual(jev.calls_used, 1)
        self.assertEqual(model.calls, [])

    def test_private_custom_flair_uses_jev_intent_override_without_reclassifying(self):
        message = FakeMessage(
            "user-custom-flair", "user", "Optimist, set my flair to 'Community volunteer'"
        )
        reddit = FakeReddit(moderators=())
        reddit.inbox = FakeInbox([message])
        model = JsonModel("This must not be used.")
        jev = RouteChoiceJev("self_custom_flair.set")
        state = {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0}

        with patch("private_chat_handler.config.USER_CUSTOM_FLAIR_ENABLED", True), patch(
            "private_chat_handler.handle_self_custom_flair_request",
            return_value=("Done.", 0, 0.0, state),
        ) as handler:
            process_private_chat(
                reddit, model, state, "OptimistPrime", jev_client=jev
            )

        handler.assert_called_once()
        self.assertEqual(handler.call_args.kwargs["intent_override"], {
            "action": "set",
            "label": "Community volunteer",
        })
        self.assertEqual(jev.calls_used, 1)
        self.assertEqual(model.calls, [])

    def test_incomplete_jev_route_never_falls_back_to_keyword_or_prose_classifier(self):
        message = FakeMessage("user-route-failure", "user", "Could you review that?")
        reddit = FakeReddit(moderators=())
        reddit.inbox = FakeInbox([message])
        model = JsonModel("This must not be used.")
        jev = RouteChoiceJev(None)
        state = {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0}

        with self.assertLogs("optimist_prime", level="ERROR"):
            replies, _tokens, _cost, state, metrics = process_private_chat(
                reddit, model, state, "OptimistPrime", jev_client=jev
            )

        self.assertEqual(replies, 0)
        self.assertEqual(metrics.messages_failed, 1)
        self.assertFalse(metrics.scan_complete)
        self.assertFalse(message.read)
        self.assertEqual(model.calls, [])
        self.assertEqual(jev.calls_used, 1)
        self.assertIn("user-route-failure", state["private_message_failures"])

    def test_private_action_choices_follow_enabled_self_service_features(self):
        message = FakeMessage("user-feature-gate", "user", "Hello Optimist")
        reddit = FakeReddit(moderators=())
        reddit.inbox = FakeInbox([message])
        model = JsonModel("A useful answer.")
        jev = RouteChoiceJev("conversation")

        with patch("private_chat_handler.config.USER_CUSTOM_FLAIR_ENABLED", False):
            process_private_chat(
                reddit,
                model,
                {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0},
                "OptimistPrime",
                jev_client=jev,
            )

        choices = jev.requests[0][1]["public_interaction"]["criteria"]
        self.assertFalse(any(name.startswith("self_custom_flair.") for name in choices))

    def test_queued_private_no_reply_marker_is_discarded_without_sending(self):
        message = FakeMessage("queued-no-reply", "user", "Follow-up after an abstention")
        reddit = FakeReddit(moderators=())
        reddit.inbox = FakeInbox([message])
        model = JsonModel("This model must not be called")
        jev = RouteChoiceJev("conversation")
        conversation_key = "private:user:queued-thread"
        state = {
            "daily_non_moderator_replies": 0,
            "_cycle_non_moderator_replies": 0,
            "private_message_pending_replies": {
                "queued-no-reply": {
                    "text": " [[NO_REPLY]] ",
                    "conversation_key": conversation_key,
                }
            },
        }

        replies, _tokens, _cost, state, metrics = process_private_chat(
            reddit, model, state, "OptimistPrime", jev_client=jev
        )

        self.assertEqual(replies, 0)
        self.assertEqual(message.replies, [])
        self.assertTrue(message.read)
        self.assertEqual(model.calls, [])
        self.assertEqual(jev.calls_used, 0)
        self.assertEqual(metrics.messages_failed, 0)
        self.assertIn("queued-no-reply", state["processed_private_messages"])
        self.assertEqual(state["private_message_pending_replies"], {})
        self.assertEqual(
            state["private_conversations"][conversation_key]["history"],
            [{
                "role": "user",
                "text": "Follow-up after an abstention",
                "source_message_id": "queued-no-reply",
            }],
        )

    def test_one_failed_private_message_does_not_block_later_messages(self):
        failed = FakeMessage("mod-failure", "mod", "Please review this")
        later = FakeMessage("user-after-failure", "user", "Hello Optimist")
        reddit = FakeReddit(moderators=("mod",))
        reddit.inbox = FakeInbox([failed, later])
        model = JsonModel("A normal conversational answer")
        processing_error = ValueError("private payload must not be logged")
        processing_error.__cause__ = RuntimeError("provider failure details")
        state = {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0}

        with patch(
            "private_chat_handler.process_moderator_turn",
            side_effect=[processing_error, None],
        ) as control, self.assertLogs("optimist_prime", level="ERROR") as captured:
            replies, _tokens, _cost, state, metrics = process_private_chat(
                reddit, model, state, "OptimistPrime",
                jev_client=RouteChoiceJev("conversation"),
            )

        self.assertEqual(replies, 1)
        self.assertFalse(failed.read)
        self.assertTrue(later.read)
        self.assertEqual(len(later.replies), 1)
        self.assertEqual(control.call_count, 1)
        self.assertFalse(metrics.scan_complete)
        self.assertEqual(metrics.messages_failed, 1)
        self.assertEqual(metrics.messages_handled, 2)
        failure = state["private_message_failures"]["mod-failure"]
        self.assertEqual(failure["error_type"], "ValueError")
        self.assertEqual(failure["stage"], "moderator_action_selection_and_response")
        self.assertNotIn("Please review this", str(failure))
        self.assertNotIn("private payload must not be logged", captured.output)
        self.assertIn("stack_trace", captured.records[0].__dict__)
        log_record = json.loads(JsonLineFormatter().format(captured.records[0]))
        self.assertIn("moderator_action_selection_and_response", log_record["operationStage"])
        self.assertEqual(log_record["causeErrorType"], "RuntimeError")
        self.assertNotIn("private payload must not be logged", str(log_record))
        self.assertNotIn("provider failure details", str(log_record))

    def test_failed_private_message_is_deferred_and_kept_retryable(self):
        message = FakeMessage("mod-retry", "mod", "Please check this request")
        reddit = FakeReddit(moderators=("mod",))
        reddit.inbox = FakeInbox([message])
        state = {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0}

        with patch(
            "private_chat_handler.process_moderator_turn",
            side_effect=ValueError("empty classifier response"),
        ) as control, self.assertLogs("optimist_prime", level="ERROR"):
            _replies, _tokens, _cost, state, first = process_private_chat(
                reddit, JsonModel("unused"), state, "OptimistPrime"
            )
            _replies, _tokens, _cost, state, second = process_private_chat(
                reddit, JsonModel("unused"), state, "OptimistPrime"
            )

        self.assertEqual(first.messages_failed, 1)
        self.assertEqual(second.messages_deferred, 1)
        self.assertFalse(first.scan_complete)
        self.assertFalse(second.scan_complete)
        self.assertEqual(control.call_count, 1)
        self.assertFalse(message.read)

    def test_jev_failure_sends_one_saved_message_without_retrying_analysis(self):
        message = FakeMessage("jev-failed-review", "mod", "review u/example_candidate")
        reddit = FakeReddit(moderators=("mod",))
        reddit.inbox = FakeInbox([message])
        state = {
            "daily_non_moderator_replies": 0,
            "_cycle_non_moderator_replies": 0,
            "private_message_failures": {
                "jev-failed-review": {
                    "attempts": 3,
                    "retry_after": datetime.now(timezone.utc).timestamp() + 3600,
                    "stage": "moderator_action_selection_and_response",
                    "error_type": "JevHistoryReviewError",
                },
            },
        }
        persist_state = Mock()

        with patch("private_chat_handler.process_moderator_turn") as turn:
            replies, _tokens, _cost, state, metrics = process_private_chat(
                reddit, JsonModel("unused"), state, "OptimistPrime",
                persist_state=persist_state,
            )

        self.assertEqual(replies, 1)
        self.assertEqual(metrics.messages_failed, 0)
        self.assertEqual(metrics.messages_deferred, 0)
        self.assertEqual(len(message.replies), 1)
        self.assertIn("Jev couldn't complete the history review", message.replies[0][0])
        self.assertIn("won't retry the same analysis", message.replies[0][0])
        self.assertTrue(message.read)
        self.assertIn("jev-failed-review", state["processed_private_messages"])
        self.assertNotIn("jev-failed-review", state["private_message_failures"])
        self.assertNotIn("jev-failed-review", state["private_message_pending_replies"])
        turn.assert_not_called()
        self.assertGreaterEqual(persist_state.call_count, 2)

    def test_private_messages_share_limits_and_moderators_bypass_call_cap(self):
        ordinary_one = FakeMessage("u1", "user1", "Hello Optimist")
        ordinary_two = FakeMessage("u2", "user2", "Hello there")
        reddit = FakeReddit(moderators=("mod",))
        reddit.inbox = FakeInbox([ordinary_one, ordinary_two])
        ordinary_model = JsonModel("First answer")
        jev = RouteChoiceJev("conversation")
        state = {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0}
        with patch("private_chat_handler.config.MAX_REPLIES_PER_RUN", 1):
            count, _tokens, _cost, state, metrics = process_private_chat(
                reddit, ordinary_model, state, "OptimistPrime", jev_client=jev
            )
        self.assertEqual(count, 1)
        self.assertEqual(len(ordinary_one.replies), 1)
        self.assertEqual(len(ordinary_two.replies), 0)
        self.assertFalse(ordinary_two.read)
        self.assertEqual(get_daily_non_moderator_reply_count(state), 1)
        self.assertEqual(state["_cycle_non_moderator_replies"], 1)
        self.assertEqual(metrics.messages_handled, 1)
        self.assertEqual(jev.calls_used, 1)
        self.assertTrue(metrics.scan_complete)

        moderator_message = FakeMessage("mod1", "mod", "Hi Optimist")
        reddit.inbox = FakeInbox([moderator_message])
        limited = RateLimitedLLMModel(JsonModel("Moderator answer"), 0)
        count, _tokens, _cost, state, metrics = process_private_chat(
            reddit, limited, state, "OptimistPrime"
        )
        self.assertEqual(count, 1)
        self.assertEqual(len(moderator_message.replies), 1)
        self.assertEqual(metrics.messages_handled, 1)

    def test_private_chat_scheduler_count_excludes_duplicates_and_limited_messages(self):
        eligible = FakeMessage("same-id", "user1", "Hello Optimist")
        limited = FakeMessage("limited-id", "user2", "Hello Optimist")
        reddit = FakeReddit(moderators=())
        reddit.inbox = FakeInbox([eligible, eligible, limited])
        model = JsonModel("Answer")
        jev = RouteChoiceJev("conversation")
        state = {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0}
        with patch("private_chat_handler.config.MAX_REPLIES_PER_RUN", 1):
            _replies, _tokens, _cost, _state, metrics = process_private_chat(
                reddit, model, state, "OptimistPrime", jev_client=jev
            )
        self.assertEqual(metrics.messages_handled, 1)
        self.assertEqual(jev.calls_used, 1)

    def test_private_chat_scan_failure_is_reported_as_incomplete(self):
        with patch(
            "private_chat_handler.unread_private_messages",
            side_effect=TimeoutError("temporary inbox failure"),
        ):
            _replies, _tokens, _cost, _state, metrics = process_private_chat(
                FakeReddit(moderators=()), JsonModel("Unused"), {}, "OptimistPrime"
            )
        self.assertFalse(metrics.scan_complete)
        self.assertEqual(metrics.messages_handled, 0)

    def test_private_context_follows_message_parent_without_cross_user_leakage(self):
        first = FakeMessage("root-a", "example_user", "I like gardening")
        second = FakeMessage("follow-a", "example_user", "Can you expand?", parent_id="answer-root-a")
        other = FakeMessage("root-b", "example_user_2", "What are the community guidelines?")
        reddit = FakeReddit(moderators=())
        reddit.inbox = FakeInbox([first])
        model = JsonModel("First answer")
        jev = RouteChoiceJev("conversation")
        state = {"daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0}
        process_private_chat(reddit, model, state, "OptimistPrime", jev_client=jev)
        model.calls.clear()
        reddit.inbox = FakeInbox([second, other])
        state["_cycle_non_moderator_replies"] = 0
        model.responses = ["Follow-up answer", "Separate answer"]
        process_private_chat(reddit, model, state, "OptimistPrime", jev_client=jev)
        prompt_followup = model.calls[0][0][0]["parts"][0]
        prompt_other = model.calls[1][0][0]["parts"][0]
        self.assertIn("I like gardening", prompt_followup)
        self.assertNotIn("I like gardening", prompt_other)


    def test_group_chat_is_explicitly_unavailable(self):
        self.assertFalse(NATIVE_GROUP_CHAT_SUPPORTED)

class SharedReplyAccountingTests(unittest.TestCase):
    def test_comment_and_private_successes_update_one_shared_counter(self):
        state = {"daily_replies": 0, "daily_non_moderator_replies": 0, "_cycle_non_moderator_replies": 0}
        record_reply_success(state, moderator=False)
        record_reply_success(state, moderator=False)
        record_reply_success(state, moderator=True)
        self.assertEqual(get_daily_non_moderator_reply_count(state), 2)
        self.assertEqual(state["_cycle_non_moderator_replies"], 2)
        self.assertEqual(state["daily_replies"], 3)


if __name__ == "__main__":
    unittest.main()
