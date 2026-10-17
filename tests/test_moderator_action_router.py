"""Tests for single-choice moderator action selection and guarded dispatch."""

from __future__ import annotations

import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from moderator_action_router import (
    interpret_moderator_turn as _interpret_moderator_turn,
    process_moderator_turn as _process_moderator_turn,
)
from history_gate_jev import JevHistoryReviewError
from bot_logging import JsonLineFormatter


def _result(route, **values):
    defaults = {
        "route": route,
        "action": "",
        "target_username": None,
        "history_question": "",
        "needs_history": False,
        "wants_flair": False,
        "exact_flair_label": "",
        "suggest_flair": False,
        "flair_label": "",
        "question": "",
        "confidence": 0.98,
        "reply_text": "",
    }
    defaults.update(values)
    return json.dumps(defaults)


class _Jev:
    def __init__(self, payload, *, choice_override=None, confidence=None):
        self.payload = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
        self.choice_override = choice_override
        self.confidence_override = confidence
        self.omit_confidence = bool(self.payload.get("omit_confidence", False))
        self.calls = []
        self.ready = True
        self.max_calls = 30
        self.calls_used = 0
        self.last_usage = {"input_tokens": 19, "cost": 0.0001}
        self.last_error = None

    def choose_many_detailed(self, *, state, questions):
        self.calls.append((state, questions))
        self.calls_used += 1
        if self.choice_override is not None:
            choice = self.choice_override
        else:
            route = str(self.payload.get("route") or "")
            action = str(self.payload.get("action") or "")
            choice = f"{route}.{action}" if route == "self_custom_flair" else route
        confidence = (
            self.confidence_override
            if self.confidence_override is not None
            else self.payload.get("confidence", 0.98)
        )
        choice_probability = self.payload.get("choice_probability", confidence)
        action_detail = {"choice": choice, "probabilities": {choice: choice_probability}}
        if not self.omit_confidence:
            action_detail["confidence"] = confidence
        else:
            action_detail["probabilities"] = {}
        return {"moderator_action": action_detail}


class _Model:
    def __init__(self, payload):
        self.payload = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
        self.calls = []
        self.jev = _Jev(self.payload)

    def generate_content(self, prompt, generation_config=None):
        if not self.jev.calls:
            raise AssertionError("prose generation ran before Jev selected a route")
        self.calls.append((prompt, generation_config))
        text = ""
        if "route is already fixed to ordinary conversation" in prompt.casefold():
            text = str(self.payload.get("reply_text") or "")
        elif "route is already fixed to clarification" in prompt.casefold():
            text = str(self.payload.get("question") or "")
        return SimpleNamespace(
            text=text,
            usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=7),
        )


def interpret_moderator_turn(text, model, **kwargs):
    kwargs.setdefault("jev_client", getattr(model, "jev", None))
    return _interpret_moderator_turn(text, model, **kwargs)


def process_moderator_turn(*args, **kwargs):
    model = args[3] if len(args) > 3 else None
    kwargs.setdefault("jev_client", getattr(model, "jev", None))
    return _process_moderator_turn(*args, **kwargs)


class ModeratorActionRouterTests(unittest.TestCase):
    def setUp(self):
        self.activity_writer = patch("moderator_action_router.record_moderator_activity")
        self.record_activity = self.activity_writer.start()
        self.addCleanup(self.activity_writer.stop)

    def test_same_moderator_decision_can_abstain_without_second_model_call(self):
        model = _Model(_result("no_reply"))
        state = {}
        result = process_moderator_turn(
            object(), object(), state, model,
            author_name="mod", text="Thanks Optimist",
            channel="comment", source_id="comment-ack",
            conversation_key="comment:mod:thread",
            bot_username="example_bot", verified_moderator=True,
        )

        self.assertEqual(result["route"], "no_reply")
        self.assertEqual(result["conversation_response"], "[[NO_REPLY]]")
        self.assertEqual(len(model.calls), 0)
        self.assertEqual(len(model.jev.calls), 1)
        self.assertIn("no_reply", model.jev.calls[0][1]["moderator_action"]["criteria"])

    def test_one_model_call_sees_full_menu_and_returns_ordinary_reply(self):
        model = _Model(_result("conversation", reply_text="A direct answer."))

        with patch("moderator_action_router.config.USER_CUSTOM_FLAIR_ENABLED", True):
            intent = interpret_moderator_turn(
                "What do you think about the updated community guidelines?",
                model,
                trusted_context="Earlier moderator discussion.",
            )

        self.assertEqual(intent["route"], "conversation")
        self.assertEqual(intent["reply_text"], "A direct answer.")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.jev.calls), 1)
        prompt = model.calls[0][0]
        self.assertIn("route is already fixed to ordinary conversation", prompt)
        criteria = model.jev.calls[0][1]["moderator_action"]["criteria"]
        for choice in (
            "history_review", "self_custom_flair.set", "conversation", "no_reply", "clarify",
        ):
            self.assertIn(choice, criteria)
        self.assertIn("sole route and action selection", model.jev.calls[0][1]["moderator_action"]["instructions"])

    def test_jev_choice_covers_every_enabled_route_and_action(self):
        cases = (
            ("history_review", "review u/example_user", "history_review", "review"),
                        ("self_custom_flair.set", "Set my custom flair to Community volunteer", "self_custom_flair", "set"),
                        ("conversation", "Hi Optimist, how are you?", "conversation", ""),
            ("no_reply", "Thanks, that's all.", "no_reply", ""),
            ("clarify", "Could you handle that thing?", "clarify", "clarify"),
        )
        with patch("moderator_action_router.config.USER_CUSTOM_FLAIR_ENABLED", True):
            for choice, text, expected_route, expected_action in cases:
                with self.subTest(choice=choice):
                    model = _Model(_result(
                        "conversation", reply_text="Hello there.",
                        question="What would you like me to handle?",
                    ))
                    model.jev.choice_override = choice
                    intent = interpret_moderator_turn(text, model, channel="private")

                    self.assertEqual(intent["route"], expected_route)
                    if expected_action:
                        self.assertEqual(intent["action"], expected_action)
                    self.assertEqual(len(model.jev.calls), 1)
                    should_generate = expected_route in {"conversation", "clarify"}
                    self.assertEqual(len(model.calls), int(should_generate))

    def test_greeting_is_conversation_only_after_jev_selects_it(self):
        model = _Model(_result("conversation", reply_text="Hi! Good to hear from you."))
        model.jev.choice_override = "conversation"

        result = interpret_moderator_turn("Hi Optimist!", model, channel="private")

        self.assertEqual(result["route"], "conversation")
        self.assertEqual(result["reply_text"], "Hi! Good to hear from you.")
        self.assertEqual(len(model.jev.calls), 1)
        self.assertEqual(len(model.calls), 1)

    def test_quoted_and_negated_actions_remain_the_current_turn_data(self):
        examples = (
            'A member wrote "review u/example_user"; I am asking what that phrase means.',
            "Don't review u/example_user; I only want to understand the request she sent.",
        )
        for text in examples:
            with self.subTest(text=text):
                model = _Model(_result("conversation", reply_text="It refers to the moderator action being discussed."))
                model.jev.choice_override = "conversation"

                result = interpret_moderator_turn(text, model, channel="private")

                self.assertEqual(result["route"], "conversation")
                self.assertEqual(model.jev.calls[0][0]["current_moderator_turn"], text)
                instructions = model.jev.calls[0][1]["moderator_action"]["instructions"].casefold()
                self.assertIn("quoted", instructions)
                self.assertIn("negated", instructions)
                self.assertEqual(len(model.jev.calls), 1)

    def test_invalid_jev_choice_does_not_fall_back_to_generation_or_another_route(self):
        model = _Model(_result("conversation", reply_text="This must not be used."))
        model.jev.choice_override = "unsupported.route_typo"

        result = interpret_moderator_turn("Review u/example_user", model, channel="private")

        self.assertEqual(result["route"], "decision_unavailable")
        self.assertEqual(len(model.jev.calls), 1)
        self.assertEqual(len(model.calls), 0)

    def test_unavailable_jev_does_not_invent_a_route_or_generate_prose(self):
        model = _Model(_result("conversation", reply_text="This must not be used."))

        result = _interpret_moderator_turn(
            "Hi Optimist", model, jev_client=None, channel="private",
        )

        self.assertEqual(result["route"], "decision_unavailable")
        self.assertEqual(result["token_info"]["total_tokens"], 0)
        self.assertEqual(len(model.jev.calls), 0)
        self.assertEqual(len(model.calls), 0)

    def test_failed_jev_turn_reports_failure_once_and_does_not_retry_same_message(self):
        model = _Model(_result("no_reply"))
        model.jev.ready = False
        state = {}
        arguments = dict(
            author_name="mod", text="review u/example_user", channel="private",
            source_id="failed-jev-message", conversation_key="private:mod",
            bot_username="example_bot", verified_moderator=True,
        )

        first = process_moderator_turn(object(), object(), state, model, **arguments)
        second = process_moderator_turn(object(), object(), state, model, **arguments)

        self.assertEqual(first["route"], "decision_unavailable")
        self.assertTrue(first["decision_failed"])
        self.assertIn("Jev couldn't process", first["response"])
        self.assertEqual(second["route"], "decision_unavailable")
        self.assertEqual(model.jev.calls, [])
        self.assertEqual(model.calls, [])

    def test_pending_followup_context_resolves_only_a_jev_selected_history_route(self):
        pending = {
            "route": "history_review", "action": "review",
            "target_username": "example_candidate", "needs_history": True,
            "history_question": "Does their history show repeated sharing of one link?",
            "created_at": time.time(),
        }
        selected_history = _Model(_result("history_review"))
        history_intent = interpret_moderator_turn(
            "review", selected_history, channel="private", prior_action=pending,
        )
        self.assertEqual(history_intent["route"], "history_review")
        self.assertEqual(history_intent["target_username"], "example_candidate")
        self.assertEqual(
            history_intent["history_question"],
            "Does their history show repeated sharing of one link?",
        )
        self.assertEqual(
            selected_history.jev.calls[0][0]["prior_action_metadata_from_code"], pending,
        )

        selected_conversation = _Model(_result("conversation", reply_text="Okay."))
        conversation_intent = interpret_moderator_turn(
            "example_candidate", selected_conversation,
            channel="private", prior_action=pending,
        )
        self.assertEqual(conversation_intent["route"], "conversation")
        self.assertEqual(len(selected_conversation.calls), 1)

    def test_top_level_moderator_post_requires_an_explicit_summon(self):
        model = _Model(_result("not_addressed"))
        intent = interpret_moderator_turn(
            "Optimist Prime made an interesting point.",
            model,
            channel="post",
        )

        self.assertEqual(intent["route"], "not_addressed")
        instructions = model.jev.calls[0][1]["moderator_action"]["instructions"]
        self.assertIn("only act or reply when the post explicitly", instructions)
        self.assertIn("A passing mention or discussion of Optimist does not invite", instructions)

    def test_history_route_validates_target_and_dispatches_without_another_router_call(self):
        profile_link = (
            "Review https://www.reddit.com/user/example_candidate/; "
            "I want to know whether they repeatedly share one link across posts."
        )
        model = _Model(_result(
            "history_review", action="review", target_username="example_candidate",
            needs_history=True,
            history_question="Do they repeatedly share one link across posts?",
        ))
        state = {}
        persisted_json = []
        interaction = SimpleNamespace(author=SimpleNamespace(name="mod"), body=profile_link)
        arguments = dict(
            author_name="mod", text=profile_link, channel="comment",
            source_id="message-2", conversation_key="private:mod:2",
            bot_username="example_bot", verified_moderator=True,
            interaction=interaction, jev_client=model.jev,
            persist_state=lambda saved: persisted_json.append(json.dumps(saved)),
        )
        with patch("moderator_action_router.refresh_moderators", return_value={"mod"}), \
             patch("moderator_history_commands.handle_moderator_history_request",
                   return_value=("History report", 9, 0.02, state)) as report:
            result = process_moderator_turn(
                object(), object(), state, model, **arguments
            )
            duplicate = process_moderator_turn(
                object(), object(), state, model, **arguments
            )

        self.assertEqual(result["response"], "History report")
        self.assertIs(result["state"], state)
        self.assertEqual(result["token_info"]["total_tokens"], 28)
        self.assertEqual(len(model.calls), 0)
        self.assertEqual(report.call_args.args[5]["target_username"], "example_candidate")
        self.assertEqual(
            report.call_args.args[5]["question"],
            "whether they repeatedly share one link across posts",
        )
        self.assertTrue(report.call_args.kwargs["requester_is_moderator"])
        self.assertNotIn("notify_moderators", report.call_args.kwargs)
        self.assertEqual(duplicate["response"], "History report")
        self.assertIs(duplicate["state"], state)
        self.assertEqual(len(model.calls), 0)
        self.assertEqual(len(persisted_json), 1)
        cached = state["processed_moderator_turns"][
            "comment:private:mod:2:message-2"
        ]
        self.assertNotIn("state", cached)
        self.assertTrue(cached["_cached_with_state"])
        json.dumps(state)  # Cached history results must not point back to their container.

    def test_jev_history_failure_returns_one_cached_nonretryable_reply(self):
        model = _Model(_result(
            "history_review", action="review", target_username="example_candidate",
            needs_history=True,
        ))
        state = {}
        arguments = dict(
            author_name="mod", text="review u/example_candidate", channel="private",
            source_id="jev-failure", conversation_key="private:mod:jev-failure",
            bot_username="example_bot", verified_moderator=True,
            jev_client=model.jev,
        )
        model.jev.last_error = {
            "type": "HTTPError", "status_code": 400, "reason": "Bad Request",
        }
        with patch("moderator_action_router.refresh_moderators", return_value={"mod"}), \
             patch("moderator_history_commands.handle_moderator_history_request",
                   side_effect=JevHistoryReviewError("Jev returned no complete result")) as review, \
             self.assertLogs("optimist_prime", level="WARNING") as captured:
            result = process_moderator_turn(object(), object(), state, model, **arguments)
            duplicate = process_moderator_turn(object(), object(), state, model, **arguments)

        self.assertFalse(result["success"])
        self.assertFalse(result["retryable"])
        self.assertIn("there isn't a pass or fail result yet", result["response"])
        self.assertIn("I didn't retry it automatically", result["response"])
        self.assertEqual(duplicate["response"], result["response"])
        self.assertEqual(review.call_count, 1)
        self.assertEqual(len(model.calls), 0)
        logged = json.loads(JsonLineFormatter().format(captured.records[0]))
        self.assertEqual(logged["providerErrorType"], "HTTPError")
        self.assertEqual(logged["providerStatusCode"], 400)
        self.assertEqual(logged["providerErrorReason"], "Bad Request")
        self.assertEqual(logged["failureReason"], "Jev returned no complete result")

    def test_private_history_followup_resumes_the_saved_review_instead_of_reasking(self):
        state = {}
        first = _Model(_result(
            "history_review", action="review", needs_history=True,
        ))
        first_result = process_moderator_turn(
            object(), object(), state, first,
            author_name="mod", text="Review the linked user's history",
            channel="private", source_id="review-request",
            conversation_key="private:mod", bot_username="example_bot",
            verified_moderator=True,
        )
        self.assertEqual(first_result["route"], "clarify")
        self.assertEqual(
            state["conversation_contexts"]["private:mod"]["pending_moderator_action"]["route"],
            "history_review",
        )

        second = _Model(_result("history_review", action="review"))
        with patch("moderator_action_router.refresh_moderators", return_value={"mod"}), \
             patch("moderator_history_commands.handle_moderator_history_request",
                   return_value=("History report ready.", 0, 0.0, state)) as review:
            result = process_moderator_turn(
                object(), object(), state, second,
                author_name="mod", text="example_candidate",
                channel="private", source_id="review-target",
                conversation_key="private:mod", bot_username="example_bot",
                verified_moderator=True,
            )

        self.assertEqual(result["route"], "history_review")
        self.assertEqual(result["response"], "History report ready.")
        self.assertEqual(review.call_args.args[5]["target_username"], "example_candidate")
        self.assertEqual(len(second.jev.calls), 1)
        self.assertEqual(len(second.calls), 0)
        self.assertNotIn(
            "pending_moderator_action",
            state["conversation_contexts"]["private:mod"],
        )

    def test_private_transcript_recovers_an_unfinished_history_review_followup(self):
        now = time.time() - 10
        history = [
            {"role": "user", "text": "review https://www.reddit.com/user/example_candidate/"},
            {"role": "assistant", "text": "Which user's history should I review? Please include their u/username."},
            {"role": "user", "text": "example_candidate"},
            {"role": "assistant", "text": "What would you like me to do with example_candidate — review their history, assign flair, or something else?"},
            {"role": "user", "text": "review"},
            {"role": "assistant", "text": "Who do you want me to review, and what specifically do you want to know about their history?"},
        ]
        model = _Model(_result("history_review", action="review"))
        state = {}
        with patch("moderator_action_router.record_moderator_activity"), \
             patch("moderator_action_router.refresh_moderators", return_value={"mod"}), \
             patch("moderator_history_commands.handle_moderator_history_request",
                   return_value=("History report ready.", 0, 0.0, state)) as review:
            result = process_moderator_turn(
                object(), object(), state, model,
                author_name="mod", text="review",
                channel="private", source_id="resume-old-transcript",
                conversation_key="private:mod",
                bot_username="example_bot", verified_moderator=True,
                conversation_history=history,
                conversation_updated_at=now,
            )

        self.assertEqual(result["route"], "history_review")
        self.assertEqual(review.call_args.args[5]["target_username"], "example_candidate")

    def test_moderator_activity_is_recorded_once_for_a_fresh_verified_turn(self):
        model = _Model(_result("no_reply"))
        arguments = dict(
            author_name="mod", text="Thanks, Optimist.", channel="private",
            source_id="activity-message", conversation_key="private:mod",
            bot_username="example_bot", verified_moderator=True,
        )
        state = {}
        process_moderator_turn(object(), object(), state, model, **arguments)
        process_moderator_turn(object(), object(), state, model, **arguments)

        self.record_activity.assert_called_once_with()

    def test_unverified_and_dry_run_turns_do_not_record_moderator_activity(self):
        model = _Model(_result("conversation", reply_text="A normal reply."))
        arguments = dict(
            author_name="user", text="Hello", channel="private",
            source_id="not-a-mod", conversation_key="private:user",
            bot_username="example_bot", verified_moderator=False,
        )
        self.assertIsNone(process_moderator_turn(object(), object(), {}, model, **arguments))
        arguments.update(
            author_name="mod", verified_moderator=True, source_id="dry-run",
        )
        process_moderator_turn(
            object(), object(), {}, model, dry_run=True, **arguments
        )
        self.record_activity.assert_not_called()

    def test_model_cannot_invent_a_third_party_target(self):
        model = _Model(_result(
            "history_review", action="review", target_username="example_user",
            needs_history=True,
        ))

        intent = interpret_moderator_turn("Review their history", model)

        self.assertEqual(intent["route"], "clarify")
        self.assertIn("Which user's history", intent["question"])
        self.assertEqual(len(model.calls), 0)

    def test_history_review_accepts_and_extracts_username_from_reddit_profile_url(self):
        message = "review https://www.reddit.com/user/example_candidate/"
        for model_target in ("example_candidate", None):
            with self.subTest(model_target=model_target):
                model = _Model(_result(
                    "history_review", action="review",
                    target_username=model_target, needs_history=True,
                ))

                intent = interpret_moderator_turn(message, model)

                self.assertEqual(intent["route"], "history_review")
                self.assertEqual(intent["target_username"], "example_candidate")
                self.assertTrue(intent["needs_history"])
                self.assertEqual(len(model.calls), 0)

    def test_history_review_route_defaults_to_general_review_when_focus_is_omitted(self):
        model = _Model(_result(
            "history_review", action="review", target_username="example_candidate",
        ))

        intent = interpret_moderator_turn(
            "review https://www.reddit.com/user/example_candidate/", model
        )

        self.assertEqual(intent["route"], "history_review")
        self.assertTrue(intent["needs_history"])
        self.assertEqual(intent["question"], "")


    def test_bare_target_in_moderator_private_chat_defaults_to_history_review(self):
        # A username by itself in private Chat selects a history review.
        model = _Model(_result(
            "history_review",
            target_username=None,
        ))
        state = {}
        with patch("moderator_action_router.refresh_moderators", return_value={"mod"}), \
             patch("moderator_history_commands.handle_moderator_history_request",
                   return_value=("History report", 9, 0.02, state)) as review:
            result = process_moderator_turn(
                object(), object(), state, model,
                author_name="mod", text="u/example_candidate", channel="private",
                source_id="target-only", conversation_key="private:mod:target-only",
                bot_username="example_bot", verified_moderator=True,
            )

        self.assertEqual(result["route"], "history_review")
        self.assertEqual(result["response"], "History report")
        self.assertEqual(review.call_args.args[5]["target_username"], "example_candidate")
        self.assertTrue(review.call_args.args[5]["needs_history"])
        self.assertEqual(len(model.calls), 0)

    def test_private_chat_history_review_accepts_bare_username_in_command(self):
        for model_target in (None, "example_candidate"):
            with self.subTest(model_target=model_target):
                model = _Model(_result(
                    "history_review", action="review", target_username=model_target,
                    needs_history=True,
                ))
                state = {}
                with patch("moderator_action_router.refresh_moderators", return_value={"mod"}), \
                     patch("moderator_history_commands.handle_moderator_history_request",
                           return_value=("History report", 9, 0.02, state)) as review:
                    result = process_moderator_turn(
                        object(), object(), state, model,
                        author_name="mod", text="review example_candidate", channel="private",
                        source_id=f"bare-target-{model_target}",
                        conversation_key=f"private:mod:bare-target-{model_target}",
                        bot_username="example_bot", verified_moderator=True,
                    )

                self.assertEqual(result["route"], "history_review")
                self.assertEqual(result["response"], "History report")
                self.assertEqual(review.call_args.args[5]["target_username"], "example_candidate")
                self.assertTrue(review.call_args.args[5]["needs_history"])
                self.assertEqual(len(model.calls), 0)

    def test_private_chat_history_review_does_not_use_username_placeholder_as_target(self):
        model = _Model(_result(
            "history_review", action="review", target_username=None,
            needs_history=True,
        ))
        with patch("moderator_action_router.refresh_moderators", return_value={"mod"}), \
             patch("moderator_history_commands.handle_moderator_history_request") as review:
            result = process_moderator_turn(
                object(), object(), {}, model,
                author_name="mod", text="review username", channel="private",
                source_id="username-placeholder", conversation_key="private:mod:placeholder",
                bot_username="example_bot", verified_moderator=True,
            )

        self.assertEqual(result["route"], "clarify")
        self.assertIn("Which user's history", result["response"])
        review.assert_not_called()

    def test_summoned_short_review_uses_parent_author_instead_of_asking_who(self):
        # Mirrors the public request: "hey optimist - review" as a reply to a user's comment.
        for action in ("review", "scan"):
            with self.subTest(action=action):
                model = _Model(_result("history_review", action="review"))
                state = {}
                with patch("moderator_action_router.refresh_moderators", return_value={"mod"}), \
                     patch("moderator_history_commands.handle_moderator_history_request",
                           return_value=("History report", 9, 0.02, state)) as review:
                    result = process_moderator_turn(
                        object(), object(), state, model,
                        author_name="mod", text=f"hey optimist - {action}", channel="comment",
                        source_id=f"short-{action}-summon", conversation_key="comment:mod:thread",
                        bot_username="example_bot", verified_moderator=True,
                        parent_author="comment_author", is_summon=True,
                    )

                self.assertEqual(result["route"], "history_review")
                self.assertEqual(result["response"], "History report")
                self.assertEqual(review.call_args.args[5]["target_username"], "comment_author")
                self.assertEqual(len(model.calls), 0)

    def test_summoned_short_review_without_parent_keeps_the_clarification(self):
        model = _Model(_result("history_review", action="review"))
        with patch("moderator_history_commands.handle_moderator_history_request") as review:
            result = process_moderator_turn(
                object(), object(), {}, model,
                author_name="mod", text="hey optimist - review", channel="comment",
                source_id="short-review-no-parent", conversation_key="comment:mod:thread",
                bot_username="example_bot", verified_moderator=True,
                parent_author=None, is_summon=True,
            )

        self.assertEqual(result["route"], "clarify")
        review.assert_not_called()

    def test_bare_profile_link_in_moderator_private_chat_defaults_to_history_review(self):
        model = _Model(_result("history_review", action="review"))
        state = {}
        with patch("moderator_action_router.refresh_moderators", return_value={"mod"}), \
             patch("moderator_history_commands.handle_moderator_history_request",
                   return_value=("History report", 9, 0.02, state)) as review:
            result = process_moderator_turn(
                object(), object(), state, model,
                author_name="mod", text="https://www.reddit.com/user/example_candidate/",
                channel="private", source_id="profile-only", conversation_key="private:mod:profile-only",
                bot_username="example_bot", verified_moderator=True,
            )

        self.assertEqual(result["route"], "history_review")
        self.assertEqual(review.call_args.args[5]["target_username"], "example_candidate")

    def test_contextual_review_comment_without_a_summon_does_not_start_history_review(self):
        model = _Model(_result(
            "clarify", question="What would you like me to do?",
        ))
        with patch("moderator_history_commands.handle_moderator_history_request") as review:
            result = process_moderator_turn(
                object(), object(), {}, model,
                author_name="mod", text="hey optimist - review", channel="comment",
                source_id="unsummoned-target", conversation_key="comment:mod:thread",
                bot_username="example_bot", verified_moderator=True,
                parent_author="parent_author", is_summon=False,
            )

        self.assertEqual(result["route"], "clarify")
        review.assert_not_called()

    def test_username_inside_a_sentence_is_not_treated_as_target_only(self):
        model = _Model(_result(
            "clarify", question="What would you like me to do?",
        ))
        with patch("moderator_history_commands.handle_moderator_history_request") as review:
            result = process_moderator_turn(
                object(), object(), {}, model,
                author_name="mod", text="I was just looking at u/example_candidate.",
                channel="private", source_id="username-in-sentence",
                conversation_key="private:mod:username-in-sentence",
                bot_username="example_bot", verified_moderator=True,
            )

        self.assertEqual(result["route"], "clarify")
        review.assert_not_called()

    def test_history_review_does_not_accept_username_from_a_lookalike_host(self):
        model = _Model(_result(
            "history_review", action="review", target_username="example_user",
            needs_history=True,
        ))

        intent = interpret_moderator_turn(
            "review https://reddit.com.example/user/example_user/", model
        )

        self.assertEqual(intent["route"], "clarify")
        self.assertIsNone(intent["target_username"])

    def test_low_confidence_action_is_clarified_before_any_execution(self):
        model = _Model(_result(
            "history_review", action="review", target_username="example_user",
            needs_history=True, confidence=0.3,
        ))
        state = {}
        with patch("moderator_action_router.refresh_moderators") as verify_mods, \
             patch("moderator_history_commands.handle_moderator_history_request") as report:
            result = process_moderator_turn(
                object(), object(), state, model,
                author_name="mod", text="If we ever review u/example_user, what would happen?",
                channel="private", source_id="message-hypothetical",
                conversation_key="private:mod", bot_username="example_bot",
                verified_moderator=True,
            )

        self.assertTrue(result["clarification"])
        self.assertIn("couldn't safely interpret", result["response"])
        verify_mods.assert_not_called()
        report.assert_not_called()
        self.assertEqual(len(model.calls), 0)
        self.assertNotIn(
            "pending_moderator_action",
            state.get("conversation_contexts", {}).get("private:mod", {}),
        )

    def test_action_without_confidence_or_selected_probability_fails_closed(self):
        model = _Model(_result(
            "history_review", action="review", target_username="example_user",
            needs_history=True, omit_confidence=True,
        ))
        with patch("moderator_action_router.refresh_moderators") as verify_mods, \
             patch("moderator_history_commands.handle_moderator_history_request") as report:
            result = process_moderator_turn(
                object(), object(), {}, model,
                author_name="mod", text="review u/example_user", channel="private",
                source_id="missing-action-confidence", conversation_key="private:mod",
                bot_username="example_bot", verified_moderator=True,
            )

        self.assertEqual(result["route"], "decision_unavailable")
        verify_mods.assert_not_called()
        report.assert_not_called()

    def test_peaked_distribution_confidence_cannot_override_a_weak_action_choice(self):
        model = _Model(_result(
            "history_review", action="review", target_username="example_user",
            needs_history=True, confidence=0.99, choice_probability=0.40,
        ))

        intent = interpret_moderator_turn("review u/example_user", model)

        self.assertEqual(intent["route"], "clarify")
        self.assertIn("couldn't safely interpret", intent["question"])
        self.assertEqual(len(model.calls), 0)

    def test_custom_flair_uses_the_selected_action_handler_after_one_selection(self):
        model = _Model(_result(
            "self_custom_flair", action="set", flair_label="Community volunteer",
        ))
        state = {}
        updated_state = {"custom_flair_seen": True}
        interaction = SimpleNamespace(author=SimpleNamespace(name="mod"), body="Set my flair to Community volunteer")
        with patch("moderator_action_router.config.USER_CUSTOM_FLAIR_ENABLED", True), \
             patch("moderator_action_router.refresh_moderators", return_value={"mod"}), \
             patch("self_flair_requests.handle_self_custom_flair_request",
                   return_value=("Flair updated.", 5, 0.01, updated_state)) as execute:
            result = process_moderator_turn(
                object(), object(), state, model,
                author_name="mod", text="Set my flair to Community volunteer", channel="private",
                source_id="message-custom", conversation_key="private:mod",
                bot_username="example_bot", verified_moderator=True,
                interaction=interaction,
            )

        self.assertEqual(result["route"], "self_custom_flair")
        self.assertEqual(result["response"], "Flair updated.")
        self.assertIs(result["state"], updated_state)
        self.assertEqual(execute.call_args.args[5], {"action": "set", "label": "Community volunteer"})
        self.assertEqual(len(model.calls), 0)

    def test_deterministic_flair_parser_cannot_change_jev_selected_action(self):
        model = _Model(_result("self_custom_flair", action="set"))
        model.jev.choice_override = "self_custom_flair.set"
        with patch("moderator_action_router.config.USER_CUSTOM_FLAIR_ENABLED", True):
            intent = interpret_moderator_turn("Suggest a custom flair for me", model)

        self.assertEqual(intent["route"], "clarify")
        self.assertIn("exact wording", intent["question"])
        self.assertEqual(len(model.jev.calls), 1)
        self.assertEqual(len(model.calls), 0)



    def test_non_moderator_never_reaches_the_action_model(self):
        model = _Model(_result("history_review", action="review", target_username="example_user"))

        result = process_moderator_turn(
            object(), object(), {}, model,
            author_name="member", text="review u/example_user", channel="private",
            source_id="message-3", conversation_key="private:member:3",
            bot_username="example_bot", verified_moderator=False,
        )

        self.assertIsNone(result)
        self.assertEqual(model.calls, [])

    def test_duplicate_comment_paths_reuse_the_same_decision_and_reply(self):
        model = _Model(_result("conversation", reply_text="One reply."))
        state = {}
        arguments = dict(
            author_name="mod", text="Optimist, what is your take?",
            channel="comment", source_id="comment-duplicate",
            conversation_key="comment:mod:post-1",
            bot_username="example_bot", verified_moderator=True,
        )

        first = process_moderator_turn(object(), object(), state, model, **arguments)
        first["_response_delivered"] = True
        second = process_moderator_turn(object(), object(), state, model, **arguments)

        self.assertIs(second, first)
        self.assertTrue(second["_response_delivered"])
        self.assertEqual(len(model.calls), 1)


if __name__ == "__main__":
    unittest.main()
