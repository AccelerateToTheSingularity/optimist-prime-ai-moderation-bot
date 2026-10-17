import unittest

from jev_interaction_router import route_public_interaction


class FakeJev:
    ready = True
    max_calls = 4
    calls_used = 0
    last_usage = {"input_tokens": 42, "cost": 0.0003}

    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error
        self.calls = []

    def choose_many_detailed(self, *, state, questions):
        self.calls.append((state, questions))
        if self.error:
            raise self.error
        return self.answer


def answer(choice, *, confidence=0.91, probabilities=None):
    return {
        "public_interaction": {
            "choice": choice,
            "confidence": confidence,
            "probabilities": probabilities or {choice: confidence},
        }
    }


class JevInteractionRouterTests(unittest.TestCase):
    def test_one_choice_request_routes_review_to_parent_author(self):
        client = FakeJev(answer("history_review.review"))

        result = route_public_interaction(
            client,
            text="hey mod bot, review",
            channel="public_comment",
            summoned=True,
            parent_author="example_target",
            parent_text="A post about community guidelines",
        )

        self.assertTrue(result["complete"])
        self.assertEqual(result["route"], "history_review")
        self.assertEqual(result["action"], "review")
        self.assertEqual(result["target_username"], "example_target")
        self.assertEqual(result["token_info"], {"input_tokens": 42, "cost": 0.0003})
        self.assertEqual(len(client.calls), 1)
        _, questions = client.calls[0]
        self.assertEqual(list(questions), ["public_interaction"])
        self.assertEqual(questions["public_interaction"]["type"], "choice")

    def test_safe_routes_remain_but_action_allow_list_is_enforced(self):
        client = FakeJev(answer("conversation"))

        result = route_public_interaction(
            client,
            text="What do you think?",
            channel="private_chat",
            summoned=True,
            allowed_actions={"self_custom_flair"},
        )

        self.assertEqual(result["route"], "conversation")
        menu = client.calls[0][1]["public_interaction"]["criteria"]
        self.assertIn("self_custom_flair.set", menu)
        self.assertNotIn("history_review.review", menu)
        self.assertIn("not_addressed", menu)
        instructions = client.calls[0][1]["public_interaction"]["instructions"]
        self.assertIn("one-to-one private Chat", instructions)
        self.assertIn("untrusted", instructions)

    def test_top_level_posts_require_jev_to_see_an_explicit_summon(self):
        client = FakeJev(answer("not_addressed"))
        result = route_public_interaction(
            client,
            text="Optimist Prime made an interesting point.",
            channel="post",
            summoned=False,
        )

        self.assertEqual(result["route"], "not_addressed")
        instructions = client.calls[0][1]["public_interaction"]["instructions"]
        self.assertIn("On a top-level post, reply only if the author explicitly", instructions)
        self.assertIn("a passing mention or discussion of Optimist is not", instructions)

    def test_specific_allow_list_action_does_not_expose_peer_actions(self):
        client = FakeJev(answer("conversation"))
        route_public_interaction(
            client,
            text="Set my flair to AI researcher",
            channel="private_chat",
            summoned=True,
            allowed_actions={"self_custom_flair.set"},
        )
        menu = client.calls[0][1]["public_interaction"]["criteria"]
        self.assertIn("self_custom_flair.set", menu)
        self.assertNotIn("self_custom_flair.suggest", menu)



    def test_explicit_custom_flair_label_is_returned_without_claiming_execution(self):
        client = FakeJev(answer("self_custom_flair.set"))

        result = route_public_interaction(
            client,
            text="Please set my flair to AI researcher",
            channel="private_chat",
            summoned=True,
            allowed_actions={"self_custom_flair"},
        )

        self.assertEqual(result["requested_flair"], "AI researcher")
        self.assertNotIn("executed", result)
        self.assertNotIn("success", result)

        quoted = FakeJev(answer("self_custom_flair.set"))
        quoted_result = route_public_interaction(
            quoted,
            text='Please set my flair to "AI researcher".',
            channel="private_chat",
            summoned=True,
            allowed_actions={"self_custom_flair.set"},
        )
        self.assertEqual(quoted_result["requested_flair"], "AI researcher")

    def test_unavailable_client_and_exhausted_budget_make_no_request(self):
        unavailable = FakeJev(answer("conversation"))
        unavailable.ready = False
        unavailable_result = route_public_interaction(
            unavailable,
            text="hi",
            channel="private_chat",
            summoned=True,
        )
        self.assertFalse(unavailable_result["complete"])
        self.assertIsNone(unavailable_result["route"])
        self.assertEqual(unavailable.calls, [])
        self.assertEqual(unavailable_result["token_info"], {"input_tokens": 0, "cost": 0.0})

        exhausted = FakeJev(answer("conversation"))
        exhausted.calls_used = exhausted.max_calls
        exhausted_result = route_public_interaction(
            exhausted,
            text="hi",
            channel="private_chat",
            summoned=True,
        )
        self.assertFalse(exhausted_result["complete"])
        self.assertEqual(exhausted.calls, [])

    def test_missing_invalid_or_failed_choice_is_incomplete_without_fallback(self):
        cases = [
            (None, None),
            ({}, None),
            (answer("history_review.review"), None),  # Not in this caller's menu.
        ]
        for response, error in cases:
            with self.subTest(response=response, error=error):
                client = FakeJev(response, error)
                result = route_public_interaction(
                    client,
                    text="hello",
                    channel="private_chat",
                    summoned=True,
                    allowed_actions={"self_custom_flair"},
                )
                self.assertFalse(result["complete"])
                self.assertIsNone(result["route"])
                self.assertEqual(result["action"], "")
                self.assertEqual(len(client.calls), 1)

        failed = FakeJev(error=RuntimeError("provider unavailable"))
        result = route_public_interaction(
            failed,
            text="hello",
            channel="private_chat",
            summoned=True,
        )
        self.assertFalse(result["complete"])
        self.assertEqual(result["error"], "jev_error:RuntimeError")
        self.assertEqual(result["token_info"], {"input_tokens": 42, "cost": 0.0003})
        self.assertEqual(len(failed.calls), 1)

    def test_invalid_input_and_missing_method_do_not_attempt_jev(self):
        client = FakeJev(answer("conversation"))
        missing_input = route_public_interaction(
            client,
            text="  ",
            channel="private_chat",
            summoned=True,
        )
        self.assertFalse(missing_input["complete"])
        self.assertEqual(client.calls, [])

        class NoChoiceClient:
            ready = True
            last_usage = {"input_tokens": 100, "cost": 1.0}

        no_method = route_public_interaction(
            NoChoiceClient(),
            text="hello",
            channel="private_chat",
            summoned=True,
        )
        self.assertFalse(no_method["complete"])
        self.assertEqual(no_method["token_info"], {"input_tokens": 0, "cost": 0.0})

    def test_none_reply_and_not_addressed_are_distinct_routes(self):
        for choice in ("no_reply", "not_addressed", "clarify", "unsupported_mod_action"):
            with self.subTest(choice=choice):
                client = FakeJev(answer(choice))
                result = route_public_interaction(
                    client,
                    text="thanks",
                    channel="public_comment",
                    summoned=True,
                )
                self.assertTrue(result["complete"])
                self.assertEqual(result["route"], choice)
                self.assertEqual(result["action"], "")

    def test_low_or_missing_confidence_cannot_publish_or_execute_a_public_route(self):
        for choice, confidence in (
            ("conversation", 0.4),
            ("self_custom_flair.set", 0.61),
            ("self_custom_flair.set", None),
        ):
            with self.subTest(choice=choice, confidence=confidence):
                client = FakeJev(answer(choice, confidence=confidence))
                result = route_public_interaction(
                    client,
                    text="Please help with my request",
                    channel="comment",
                    summoned=True,
                    allowed_actions={"self_custom_flair"},
                )
                self.assertTrue(result["complete"])
                self.assertEqual(result["route"], "no_reply")
                self.assertEqual(result["action"], "")

    def test_reply_uses_selected_option_probability_not_distribution_confidence(self):
        client = FakeJev(answer(
            "conversation",
            confidence=0.50,
            probabilities={
                "conversation": 0.60,
                "clarify": 0.39,
                "not_addressed": 0.01,
            },
        ))

        result = route_public_interaction(
            client,
            text="Hey Optimist, could you explain this?",
            channel="public_comment",
            summoned=True,
        )

        self.assertEqual(result["route"], "conversation")
        self.assertEqual(result["confidence"], 0.50)
        self.assertEqual(result["choice_probability"], 0.60)

    def test_action_needs_its_own_probability_even_when_distribution_is_peaked(self):
        client = FakeJev(answer(
            "self_custom_flair.set",
            confidence=0.99,
            probabilities={"self_custom_flair.set": 0.61},
        ))

        result = route_public_interaction(
            client,
            text="Please set my flair to Community volunteer.",
            channel="public_comment",
            summoned=True,
            allowed_actions={"self_custom_flair"},
        )

        self.assertEqual(result["route"], "no_reply")

    def test_selected_probability_can_supply_missing_confidence(self):
        client = FakeJev({
            "public_interaction": {
                "choice": "conversation",
                "probabilities": {"conversation": 0.91},
            }
        })
        result = route_public_interaction(
            client,
            text="What do you think?",
            channel="comment",
            summoned=True,
        )
        self.assertEqual(result["route"], "conversation")
        self.assertEqual(result["confidence"], 0.91)


    def test_one_source_uses_one_jev_route_across_worker_phases(self):
        client = FakeJev(answer("conversation"))
        cache = {}
        first = route_public_interaction(
            client,
            text="Optimist made a good point.",
            channel="comment",
            summoned=False,
            allowed_actions={"self_custom_flair"},
            interaction_cache=cache,
            source_id="same-comment",
        )
        second = route_public_interaction(
            client,
            text="Optimist made a good point.",
            channel="comment",
            summoned=True,
            allowed_actions={"self_custom_flair"},
            interaction_cache=cache,
            source_id="same-comment",
        )

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(first["route"], second["route"])
        self.assertTrue(second["_cycle_cache_hit"])
        self.assertEqual(
            second["token_info"],
            {"input_tokens": 0, "total_tokens": 0, "cost": 0.0},
        )

    def test_failed_jev_route_is_cached_only_for_the_current_cycle(self):
        client = FakeJev(error=RuntimeError("provider error"))
        cache = {}
        first = route_public_interaction(
            client, text="Please help", channel="comment", summoned=True,
            interaction_cache=cache, source_id="failed-comment",
        )
        second = route_public_interaction(
            client, text="Please help", channel="comment", summoned=True,
            interaction_cache=cache, source_id="failed-comment",
        )
        next_cycle = route_public_interaction(
            client, text="Please help", channel="comment", summoned=True,
            interaction_cache={}, source_id="failed-comment",
        )

        self.assertFalse(first["complete"])
        self.assertFalse(second["complete"])
        self.assertTrue(second["_cycle_cache_hit"])
        self.assertFalse(next_cycle["complete"])
        self.assertEqual(len(client.calls), 2)

    def test_failed_route_is_not_retried_by_another_phase_with_a_wider_menu(self):
        client = FakeJev(error=RuntimeError("provider error"))
        cache = {}
        first = route_public_interaction(
            client,
            text="Optimist, could you help?",
            channel="comment",
            summoned=False,
            allowed_actions={"self_custom_flair"},
            interaction_cache=cache,
            source_id="shared-failure",
        )
        later_phase = route_public_interaction(
            client,
            text="Optimist, could you help?",
            channel="comment",
            summoned=True,
            allowed_actions={"self_custom_flair"},
            interaction_cache=cache,
            source_id="shared-failure",
        )

        self.assertFalse(first["complete"])
        self.assertFalse(later_phase["complete"])
        self.assertTrue(later_phase["_cycle_cache_hit"])
        self.assertEqual(len(client.calls), 1)

    def test_edited_source_can_be_retried_in_the_same_cycle(self):
        client = FakeJev(error=RuntimeError("provider error"))
        cache = {}
        route_public_interaction(
            client,
            text="Optimist, could you help?",
            channel="comment",
            summoned=True,
            interaction_cache=cache,
            source_id="edited-source",
        )
        route_public_interaction(
            client,
            text="Optimist, could you explain this?",
            channel="comment",
            summoned=True,
            interaction_cache=cache,
            source_id="edited-source",
        )

        self.assertEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
