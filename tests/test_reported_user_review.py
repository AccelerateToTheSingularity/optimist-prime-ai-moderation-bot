"""Isolated tests for reported-user history review."""

import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import config
from reported_user_review import (
    MalformedHistoryReview,
    ReportedItem,
    ReviewDecision,
    calculate_review_decision,
    build_history_review_prompt,
    collect_reported_items,
    evaluate_history,
    format_review_modmail,
    parse_history_review_response,
    process_reported_user_reviews,
    _parse_history_review_document,
)
from history_criteria_test_support import current_history_criteria_revisions
from history_gate_jev import (
    HistoryGateAssessment,
    JevHistoryReviewError,
)
from history_criteria_test_support import (
    ConfiguredHistoryCriteriaTestCase,
    synthetic_history_criteria,
    scores_for_questions,
)
from user_history import HistoryItem, UserHistory, collect_user_history
from history_review_ledger import HistoryReviewLedger


class _Listing:
    def __init__(self, values):
        self.values = values
        self.calls = 0
        self.limits = []

    def new(self, limit=None):
        self.calls += 1
        self.limits.append(limit)
        return self.values


class _Reddit:
    def __init__(self, comments, posts):
        self._redditor = SimpleNamespace(
            comments=_Listing(comments),
            submissions=_Listing(posts),
        )

    def redditor(self, username):
        return self._redditor


class _FakeJev:
    ready = True
    max_calls = 5

    def __init__(self, choice="yes", *, selected_criterion=None):
        self.choice = choice
        self.selected_criterion = selected_criterion
        self.calls_used = 0
        self.last_usage = {"input_tokens": 12, "cost": 0.0002}
        self.served_model = "typesafe/jev-test"
        self.questions = []

    def decide_many(self, *, state, questions):
        self.calls_used += 1
        self.questions.append((state, questions))
        positives = ()
        if self.choice == "yes":
            target = self.selected_criterion or next(iter(questions))
            positives = (target,) if target in questions else ()
        return scores_for_questions(questions, positives=positives)


class _Subreddit:
    def __init__(self, reports):
        self.messages = []
        self.report_limits = []
        self._reports = reports
        self.mod = SimpleNamespace(reports=self.reports)
        self._conversations = []
        self.fail_create_once = False
        self.modmail = SimpleNamespace(
            conversations=self.conversations,
            create=self.create_internal,
        )

    def reports(self, limit=None):
        self.report_limits.append(limit)
        return self._reports

    def moderator(self):
        return []

    def conversations(self, **kwargs):
        return iter(self._conversations)

    def create_internal(self, *, subject, body, recipient):
        if recipient is not None:
            raise AssertionError("reported-user notes must have no outside recipient")
        if self.fail_create_once:
            self.fail_create_once = False
            raise RuntimeError("temporary Modmail failure")
        conversation = SimpleNamespace(
            id=f"internal-{len(self._conversations) + 1}",
            subject=subject,
            is_internal=True,
            body=body,
        )
        self._conversations.append(conversation)
        self.messages.append((subject, body))
        return conversation


class _Model:
    def __init__(self, text):
        self.text = text
        self.generation_config = None
        self.contents = None
        self.calls = 0

    def generate_content(self, contents, generation_config=None):
        self.calls += 1
        self.contents = contents
        self.generation_config = generation_config
        return SimpleNamespace(
            text=self.text,
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
        )


def _reported_comment(item_id, username="example_user", reason="rule report", count=1):
    return SimpleNamespace(
        id=item_id,
        link_id="t3_parent-post",
        body="reported content",
        author=SimpleNamespace(name=username),
        subreddit=SimpleNamespace(display_name="example_community"),
        permalink=f"/r/example_community/comments/{item_id}/x/",
        created_utc=2_000_000_000.0,
        user_reports=[{"reason": reason, "count": count}],
        mod_reports=[],
    )


def _history_item(item_id, kind, subreddit, created):
    return HistoryItem(
        item_id=item_id,
        kind=kind,
        subreddit=subreddit,
        created_utc=created,
        score=-2,
        permalink=f"https://www.reddit.com/r/{subreddit}/comments/{item_id}/x/",
        text="evidence text",
    )


def _example_criterion(name):
    return {
        "name": name,
        "title": "Example criterion",
        "description": "The author repeatedly shares the same example link in unrelated posts.",
        "high_confidence_yes": "Several separate posts repeat the same link without relevant discussion.",
        "not_sufficient_by_itself": "One relevant link or a single ambiguous example is not enough.",
        "closest_boundaries": "Distinguish repeated promotion from ordinary discussion or quoted links.",
        "active": True,
        "revision": 1,
        "order": 1,
    }


def _synthetic_moderation_rules_json():
    """Return neutral test-only rules for code paths that load content rules."""
    return json.dumps([
        {
            "name": name,
            "description": guide["description"],
            "active": True,
            "order": index,
            "target": "both",
            "actions": ["report"],
        }
        for index, (name, guide) in enumerate(
            synthetic_history_criteria().items(),
            start=1,
        )
    ])


class TestReportedUserReview(ConfiguredHistoryCriteriaTestCase):
    def setUp(self):
        super().setUp()
        self.ledger_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.ledger_dir.cleanup)
        self.review_ledger = HistoryReviewLedger(
            Path(self.ledger_dir.name) / "history-review.sqlite3"
        )
        self.ledger_patch = patch(
            "reported_user_review.HistoryReviewLedger",
            return_value=self.review_ledger,
        )
        self.ledger_patch.start()
        self.addCleanup(self.ledger_patch.stop)
        self.threshold_patch = patch.object(config, "HISTORY_REVIEW_MATCH_THRESHOLD_PERCENT", 67)
        self.threshold_patch.start()
        self.addCleanup(self.threshold_patch.stop)
        self.disable_autoban = patch.object(config, "AUTOMATED_BAN_ENABLED", False)
        self.disable_autoban.start()
        self.addCleanup(self.disable_autoban.stop)

    def test_modmail_report_has_no_boilerplate_disclaimer(self):
        report = format_review_modmail(
            "example_user",
            [],
            UserHistory("example_user", (), 0, 0),
            [],
            ReviewDecision("no_actionable_match", 0, 0, (), 0.0, ()),
        )
        self.assertTrue(report.startswith("**Reported-user history review for u/example_user**"))
        self.assertNotIn("This is a recommendation for moderator review", report)
        self.assertNotIn("automatic ban decision", report)
        self.assertNotIn("Please check the linked content", report)
        self.assertIn("Review incomplete: no eligible history was available.", report)

    def test_normal_modmail_is_a_compact_criteria_first_quick_reference(self):
        assessment = HistoryGateAssessment(
            criterion_score_revisions=current_history_criteria_revisions(),
            rule_names=("sample_repeated_link_posting", "sample_direct_personal_attack"),
            bundle_size=7,
            bundles=(),
            matched_rules=("sample_repeated_link_posting",),
            calls=1,
            input_tokens=99,
            cost=0.001,
            served_model="example/provider-model",
        )
        finding = SimpleNamespace(
            rule_name="sample_direct_personal_attack",
            confidence=0.99,
            excerpt="diagnostic-only evidence",
        )
        report_item = ReportedItem(
            item_id="c1",
            kind="comment",
            username="example_user",
            subreddit="example_community",
            permalink="https://reddit.test/r/example_community/comments/c1/x/",
            title="",
            text="reported text",
            report_reasons=("reported reason",),
        )
        report = format_review_modmail(
            "example_user",
            [report_item],
            UserHistory("example_user", (HistoryItem("c1", "comment", "example_community", 1.0, 1, "", "evidence"),), 1, 0),
            [finding],
            ReviewDecision("moderator_review", 1, 1, ("example_community",), 0.99, ("sample_direct_personal_attack",)),
            summary="A short criterion-focused summary.",
            gate_assessment=assessment,
        )
        self.assertIn("Sample Repeated Link Posting", report)
        self.assertNotIn("82%", report)
        self.assertNotIn("Sample Direct Personal Attack", report)
        self.assertNotIn("Jev", report)
        self.assertNotIn("example/provider-model", report)
        self.assertNotIn("average", report.casefold())
        self.assertNotIn("themes", report.casefold())
        self.assertNotIn("diagnostic-only evidence", report)
        self.assertNotIn("History date range", report)
        self.assertLess(report.index("Sample Repeated Link Posting"), report.index("Reported items (1)"))
        self.assertNotIn("A short criterion-focused summary.", report)
        self.assertIn("[c1](https://reddit.test/r/example_community/comments/c1/x/)", report)

    def test_no_match_report_shows_configured_criteria_but_keeps_summary_concise(self):
        criteria = synthetic_history_criteria()
        names = tuple(criteria)
        assessment = HistoryGateAssessment(
            criterion_score_revisions=current_history_criteria_revisions(),
            rule_names=names,
            bundle_size=0,
            bundles=(),
            matched_rules=(),
            calls=1,
            input_tokens=0,
            cost=0.0,
            served_model="example/provider-model",
            item_attribution_available=False,
            criterion_probabilities=scores_for_questions({name: {} for name in names}),
        )
        report = format_review_modmail(
            "example_user", [], UserHistory("example_user", (HistoryItem("c1", "comment", "example_community", 1.0, 1, "", "evidence"),), 1, 0), [],
            ReviewDecision("no_actionable_match", 0, 0, (), 0.0, ()),
            summary="General background that does not change the criterion result.",
            gate_assessment=assessment,
        )
        self.assertIn(
            "1 comment scanned for [u/example_user](https://www.reddit.com/user/example_user/). No configured criterion matched.",
            report,
        )
        self.assertNotIn("| Configured criterion |", report)
        self.assertNotIn("General background", report)

    def test_supplied_report_scan_is_reused_without_a_second_reddit_request(self):
        subreddit = _Subreddit([])
        with patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True):
            process_reported_user_reviews(
                subreddit,
                _Reddit([], []),
                {},
                _Model('{"matches":[]}'),
                reported_items=[],
                dry_run=True,
            )
        self.assertEqual(subreddit.report_limits, [])

    def test_failed_shared_report_scan_does_not_trigger_an_internal_retry(self):
        subreddit = _Subreddit([])
        with patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True):
            metrics, _, _, state = process_reported_user_reviews(
                subreddit,
                _Reddit([], []),
                {},
                _Model('{"matches":[]}'),
                report_scan_error="temporary scan failure",
                dry_run=True,
            )
        self.assertEqual(subreddit.report_limits, [])
        self.assertEqual(metrics.retryable_failures, 1)
        self.assertIn("temporary scan failure", state["reported_user_reviews"]["last_error"])

    def test_report_items_are_deduplicated_and_reasons_merged(self):
        first = _reported_comment("c1", reason="reason one")
        second = _reported_comment("c1", reason="reason two")
        second.mod_reports = [{"reason": "mod reason"}]
        items = collect_reported_items(_Subreddit([first, second]))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].all_report_reasons, ("reason one", "reason two", "mod reason"))
        self.assertEqual(items[0].post_id, "parent-post")
        self.assertEqual(items[0].user_report_count, 1)

    def test_report_counts_sum_user_report_rows_but_not_moderator_reports(self):
        item = _reported_comment("c1", count=2)
        item.user_reports.append(("second reason", 3))
        item.mod_reports = [("mod report", 9)]
        reported = collect_reported_items(_Subreddit([item]))[0]
        self.assertEqual(reported.user_report_count, 5)

    def test_authorless_reported_comment_is_retained_for_other_consumers(self):
        item = _reported_comment("deleted-comment")
        item.author = None
        reported = collect_reported_items(_Subreddit([item]))[0]
        self.assertEqual(reported.username, "")
        self.assertEqual(reported.post_id, "parent-post")
        self.assertEqual(reported.user_report_count, 1)

    def test_default_report_scan_requests_all_visible_reports(self):
        subreddit = _Subreddit([_reported_comment("c1")])
        collect_reported_items(subreddit)
        self.assertEqual(subreddit.report_limits, [None])

    def test_strict_parser_rejects_unknown_rule_and_bad_index(self):
        now = 2_000_000_000.0
        history = UserHistory("example_user", (_history_item("c1", "comment", "one", now - 1),), 1, 0)
        rules = [_example_criterion("known")]
        with self.assertRaises(MalformedHistoryReview):
            parse_history_review_response('{"matches":[{"evidence_index":0,"rule_name":"unknown","confidence":1,"explanation":"x"}]}', history, rules)
        with self.assertRaises(MalformedHistoryReview):
            parse_history_review_response('{"matches":[{"evidence_index":4,"rule_name":"known","confidence":1,"explanation":"x"}]}', history, rules)
        with self.assertRaises(MalformedHistoryReview):
            parse_history_review_response("```json {} ```", history, rules)
        with self.assertRaises(MalformedHistoryReview):
            parse_history_review_response(
                '{"matches": [], "action": "ban"}', history, rules
            )

    def test_concise_parser_bounds_long_summary_without_discarding_criterion_matches(self):
        now = 2_000_000_000.0
        history = UserHistory(
            "example_user", (_history_item("c1", "comment", "example_community", now - 1),), 1, 0
        )
        rule = _example_criterion("known")
        response = json.dumps({
            "summary": "The author gives a clear explanation. " + "More context. " * 40,
            "themes": [],
            "matches": [{
                "evidence_index": 0,
                "rule_name": "known",
                "confidence": 0.95,
                "explanation": "The comment directly describes the example behavior.",
            }],
        })

        findings, summary, _themes, _answer = _parse_history_review_document(
            response,
            history,
            [rule],
            summary_limit=350,
            answer_limit=300,
            theme_limit=4,
            one_match_per_rule=True,
        )

        self.assertLessEqual(len(summary), 350)
        self.assertTrue(summary.endswith("…"))
        self.assertEqual([finding.rule_name for finding in findings], ["known"])

    def test_history_review_evidence_contains_auditable_item_fields(self):
        now = 2_000_000_000.0
        history = UserHistory(
            "example_user", (_history_item("c1", "comment", "one", now - 1),), 1, 0
        )
        rule = _example_criterion("known")
        findings = parse_history_review_response(
            '{"matches":[{"evidence_index":0,"rule_name":"known",'
            '"confidence":0.9,"explanation":"evidence"}]}',
            history,
            [rule],
        )
        self.assertEqual(findings[0].item_id, "c1")
        self.assertEqual(findings[0].score, -2)
        self.assertEqual(findings[0].excerpt, "evidence text")
        self.assertTrue(findings[0].timestamp.startswith("2033-"))

    def test_history_prompt_uses_operator_supplied_criteria(self):
        criteria = [{
            "name": "sample_direct_personal_attack",
            **synthetic_history_criteria()["sample_direct_personal_attack"],
        }]
        history = UserHistory("example_user", (), 0, 0)
        prompt = build_history_review_prompt("example_user", [], history, criteria)
        self.assertIn("configured criteria supplied below", prompt)
        self.assertIn('"sample_direct_personal_attack"', prompt)
        self.assertIn("Repeated direct personal attacks", prompt)
        self.assertIn("the author's own words", prompt)
        self.assertIn("configured criteria supplied below", prompt)
        self.assertIn("stop after the first clear YES", prompt)
        self.assertIn('"summary":"short neutral summary"', prompt)
        self.assertIn('Each object in "matches" must have this shape', prompt)

    def test_concise_report_prompt_keeps_broad_sample_with_compact_evidence(self):
        now = 2_000_000_000.0
        history = UserHistory(
            "example_user",
            tuple(
                HistoryItem(
                    item_id=f"c{index}", kind="comment", subreddit="example_community",
                    created_utc=now - index * 86400, score=1,
                    permalink=f"https://reddit.test/comments/c{index}",
                    text=("opening words " + "long evidence " * 100 + " conclusion"),
                )
                for index in range(200)
            ),
            200,
            0,
        )
        rule = _example_criterion("example_detector")
        concise = build_history_review_prompt(
            "example_user", [], history, [rule], concise_output=True
        )
        payload_text = concise.rsplit("<history>", 1)[1].split("</history>", 1)[0]
        payload = json.loads(payload_text)
        full = build_history_review_prompt("example_user", [], history, [rule])

        self.assertEqual(len(payload), 200)
        self.assertLessEqual(max(len(item["excerpt"]) for item in payload), 300)
        self.assertIn("opening words", payload[0]["excerpt"])
        self.assertIn("conclusion", payload[0]["excerpt"])
        self.assertNotIn("permalink", payload[0])
        self.assertNotIn("score", payload[0])
        self.assertIn("Do not add generic disclaimers", concise)
        self.assertLess(len(concise), len(full) * 0.6)
        self.assertIn("summarize only evidence related to the supplied criteria", concise)
        self.assertIn("include nuance that changes interpretation", concise)
        self.assertIn("do not let one item define the whole person", concise)
        self.assertIn('"summary":"', concise)

    def test_concise_report_keeps_strongest_example_for_repeated_criterion_matches(self):
        now = 2_000_000_000.0
        history = UserHistory(
            "example_user",
            (
                _history_item("c1", "comment", "example_community", now - 1),
                _history_item("c2", "comment", "example_other_community", now - 2),
            ),
            2,
            0,
        )
        rule = _example_criterion("example_detector")

        class RecordingModel:
            def __init__(self):
                self.generation_config = None

            def generate_content(self, _contents, generation_config=None):
                self.generation_config = generation_config
                return SimpleNamespace(
                    text=json.dumps({
                        "summary": "Mixed views.",
                        "themes": ["gardening"],
                        "matches": [
                            {"evidence_index": 0, "rule_name": rule["name"], "confidence": 0.9, "explanation": "first"},
                            {"evidence_index": 1, "rule_name": rule["name"], "confidence": 0.8, "explanation": "second"},
                        ],
                    }),
                    usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=50),
                )

        model = RecordingModel()
        findings, _document = evaluate_history(
            "example_user", [], history, [rule], model, concise_output=True
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].item_id, "c1")
        self.assertEqual(findings[0].confidence, 0.9)
        self.assertEqual(model.generation_config["max_output_tokens"], 6000)
        self.assertIs(model.generation_config["thinking_enabled"], False)

    def test_history_parser_rejects_more_than_one_rule_for_one_item(self):
        now = 2_000_000_000.0
        history = UserHistory(
            "example_user", (_history_item("c1", "comment", "one", now - 1),), 1, 0
        )
        rules = [_example_criterion("first"), _example_criterion("second")]
        response = json.dumps({
            "summary": "summary",
            "themes": ["community gardening"],
            "matches": [
                {"evidence_index": 0, "rule_name": "first", "confidence": 0.9, "explanation": "first match"},
                {"evidence_index": 0, "rule_name": "second", "confidence": 0.8, "explanation": "later match"},
            ],
        })
        with self.assertRaises(MalformedHistoryReview):
            parse_history_review_response(response, history, rules)

    def test_history_parser_rejects_duplicate_json_fields(self):
        now = 2_000_000_000.0
        history = UserHistory(
            "example_user", (_history_item("c1", "comment", "one", now - 1),), 1, 0
        )
        rule = _example_criterion("known")
        response = '{"summary":"one","summary":"two","themes":[],"matches":[]}'
        with self.assertRaises(MalformedHistoryReview):
            parse_history_review_response(response, history, [rule])

    def test_qualifying_evidence_is_a_recommendation_and_never_a_ban(self):
        now = 2_000_000_000.0
        items = (
            _history_item("c1", "comment", "one", now - 10),
            _history_item("c2", "comment", "one", now - 20),
            _history_item("p1", "post", "two", now - 30),
        )
        history = UserHistory("example_user", items, 2, 1)
        rule = _example_criterion("example_detector")
        response = json.dumps({
            "matches": [
                {"evidence_index": 0, "rule_name": rule["name"], "confidence": 0.95, "explanation": "match 1"},
                {"evidence_index": 1, "rule_name": rule["name"], "confidence": 0.90, "explanation": "match 2"},
                {"evidence_index": 2, "rule_name": rule["name"], "confidence": 0.90, "explanation": "match 3"},
            ]
        })
        findings = parse_history_review_response(response, history, [rule])
        decision = calculate_review_decision(findings, [rule])
        self.assertEqual(decision.recommendation, "moderator_review")
        self.assertEqual(decision.matching_items, 3)

    def test_feature_groups_reports_and_sends_one_modmail_without_ban_api(self):
        now = 2_000_000_000.0
        reports = [_reported_comment("r1"), _reported_comment("r2", reason="second report")]
        comments = [
            SimpleNamespace(id="c1", body="one", score=1, subreddit=SimpleNamespace(display_name="one"), created_utc=now - 10, permalink="/c1"),
            SimpleNamespace(id="c2", body="two", score=1, subreddit=SimpleNamespace(display_name="one"), created_utc=now - 20, permalink="/c2"),
        ]
        posts = [
            SimpleNamespace(id="p1", title="three", selftext="body", score=1, subreddit=SimpleNamespace(display_name="two"), created_utc=now - 30, permalink="/p1"),
        ]
        model = _Model(json.dumps({
            "summary": "The supplied sample history supports one possible match.",
            "themes": ["community gardening"],
            "matches": [
                {"evidence_index": 0, "rule_name": "sample_repeated_link_posting", "confidence": 0.95, "explanation": "evidence one"},
            ]
        }))
        subreddit = _Subreddit(reports)
        reddit = _Reddit(comments, posts)
        state = {}
        rules_json = _synthetic_moderation_rules_json()
        with patch.dict(os.environ, {"BOT_MODERATION_RULES_JSON": rules_json}, clear=False), \
             patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True), \
             patch.object(config, "REPORTED_USER_REVIEW_ACTION", "modmail"), \
             patch.object(config, "REPORTED_USER_REVIEW_MAX_LLM_CALLS_PER_RUN", 3), \
             patch.object(config, "HISTORY_REVIEW_LOOKBACK_DAYS", 365), \
             patch.object(config, "HISTORY_REVIEW_MAX_ITEMS", 90), \
             patch("reported_user_review.record_review_note", return_value="delivered") as note, \
             patch("reported_user_review.collect_user_history", wraps=collect_user_history) as collect_history, \
             patch("reported_user_review.load_history_subreddits", return_value=frozenset({"one", "two"})):
            metrics, _, _, updated = process_reported_user_reviews(
                subreddit, reddit, state, model, bot_username="optimist-prime", dry_run=False,
                jev_client=_FakeJev(selected_criterion="sample_repeated_link_posting"),
            )
        self.assertEqual(note.call_count, 2)
        self.assertEqual(
            [call.kwargs["thing"] for call in note.call_args_list],
            ["t1_r1", "t1_r2"],
        )
        expected = (
            "2 comments and 1 post scanned for [u/example_user]"
            "(https://www.reddit.com/user/example_user/). Matched configured criteria: "
            "Sample Repeated Link Posting."
        )
        for call in note.call_args_list:
            self.assertEqual(call.kwargs["result_text"], expected)
            self.assertTrue(call.kwargs["review_key"].startswith("reported:"))
        self.assertEqual(metrics.users_reviewed, 1)
        self.assertEqual(metrics.reviews_completed, 1)
        self.assertEqual(collect_history.call_args.kwargs["max_comments"], 90)
        self.assertEqual(collect_history.call_args.kwargs["max_posts"], 90)
        self.assertEqual(collect_history.call_args.kwargs["max_items"], 90)
        self.assertEqual(len(subreddit.messages), 1)
        self.assertIn(expected, subreddit.messages[0][1])
        self.assertNotIn("| Configured criterion |", subreddit.messages[0][1])
        self.assertNotIn("The supplied sample history supports one possible match.", subreddit.messages[0][1])
        self.assertNotIn("Themes found", subreddit.messages[0][1])
        self.assertNotIn("raw criterion scores", subreddit.messages[0][1])
        self.assertEqual(model.generation_config["max_output_tokens"], 6000)
        self.assertIn("Apply only the configured criteria", model.contents[0]["parts"][0])
        self.assertFalse(hasattr(subreddit, "banned"))
        self.assertTrue(all(entry["status"] == "completed" for entry in updated["reported_user_reviews"]["reports"].values()))

    def test_reported_user_with_a_score_above_the_setting_is_banned(self):
        now = 2_000_000_000.0
        subreddit = _Subreddit([_reported_comment("r1")])

        class BanList:
            def __init__(self):
                self.users = set()
                self.added = []

            def __call__(self, *, redditor, limit):
                self.assert_limit = limit
                return iter([redditor] if redditor in self.users else [])

            def add(self, username, **settings):
                self.added.append((username, settings))
                self.users.add(username)

        ban_list = BanList()
        subreddit.banned = ban_list
        reddit = _Reddit([
            SimpleNamespace(
                id="c1", body="history evidence", score=1,
                subreddit=SimpleNamespace(display_name="one"),
                created_utc=now - 10, permalink="/c1",
            ),
        ], [])
        model = _Model('{"summary":"","themes":[],"matches":[]}')
        jev = _FakeJev(selected_criterion="sample_repeated_link_posting")
        rules_json = _synthetic_moderation_rules_json()

        with patch.dict(os.environ, {"BOT_MODERATION_RULES_JSON": rules_json}, clear=False), \
             patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True), \
             patch.object(config, "REPORTED_USER_REVIEW_ACTION", "modmail"), \
             patch.object(config, "AUTOMATED_BAN_ENABLED", True), \
             patch.object(config, "SAFE_MODE", False), \
             patch.object(config, "HISTORY_REVIEW_BAN_THRESHOLD_PERCENT", 87), \
             patch("reported_user_review.load_history_subreddits", return_value=frozenset({"one"})), \
             patch("reported_user_review.record_review_note", return_value="delivered"), \
             patch("reddit_observability.single_write_attempt", return_value=nullcontext()):
            metrics, _, _, state = process_reported_user_reviews(
                subreddit, reddit, {}, model,
                jev_client=jev,
                persist_state=lambda _changed: None,
            )

        self.assertEqual(len(ban_list.added), 1)
        username, settings = ban_list.added[0]
        self.assertEqual(username, "example_user")
        self.assertNotIn("duration", settings)
        self.assertNotIn("Sample Repeated Link Posting", settings["ban_message"])
        self.assertEqual(ban_list.assert_limit, 1)
        self.assertEqual(len(subreddit.messages), 1)
        self.assertIn("was banned after a history review", subreddit.messages[0][1])
        self.assertIn("Sample Repeated Link Posting (93%)", subreddit.messages[0][1])
        self.assertEqual(metrics.reviews_completed, 1)
        self.assertEqual(
            next(iter(state["reported_user_reviews"]["reports"].values()))["status"],
            "completed",
        )

    def test_manual_review_user_uses_the_same_autoban_rule(self):
        now = 2_000_000_000.0
        subreddit = _Subreddit([])

        class BanList:
            def __init__(self):
                self.users = set()
                self.added = []

            def __call__(self, *, redditor, limit):
                return iter([redditor] if redditor in self.users else [])

            def add(self, username, **settings):
                self.added.append((username, settings))
                self.users.add(username)

        ban_list = BanList()
        subreddit.banned = ban_list
        reddit = _Reddit([
            SimpleNamespace(
                id="c1", body="history evidence", score=1,
                subreddit=SimpleNamespace(display_name="one"),
                created_utc=now - 10, permalink="/c1",
            ),
        ], [])
        model = _Model('{"summary":"","themes":[],"matches":[]}')
        jev = _FakeJev(selected_criterion="sample_repeated_link_posting")
        with patch.object(config, "REPORTED_USER_REVIEW_ENABLED", False), \
             patch.object(config, "AUTOMATED_BAN_ENABLED", True), \
             patch.object(config, "SAFE_MODE", False), \
             patch.object(config, "HISTORY_REVIEW_BAN_THRESHOLD_PERCENT", 87), \
             patch("reported_user_review.load_history_subreddits", return_value=frozenset({"one"})), \
             patch("reported_user_review.record_review_note", return_value="delivered"), \
             patch("reddit_observability.single_write_attempt", return_value=nullcontext()):
            metrics, _tokens, _cost, state = process_reported_user_reviews(
                subreddit, reddit, {}, model,
                manual_username="example_user",
                jev_client=jev,
                persist_state=lambda _changed: None,
            )

        self.assertEqual(len(jev.questions), 1)
        self.assertEqual(len(ban_list.added), 1)
        self.assertEqual(ban_list.added[0][0], "example_user")
        self.assertNotIn("duration", ban_list.added[0][1])
        self.assertEqual(metrics.reviews_completed, 1)
        self.assertTrue(any(
            record.get("username") == "example_user" and record.get("status") == "banned"
            for record in state["history_criteria_ban_actions"].values()
        ))

    def test_scheduled_review_uses_one_request_for_configured_criteria_without_item_attribution(self):
        reports = [_reported_comment("r1")]
        subreddit = _Subreddit(reports)
        now = 2_000_000_000.0
        comments = [
            SimpleNamespace(
                id=f"c{index}",
                body=f"synthetic profile comment {index}",
                score=1,
                subreddit=SimpleNamespace(display_name="one"),
                created_utc=now - index,
                permalink=f"/r/one/comments/c{index}/x/",
            )
            for index in range(250)
        ]
        posts = [
            SimpleNamespace(
                id=f"p{index}",
                title=f"synthetic profile post {index}",
                selftext="post text",
                score=1,
                subreddit=SimpleNamespace(display_name="one"),
                created_utc=now - 1000 - index,
                permalink=f"/r/one/comments/p{index}/x/",
            )
            for index in range(3)
        ]
        reddit = _Reddit(comments, posts)
        model = _Model(json.dumps({
            "summary": "The sample history may support the configured criterion.",
            "themes": [],
            "matches": [],
        }))
        jev = _FakeJev(selected_criterion="sample_repeated_link_posting")

        with patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True), \
             patch.object(config, "REPORTED_USER_REVIEW_MAX_LLM_CALLS_PER_RUN", 3), \
             patch.object(config, "HISTORY_REVIEW_MAX_ITEMS", 250), \
             patch("reported_user_review.collect_user_history", wraps=collect_user_history) as collect_history, \
             patch("reported_user_review.load_history_subreddits", return_value=frozenset({"one"})):
            metrics, _, _, updated = process_reported_user_reviews(
                subreddit, reddit, {}, model, dry_run=False, jev_client=jev,
            )

        self.assertEqual(metrics.reviews_completed, 1)
        self.assertEqual(collect_history.call_args.kwargs["max_comments"], 250)
        self.assertEqual(collect_history.call_args.kwargs["max_posts"], 250)
        self.assertEqual(collect_history.call_args.kwargs["max_items"], 250)
        self.assertEqual(len(jev.questions), 1)
        state, questions = jev.questions[0]
        self.assertEqual(len(state["reddit_history"]), 250)
        self.assertEqual(
            [item["kind"] for item in state["reddit_history"][:3]],
            ["post", "post", "post"],
        )
        self.assertEqual(
            [item["kind"] for item in state["reddit_history"][3:]],
            ["comment"] * 247,
        )
        self.assertEqual(reddit._redditor.submissions.limits, [250])
        self.assertEqual(len(questions), 3)
        self.assertTrue(all(question["type"] == "noul" for question in questions.values()))
        audit = updated["reported_user_reviews"]["audit"][-1]
        gate = audit["history_criteria_assessment"]
        self.assertEqual(gate["classifier"], "jev_full_history_noul")
        self.assertFalse(gate["item_attribution_available"])
        self.assertEqual(gate["matched_rules"], ["sample_repeated_link_posting"])
        self.assertEqual(len(gate["criterion_probabilities"]), 3)
        self.assertEqual(gate["match_threshold"], 0.67)
        self.assertEqual(gate["positive_criterion_count"], 1)
        self.assertIn("evidence_truncated_items", gate)
        self.assertEqual(audit["history_count"], 250)
        report_state = next(iter(updated["reported_user_reviews"]["reports"].values()))
        self.assertFalse(report_state["matching_item_count_known"])
        self.assertIsNone(report_state["matching_items"])

    def test_failed_modmail_delivery_reuses_saved_report_without_second_ai_call(self):
        now = 2_000_000_000.0
        reports = [_reported_comment("r1"), _reported_comment("r2", reason="second report")]
        comments = [
            SimpleNamespace(id="c1", body="one", score=1, subreddit=SimpleNamespace(display_name="one"), created_utc=now - 10, permalink="/c1"),
            SimpleNamespace(id="c2", body="two", score=1, subreddit=SimpleNamespace(display_name="one"), created_utc=now - 20, permalink="/c2"),
        ]
        subreddit = _Subreddit(reports)
        subreddit.fail_create_once = True
        reddit = _Reddit(comments, [])
        state = {}
        rules_json = _synthetic_moderation_rules_json()
        model = _Model(json.dumps({
            "summary": "The history contains several posts about gardening.",
            "themes": ["community gardening"],
            "matches": [{
                "evidence_index": 0,
                "rule_name": "sample_repeated_link_posting",
                "confidence": 0.95,
                "explanation": "Repeated evidence.",
            }],
        }))
        persisted = []
        with patch.dict(os.environ, {"BOT_MODERATION_RULES_JSON": rules_json}, clear=False), \
             patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True), \
             patch.object(config, "REPORTED_USER_REVIEW_ACTION", "modmail"), \
             patch.object(config, "REPORTED_USER_REVIEW_MAX_LLM_CALLS_PER_RUN", 3), \
             patch("reported_user_review.load_history_subreddits", return_value=frozenset({"one"})):
            first, _, _, state = process_reported_user_reviews(
                subreddit,
                reddit,
                state,
                model,
                bot_username="optimist-prime",
                jev_client=_FakeJev(selected_criterion="sample_repeated_link_posting"),
                persist_state=lambda changed: persisted.append(
                    dict(changed["reported_user_reviews"]["pending_notifications"])
                ),
            )

            class NoSecondAnalysis(_Model):
                def generate_content(self, *args, **kwargs):
                    raise AssertionError("saved report must be delivered without another AI call")

            second, _, _, state = process_reported_user_reviews(
                subreddit,
                reddit,
                state,
                NoSecondAnalysis(""),
                bot_username="optimist-prime",
            )

        self.assertEqual(first.retryable_failures, 1)
        self.assertEqual(model.calls, 1)
        self.assertTrue(any(persisted))
        self.assertEqual(second.moderator_notifications, 1)
        self.assertEqual(len(subreddit.messages), 1)
        self.assertEqual(state["reported_user_reviews"]["pending_notifications"], {})
        self.assertTrue(all(
            entry["status"] == "completed"
            for entry in state["reported_user_reviews"]["reports"].values()
        ))

    def test_dry_run_does_not_send_or_change_input_state(self):
        reports = [_reported_comment("r1")]
        subreddit = _Subreddit(reports)
        reddit = _Reddit([], [])
        state = {"existing": "value"}
        rules_json = json.dumps([{"name": "history_rule", "description": "rule", "actions": ["report"]}])
        model = _Model('{"matches":[]}')
        with patch.dict(os.environ, {"BOT_MODERATION_RULES_JSON": rules_json}, clear=False), \
             patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True):
            metrics, _, _, updated = process_reported_user_reviews(
                subreddit, reddit, state, model, dry_run=True,
            )
        self.assertEqual(state, {"existing": "value"})
        self.assertEqual(subreddit.messages, [])
        self.assertEqual(metrics.safe_mode_prevented_delivery, 0)
        self.assertNotEqual(updated, state)

    def test_suspended_account_is_skipped_without_history_or_llm(self):
        reports = [_reported_comment("r1", username="suspended_user")]
        subreddit = _Subreddit(reports)
        reddit = _Reddit([], [])
        reddit._redditor.is_suspended = True

        class FailingModel(_Model):
            def generate_content(self, *args, **kwargs):
                raise AssertionError("suspended accounts must not reach the LLM")

        with patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True):
            metrics, _, _, updated = process_reported_user_reviews(
                subreddit, reddit, {}, FailingModel('{"matches":[]}'),
                dry_run=True,
            )
        self.assertEqual(metrics.reports_skipped, 1)
        self.assertEqual(metrics.users_reviewed, 0)
        self.assertEqual(subreddit.messages, [])
        self.assertEqual(
            next(iter(updated["reported_user_reviews"]["reports"].values()))["status"],
            "skipped",
        )

    def test_empty_history_stays_incomplete_without_an_llm_call(self):
        reports = [_reported_comment("r1")]
        subreddit = _Subreddit(reports)
        reddit = _Reddit([], [])

        class FailingModel(_Model):
            def generate_content(self, *args, **kwargs):
                raise AssertionError("empty history must not reach the LLM")

        with patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True), \
             patch.object(config, "REPORTED_USER_REVIEW_MAX_LLM_CALLS_PER_RUN", 1):
            metrics, tokens, cost, updated = process_reported_user_reviews(
                subreddit,
                reddit,
                {},
                FailingModel('{"matches":[]}'),
                dry_run=True,
                jev_client=_FakeJev(),
            )

        self.assertEqual(metrics.users_reviewed, 1)
        self.assertEqual(metrics.reviews_completed, 0)
        self.assertEqual(metrics.retryable_failures, 1)
        self.assertEqual(metrics.history_items_reviewed, 0)
        self.assertEqual(tokens, 0)
        self.assertEqual(cost, 0.0)
        self.assertEqual(subreddit.messages, [])

    def test_failed_attempt_stays_incomplete_and_later_reports_only_use_unseen_items(self):
        now = 2_000_000_000.0
        subreddit = _Subreddit([])
        c1 = SimpleNamespace(
            id="c1", body="history c1", score=1,
            subreddit=SimpleNamespace(display_name="one"),
            created_utc=now - 1, permalink="/r/one/comments/c1/x/",
        )
        c2 = SimpleNamespace(
            id="c2", body="history c2", score=1,
            subreddit=SimpleNamespace(display_name="one"),
            created_utc=now - 2, permalink="/r/one/comments/c2/x/",
        )
        reddit = _Reddit([c1], [])
        criteria = synthetic_history_criteria()
        probabilities = {name: 0.03 for name in criteria}
        complete = HistoryGateAssessment(
            criterion_score_revisions=current_history_criteria_revisions(),
            rule_names=tuple(criteria),
            bundle_size=0,
            bundles=(),
            matched_rules=(),
            calls=1,
            input_tokens=12,
            cost=0.0002,
            served_model="test",
            classifier="jev_full_history_noul",
            item_attribution_available=False,
            criterion_probabilities=probabilities,
            scores_complete=True,
        )
        jev = _FakeJev()
        calls = []

        def assess(history, *_args, **_kwargs):
            calls.append([item.item_id for item in history.items])
            if len(calls) == 1:
                raise JevHistoryReviewError("provider failed")
            return complete

        options = [
            patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True),
            patch.object(config, "REPORTED_USER_REVIEW_ACTION", "log"),
            patch.object(config, "REPORTED_USER_REVIEW_MAX_LLM_CALLS_PER_RUN", 0),
            patch.object(config, "HISTORY_REVIEW_MAX_ITEMS", 250),
            patch("reported_user_review.load_history_subreddits", return_value=frozenset({"one"})),
            patch("reported_user_review.assess_history_nouls_with_jev", side_effect=assess),
            patch("reported_user_review.record_review_note", return_value="delivered"),
        ]
        with options[0], options[1], options[2], options[3], options[4], options[5], options[6]:
            first_item = collect_reported_items(_Subreddit([_reported_comment("r1")]))[0]
            first, _, _, state = process_reported_user_reviews(
                subreddit, reddit, {}, _Model('{"summary":"","themes":[],"matches":[]}'),
                reported_items=[first_item], jev_client=jev,
            )
            self.assertEqual(
                state["reported_user_reviews"]["reports"][first_item.report_key]["status"],
                "awaiting_new_history",
            )

            same_report, _, _, state = process_reported_user_reviews(
                subreddit, reddit, state, _Model('{"summary":"","themes":[],"matches":[]}'),
                reported_items=[first_item], jev_client=jev,
            )
            self.assertEqual(same_report.reports_skipped, 1)
            self.assertEqual(len(calls), 1)

            second_item = collect_reported_items(_Subreddit([_reported_comment("r2")]))[0]
            no_new, _, _, state = process_reported_user_reviews(
                subreddit, reddit, state, _Model('{"summary":"","themes":[],"matches":[]}'),
                reported_items=[second_item], jev_client=jev,
            )
            self.assertEqual(no_new.reviews_completed, 0)
            self.assertEqual(no_new.retryable_failures, 1)
            self.assertEqual(
                state["reported_user_reviews"]["reports"][second_item.report_key]["status"],
                "awaiting_new_history",
            )
            self.assertEqual(len(calls), 1)

            reddit._redditor.comments.values.append(c2)
            third_item = collect_reported_items(_Subreddit([_reported_comment("r3")]))[0]
            fresh, _, _, state = process_reported_user_reviews(
                subreddit, reddit, state, _Model('{"summary":"","themes":[],"matches":[]}'),
                reported_items=[third_item], jev_client=jev,
            )

        self.assertEqual(first.retryable_failures, 1)
        self.assertEqual(fresh.reviews_completed, 1)
        self.assertEqual(calls, [["c1"], ["c2"]])

    def test_partial_reddit_history_keeps_ledger_attempt_incomplete(self):
        report = collect_reported_items(_Subreddit([_reported_comment("r-gap")]))[0]
        subreddit = _Subreddit([])
        item = _history_item("c-gap", "comment", "one", 1_999_999_999.0)
        history = UserHistory(
            "example_user", (item,), 1, 0, errors=("comment listing was incomplete",)
        )
        criteria = synthetic_history_criteria()
        assessment = HistoryGateAssessment(
            criterion_score_revisions=current_history_criteria_revisions(),
            rule_names=tuple(criteria),
            bundle_size=0,
            bundles=(),
            matched_rules=(),
            calls=1,
            input_tokens=12,
            cost=0.0002,
            served_model="test",
            classifier="jev_full_history_noul",
            item_attribution_available=False,
            criterion_probabilities={name: 0.03 for name in criteria},
            scores_complete=True,
        )
        with patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True), \
             patch.object(config, "REPORTED_USER_REVIEW_ACTION", "log"), \
             patch.object(config, "REPORTED_USER_REVIEW_MAX_LLM_CALLS_PER_RUN", 0), \
             patch("reported_user_review.load_history_subreddits", return_value=frozenset({"one"})), \
             patch("reported_user_review.collect_user_history", return_value=history), \
             patch("reported_user_review.assess_history_nouls_with_jev", return_value=assessment):
            metrics, _, _, state = process_reported_user_reviews(
                subreddit, _Reddit([], []), {}, _Model('{"summary":"","themes":[],"matches":[]}'),
                reported_items=[report], jev_client=_FakeJev(),
            )

        self.assertEqual(metrics.reviews_completed, 0)
        self.assertEqual(metrics.retryable_failures, 1)
        self.assertEqual(self.review_ledger.metrics()["incomplete_jev_attempts"], 1)
        self.assertEqual(
            state["reported_user_reviews"]["reports"][report.report_key]["status"],
            "awaiting_new_history",
        )

    def test_report_review_defers_when_current_moderator_roster_cannot_be_verified(self):
        raw_report = _reported_comment("r-roster")
        report = collect_reported_items(_Subreddit([raw_report]))[0]
        subreddit = _Subreddit([raw_report])

        def fail_roster():
            raise RuntimeError("Reddit unavailable")

        subreddit.moderator = fail_roster
        reddit = _Reddit([], [])
        with patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True):
            metrics, _, _, state = process_reported_user_reviews(
                subreddit, reddit, {}, _Model('{"matches":[]}'),
                jev_client=_FakeJev(),
            )

        self.assertEqual(metrics.retryable_failures, 1)
        self.assertEqual(metrics.users_reviewed, 0)
        self.assertEqual(reddit._redditor.comments.calls, 0)
        self.assertEqual(
            state["reported_user_reviews"]["reports"][report.report_key]["status"],
            "retryable",
        )

    def test_report_review_runs_jev_when_narrative_call_cap_is_exhausted(self):
        now = 2_000_000_000.0
        reports = [_reported_comment("r1")]
        comments = [SimpleNamespace(
            id="c1", body="a usable history item", score=1,
            subreddit=SimpleNamespace(display_name="one"), created_utc=now - 10,
            permalink="/c1",
        )]
        jev = _FakeJev(selected_criterion="sample_repeated_link_posting")
        model = _Model('{"summary":"unused","themes":[],"matches":[]}')
        with patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True), \
             patch.object(config, "REPORTED_USER_REVIEW_MAX_LLM_CALLS_PER_RUN", 0), \
             patch("reported_user_review.load_history_subreddits", return_value=frozenset({"one"})), \
             patch("reported_user_review.evaluate_history") as evaluate:
            metrics, _tokens, _cost, _state = process_reported_user_reviews(
                _Subreddit(reports), _Reddit(comments, []), {}, model,
                jev_client=jev, dry_run=True,
            )

        self.assertEqual(len(jev.questions), 1)
        evaluate.assert_not_called()
        self.assertEqual(model.calls, 0)
        self.assertEqual(metrics.reviews_completed, 1)
        self.assertEqual(metrics.retryable_failures, 0)

    def test_empty_operator_criteria_skip_history_scanning(self):
        reports = [_reported_comment("r1")]
        subreddit = _Subreddit(reports)
        now = 2_000_000_000.0
        reddit = _Reddit(
            [SimpleNamespace(
                id="c1",
                body="evidence",
                score=1,
                subreddit=SimpleNamespace(display_name="one"),
                created_utc=now - 10,
                permalink="/c1",
            )],
            [],
        )
        model = _Model('{"matches":[]}')
        with patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True), \
             patch("reported_user_review.load_history_criteria", return_value={}), \
             patch.object(config, "REPORTED_USER_REVIEW_MAX_LLM_CALLS_PER_RUN", 1):
            metrics, _, _, updated = process_reported_user_reviews(
                subreddit,
                reddit,
                {},
                model,
                dry_run=True,
                jev_client=_FakeJev(),
            )

        self.assertEqual(metrics.users_reviewed, 0)
        self.assertEqual(metrics.retryable_failures, 0)
        self.assertEqual(reddit._redditor.comments.calls, 0)
        self.assertEqual(reddit._redditor.submissions.calls, 0)
        self.assertEqual(model.calls, 0)
        self.assertEqual(updated["reported_user_reviews"]["reports"], {})

    def test_missing_jev_retries_before_fetching_history(self):
        reports = [_reported_comment("r1")]
        subreddit = _Subreddit(reports)
        reddit = _Reddit(
            [SimpleNamespace(
                id="c1",
                body="evidence",
                score=1,
                subreddit=SimpleNamespace(display_name="one"),
                created_utc=2_000_000_000.0,
                permalink="/c1",
            )],
            [],
        )
        model = _Model('{"matches":[]}')
        with patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True):
            metrics, _, _, updated = process_reported_user_reviews(
                subreddit, reddit, {}, model, dry_run=True, jev_client=None,
            )

        self.assertEqual(metrics.retryable_failures, 1)
        self.assertEqual(reddit._redditor.comments.calls, 0)
        self.assertEqual(reddit._redditor.submissions.calls, 0)
        self.assertEqual(model.calls, 0)
        self.assertEqual(subreddit.messages, [])
        entry = next(iter(updated["reported_user_reviews"]["reports"].values()))
        self.assertEqual(entry["status"], "retryable")

    def test_text_model_only_match_does_not_override_the_configured_history_result(self):
        reports = [_reported_comment("r1")]
        subreddit = _Subreddit(reports)
        now = 2_000_000_000.0
        reddit = _Reddit(
            [SimpleNamespace(
                id="c1",
                body="evidence",
                score=1,
                subreddit=SimpleNamespace(display_name="one"),
                created_utc=now - 10,
                permalink="/c1",
            )],
            [],
        )
        model = _Model(json.dumps({
            "summary": "Mostly discusses gardening and community events.",
            "themes": ["gardening"],
            "matches": [{
                "evidence_index": 0,
                "rule_name": "sample_repeated_link_posting",
                "confidence": 0.9,
                "explanation": "Text-model reference match.",
            }],
        }))
        jev = _FakeJev(choice="no")
        with patch.object(config, "REPORTED_USER_REVIEW_ENABLED", True), \
             patch.object(config, "REPORTED_USER_REVIEW_MAX_LLM_CALLS_PER_RUN", 1), \
             patch("reported_user_review.load_history_subreddits", return_value=frozenset({"one"})):
            metrics, tokens, cost, updated = process_reported_user_reviews(
                subreddit, reddit, {}, model, dry_run=False, jev_client=jev,
            )

        self.assertEqual(metrics.moderator_notifications, 0)
        self.assertEqual(subreddit.messages, [])
        self.assertEqual(tokens, 27)  # 15 text-model tokens plus 12 structured-decision input tokens.
        self.assertAlmostEqual(cost, 0.000203)
        audit = updated["reported_user_reviews"]["audit"][-1]
        self.assertEqual(audit["recommendation"], "no_actionable_match")
        self.assertEqual(audit["history_criteria_assessment"]["matched_rules"], [])
        self.assertIn("sample_repeated_link_posting", audit["findings"][0]["rule_name"])
        self.assertEqual(audit["history_criteria_assessment"]["classifier"], "jev_full_history_noul")
        self.assertEqual(len(audit["history_criteria_assessment"]["criterion_probabilities"]), 3)
        self.assertEqual(audit["history_criteria_assessment"]["positive_criterion_count"], 0)


if __name__ == "__main__":
    unittest.main()
