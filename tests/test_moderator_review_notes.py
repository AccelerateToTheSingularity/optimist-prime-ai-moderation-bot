import unittest
from unittest.mock import patch

from moderator_review_notes import (
    record_review_note,
    reconcile_review_notes,
)


class FakeNote:
    def __init__(self, note_id, note):
        self.id = note_id
        self.note = note


class FakeNotes:
    def __init__(self):
        self.created = []
        self.visible = []
        self.create_error = None
        self.list_error = None
        self.events = []

    def create(self, **kwargs):
        self.events.append("create")
        self.created.append(kwargs)
        if self.create_error is not None:
            raise self.create_error
        note = FakeNote(f"note-{len(self.created)}", kwargs["note"])
        self.visible.append(note)
        return note

    def redditors(self, username, **kwargs):
        self.events.append("list")
        if self.list_error is not None:
            raise self.list_error
        self.list_args = (username, kwargs)
        return list(self.visible)


class FakeSubreddit:
    display_name = "example_community"

    def __init__(self, notes=None):
        self.mod = type("Mod", (), {})()
        if notes is not None:
            self.mod.notes = notes


class RedditAPIException(Exception):
    pass


class ModeratorReviewNotesTests(unittest.TestCase):
    def setUp(self):
        self.notes = FakeNotes()
        self.subreddit = FakeSubreddit(self.notes)
        self.state = {}
        self.persisted = []

    def persist(self, state):
        self.persisted.append(state)
        return True

    def record(self, **kwargs):
        args = {
            "username": "ExampleUser",
            "review_key": "comment:abc123",
            "result_text": "Review complete: no matching evidence. Scanned 4 comments.",
            "persist_state": self.persist,
        }
        args.update(kwargs)
        return record_review_note(self.subreddit, self.state, **args)

    def test_success_persists_intent_before_user_summary_post_and_verifies_response(self):
        def persist_with_order(state):
            self.notes.events.append("persist")
            self.persisted.append(state)
            return True

        result = record_review_note(
            self.subreddit,
            self.state,
            username="ExampleUser",
            review_key="comment:abc123",
            result_text="Review complete: no matching evidence. Scanned 4 comments.",
            persist_state=persist_with_order,
        )

        self.assertEqual(result, "delivered")
        self.assertEqual(self.notes.events, ["persist", "create", "persist"])
        self.assertEqual(self.notes.created[0]["label"], "USER_SUMMARY")
        self.assertEqual(self.notes.created[0]["redditor"], "ExampleUser")
        body = self.notes.created[0]["note"]
        self.assertIn("[review-ref:", body)
        self.assertLessEqual(len(body), 250)
        record = next(iter(self.state["moderator_review_notes"]["records"].values()))
        self.assertEqual(record["status"], "delivered")
        self.assertEqual(record["note_id"], "note-1")

    def test_repeat_of_delivered_review_does_not_post_twice(self):
        self.assertEqual(self.record(), "delivered")

        self.assertEqual(self.record(), "already_delivered")
        self.assertEqual(len(self.notes.created), 1)

    def test_note_links_to_its_report_item(self):
        self.assertEqual(self.record(thing="t1_test000005"), "delivered")

        self.assertEqual(self.notes.created[0]["thing"], "t1_test000005")
        record = next(iter(self.state["moderator_review_notes"]["records"].values()))
        self.assertEqual(record["thing"], "t1_test000005")

    def test_linked_items_have_independent_idempotency_keys(self):
        self.assertEqual(self.record(thing="t1_test000005"), "delivered")
        self.assertEqual(self.record(thing="t3_test002"), "delivered")

        self.assertEqual(len(self.notes.created), 2)
        self.assertEqual(
            [entry["thing"] for entry in self.notes.created],
            ["t1_test000005", "t3_test002"],
        )
        self.assertEqual(self.record(thing="t1_test000005"), "already_delivered")
        self.assertEqual(len(self.notes.created), 2)

    def test_invalid_link_target_is_rejected_without_post(self):
        self.assertEqual(self.record(thing="t2_user123"), "invalid")
        self.assertEqual(self.notes.created, [])

    def test_marker_is_stable_but_scoped_to_username_subreddit_and_review_key(self):
        self.assertEqual(self.record(), "delivered")
        marker = self.notes.created[0]["note"].split()[-1]

        other_review = self.record(review_key="comment:other")
        self.assertEqual(other_review, "delivered")
        other_user = self.record(username="AnotherUser")
        self.assertEqual(other_user, "delivered")
        self.assertEqual(len(self.notes.created), 3)
        self.assertEqual(len({entry["note"].split()[-1] for entry in self.notes.created}), 3)
        self.assertTrue(marker.startswith("[review-ref:"))

    def test_persistence_failure_escapes_before_create(self):
        def failing_persist(_state):
            raise SystemExit(1)

        with self.assertRaises(SystemExit):
            self.record(persist_state=failing_persist)

        self.assertEqual(self.notes.created, [])

    def test_uncertain_post_is_reconciled_without_resending(self):
        self.notes.create_error = TimeoutError("response timed out")
        self.assertEqual(self.record(), "uncertain")
        self.assertEqual(len(self.notes.created), 1)

        self.notes.create_error = None
        self.assertEqual(self.record(), "uncertain")
        self.assertEqual(len(self.notes.created), 1)
        self.assertEqual(self.notes.list_args[0], "ExampleUser")
        self.assertEqual(self.notes.list_args[1]["all_notes"], True)

    def test_reconciliation_confirms_a_visible_marker_and_persists_note_id(self):
        self.notes.create_error = TimeoutError("response timed out")
        self.assertEqual(self.record(), "uncertain")

        self.notes.create_error = None
        marker = self.notes.created[0]["note"].split()[-1]
        self.notes.visible.append(FakeNote("visible-42", f"Reddit's stored form {marker}"))
        counts = reconcile_review_notes(
            self.subreddit,
            self.state,
            persist_state=self.persist,
        )

        self.assertEqual(counts["delivered"], 1)
        record = next(iter(self.state["moderator_review_notes"]["records"].values()))
        self.assertEqual(record["status"], "delivered")
        self.assertEqual(record["note_id"], "visible-42")
        self.assertEqual(len(self.notes.created), 1)

    def test_missing_marker_and_listing_failure_remain_uncertain_without_post(self):
        self.notes.create_error = TimeoutError("response timed out")
        self.assertEqual(self.record(), "uncertain")
        self.notes.create_error = None

        counts = reconcile_review_notes(self.subreddit, self.state, persist_state=self.persist)
        self.assertEqual(counts["uncertain"], 1)
        self.assertEqual(len(self.notes.created), 1)

        self.notes.list_error = OSError("listing unavailable")
        counts = reconcile_review_notes(self.subreddit, self.state, persist_state=self.persist)
        self.assertEqual(counts["uncertain"], 1)
        self.assertEqual(len(self.notes.created), 1)

    def test_reconciliation_stops_after_bounded_checks_and_never_reposts(self):
        self.notes.create_error = TimeoutError("response timed out")
        self.assertEqual(self.record(), "uncertain")
        self.notes.create_error = None

        for _ in range(5):
            reconcile_review_notes(self.subreddit, self.state, persist_state=self.persist)

        record = next(iter(self.state["moderator_review_notes"]["records"].values()))
        self.assertEqual(record["status"], "needs_attention")
        self.assertEqual(record["reconcile_checks"], 5)
        self.assertEqual(len(self.notes.created), 1)
        counts = reconcile_review_notes(self.subreddit, self.state, persist_state=self.persist)
        self.assertEqual(counts["checked"], 0)

    def test_completed_record_retention_exceeds_callers_replay_windows(self):
        with patch("moderator_review_notes._MAX_COMPLETED_RECORDS", 2):
            for index in range(3):
                self.assertEqual(
                    self.record(review_key=f"comment:{index}"),
                    "delivered",
                )

        records = self.state["moderator_review_notes"]["records"]
        self.assertEqual(len(records), 2)
        self.assertEqual(
            {record["status"] for record in records.values()},
            {"delivered"},
        )

    def test_warning_logs_include_status_and_type_but_no_note_content(self):
        self.notes.create_error = TimeoutError("private note content must not be logged")

        with self.assertLogs("moderator_review_notes", level="WARNING") as captured:
            self.assertEqual(
                self.record(result_text="Sensitive review text"),
                "uncertain",
            )

        output = "\n".join(captured.output)
        self.assertIn("status=uncertain", output)
        self.assertIn("TimeoutError", output)
        self.assertNotIn("Sensitive review text", output)
        self.assertNotIn("private note content", output)

    def test_definite_reddit_api_rejection_is_failed_and_not_retried(self):
        self.notes.create_error = RedditAPIException("invalid note")

        self.assertEqual(self.record(), "failed")
        self.assertEqual(self.record(), "failed")
        self.assertEqual(len(self.notes.created), 1)

    def test_http_client_error_is_definite_rejection(self):
        error = RuntimeError("bad request")
        error.response = type("Response", (), {"status_code": 400})()
        self.notes.create_error = error

        self.assertEqual(self.record(), "failed")
        self.assertEqual(len(self.notes.created), 1)

    def test_unverified_create_response_is_uncertain(self):
        self.notes.create = lambda **kwargs: FakeNote("note-1", "different body")

        self.assertEqual(self.record(), "uncertain")
        record = next(iter(self.state["moderator_review_notes"]["records"].values()))
        self.assertEqual(record["status"], "uncertain")

    def test_safe_skips_for_dry_run_missing_store_or_missing_interface(self):
        self.assertEqual(self.record(dry_run=True), "skipped_dry_run")
        self.assertEqual(self.record(persist_state=None), "skipped_no_durable_store")
        self.assertEqual(record_review_note(
            FakeSubreddit(),
            self.state,
            username="ExampleUser",
            review_key="comment:abc123",
            result_text="Review complete.",
            persist_state=self.persist,
        ), "skipped_unavailable")
        self.assertEqual(self.notes.created, [])

        counts = reconcile_review_notes(self.subreddit, self.state, dry_run=True)
        self.assertEqual(counts["checked"], 0)
        self.assertEqual(counts["skipped"], 0)

    def test_note_over_limit_is_rejected_without_post(self):
        self.assertEqual(self.record(result_text="x" * 251), "invalid")
        self.assertEqual(self.notes.created, [])


if __name__ == "__main__":
    unittest.main()
