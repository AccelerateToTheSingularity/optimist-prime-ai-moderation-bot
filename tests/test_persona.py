"""Tests for persona.py."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from persona import (
    _captured_thread_comments,
    build_full_context,
    generate_conversation_response_from_text,
    generate_conversational_response,
    generate_post_summon_response,
    get_reply_prompt,
)


class TestGetReplyPrompt(unittest.TestCase):
    def test_includes_context(self):
        prompt = get_reply_prompt("test message", "test context")
        self.assertIn("test context", prompt)
        self.assertIn("test message", prompt)

    def test_includes_persona(self):
        prompt = get_reply_prompt("test", "context")
        self.assertIn("Optimist Prime", prompt)

    def test_persona_blocks_generic_disclaimers(self):
        prompt = get_reply_prompt("test", "context")
        self.assertIn("generic disclaimers", prompt)
        self.assertIn("avoid filler", prompt)

    def test_summon_note(self):
        prompt = get_reply_prompt("test", "context", is_summon=True)
        self.assertIn("SPECIAL NOTE", prompt)

    def test_no_summon_note(self):
        prompt = get_reply_prompt("test", "context", is_summon=False)
        self.assertNotIn("SPECIAL NOTE", prompt)

    def test_prompt_allows_short_acknowledgments_and_limits_unrequested_replies(self):
        prompt = get_reply_prompt("Ok good", "A short conversation.")
        self.assertIn("brief acknowledgment may be one short sentence", prompt)
        self.assertIn("directly address or summon you", prompt)


class TestBuildFullContext(unittest.TestCase):
    def test_includes_title(self):
        submission = MagicMock()
        submission.title = "Test Post Title"
        submission.selftext = "Test body"

        comment = MagicMock()
        comment.parent.return_value = MagicMock()
        comment.parent.return_value.body = None
        comment.parent.return_value.author = None

        context = build_full_context(comment, submission)
        self.assertIn("Test Post Title", context)
        self.assertIn("Test body", context)

    def test_parent_chain_is_not_hydrated_from_reddit(self):
        submission = SimpleNamespace(id="post-1", title="Title", selftext="Body")
        comment = MagicMock()
        comment.id = "reply"
        comment.submission_id = "post-1"
        comment.parent_id = "t1_test001"
        comment.created_utc = 2
        comment.parent.side_effect = AssertionError("must not fetch parent")
        context = build_full_context(comment, submission, captured_comments=[])
        self.assertIn("Title", context)
        comment.parent.assert_not_called()

    def test_captured_comment_selection_excludes_other_posts(self):
        post = SimpleNamespace(id="post-1")
        parent = SimpleNamespace(
            id="parent", parent_id="t3_post-1", submission_id="post-1",
            created_utc=10, body="Earlier context", author=SimpleNamespace(name="example_user"),
        )
        unrelated = SimpleNamespace(
            id="other", parent_id="t3_test03", submission_id="post-2",
            created_utc=11, body="Wrong post", author=SimpleNamespace(name="bob"),
        )
        incoming = SimpleNamespace(
            id="reply", parent_id="t1_test04", submission_id="post-1", created_utc=12,
        )

        result = _captured_thread_comments(incoming, post, [parent, unrelated])

        self.assertEqual(len(result), 1)
        self.assertIn("Earlier context", result[0])
        self.assertNotIn("Wrong post", result[0])


class TestThreadContextDefault(unittest.TestCase):
    class Model:
        def __init__(self):
            self.prompt = None

        def generate_content(self, contents, generation_config=None):
            self.prompt = "\n".join(contents[0]["parts"])
            usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=4)
            return SimpleNamespace(text="A helpful answer.", usage_metadata=usage)

    def test_reply_includes_already_captured_thread_context_by_default(self):
        model = self.Model()
        submission = SimpleNamespace(id="p1", title="Thread title", selftext="Post text")
        comment = SimpleNamespace(
            body="Hello", id="reply", submission_id="p1", parent_id="t1-parent",
            created_utc=2,
        )
        captured = [SimpleNamespace(
            id="parent", submission_id="p1", parent_id="t3-p1",
            body="Earlier conversation", created_utc=1,
            score=1, author=SimpleNamespace(name="member"),
        )]

        _response, usage = generate_conversational_response(
            comment, submission, model, captured_comments=captured
        )

        self.assertIn("Thread title", model.prompt)
        self.assertIn("Post text", model.prompt)
        self.assertIn("Earlier conversation", model.prompt)
        self.assertNotIn("jev_input_tokens", usage)

    def test_private_conversation_disables_internal_reasoning(self):
        class Model:
            def __init__(self):
                self.generation_config = None

            def generate_content(self, _contents, generation_config=None):
                self.generation_config = generation_config
                usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=4)
                return SimpleNamespace(text="A helpful answer.", usage_metadata=usage)

        model = Model()
        response, _usage = generate_conversation_response_from_text(
            "What is new?", "Private conversation with u/member.", model
        )

        self.assertEqual(response, "A helpful answer.")
        self.assertIs(model.generation_config["thinking_enabled"], False)

    def test_context_dependent_comment_reply_includes_captured_thread(self):
        model = self.Model()
        submission = SimpleNamespace(id="p1", title="Thread title", selftext="Post body")
        comment = SimpleNamespace(
            body="What do you mean by that?", id="reply", submission_id="p1",
            parent_id="t1-parent", created_utc=2,
        )
        captured = [
            SimpleNamespace(
                id="parent", submission_id="p1", parent_id="t3-p1",
                body="The claim we were discussing", created_utc=1,
                score=4, author=SimpleNamespace(name="member"),
            ),
            SimpleNamespace(
                id="top-low", submission_id="p1", parent_id="t3_p1",
                body="Low-ranked top-level comment", created_utc=0,
                score=1, author=SimpleNamespace(name="low"),
            ),
            SimpleNamespace(
                id="top-high", submission_id="p1", parent_id="t3_p1",
                body="Most-upvoted top-level comment", created_utc=1,
                score=25, author=SimpleNamespace(name="high"),
            ),
            SimpleNamespace(
                id="top-mid", submission_id="p1", parent_id="t3_p1",
                body="Second-most-upvoted top-level comment", created_utc=1,
                score=12, author=SimpleNamespace(name="mid"),
            ),
            *[
                SimpleNamespace(
                    id=f"recent{i}", submission_id="p1", parent_id="t1-other",
                    body=f"Recent captured comment {i}", created_utc=1.5 + i / 10,
                    score=0, author=SimpleNamespace(name=f"recent{i}"),
                )
                for i in range(1, 4)
            ],
        ]

        generate_conversational_response(comment, submission, model, captured_comments=captured)

        self.assertIn("Thread title", model.prompt)
        self.assertIn("The claim we were discussing", model.prompt)
        self.assertIn("Most-upvoted top-level comment", model.prompt)
        self.assertIn("Second-most-upvoted top-level comment", model.prompt)
        self.assertNotIn("Low-ranked top-level comment", model.prompt)

    def test_post_summon_uses_post_title_text_and_best_captured_comments(self):
        model = self.Model()
        submission = SimpleNamespace(id="p1", title="The post title", selftext="The post text")
        captured = [
            SimpleNamespace(
                id=f"c{i}", submission_id="p1", body=f"Captured view {i}",
                created_utc=i, score=i, author=SimpleNamespace(name=f"member{i}"),
            )
            for i in range(1, 8)
        ]

        generate_post_summon_response(submission, model, captured_comments=captured)

        self.assertIn("The post title", model.prompt)
        self.assertIn("The post text", model.prompt)
        self.assertIn("Captured view 7", model.prompt)
        self.assertLessEqual(model.prompt.count("[Captured comment"), 4)

    def test_post_summon_includes_captured_comments_without_a_jev_call(self):
        model = self.Model()
        submission = SimpleNamespace(id="p1", title="Thread title", selftext="Post body")
        captured = [
            SimpleNamespace(id=f"c{i}", submission_id="p1", body=f"Relevant {i}", created_utc=i, author=SimpleNamespace(name="member"))
            for i in range(1, 8)
        ]
        generate_post_summon_response(submission, model, captured_comments=captured)
        self.assertIn("Thread title", model.prompt)
        self.assertIn("Post body", model.prompt)
        self.assertIn("Already-captured comments", model.prompt)
        self.assertLessEqual(model.prompt.count("[u/member]"), 4)


if __name__ == "__main__":
    unittest.main()
