"""Tests for reddit_reference.py URL parsing."""

import unittest

from reddit_reference import parse_reddit_url, find_reddit_urls_in_text


class TestParseRedditUrl(unittest.TestCase):
    def test_valid_post_url(self):
        result = parse_reddit_url("https://www.reddit.com/r/example_community/comments/abc123/my_post/")
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "post")
        self.assertEqual(result["id"], "t3_abc123")

    def test_valid_comment_url(self):
        result = parse_reddit_url("https://www.reddit.com/r/example_community/comments/abc123/my_post/def456/")
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "comment")
        self.assertEqual(result["id"], "t1_def456")

    def test_old_reddit_url(self):
        result = parse_reddit_url("https://old.reddit.com/r/example_community/comments/abc123/post/")
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "post")

    def test_np_reddit_url(self):
        result = parse_reddit_url("https://np.reddit.com/r/example_community/comments/abc123/post/")
        self.assertIsNotNone(result)

    def test_non_reddit_url(self):
        self.assertIsNone(parse_reddit_url("https://example.com/something"))

    def test_invalid_url(self):
        self.assertIsNone(parse_reddit_url("not a url"))

    def test_empty_string(self):
        self.assertIsNone(parse_reddit_url(""))


class TestFindRedditUrls(unittest.TestCase):
    def test_finds_url_in_text(self):
        text = "Check out this post: https://www.reddit.com/r/example_community/comments/abc123/test/"
        result = find_reddit_urls_in_text(text)
        self.assertIsNotNone(result)
        self.assertIn("reddit.com", result)

    def test_no_url_in_text(self):
        result = find_reddit_urls_in_text("No urls here")
        self.assertIsNone(result)

    def test_returns_first_url(self):
        text = "First https://www.reddit.com/r/a/comments/111/a/ and second https://www.reddit.com/r/b/comments/222/b/"
        result = find_reddit_urls_in_text(text)
        self.assertIn("111", result)


if __name__ == "__main__":
    unittest.main()
