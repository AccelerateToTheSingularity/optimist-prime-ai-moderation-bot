"""Offline tests for the local Reddit authorization helper."""

import os
import unittest
from unittest.mock import patch

from obtain_refresh_token import (
    _REQUIRED_REDDIT_SCOPES,
    _expected_bot_username,
    _has_required_scopes,
    _new_oauth_state,
    _parse_reddit_scopes,
    _parse_redirect_request,
)


class TestRefreshTokenHelper(unittest.TestCase):
    def test_redirect_parser_decodes_query_values(self):
        request = "GET /?state=abc%20123&code=a%2Bb%3Dc HTTP/1.1\r\nHost: localhost\r\n\r\n"

        self.assertEqual(
            _parse_redirect_request(request),
            {"state": "abc 123", "code": "a+b=c"},
        )

    def test_redirect_parser_rejects_missing_request_target(self):
        with self.assertRaises(ValueError):
            _parse_redirect_request("not an HTTP request")

    def test_oauth_state_is_long_url_safe_and_fresh(self):
        state = _new_oauth_state()

        self.assertGreaterEqual(len(state), 40)
        self.assertRegex(state, r"^[A-Za-z0-9_-]+$")
        self.assertNotEqual(state, _new_oauth_state())

    def test_expected_account_must_be_configured_explicitly(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_expected_bot_username(), "")

    def test_expected_account_uses_only_the_explicit_expected_username(self):
        with patch.dict(os.environ, {
            "BOT_EXPECTED_USERNAME": "example_bot",
        }, clear=True):
            self.assertEqual(_expected_bot_username(), "example_bot")

    def test_legacy_reddit_username_does_not_configure_the_expected_account(self):
        with patch.dict(os.environ, {"REDDIT_USERNAME": "example_bot"}, clear=True):
            self.assertEqual(_expected_bot_username(), "")

    def test_fresh_oauth_requests_only_account_identity_and_read_access(self):
        self.assertEqual(_REQUIRED_REDDIT_SCOPES, frozenset({"identity", "read"}))
        self.assertEqual(_parse_reddit_scopes(None), _REQUIRED_REDDIT_SCOPES)
        self.assertTrue(_has_required_scopes(_REQUIRED_REDDIT_SCOPES))
        for scope in _REQUIRED_REDDIT_SCOPES:
            with self.subTest(missing_scope=scope):
                self.assertFalse(_has_required_scopes(_REQUIRED_REDDIT_SCOPES - {scope}))
        self.assertFalse(_has_required_scopes({"*"}))

    def test_operator_can_request_only_explicit_additional_scopes(self):
        requested = _parse_reddit_scopes("identity, read submit modposts")
        self.assertEqual(requested, frozenset({"identity", "read", "submit", "modposts"}))
        self.assertTrue(_has_required_scopes(requested, requested))
        self.assertFalse(_has_required_scopes(requested - {"modposts"}, requested))

    def test_scope_configuration_rejects_wildcards_and_missing_required_scopes(self):
        for value in ("*", "read submit", "identity read bad-scope"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _parse_reddit_scopes(value)


if __name__ == "__main__":
    unittest.main()
