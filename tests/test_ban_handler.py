
import unittest
from unittest.mock import MagicMock, Mock, patch
import sys
import os
from io import StringIO
from datetime import datetime, timezone

# Add parent directory to path to import ban_handler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ban_handler import check_and_ban_negative_karma_users

class TestBanHandler(unittest.TestCase):
    def test_does_not_fetch_ban_list(self):
        """Verify that ban list is NOT fetched."""
        subreddit = MagicMock()

        # Mock mod log to return 3 users to ban
        log_entries = []
        users_to_ban_names = ["user1", "user2", "user3"]

        for name in users_to_ban_names:
            log = Mock()
            log.action = "removelink"
            log.details = "User has negative local reputation"
            log.target_author = name
            log.created_utc = datetime.now(timezone.utc).timestamp()
            log_entries.append(log)

        subreddit.mod.log.return_value = log_entries

        banned_mock = MagicMock()
        subreddit.banned = banned_mock

        state = {"banned_users": [], "stats": {}}

        # Capture stdout using mock.patch for proper test isolation
        with patch('sys.stdout', new_callable=StringIO):
            check_and_ban_negative_karma_users(subreddit, state, dry_run=False)

        # Assert that subreddit.banned is called 0 times (optimized)
        self.assertEqual(banned_mock.call_count, 0)
        # Verify add was called 3 times
        self.assertEqual(banned_mock.add.call_count, 3)

    def test_attempts_ban_for_all_users(self):
        """Verify that we attempt to ban all users, relying on PRAW/API to handle existing bans."""
        subreddit = MagicMock()

        # Log has 2 users: user1 (already banned), user2 (new)
        log1 = Mock()
        log1.action = "removelink"
        log1.details = "User has negative local reputation"
        log1.target_author = "user1"
        log1.created_utc = datetime.now(timezone.utc).timestamp()

        log2 = Mock()
        log2.action = "removelink"
        log2.details = "User has negative local reputation"
        log2.target_author = "user2"
        log2.created_utc = datetime.now(timezone.utc).timestamp()

        subreddit.mod.log.return_value = [log1, log2]

        banned_mock = MagicMock()
        subreddit.banned = banned_mock

        state = {"banned_users": [], "stats": {}}

        # Run with captured output
        with patch('sys.stdout', new_callable=StringIO) as captured_output:
            users_banned, new_state = check_and_ban_negative_karma_users(subreddit, state, dry_run=False)

        # Verify call to subreddit.banned.add
        banned_calls = banned_mock.add.call_args_list
        # We expect 2 calls (for user1 and user2)
        self.assertEqual(len(banned_calls), 2)

        # Verify arguments
        call_args = [c[0][0] for c in banned_calls]
        self.assertIn("user1", call_args)
        self.assertIn("user2", call_args)

        # Output should confirm attempts to ban both
        output = captured_output.getvalue()
        self.assertIn("Banned u/user1", output)
        self.assertIn("Banned u/user2", output)

if __name__ == "__main__":
    unittest.main()
