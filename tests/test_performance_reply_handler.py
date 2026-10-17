import unittest

from unittest.mock import MagicMock, patch

from datetime import datetime, timezone



from reply_handler import check_inbox_replies





class TestReplyHandlerPerformance(unittest.TestCase):

    def test_no_parent_calls(self):

        reddit = MagicMock()

        gemini_model = MagicMock()

        state = {}

        bot_username = "OptimistPrimeBot"



        num_items = 5

        items = []

        for i in range(num_items):

            item = MagicMock()

            item.id = f"item_{i}"

            item.created_utc = datetime.now(timezone.utc).timestamp()

            item.subreddit.display_name = "example_community"

            item.body = "Hello bot"

            item.author.name = f"user_{i}"



            parent = MagicMock()

            parent.author.name = bot_username

            item.parent.return_value = parent

            item.parent_id = f"t1_parent_{i}"



            items.append(item)



        reddit.inbox.comment_replies.return_value = items



        replies_sent, _, _, _ = check_inbox_replies(

            reddit, gemini_model, state, bot_username, dry_run=True

        )



        total_parent_calls = sum(item.parent.call_count for item in items)

        self.assertEqual(replies_sent, 0)

        self.assertEqual(total_parent_calls, 0)





if __name__ == "__main__":

    unittest.main()

