
import unittest
import unittest.mock
from acceleration_handler import calculate_pro_ai_karma
from config import ACCELERATION_PRO_AI_SUBS

class TestAccelerationHandler(unittest.TestCase):
    def test_calculate_pro_ai_karma(self):
        mock_reddit = unittest.mock.Mock()
        mock_redditor = unittest.mock.Mock()

        # Create some mock comments
        mock_comment_pro = unittest.mock.Mock()
        mock_comment_pro.subreddit.display_name = ACCELERATION_PRO_AI_SUBS[0] # Should match
        mock_comment_pro.score = 10
        mock_comment_pro.created_utc = 2000000000 # Future date, so not filtered by age

        mock_comment_anti = unittest.mock.Mock()
        mock_comment_anti.subreddit.display_name = "SomeOtherSub"
        mock_comment_anti.score = 5
        mock_comment_anti.created_utc = 2000000000

        mock_redditor.comments.new.return_value = [mock_comment_pro, mock_comment_anti]
        mock_redditor.submissions.new.return_value = []

        # Patch datetime to ensure "one year ago" check passes
        with unittest.mock.patch('acceleration_handler.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value.timestamp.return_value = 2000000000

            pro_ai_karma, total_karma = calculate_pro_ai_karma(mock_redditor, mock_reddit)

            self.assertEqual(pro_ai_karma, 10)
            self.assertEqual(total_karma, 15)

if __name__ == '__main__':
    unittest.main()
