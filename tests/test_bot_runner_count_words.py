import unittest
import sys
import os

# Add parent directory to path to allow importing bot_runner
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot_runner import count_words

class TestCountWords(unittest.TestCase):
    def test_count_words_basic(self):
        self.assertEqual(count_words("hello world"), 2)

    def test_count_words_markdown(self):
        text = "**bold** *italic* `code` [link](url)"
        # "bold italic code link" -> 4 words
        self.assertEqual(count_words(text), 4)

    def test_count_words_empty(self):
        self.assertEqual(count_words(""), 0)
        self.assertEqual(count_words(None), 0)

    def test_count_words_complex(self):
        text = "This is a **bold statement** with [a link](http://example.com) inside."
        # "This is a bold statement with a link inside." -> 9 words
        self.assertEqual(count_words(text), 9)

if __name__ == '__main__':
    unittest.main()
