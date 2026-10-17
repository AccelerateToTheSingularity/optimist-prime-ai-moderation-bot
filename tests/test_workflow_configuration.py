"""Static checks that GitHub-hosted execution stays disabled."""

import unittest
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"


class TestWorkflowConfiguration(unittest.TestCase):
    def test_public_showcase_has_no_github_actions_workflows(self):
        self.assertFalse(WORKFLOW_DIR.exists())


if __name__ == "__main__":
    unittest.main()
