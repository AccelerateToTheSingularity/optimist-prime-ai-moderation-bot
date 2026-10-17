"""Synthetic operator-authored history criteria shared by isolated tests."""

from copy import deepcopy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config

SYNTHETIC_HISTORY_CRITERIA = {
    "sample_repeated_link_posting": {
        "title": "Repeated link posting",
        "description": "The author repeatedly shares the same link across unrelated posts.",
        "high_confidence_yes": "Several separate posts repeat the same link without relevant discussion.",
        "not_sufficient_by_itself": "One link, a relevant discussion, or a disclosed announcement does not meet this example criterion.",
        "closest_boundaries": "Distinguish repeated promotion from ordinary discussion and quoted links.",
        "active": True,
        "revision": 2,
    },
    "sample_direct_personal_attack": {
        "title": "Repeated direct personal attacks",
        "description": "The author repeatedly directs personal insults at another participant.",
        "high_confidence_yes": "Multiple direct statements target the same fictional participant with insults.",
        "not_sufficient_by_itself": "A quotation, hypothetical, or single ambiguous phrase is not enough.",
        "closest_boundaries": "Separate direct attacks from criticism of an idea or a quoted example.",
        "active": True,
        "revision": 2,
    },
    "sample_contact_information": {
        "title": "Repeated sharing of contact details",
        "description": "The author repeatedly shares another person's contact details without permission.",
        "high_confidence_yes": "Multiple posts disclose clearly identifying contact details about a fictional person.",
        "not_sufficient_by_itself": "Public business contact information or fictional placeholder data is not enough.",
        "closest_boundaries": "Distinguish a real disclosure from a harmless fictional example.",
        "active": True,
        "revision": 2,
    },
}

SYNTHETIC_HISTORY_CRITERIA_IDS = frozenset(SYNTHETIC_HISTORY_CRITERIA)


class ConfiguredHistoryCriteriaTestCase(unittest.TestCase):
    """Give each history-review test an isolated operator-authored catalog."""

    def setUp(self):
        super().setUp()
        self._criteria_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._criteria_tmp.cleanup)
        criteria_path = Path(self._criteria_tmp.name) / "history_criteria.json"
        criteria_path.write_text(
            json.dumps({"schema_version": 1, "guides": synthetic_history_criteria()}),
            encoding="utf-8",
        )
        env_patch = patch.dict(
            os.environ,
            {"BOT_HISTORY_CRITERIA_FILE": str(criteria_path)},
            clear=False,
        )
        env_patch.start()
        self.addCleanup(env_patch.stop)
        lookback_patch = patch.object(config, "HISTORY_REVIEW_LOOKBACK_DAYS", 90)
        lookback_patch.start()
        self.addCleanup(lookback_patch.stop)
        sample_limit_patch = patch.object(config, "HISTORY_REVIEW_MAX_ITEMS", 100)
        sample_limit_patch.start()
        self.addCleanup(sample_limit_patch.stop)
        subreddit_patch = patch.object(config, "SUBREDDIT", "example_community")
        subreddit_patch.start()
        self.addCleanup(subreddit_patch.stop)


def synthetic_history_criteria():
    """Return a copy so tests cannot mutate the shared synthetic fixture."""
    return deepcopy(SYNTHETIC_HISTORY_CRITERIA)


def synthetic_history_criteria_rules():
    """Return active criteria in the input form used by history evaluation."""
    return [
        {"name": name, **guide, "order": index}
        for index, (name, guide) in enumerate(
            synthetic_history_criteria().items(), start=1
        )
    ]


def current_history_criteria_revisions():
    return {
        name: guide["revision"]
        for name, guide in SYNTHETIC_HISTORY_CRITERIA.items()
    }


def scores_for_questions(questions, *, positives=(), positive_score=0.93, negative_score=0.03):
    positives = set(positives)
    return {
        name: positive_score if name in positives else negative_score
        for name in questions
    }
