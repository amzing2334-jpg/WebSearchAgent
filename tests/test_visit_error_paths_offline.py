"""Offline regression tests for Visit's summary failure and recovery paths."""

import io
import json
import os
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from offline_test_support import blocked_network


OFFLINE_ENV = {
    "DASHSCOPE_API_KEY": "offline-test-key",
    "DASHSCOPE_SUMMARY_MODEL": "offline-test-model",
    "JINA_API_KEY": "offline-test-key",
    "WEBCONTENT_MAXLENGTH": "150000",
}

# Guard imports as well as test execution. No .env file is loaded.
with blocked_network(), patch.dict(os.environ, OFFLINE_ENV):
    import tool_visit


class VisitErrorPathsOfflineTests(unittest.TestCase):
    URL = "https://example.invalid/offline-visit-test"
    GOAL = "Extract the fact from the fixed offline page."
    PAGE = "The offline reference page says that the sample has seven parts."
    EVIDENCE = "OFFLINE_EVIDENCE: the sample has seven parts."
    SUMMARY = "OFFLINE_SUMMARY: seven parts."
    INVALID_JSON = "This reply is not valid JSON."

    def setUp(self):
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(blocked_network())
        contexts.enter_context(patch.dict(os.environ, OFFLINE_ENV))
        contexts.enter_context(redirect_stdout(io.StringIO()))

        self.create_completion = Mock(name="offline_completions_create")
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )
        )
        contexts.enter_context(
            patch.object(tool_visit, "OpenAI", return_value=fake_client)
        )
        self.visit = tool_visit.Visit()
        contexts.enter_context(
            patch.object(self.visit, "jina_readpage", return_value=self.PAGE)
        )
        self.valid_json = json.dumps(
            {"evidence": self.EVIDENCE, "summary": self.SUMMARY}
        )

    @staticmethod
    def completion(content):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def assert_summary_is_used(self, result):
        self.assertIn(self.EVIDENCE, result)
        self.assertIn(self.SUMMARY, result)

    def test_continuous_empty_replies_return_failure_text_without_crashing(self):
        self.create_completion.return_value = self.completion("")

        # Exercise the real call_server -> readpage path, including exhaustion.
        result = self.visit.readpage(self.URL, self.GOAL)

        self.assertIsInstance(result, str)
        self.assertTrue(result.strip(), "An exhausted request needs a failure message.")

    def test_valid_reply_from_last_json_retry_is_processed(self):
        self.create_completion.side_effect = [
            self.completion(self.INVALID_JSON),
            self.completion(self.INVALID_JSON),
            self.completion(self.INVALID_JSON),
            self.completion(self.valid_json),
        ]

        result = self.visit.readpage(self.URL, self.GOAL)

        self.assertEqual(self.create_completion.call_count, 4)
        self.assert_summary_is_used(result)

    def test_first_valid_reply_produces_a_summary(self):
        self.create_completion.return_value = self.completion(self.valid_json)

        result = self.visit.readpage(self.URL, self.GOAL)

        self.assertEqual(self.create_completion.call_count, 1)
        self.assert_summary_is_used(result)

    def test_json_retry_recovers_before_the_last_attempt(self):
        self.create_completion.side_effect = [
            self.completion(self.INVALID_JSON),
            self.completion(self.INVALID_JSON),
            self.completion(self.valid_json),
        ]

        result = self.visit.readpage(self.URL, self.GOAL)

        self.assertEqual(self.create_completion.call_count, 3)
        self.assert_summary_is_used(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
