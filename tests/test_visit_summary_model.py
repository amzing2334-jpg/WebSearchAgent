"""Offline tests for the configurable Visit summary model and Qwen3.5 mode."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tool_visit


class VisitSummaryModelTests(unittest.TestCase):
    def request_for(self, model):
        client = Mock()
        client.chat.completions.create.return_value = Mock(choices=[
            Mock(message=Mock(content='{"evidence": "text", "summary": "summary"}'))
        ])
        env = {'DASHSCOPE_API_KEY': 'unit-test-key', 'DASHSCOPE_SUMMARY_MODEL': model,
               'DASHSCOPE_MAIN_MODEL': 'qwen3-max', 'CHECKER_MODEL': 'qwen3.5-flash'}
        messages = [{'role': 'user', 'content': 'Extract evidence and summary as JSON.'}]
        with patch.dict(os.environ, env), patch.object(tool_visit, 'OpenAI', return_value=client) as factory:
            before = dict(os.environ)
            visit = object.__new__(tool_visit.Visit)
            result = visit.call_server(messages, max_tries=1)
            self.assertEqual(json.loads(result), {'evidence': 'text', 'summary': 'summary'})
            self.assertEqual(dict(os.environ), before)
            factory.assert_called_once_with(api_key='unit-test-key',
                                           base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
            client.chat.completions.create.assert_called_once()
            kwargs = client.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs['model'], model)
            self.assertEqual(kwargs['messages'], messages)
            return kwargs

    def test_flash_disables_thinking(self):
        self.assertEqual(self.request_for('qwen3.5-flash')['extra_body'], {'enable_thinking': False})

    def test_qwen35_snapshot_disables_thinking(self):
        self.assertEqual(self.request_for('qwen3.5-flash-2026-02-23')['extra_body'], {'enable_thinking': False})

    def test_other_models_keep_existing_thinking_behavior(self):
        for model in ('qwen-plus', 'qwen3-max'):
            with self.subTest(model=model):
                self.assertEqual(self.request_for(model)['extra_body'], {})


if __name__ == '__main__':
    unittest.main()
