"""Test the real Agent loop with scripted external responses, without network.

Real code: _run, decompose_question, update_scratchpad, JSON parsing,
Qwen-Agent tool registration/dispatch, message writes, and history trimming.
Replaced boundaries: main/auxiliary model responses, Search.call, Visit.call,
and token counting (no tokenizer download and no token-limit branch in this suite).

The fixtures below are synthetic test data, not a real research run or evidence.
"""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
from offline_test_support import OfflineNetworkError, blocked_network

# Import the library, NOT run_local.py: that entry point loads .env and prompts.
with blocked_network():
    import tool_search
    import tool_visit
    import react_agent


QUESTION = "离线测试问题：检查程序的数据流，不验证真实知识。"
SYSTEM = "OFFLINE_SYSTEM_RULES: keep the original rules."
USER_PREFIX = "OFFLINE_USER_PROMPT\n"
MODEL = "offline-main-model"
FINAL = "<verify>Checked synthetic fixtures.</verify><answer>测试答案</answer>"
PLAN_STEPS = [
    {"step": 1, "task": "Search the synthetic fixture."},
    {"step": 2, "task": "Visit the synthetic fixture."},
]
PLAN_JSON = json.dumps(PLAN_STEPS, ensure_ascii=False)
SUMMARY_1 = "OFFLINE_SUMMARY_1: facts from calls 1, 2, 3."
SUMMARY_2 = "OFFLINE_SUMMARY_2: preserved facts from calls 1 through 6."
REMINDER = (
    "\n\n[Reminder] Key confirmed facts are in the system Scratchpad above. "
    "Do NOT re-search what is already confirmed there."
)
ENV = {
    "DECOMPOSER_MODEL": "offline-decomposer",
    "CHECKER_MODEL": "offline-checker",
    "SCRATCHPAD_MODEL": "offline-scratchpad",
    "DASHSCOPE_API_KEY": "offline-not-a-real-key",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def tool_output(name, arguments):
    """Produce a model-output fixture in the project's real XML/JSON format."""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>{payload}</tool_call>"


def six_tool_calls():
    calls = []
    for number in range(1, 7):
        if number % 2:
            calls.append(("search", {"query": [f"OFFLINE_QUERY_{number}"]}))
        else:
            calls.append(("visit", {
                "url": f"https://example.invalid/OFFLINE_PAGE_{number}",
                "goal": f"OFFLINE_GOAL_{number}",
            }))
    return calls


def wrapped_result(number, name):
    return f"<tool_response>\nOFFLINE_TOOL_RESULT_{number}_{name}\n</tool_response>"


class ScriptExhausted(BaseException):
    """Fail immediately if the program makes an unplanned model/tool request."""


class ScriptedRun:
    """Recording harness. It does not reimplement the production Agent loop."""

    def __init__(self, outputs, *, summaries=(SUMMARY_1, SUMMARY_2), plan=PLAN_JSON,
                 checked_plan=None):
        self.outputs = list(outputs)
        self.summary_outputs = list(summaries)
        self.plan = plan
        self.checked_plan = plan if checked_plan is None else checked_plan
        self.model_inputs = []
        self.aux_inputs = []
        self.tool_events = []
        self.token_inputs = []
        self.events = []
        self.violations = []
        self.log = ""

        # Avoid initializing API-backed models. Initialize only the state used by
        # _run and the actual framework dispatch; this is not an __init__ test.
        self.agent = object.__new__(react_agent.MultiTurnReactAgent)
        self.agent.system_message = SYSTEM
        self.agent.function_map = {}
        # Real registration lookup and real BaseTool constructors, no SDK calls.
        self.agent._init_tool("search")
        self.agent._init_tool("visit")

    def unexpected(self, message):
        self.violations.append(message)
        raise ScriptExhausted(message)

    def main_response(self, messages, model, **kwargs):
        # Deepcopy is essential: the live list AND its dictionaries are mutated
        # later. Mock.call_args_list alone would not preserve historical inputs.
        snapshot = deepcopy(messages)
        self.model_inputs.append(snapshot)
        self.events.append({"kind": "model", "message_count": len(snapshot)})
        if model != MODEL:
            self.unexpected("Unexpected main-model name")
        index = len(self.model_inputs) - 1
        if index >= len(self.outputs):
            self.unexpected("The Agent requested more main-model outputs than scripted")
        return self.outputs[index]

    def auxiliary_response(self, messages, model, **kwargs):
        snapshot = {
            "messages": deepcopy(messages), "model": model,
            "kwargs": deepcopy(kwargs), "tool_count": len(self.tool_events),
        }
        self.aux_inputs.append(snapshot)
        self.events.append({"kind": "auxiliary", "model": model,
                            "tool_count": len(self.tool_events)})
        if model == ENV["DECOMPOSER_MODEL"]:
            return self.plan
        if model == ENV["CHECKER_MODEL"]:
            return self.checked_plan
        if model == ENV["SCRATCHPAD_MODEL"]:
            index = sum(x["model"] == model for x in self.aux_inputs) - 1
            if index >= len(self.summary_outputs):
                self.unexpected("The Agent requested more summaries than scripted")
            return self.summary_outputs[index]
        self.unexpected("Unexpected auxiliary-model name")

    def tool_response(self, name, params, **kwargs):
        number = len(self.tool_events) + 1
        event = {"name": name, "arguments": deepcopy(params), "number": number}
        self.tool_events.append(event)
        self.events.append({"kind": "tool", **event})
        return f"OFFLINE_TOOL_RESULT_{number}_{name}"

    def count_tokens(self, messages, **kwargs):
        self.token_inputs.append(deepcopy(messages))
        return 1

    def run(self):
        output = io.StringIO()
        with blocked_network(), ExitStack() as stack, redirect_stdout(output):
            stack.enter_context(patch.dict(os.environ, ENV))
            stack.enter_context(patch.object(react_agent, "MAX_LLM_CALL_PER_RUN", 20))
            stack.enter_context(patch.object(react_agent, "MAX_TOKEN_LENGTH", 10000))
            # If a new code path accidentally bypasses the scripts, never create
            # a real client; the network guard also protects HTTP/socket layers.
            for module in (react_agent, tool_visit):
                stack.enter_context(patch.object(
                    module, "OpenAI",
                    side_effect=lambda *a, **k: self.unexpected("Unexpected API client creation"),
                ))
            self.model_mock = stack.enter_context(patch.object(
                self.agent, "call_server", side_effect=self.main_response))
            stack.enter_context(patch.object(
                self.agent, "_decomposer_call", side_effect=self.auxiliary_response))
            stack.enter_context(patch.object(
                self.agent, "count_tokens", side_effect=self.count_tokens))
            self.search_mock = stack.enter_context(patch.object(
                self.agent.function_map["search"], "call",
                side_effect=lambda params, **kw: self.tool_response("search", params, **kw),
            ))
            self.visit_mock = stack.enter_context(patch.object(
                self.agent.function_map["visit"], "call",
                side_effect=lambda params, **kw: self.tool_response("visit", params, **kw),
            ))
            result = self.agent._run(
                {"item": {"question": QUESTION, "answer": ""}, "rollout_id": 1},
                MODEL, USER_PREFIX,
            )
        self.log = output.getvalue()
        # Bare except handlers must not hide exhausted fixtures or stray SDK calls.
        if self.violations:
            raise AssertionError("; ".join(self.violations))
        if len(self.model_inputs) != len(self.outputs):
            raise AssertionError("The Agent terminated without consuming all scripted outputs")
        self.result = result
        return result

    @property
    def summary_calls(self):
        return [x for x in self.aux_inputs if x["model"] == ENV["SCRATCHPAD_MODEL"]]


class AgentFlowOfflineTests(unittest.TestCase):
    def full_run(self):
        calls = six_tool_calls()
        harness = ScriptedRun([tool_output(*call) for call in calls] + [FINAL])
        harness.run()
        return harness, calls

    def test_initial_messages_and_real_plan_formatting(self):
        h = ScriptedRun([FINAL])
        result = h.run()
        first = h.model_inputs[0]
        self.assertEqual([m["role"] for m in first], ["system", "user"])
        self.assertEqual(first[0]["content"], SYSTEM)
        self.assertTrue(first[1]["content"].startswith(USER_PREFIX + QUESTION))
        self.assertIn("[Research Plan]", first[1]["content"])
        self.assertIn("Step 1: Search the synthetic fixture.", first[1]["content"])
        self.assertEqual([x["model"] for x in h.aux_inputs],
                         [ENV["DECOMPOSER_MODEL"], ENV["CHECKER_MODEL"]])
        self.assertEqual(h.aux_inputs[0]["kwargs"],
                         {"enable_thinking": True, "thinking_budget": 4096})
        self.assertEqual(h.aux_inputs[1]["kwargs"], {})
        self.assertEqual(h.tool_events, [])
        self.assertEqual(result["prediction"], "测试答案")

    def test_missing_plan_keeps_question_and_skips_checker(self):
        h = ScriptedRun([FINAL], plan="")
        h.run()
        self.assertEqual(h.model_inputs[0][1]["content"], USER_PREFIX + QUESTION)
        self.assertEqual(len(h.aux_inputs), 1)

    def test_real_dispatch_selects_tools_and_preserves_arguments(self):
        h, calls = self.full_run()
        self.assertIsInstance(h.agent.function_map["search"], tool_search.Search)
        self.assertIsInstance(h.agent.function_map["visit"], tool_visit.Visit)
        self.assertEqual([(e["name"], e["arguments"]) for e in h.tool_events], calls)
        self.assertEqual(h.search_mock.call_count, 3)
        self.assertEqual(h.visit_mock.call_count, 3)

    def test_tool_results_are_written_back_once_before_next_model_call(self):
        h, calls = self.full_run()
        for number, (name, arguments) in enumerate(calls, start=1):
            with self.subTest(tool_call=number):
                next_input = h.model_inputs[number]
                self.assertEqual(next_input[-2], {
                    "role": "assistant", "content": tool_output(name, arguments),
                })
                expected = wrapped_result(number, name)
                if number % 3 == 0:
                    expected += REMINDER
                self.assertEqual(next_input[-1], {"role": "user", "content": expected})
                marker = f"OFFLINE_TOOL_RESULT_{number}_{name}"
                self.assertEqual(sum(marker in m["content"] for m in next_input), 1)

    def test_summary_batches_run_after_calls_3_and_6_and_keep_previous_facts(self):
        h, _ = self.full_run()
        self.assertEqual([x["tool_count"] for x in h.summary_calls], [3, 6])
        first, second = [x["messages"][0]["content"] for x in h.summary_calls]
        # Inspect only the new-results section: previous facts are allowed to
        # refer to older calls and are deliberately carried into the next prompt.
        first_batch = first.split("## New Tool Results (since last update)\n", 1)[1]
        second_batch = second.split("## New Tool Results (since last update)\n", 1)[1]
        for number in (1, 2, 3):
            self.assertIn(f"[Call #{number}:", first_batch)
            self.assertNotIn(f"[Call #{number}:", second_batch)
        for number in (4, 5, 6):
            self.assertNotIn(f"[Call #{number}:", first_batch)
            self.assertIn(f"[Call #{number}:", second_batch)
        self.assertIn(SUMMARY_1, second)
        self.assertIn(QUESTION, first)
        self.assertIn("[Research Plan]", first)
        for call in h.summary_calls:
            self.assertEqual(call["kwargs"],
                             {"max_tries": 2, "enable_thinking": True, "thinking_budget": 4096})
        # These are independent local inputs, not the Agent's full history.
        self.assertEqual([len(x["messages"]) for x in h.summary_calls], [1, 1])

    def test_history_is_not_trimmed_at_8_then_keeps_exact_calls_4_5_6(self):
        h, calls = self.full_run()
        self.assertEqual([len(x) for x in h.model_inputs], [2, 4, 6, 8, 10, 12, 8])
        after_first_summary = h.model_inputs[3]
        self.assertIn("OFFLINE_TOOL_RESULT_1_search", after_first_summary[3]["content"])
        self.assertIn(SUMMARY_1, after_first_summary[0]["content"])
        after_trim = h.model_inputs[6]
        self.assertEqual(after_trim[1], h.model_inputs[0][1])
        self.assertEqual(after_trim[0]["content"],
                         SYSTEM + "\n\n## 📋 Current Research Facts (Scratchpad)\n" + SUMMARY_2 + "\n")
        expected_tail = []
        for number in (4, 5, 6):
            name, arguments = calls[number - 1]
            expected_tail.extend([
                {"role": "assistant", "content": tool_output(name, arguments)},
                {"role": "user", "content": wrapped_result(number, name)
                 + (REMINDER if number == 6 else "")},
            ])
        self.assertEqual(after_trim[2:], expected_tail)
        self.assertEqual(h.log.count("History compressed:"), 1)
        self.assertEqual([len(x) for x in h.token_inputs], [4, 6, 8, 10, 12, 8])
        self.assertEqual(h.agent.system_message, SYSTEM)
        self.assertEqual(len(h.result["messages"]), 9)  # Final assistant is appended after trimming.
        self.assertEqual(h.result["prediction"], "测试答案")
        # This would fail if recordings held references to the mutated live list.
        self.assertEqual(h.model_inputs[0], [
            {"role": "system", "content": SYSTEM}, after_trim[1],
        ])

    def test_nonempty_summary_at_9_messages_also_triggers_trimming(self):
        calls = six_tool_calls()[:3]
        # A non-tool assistant reply adds one extra message before three tool calls.
        h = ScriptedRun(["Offline non-tool reply"] + [tool_output(*c) for c in calls] + [FINAL])
        h.run()
        self.assertEqual([len(x) for x in h.model_inputs], [2, 3, 5, 7, 8])
        self.assertNotIn("Offline non-tool reply", str(h.model_inputs[-1]))
        self.assertEqual(h.log.count("History compressed:"), 1)
        self.assertEqual([x["tool_count"] for x in h.summary_calls], [3])

    def test_empty_summaries_do_not_delete_unsummarized_history(self):
        calls = six_tool_calls()
        h = ScriptedRun([tool_output(*c) for c in calls] + [FINAL], summaries=("", ""))
        h.run()
        self.assertEqual([len(x) for x in h.model_inputs], [2, 4, 6, 8, 10, 12, 14])
        self.assertEqual(h.model_inputs[-1][0]["content"], SYSTEM)
        self.assertNotIn("History compressed:", h.log)

    def test_malformed_tool_json_does_not_dispatch_a_tool(self):
        h = ScriptedRun(["<tool_call>not JSON</tool_call>", FINAL])
        h.run()
        self.assertEqual(h.tool_events, [])
        self.assertEqual(h.model_inputs[1][-1]["role"], "user")
        self.assertIn("not a valid JSON", h.model_inputs[1][-1]["content"])

    def test_unknown_tool_reports_error_without_calling_search_or_visit(self):
        h = ScriptedRun([tool_output("not_registered", {}), FINAL])
        h.run()
        self.assertEqual(h.tool_events, [])
        self.assertIn("Tool not_registered does not exists.", h.model_inputs[1][-1]["content"])

    def test_fabricated_tool_response_is_removed_before_real_result_is_written(self):
        call = six_tool_calls()[0]
        fake = tool_output(*call) + "<tool_response>FABRICATED_RESULT</tool_response>"
        h = ScriptedRun([fake, FINAL])
        h.run()
        self.assertEqual(h.model_inputs[1][-2]["content"], tool_output(*call))
        self.assertEqual(h.model_inputs[1][-1]["content"], wrapped_result(1, "search"))
        self.assertNotIn("FABRICATED_RESULT", str(h.result["messages"]))

    def test_answer_without_verify_requests_one_more_model_reply(self):
        h = ScriptedRun([tool_output(*six_tool_calls()[0]), "<answer>过早答案</answer>", FINAL])
        h.run()
        self.assertEqual(len(h.model_inputs), 3)
        self.assertIn("FORMAT VIOLATION", h.model_inputs[2][-1]["content"])
        self.assertEqual(len(h.tool_events), 1)
        self.assertEqual(h.result["prediction"], "测试答案")


class OfflineGuardTests(unittest.TestCase):
    def test_network_attempt_fails_even_when_application_catches_it(self):
        import requests
        with self.assertRaises(OfflineNetworkError):
            with blocked_network():
                try:
                    requests.get("https://example.invalid/never-send")
                except BaseException:
                    pass

    def test_main_response_script_exhaustion_cannot_silently_pass(self):
        with self.assertRaises(ScriptExhausted):
            ScriptedRun([]).run()


if __name__ == "__main__":
    unittest.main()
