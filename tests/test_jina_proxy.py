"""Offline regression tests for Jina's Windows proxy fallback.

Run from the project root with:
    python -m unittest discover -s tests -p test_jina_proxy.py -v

These tests never load .env, read real Windows proxy settings, or send requests.
"""

import io
import os
import sys
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
import urllib.request

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import tool_visit  # noqa: E402 -- import the project module, not run_local.py


READER_URL = "https://r.jina.ai/https://example.invalid/article"
PAGE_URL = "https://example.invalid/article"
BYPASS = "dashscope.aliyuncs.com,google.serper.dev"
SYSTEM_PROXIES = {"http": "http://proxy.invalid:8080", "https": "http://proxy.invalid:8080"}


@contextmanager
def routing_context(env=None, platform="win32", registry=None, registry_bypass=False):
    """Provide deterministic routing on both Windows and non-Windows hosts."""
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, env or {}, clear=True))
        stack.enter_context(patch.object(tool_visit.sys, "platform", platform))
        get_registry = stack.enter_context(patch.object(
            urllib.request, "getproxies_registry", create=True,
            return_value=SYSTEM_PROXIES.copy() if registry is None else registry,
        ))
        bypass_registry = stack.enter_context(patch.object(
            urllib.request, "proxy_bypass_registry", create=True,
            return_value=registry_bypass,
        ))
        # Keep Requests' NO_PROXY parsing, including '*', but never query the
        # host's actual registry through its imported proxy_bypass function.
        stack.enter_context(patch.object(
            requests.utils, "proxy_bypass",
            side_effect=lambda host: urllib.request.proxy_bypass_environment(
                host, proxies=urllib.request.getproxies_environment()
            ),
        ))
        # Fail closed if a regression accidentally tries a real network request.
        stack.enter_context(patch.object(
            requests.sessions.Session, "send",
            side_effect=AssertionError("Network requests are forbidden in this test"),
        ))
        yield get_registry, bypass_registry


class JinaProxyRoutingTests(unittest.TestCase):
    def test_no_proxy_only_recovers_registry_for_jina(self):
        with routing_context({"NO_PROXY": BYPASS}) as (get_registry, bypass_registry):
            before = dict(os.environ)
            self.assertEqual(tool_visit._jina_request_proxies(READER_URL), SYSTEM_PROXIES)
            get_registry.assert_called_once_with()
            bypass_registry.assert_called_once_with("r.jina.ai")
            self.assertEqual(dict(os.environ), before)

    def test_no_environment_proxy_recovers_system_settings(self):
        with routing_context() as (get_registry, _):
            self.assertEqual(tool_visit._jina_request_proxies(READER_URL), SYSTEM_PROXIES)
            get_registry.assert_called_once_with()

    def test_explicit_proxy_names_and_empty_values_preserve_requests_default(self):
        names = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
                 "ALL_PROXY", "all_proxy", "HtTpS_PrOxY")
        for name in names:
            for value in ("http://explicit.invalid:9090", ""):
                with self.subTest(name=name, empty=value == ""):
                    with routing_context({"NO_PROXY": BYPASS, name: value}) as (get_registry, bypass_registry):
                        before = dict(os.environ)
                        self.assertIsNone(tool_visit._jina_request_proxies(READER_URL))
                        get_registry.assert_not_called()
                        bypass_registry.assert_not_called()
                        self.assertEqual(dict(os.environ), before)

    def test_non_windows_never_uses_registry(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                with routing_context({"NO_PROXY": BYPASS}, platform=platform) as (get_registry, bypass_registry):
                    self.assertIsNone(tool_visit._jina_request_proxies(READER_URL))
                    get_registry.assert_not_called()
                    bypass_registry.assert_not_called()

    def test_no_proxy_matching_host_domain_and_wildcard_take_precedence(self):
        for name in ("NO_PROXY", "no_proxy"):
            for value in ("r.jina.ai", ".jina.ai", "*", BYPASS + ",r.jina.ai"):
                with self.subTest(name=name, value=value):
                    with routing_context({name: value}) as (get_registry, bypass_registry):
                        before = dict(os.environ)
                        self.assertIsNone(tool_visit._jina_request_proxies(READER_URL))
                        get_registry.assert_not_called()
                        bypass_registry.assert_not_called()
                        self.assertEqual(dict(os.environ), before)

    def test_registry_bypass_still_honored_when_no_proxy_masks_it(self):
        with routing_context({"NO_PROXY": BYPASS}, registry_bypass=True) as (get_registry, bypass_registry):
            self.assertIsNone(tool_visit._jina_request_proxies(READER_URL))
            bypass_registry.assert_called_once_with("r.jina.ai")
            get_registry.assert_not_called()

    def test_disabled_or_missing_registry_proxy_keeps_requests_default(self):
        with routing_context({"NO_PROXY": BYPASS}, registry={}) as (get_registry, _):
            self.assertIsNone(tool_visit._jina_request_proxies(READER_URL))
            get_registry.assert_called_once_with()

    def test_existing_dashscope_and_serper_bypass_is_unchanged(self):
        with routing_context({"NO_PROXY": BYPASS}):
            before = dict(os.environ)
            tool_visit._jina_request_proxies(READER_URL)
            self.assertTrue(requests.utils.should_bypass_proxies(
                "https://dashscope.aliyuncs.com/compatible-mode/v1", no_proxy=None))
            self.assertTrue(requests.utils.should_bypass_proxies(
                "https://google.serper.dev/search", no_proxy=None))
            self.assertFalse(requests.utils.should_bypass_proxies(READER_URL, no_proxy=None))
            self.assertEqual(dict(os.environ), before)


class JinaReadPageTests(unittest.TestCase):
    def setUp(self):
        # jina_readpage has no BaseTool instance state or model dependency.
        self.visit = object.__new__(tool_visit.Visit)
        self.key_patch = patch.object(tool_visit, "JINA_API_KEY", "unit-test-secret")
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)

    def test_http_200_returns_body_and_supplies_proxy_only_to_reader_request(self):
        with routing_context({"NO_PROXY": BYPASS}):
            before = dict(os.environ)
            with patch.object(requests, "get", return_value=Mock(status_code=200, text="Page body")) as get:
                with patch.object(self.visit, "call_server", side_effect=AssertionError("No model call expected")):
                    result = self.visit.jina_readpage(PAGE_URL)
            self.assertEqual(result, "Page body")
            get.assert_called_once_with(
                READER_URL, headers={"Authorization": "Bearer unit-test-secret"},
                timeout=20, proxies=SYSTEM_PROXIES,
            )
            self.assertEqual(dict(os.environ), before)

    def test_explicit_environment_proxy_leaves_requests_proxy_resolution_alone(self):
        with routing_context({"NO_PROXY": BYPASS, "HTTPS_PROXY": "http://explicit.invalid:9090"}):
            with patch.object(requests, "get", return_value=Mock(status_code=200, text="Page")) as get:
                self.assertEqual(self.visit.jina_readpage(PAGE_URL), "Page")
                self.assertIsNone(get.call_args.kwargs["proxies"])

    def test_connect_timeout_retries_three_times_with_safe_logs(self):
        secret_exception = requests.exceptions.ConnectTimeout(
            "unit-test-secret http://proxy-user:proxy-password@proxy.invalid"
        )
        output = io.StringIO()
        with routing_context({"NO_PROXY": BYPASS}):
            with patch.object(requests, "get", side_effect=secret_exception) as get, redirect_stdout(output):
                result = self.visit.jina_readpage(PAGE_URL)
                self.assertEqual(get.call_count, 3)
        self.assertEqual(result, "[visit] Failed to read page.")
        log = output.getvalue()
        for attempt in (1, 2, 3):
            self.assertIn(f"attempt {attempt}/3 failed: ConnectTimeout", log)
        for secret in ("unit-test-secret", "proxy-password", "proxy.invalid"):
            self.assertNotIn(secret, log)

    def test_http_401_retries_and_logs_status_without_response_body(self):
        response = Mock(status_code=401, text="unit-test-secret proxy-password")
        output = io.StringIO()
        with routing_context({"NO_PROXY": BYPASS}):
            with patch.object(requests, "get", return_value=response) as get, redirect_stdout(output):
                result = self.visit.jina_readpage(PAGE_URL)
                self.assertEqual(get.call_count, 3)
        self.assertEqual(result, "[visit] Failed to read page.")
        self.assertEqual(output.getvalue().count("HTTP 401"), 3)
        self.assertNotIn("unit-test-secret", output.getvalue())
        self.assertNotIn("proxy-password", output.getvalue())

    def test_success_after_timeout_stops_retrying(self):
        output = io.StringIO()
        outcomes = [requests.exceptions.ConnectTimeout("hidden"), Mock(status_code=200, text="Recovered page")]
        with routing_context({"NO_PROXY": BYPASS}):
            with patch.object(requests, "get", side_effect=outcomes) as get, redirect_stdout(output):
                result = self.visit.jina_readpage(PAGE_URL)
                self.assertEqual(get.call_count, 2)
        self.assertEqual(result, "Recovered page")


if __name__ == "__main__":
    unittest.main()
