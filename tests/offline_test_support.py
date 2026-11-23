"""Network guard shared by the deterministic, offline regression tests.

This is test infrastructure, not a production request/proxy configuration.
No .env is loaded and no request URL, headers, or credentials are recorded.
"""

from contextlib import ExitStack, contextmanager
import socket
from unittest.mock import patch

import httpx
import requests


class OfflineNetworkError(BaseException):
    """Escape ordinary SDK retry handlers if an offline test reaches the network."""


@contextmanager
def blocked_network():
    """Block common HTTP clients and socket connections, even if code swallows errors.

The attempt ledger is checked on exit because the application contains bare
``except:`` handlers. A swallowed network error must never become a passing test.
Only entry-point names are recorded, never arguments that might contain secrets.
"""
    attempts = []

    def reject(label):
        def denied(*args, **kwargs):
            attempts.append(label)
            raise OfflineNetworkError(f"Offline test blocked network entry: {label}")

        return denied

    async def reject_async(*args, **kwargs):
        attempts.append("httpx.AsyncClient.send")
        raise OfflineNetworkError("Offline test blocked network entry: httpx.AsyncClient.send")

    with ExitStack() as stack:
        for target, attribute, label in (
            (requests.sessions.Session, "request", "requests.Session.request"),
            (requests.sessions.Session, "send", "requests.Session.send"),
            (httpx.Client, "send", "httpx.Client.send"),
            (socket, "getaddrinfo", "socket.getaddrinfo"),
            (socket, "create_connection", "socket.create_connection"),
            (socket.socket, "connect", "socket.connect"),
            (socket.socket, "connect_ex", "socket.connect_ex"),
        ):
            stack.enter_context(patch.object(target, attribute, new=reject(label)))
        stack.enter_context(patch.object(httpx.AsyncClient, "send", new=reject_async))
        try:
            yield attempts
        finally:
            if attempts:
                raise OfflineNetworkError(
                    "Offline test attempted network access: " + ", ".join(attempts)
                )
