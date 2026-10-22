"""The API answers the dashboard only: no other site's page (a form POST needs no CORS preflight)
and no DNS-rebound name may switch rules off.  Plain ASGI, so no web framework is needed here."""
from __future__ import annotations

import asyncio

import pytest

from sayguard.netguard import LocalGuard, host_allowed, write_allowed


@pytest.mark.parametrize("host, ok", [
    ("127.0.0.1:8000", True), ("localhost:8000", True), ("[::1]:8000", True), ("192.168.1.20:8000", True),
    ("evil.example:8000", False), ("", False), ("127.0.0.1.nip.io:8000", False),
])
def test_host_check(host, ok):
    assert host_allowed(host) is ok


def test_configured_names_are_allowed():
    assert host_allowed("sayguard.local:8000", {"sayguard.local"})


@pytest.mark.parametrize("method, origin, site, ok", [
    ("GET", "https://evil.example", "cross-site", True),           # reads are not writes
    ("POST", "http://127.0.0.1:8000", "same-origin", True),        # the dashboard itself
    ("POST", None, None, True),                                    # curl / scripts on this machine
    ("POST", "https://evil.example", "cross-site", False),
    ("DELETE", "http://localhost:8000", None, False),              # another origin, even local
    ("PATCH", "null", None, False),                                # sandboxed frame
    ("POST", None, "cross-site", False),
])
def test_write_check(method, origin, site, ok):
    assert write_allowed(method, "127.0.0.1:8000", origin, site) is ok


def call(method, headers):
    sent, called = [], []

    async def app(scope, receive, send):
        called.append(True)

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "method": method, "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}
    asyncio.run(LocalGuard(app)(scope, None, send))
    return (sent[0]["status"] if sent else 200), bool(called)


def test_middleware_refuses_before_the_app_runs():
    assert call("DELETE", {"host": "127.0.0.1:8000", "origin": "https://evil.example"}) == (403, False)
    assert call("GET", {"host": "rebound.example:8000"}) == (421, False)
    assert call("POST", {"host": "127.0.0.1:8000", "origin": "http://127.0.0.1:8000"}) == (200, True)
