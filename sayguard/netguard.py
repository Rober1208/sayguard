"""Keep the dashboard's API to the dashboard.  The server binds to 127.0.0.1, but a web page the
officer happens to open can still reach it: a cross-site form POST needs no CORS preflight, and DNS
rebinding makes an attacker's domain resolve to 127.0.0.1 so the page counts as same-origin.  Either
could switch every safety rule off.  Two checks, as a plain ASGI middleware (no streaming caveats for
the MJPEG tiles):

  Host     must be localhost / a loopback or LAN IP literal / a name listed in server.allowed_hosts;
           a rebound attacker domain arrives with its own name in Host and is refused (421)
  writes   POST/PUT/PATCH/DELETE from another site are refused (403): Sec-Fetch-Site cross-site, or
           an Origin whose host:port is not the Host the request was sent to
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

LOCAL_NAMES = {"localhost", "127.0.0.1", "[::1]", "::1"}
WRITES = {"POST", "PUT", "PATCH", "DELETE"}


def host_name(host: str) -> str:
    """'127.0.0.1:8000' -> '127.0.0.1', '[::1]:8000' -> '::1', 'Example.com' -> 'example.com'."""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def host_allowed(host: str, extra: set[str] = frozenset()) -> bool:
    name = host_name(host)
    if not name:
        return False
    if name in LOCAL_NAMES or name in extra:
        return True
    try:
        ipaddress.ip_address(name)        # an IP literal (LAN access by address) cannot be rebound
        return True
    except ValueError:
        return False


def write_allowed(method: str, host: str, origin: str | None, fetch_site: str | None) -> bool:
    if method.upper() not in WRITES:
        return True
    if fetch_site == "cross-site":
        return False
    if origin and origin != "null":
        return urlsplit(origin).netloc.lower() == host.strip().lower()
    return origin != "null"


class LocalGuard:
    def __init__(self, app, allowed_hosts: list[str] | tuple[str, ...] = ()):
        self.app = app
        self.extra = {h.strip().lower() for h in allowed_hosts if h}

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            h = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
            host = h.get("host", "")
            if not host_allowed(host, self.extra):
                return await _refuse(send, 421, b"host not allowed")
            if not write_allowed(scope.get("method", "GET"), host, h.get("origin"), h.get("sec-fetch-site")):
                return await _refuse(send, 403, b"cross-site request refused")
        await self.app(scope, receive, send)


async def _refuse(send, status: int, body: bytes) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"text/plain; charset=utf-8"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})
