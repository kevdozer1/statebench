"""Labelled deviation (Turn 6): a declared User-Agent on Jev calls only.

Partway through the Turn 6 grid, TypeSafe's Cloudflare front began refusing every
Jev request with HTTP 403, error 1010 (access denied on the client's signature;
the client sent Python's default ``Python-urllib`` User-Agent). The key's owner
authorised sending an honest, declared client identifier instead. Importing this
module makes every request from ``backends.jev`` (decisions and the smoke call)
carry ``User-Agent: statebench/0.2 (kevdozer1)``. Nothing else about the request
changes: same URL, key, body, timeout and retry policy. No pinned file's bytes
change; the request path changes at runtime, which is why it is a deviation.
OpenRouter calls are untouched.
"""

from __future__ import annotations

from .backends import http as _http
from .backends import jev as _jev

USER_AGENT = "statebench/0.2 (kevdozer1)"


def _post_json_declared(url, body, api_key, **kwargs):
    headers = dict(kwargs.pop("extra_headers", None) or {})
    headers["User-Agent"] = USER_AGENT
    return _http.post_json(url, body, api_key, extra_headers=headers, **kwargs)


_jev.post_json = _post_json_declared
INSTALLED = True
