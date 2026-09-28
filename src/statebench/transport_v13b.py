"""Labelled deviation (Turn 13 confirmation): transport retries and rate spacing, requests unchanged.

During the first minutes of the confirmation two transport failures appeared that did not occur on dev:
* OpenRouter answered Sonnet calls with HTTP 429 ("new accounts are limited to 20 requests per minute for this
  model"); after the pinned 3 attempts the step became an ``inspect`` with an error. All 7 Sonnet episodes written
  so far had such steps.
* TypeSafe answered 11 of 651 Jev calls with HTTP 529 ("system_overloaded"), a status the pinned retry policy
  (``backends/http.py``, Turn 2) does not retry, so each became an ``inspect`` with an error (9 of 31 episodes).

Importing this module replaces ``backends.http.post_json`` at runtime with a wrapper that:
* spaces OpenRouter calls to Sonnet at least SONNET_MIN_INTERVAL_S apart across all threads (18.75 per minute);
* retries HTTP 429, 529 and 5xx and transport timeouts up to MAX_ATTEMPTS times with backoff
  min(BACKOFF_CAP_S, 5 s x 2^k).
The URL, key, body (prompt, schema, model, temperature) and timeout are unchanged. No pinned file's bytes change;
the request path changes at runtime, which is why it is a deviation.

Episodes containing any transport error step are not scored: they are moved to
``confirm_llm_<reader>_transport_errors.jsonl`` and their (condition, seed) pairs are re-run under this transport.
"""

from __future__ import annotations

import threading
import time

from .backends import http as _http

SONNET_MIN_INTERVAL_S = 3.2
MAX_ATTEMPTS = 8
BACKOFF_BASE_S, BACKOFF_CAP_S = 5.0, 60.0
RETRY = (429, 500, 502, 503, 504, 520, 522, 524, 529)

_orig = _http.post_json
_lock = threading.Lock()
_last = {"t": 0.0}
STATS = {"calls": 0, "retries": 0, "by_status": {}}


def _post_json_v13b(url, body, api_key, **kwargs):
    kwargs.pop("max_attempts", None)
    spaced = "openrouter.ai" in url and "claude-sonnet" in str(body.get("model", ""))
    res = None
    for k in range(MAX_ATTEMPTS):
        if spaced:
            with _lock:
                wait = _last["t"] + SONNET_MIN_INTERVAL_S - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                _last["t"] = time.monotonic()
        res = _orig(url, body, api_key, max_attempts=1, **kwargs)
        STATS["calls"] += 1
        if res.ok or (res.status is not None and res.status not in RETRY):
            return res
        STATS["retries"] += 1
        STATS["by_status"][str(res.status)] = STATS["by_status"].get(str(res.status), 0) + 1
        time.sleep(min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** k)))
    return res


_http.post_json = _post_json_v13b
INSTALLED = True
