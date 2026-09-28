"""Shared HTTP transport and the pre-registered retry policy.

Retry policy (Turn 2, pre-registered in ``rules_turn2.py``): up to 3 attempts,
exponential backoff with jitter, retrying only on HTTP 429 and 5xx and on
transport-level timeouts. A 4xx other than 429 is a request defect and is not
retried. After the last attempt the caller records a ``BACKEND_ERROR`` step.

Keys never appear in a log line, an exception message or a receipt: the only
place the credential is used is the ``Authorization`` header built here.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 1.5
BACKOFF_CAP_S = 20.0
RETRY_STATUSES = (429, 500, 502, 503, 504, 520, 522, 524)


@dataclass
class HttpResult:
    ok: bool
    status: int | None
    payload: Any = None
    latency_seconds: float = 0.0
    attempts: int = 1
    retries: int = 0
    error: str | None = None
    attempt_log: list[dict[str, Any]] = field(default_factory=list)


def _sleep_for(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(BACKOFF_CAP_S, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** (attempt - 1))) * (0.7 + 0.6 * random.random())


def post_json(
    url: str,
    body: dict[str, Any],
    api_key: str,
    *,
    timeout: float = 120.0,
    max_attempts: int = MAX_ATTEMPTS,
    extra_headers: dict[str, str] | None = None,
) -> HttpResult:
    """POST JSON with the pre-registered retry policy. Never logs the key."""
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    encoded = json.dumps(body).encode("utf-8")

    total_started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        request = urllib.request.Request(url, encoded, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
                status = int(response.status)
            latency = time.perf_counter() - started
            attempts.append({"attempt": attempt, "status": status,
                             "latency_seconds": round(latency, 4)})
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                return HttpResult(False, status, None, time.perf_counter() - total_started,
                                  attempt, attempt - 1,
                                  f"JSONDecodeError: {exc}", attempts)
            return HttpResult(True, status, payload, time.perf_counter() - total_started,
                              attempt, attempt - 1, None, attempts)
        except urllib.error.HTTPError as exc:
            latency = time.perf_counter() - started
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001
                pass
            status = int(exc.code)
            attempts.append({"attempt": attempt, "status": status,
                             "latency_seconds": round(latency, 4), "detail": detail})
            if status in RETRY_STATUSES and attempt < max_attempts:
                time.sleep(_sleep_for(attempt, exc.headers.get("Retry-After")))
                continue
            return HttpResult(False, status, None, time.perf_counter() - total_started,
                              attempt, attempt - 1, f"HTTP {status}: {detail}", attempts)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            latency = time.perf_counter() - started
            attempts.append({"attempt": attempt, "status": None,
                             "latency_seconds": round(latency, 4),
                             "detail": f"{type(exc).__name__}"})
            if attempt < max_attempts:
                time.sleep(_sleep_for(attempt, None))
                continue
            return HttpResult(False, None, None, time.perf_counter() - total_started,
                              attempt, attempt - 1, f"{type(exc).__name__}: {exc}", attempts)
    return HttpResult(False, None, None, time.perf_counter() - total_started,
                      max_attempts, max_attempts - 1, "exhausted attempts", attempts)
