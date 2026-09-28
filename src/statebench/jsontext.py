"""JSON from a model's answer, tolerating code fences and prose around it (from robolabel's provider base)."""
from __future__ import annotations

import json
from typing import Any


def extract_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [i for i in (cleaned.find("["), cleaned.find("{")) if i >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(cleaned.rfind("]"), cleaned.rfind("}"))
        if end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def try_extract_json(text: str) -> Any:
    try:
        return extract_json(text)
    except (ValueError, json.JSONDecodeError):
        return None
