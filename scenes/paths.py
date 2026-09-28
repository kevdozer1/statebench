"""Paths for the scene servers. They run in the playground's own environment, so they cannot import statebench.config.

``STATEBENCH_DATA`` (default ``./data``) and ``STATEBENCH_PLAYGROUND`` (default ``./data/llm-robotics-playground``) as
in ``statebench.config``; ``.env`` at the repository root is read the same way.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    f = ROOT / ".env"
    if f.is_file():
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() == name and v.strip():
                return v.strip().strip('"').strip("'")
    return None


DATA = Path(_env("STATEBENCH_DATA") or ROOT / "data")
PLAYGROUND = Path(_env("STATEBENCH_PLAYGROUND") or DATA / "llm-robotics-playground")
ASSETS = ROOT / "assets"
