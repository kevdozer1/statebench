"""Paths and environment.

Everything the code writes goes under ``$STATEBENCH_DATA`` (default ``./data``). Keys and paths can also sit in a
``.env`` file at the repository root; values already in the environment win.

* ``STATEBENCH_DATA``: run outputs (default ``./data``).
* ``STATEBENCH_PLAYGROUND``: a checkout of llm-robotics-playground (default ``./data/llm-robotics-playground``).
* ``STATEBENCH_PLAYGROUND_PYTHON``: the Python of that checkout's environment (default its ``.venv``).
* ``STATEBENCH_XARM7``: MuJoCo Menagerie's ``ufactory_xarm7`` folder (default ``./data/mujoco_menagerie/ufactory_xarm7``).
* ``OPENROUTER_API_KEY`` (Sol) and ``TYPESAFE_JEV_API_KEY`` (Jev) for model calls.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "statebench/state/v0.1"


def _load_dotenv() -> dict[str, str]:
    """Read ``.env`` at the project root. Values already in the process win."""
    values: dict[str, str] = {}
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return values
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_DOTENV = _load_dotenv()


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name) or _DOTENV.get(name)
    return value if value else default


def data_root() -> Path:
    return Path(env("STATEBENCH_DATA", str(PROJECT_ROOT / "data")))


def playground_dir() -> Path:
    return Path(env("STATEBENCH_PLAYGROUND", str(data_root() / "llm-robotics-playground")))


def playground_python() -> Path:
    given = env("STATEBENCH_PLAYGROUND_PYTHON")
    if given:
        return Path(given)
    venv = playground_dir() / ".venv"
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def assets_dir() -> Path:
    """The xArm7 model used by the pick-and-place and button scenes."""
    return Path(env("STATEBENCH_XARM7", str(data_root() / "mujoco_menagerie" / "ufactory_xarm7")))


def run_dir(name: str = "turn1") -> Path:
    path = data_root() / "runs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(name: str = "turn1") -> Path:
    path = run_dir(name) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def renders_dir(name: str = "turn1") -> Path:
    path = data_root() / "renders" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


RUNS = data_root() / "runs"
SCENES = PROJECT_ROOT / "scenes"
ASSETS = PROJECT_ROOT / "assets"
