"""Line-ending-normalized file hashes, used from Turn 4 on.

Turns 1-3 hashed files with ``sha256(path.read_bytes())``. On this Windows
machine git's system config sets ``core.autocrlf=true``, so a working copy can
carry CRLF endings while the committed blob carries LF. A hash of the working
copy then cannot be reproduced from a clone on another machine. Three recorded
hashes (``rules.py``, ``rules_turn3b.py`` and ``executor.py`` at Turn 3b) were
taken that way.

From v0.2 every hash is taken over the file with CRLF converted to LF, which
equals the committed blob once ``.gitattributes`` pins ``eol=lf``. The old hash
functions in ``rules_turn*.py`` are left exactly as they were; this module is
new and nothing earlier imports it.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
HASH_SCHEME = "sha256 over bytes with CRLF converted to LF (statebench v0.2)"


def normalize_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_sha256(path: Path | str) -> str:
    """The v0.2 hash: CRLF -> LF, then sha256."""
    return sha256_bytes(normalize_bytes(Path(path).read_bytes()))


def raw_sha256(path: Path | str) -> str:
    """The Turn 1-3 hash of the working copy, bytes as they are on disk."""
    return sha256_bytes(Path(path).read_bytes())


def committed_blob(relpath: str, rev: str = "HEAD") -> bytes | None:
    """Bytes of ``relpath`` as committed at ``rev`` (``git show rev:path``)."""
    try:
        out = subprocess.run(
            ["git", "show", f"{rev}:{relpath}"],
            cwd=REPO_ROOT, capture_output=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout


def module_hash(name: str) -> str:
    """Normalized hash of a module in the statebench package, by file name."""
    return normalized_sha256(PACKAGE_DIR / name)
