"""Provenance check on reads, not only on declarations (v0.2 fix 8).

``predicates.check_provenance`` checks the field paths each predicate *declares*.
This module checks what ``build_predicates`` actually *reads*: the state and
proprio dicts are wrapped in an access-recording mapping, every lookup is logged
as a dotted path, and each path read must be in ``predicates.ALLOWED_PATHS``.

Path normalization, declared here:

* anything read inside a history snapshot is the path ``history``;
* anything read inside the ``events_recent`` list is the path ``events_recent``;
* a lookup that finds no value (a missing key) counts as a read of that path,
  and a missing *container* path is allowed when some allowed path lies under it,
  since reading "is there a geometry block at all" is how a reader learns it has
  none.
"""

from __future__ import annotations

from typing import Any

from .predicates import ALLOWED_PATHS, build_predicates

_MISSING = object()


class RecordingDict(dict):
    """A dict that records the dotted path of every key looked up in it."""

    def __init__(self, data: dict[str, Any], prefix: str, log: set[str], collapse: str | None = None):
        super().__init__(data)
        self._prefix = prefix
        self._log = log
        self._collapse = collapse

    def _path(self, key: Any) -> str:
        if self._collapse is not None:
            return self._collapse
        return f"{self._prefix}.{key}" if self._prefix else str(key)

    def _wrap(self, key: Any, value: Any) -> Any:
        path = self._path(key)
        if self._collapse is not None:
            self._log.add(self._collapse)
            if isinstance(value, dict):
                return RecordingDict(value, path, self._log, collapse=self._collapse)
            return value
        if isinstance(value, dict):
            return RecordingDict(value, path, self._log)
        # A leaf, or a list: the path is read.
        self._log.add(path)
        if key == "events_recent" and isinstance(value, list):
            return list(value)
        return value

    def __getitem__(self, key):
        if not dict.__contains__(self, key):
            self._log.add(self._path(key))
            raise KeyError(key)
        return self._wrap(key, dict.__getitem__(self, key))

    def get(self, key, default=None):
        if not dict.__contains__(self, key):
            self._log.add(self._path(key))
            return default
        return self._wrap(key, dict.__getitem__(self, key))

    def __contains__(self, key):
        self._log.add(self._path(key))
        return dict.__contains__(self, key)

    def items(self):
        return [(k, self._wrap(k, v)) for k, v in dict.items(self)]

    def values(self):
        return [self._wrap(k, v) for k, v in dict.items(self)]


def _allowed(path: str, allowed: frozenset[str]) -> bool:
    return path in allowed or any(a.startswith(path + ".") for a in allowed)


def audited_build_predicates(
    state: dict[str, Any],
    proprio: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
    allowed: frozenset[str] = ALLOWED_PATHS,
) -> tuple[dict[str, Any], set[str], set[str]]:
    """Run ``build_predicates`` under recording. Returns (block, paths read, undeclared)."""
    log: set[str] = set()
    wrapped_state = RecordingDict(state, "", log)
    wrapped_proprio = None if proprio is None else RecordingDict(proprio, "proprio", log)
    wrapped_history = (
        None if history is None
        else [RecordingDict(s, "history", log, collapse="history") for s in history]
    )
    block = build_predicates(wrapped_state, wrapped_proprio, wrapped_history)
    undeclared = {p for p in log if not _allowed(p, allowed)}
    # The block must equal the unwrapped computation, or the wrapper changed behaviour.
    plain = build_predicates(state, proprio, history)
    if plain["values"] != block["values"]:
        raise AssertionError("recording wrapper changed predicate values")
    return plain, log, undeclared
