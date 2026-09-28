"""R4d: the R4c decision table with rules_v2's explicit first-match order and final fallback.

Prompt engineering, labelled as such wherever it appears. With R4d, Jev and the
``rules_v2`` lookup table share one policy contract: the same clauses, the same
order, the same default. Built from R4c's own sentences so no clause is reworded.
"""

from __future__ import annotations

from .rules_turn3b import R4C_NEXT_ACTION_INSTRUCTION

_ORDER = ("Choose inspect", "Choose release if gripper_closed_empty", "Choose retreat",
          "Choose release if object_at_receptacle", "Choose transport", "Choose test_lift",
          "Choose close", "Choose approach")


def _r4d() -> str:
    sentences = [s.strip() + "." for s in R4C_NEXT_ACTION_INSTRUCTION.split(".") if s.strip()]
    ordered = [next(s for s in sentences if s.startswith(prefix)) for prefix in _ORDER]
    assert sorted(ordered) == sorted(sentences)
    return (" ".join(ordered) + " Apply the first rule that matches, in the order listed."
            " If no rule matches, choose approach.")


R4D_NEXT_ACTION_INSTRUCTION = _r4d()
