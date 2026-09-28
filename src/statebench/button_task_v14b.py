"""Turn 14 Phase 3 addendum: the button's T+M+P minus-one conditions (same recipe as pick and place).

Added after the first button dev results, because Phase 4 names, for Jev on both tasks, the leave-one-out condition
that dev shows matters most, and Phase 3 had defined none for the button. ``button_task_v14`` (frozen, 17783aca) is
unchanged; importing this module registers three more conditions in its CONDITIONS table:
T+M+P-A (without aligned_for_press), T+M+P-S (without estimate_stale), T+M+P-C (without
press_attempts_since_activation).
"""
from . import button_task_v14 as B

B.CONDITIONS.setdefault("T+M+P-A", B.T + B.M + ("estimate_stale", "press_attempts_since_activation"))
B.CONDITIONS.setdefault("T+M+P-S", B.T + B.M + ("aligned_for_press", "press_attempts_since_activation"))
B.CONDITIONS.setdefault("T+M+P-C", B.T + B.M + ("aligned_for_press", "estimate_stale"))
