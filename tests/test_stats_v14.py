"""Turn 14: paired-difference intervals."""
from statebench.stats_v14 import newcombe, paired, tango


def test_newcombe_textbook_example():
    lo, hi = newcombe(12, 9, 2, 21)  # Newcombe (1998), method 10: about 0.018 to 0.288
    assert abs(lo - 0.018) < 0.002 and abs(hi - 0.288) < 0.002


def test_all_concordant_interval_brackets_zero():
    for f in (tango, newcombe):
        lo, hi = f(50, 0, 0, 0)
        assert lo < 0 < hi


def test_mirror_symmetry():
    lo, hi = tango(20, 5, 1, 24)
    lo2, hi2 = tango(20, 1, 5, 24)
    assert abs(lo + hi2) < 1e-6 and abs(hi + lo2) < 1e-6


def test_words():
    A = {s: {"success": True} for s in range(50)}
    B = {s: {"success": s < 28} for s in range(50)}
    assert paired(A, B)["word"] == "CLEAR DIFFERENCE"
    assert paired(A, A)["word"] == "NO CLEAR DIFFERENCE"
    assert paired({0: {"success": True}}, {0: {"success": False}})["word"] == "INCONCLUSIVE"
