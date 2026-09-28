"""Turn 14: paired binary contrasts (the difference of two success rates on the same seeds).

* ``tango``: Tango's (1998) score interval for a difference of paired proportions, found by bisection on the score
  statistic. The primary interval.
* ``newcombe``: Newcombe's (1998) hybrid score interval (method 10), from the two Wilson intervals and the phi
  correlation. Reported beside it.
* ``paired``: the 2 x 2 counts, the difference, both intervals, the exact two-sided sign test on the discordant
  pairs, and the outcome word: CLEAR DIFFERENCE if Tango's 95% interval excludes 0, NO CLEAR DIFFERENCE if it
  contains 0, INCONCLUSIVE if fewer than 30 paired seeds completed.
"""

from __future__ import annotations

import math

Z95 = 1.959963984540054


def _wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def newcombe(a: int, b: int, c: int, d: int, z: float = Z95) -> tuple[float, float]:
    """a both succeed, b first only, c second only, d neither; difference = first minus second."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = _wilson(a + b, n, z)
    l2, u2 = _wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if den == 0 else (a * d - b * c) / math.sqrt(den)
    th = p1 - p2
    lo = th - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = th + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return max(-1.0, lo), min(1.0, hi)


def _tango_z(b: int, c: int, n: int, delta: float) -> float:
    A = 2 * n
    B = -b - c + (2 * n - b + c) * delta
    C = -c * delta * (1 - delta)
    q = (math.sqrt(max(0.0, B * B - 4 * A * C)) - B) / (2 * A)
    var = n * (2 * q + delta * (1 - delta))
    if var <= 0:
        return math.copysign(math.inf, (b - c - n * delta)) if (b - c - n * delta) != 0 else 0.0
    return (b - c - n * delta) / math.sqrt(var)


def tango(a: int, b: int, c: int, d: int, z: float = Z95) -> tuple[float, float]:
    n = a + b + c + d
    th = (b - c) / n

    def root(lo, hi, want_upper):
        # Z(delta) decreases in delta; find where Z = -z (upper bound) or Z = +z (lower bound)
        target = -z if want_upper else z
        for _ in range(200):
            mid = (lo + hi) / 2
            if (_tango_z(b, c, n, mid) > target):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = -1.0 if _tango_z(b, c, n, -1.0 + 1e-12) <= z else root(-1.0 + 1e-12, th, False)
    upper = 1.0 if _tango_z(b, c, n, 1.0 - 1e-12) >= -z else root(th, 1.0 - 1e-12, True)
    return lower, upper


def sign_test(x: int, y: int) -> float | None:
    n = x + y
    if n == 0:
        return None
    k = min(x, y)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(A: dict, B: dict, min_n: int = 30) -> dict:
    """A, B: {seed: record with 'success'}; paired on the shared seeds; difference = A minus B."""
    sh = sorted(set(A) & set(B))
    a = sum(1 for s in sh if A[s]["success"] and B[s]["success"])
    b = sum(1 for s in sh if A[s]["success"] and not B[s]["success"])
    c = sum(1 for s in sh if not A[s]["success"] and B[s]["success"])
    d = len(sh) - a - b - c
    n = len(sh)
    if n == 0:
        return {"n": 0, "word": "INCONCLUSIVE"}
    tl, tu = tango(a, b, c, d)
    nl, nu = newcombe(a, b, c, d)
    word = "INCONCLUSIVE" if n < min_n else ("CLEAR DIFFERENCE" if (tl > 0 or tu < 0) else "NO CLEAR DIFFERENCE")
    return {"n": n, "a_success": a + b, "b_success": a + c, "both": a, "a_only": b, "b_only": c, "neither": d,
            "difference": round((b - c) / n, 4), "tango95": [round(tl, 4), round(tu, 4)],
            "newcombe95": [round(nl, 4), round(nu, 4)], "sign_test_p": sign_test(b, c), "word": word}
