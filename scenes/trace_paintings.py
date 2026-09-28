"""Turn 15 Phase 0: outlines traced from 3 public-domain paintings, as pen targets (statebench venv; numpy, scipy, PIL,
matplotlib only).

Sources (Wikimedia Commons, downloaded at 960 px width with Special:FilePath; files in runs/turn15/paintings/):
* The Starry Night, Vincent van Gogh, 1889 (file "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg");
* The Great Wave off Kanagawa, Katsushika Hokusai, about 1831 (file "Tsunami_by_hokusai_19th_century.jpg");
* Mona Lisa, Leonardo da Vinci, about 1503-1519 (file "Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg").
Why public domain: each artist died more than 100 years ago (van Gogh 1890, Hokusai 1849, Leonardo 1519), so the
works are out of copyright worldwide; Wikimedia Commons marks faithful photographic reproductions of 2D public-domain
works as public domain too.

Tracing method (declared):
1. grayscale; resize so the longer side is 200 px; Gaussian blur, sigma 2 px;
2. edge extraction as iso-contours of the blurred image at its median grey level (matplotlib's contour generator);
3. Douglas-Peucker simplification, tolerance 1.5 px; contours shorter than 15 px are dropped;
4. at most 20 strokes: the 20 longest; then the shortest are dropped until the total drawn length is at most
   MAX_TOTAL_M = 0.45 m at the drawing scale (the longer image side = 0.075 m);
5. stored as polylines in unit coordinates (x right, y up, the longer side spanning 0 to 1).
Placement on the paper (per seed, in ``targets15``) follows the Turn 10 convention.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import paths  # noqa: E402

D = paths.ASSETS / "paintings"
PAINTINGS = {"starry_night": "The Starry Night (van Gogh, 1889)", "great_wave": "The Great Wave off Kanagawa (Hokusai, c. 1831)",
             "mona_lisa": "Mona Lisa (Leonardo da Vinci, c. 1503-1519)"}
LONG_SIDE_PX, SIGMA_PX, DP_TOL_PX, MIN_LEN_PX = 200, 2.0, 1.5, 15.0
MAX_STROKES, SCALE_M, MAX_TOTAL_M = 20, 0.075, 0.45


def _dp(pts: np.ndarray, tol: float) -> np.ndarray:
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    d = (np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n) if n > 1e-12 else np.linalg.norm(pts - a, axis=1)
    i = int(np.argmax(d))
    if d[i] > tol:
        return np.vstack([_dp(pts[: i + 1], tol)[:-1], _dp(pts[i:], tol)])
    return np.vstack([a, b])


def trace(name: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    im = Image.open(D / f"{name}.jpg").convert("L")
    w, h = im.size
    s = LONG_SIDE_PX / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    g = gaussian_filter(np.asarray(im, dtype=float), SIGMA_PX)
    level = float(np.median(g))
    fig = plt.figure()
    cs = plt.contour(g, levels=[level])
    segs = [np.asarray(p) for p in cs.allsegs[0]]
    plt.close(fig)
    H = g.shape[0]
    lines = []
    for p in segs:
        p = _dp(p, DP_TOL_PX)
        L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        if L >= MIN_LEN_PX:
            lines.append((L, p))
    lines.sort(key=lambda t: -t[0])
    lines = lines[:MAX_STROKES]
    px_to_m = SCALE_M / LONG_SIDE_PX
    while lines and sum(L for L, _ in lines) * px_to_m > MAX_TOTAL_M:
        lines.pop()
    long_side = max(g.shape)
    unit = [[[round(float(x) / long_side, 5), round(float(H - 1 - y) / long_side, 5)] for x, y in p] for _, p in lines]
    return {"name": name, "title": PAINTINGS[name], "strokes_unit": unit, "n_strokes": len(unit),
            "total_length_m": round(sum(L for L, _ in lines) * px_to_m, 4), "iso_level": round(level, 2),
            "method": {"long_side_px": LONG_SIDE_PX, "sigma_px": SIGMA_PX, "dp_tol_px": DP_TOL_PX,
                       "min_len_px": MIN_LEN_PX, "max_strokes": MAX_STROKES, "scale_m": SCALE_M, "max_total_m": MAX_TOTAL_M}}


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, n in zip(axes, PAINTINGS):
        t = trace(n)
        (D / f"{n}_trace.json").write_text(json.dumps(t))
        for s in t["strokes_unit"]:
            a = np.asarray(s)
            ax.plot(a[:, 0], a[:, 1], lw=1)
        ax.set_aspect("equal")
        ax.set_title(f"{n}: {t['n_strokes']} strokes, {t['total_length_m']} m", fontsize=9)
        print(n, t["n_strokes"], t["total_length_m"], sum(len(s) for s in t["strokes_unit"]), "points")
    fig.savefig(D / "traces_preview.png", dpi=100)
