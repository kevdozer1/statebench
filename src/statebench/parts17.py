"""Turn 17 Phase 2 (drawing): does the goal steer what Sol draws? The same painting, three goals.

Goals per painting (declared before any run): the whole picture, and 2 single parts from the Turn 16 chunk lists:
* Mona Lisa: "the head and hair", "the folded hands";
* Starry Night: "the tall cypress tree", "the moon and the stars";
* Great Wave: "the big wave on the left", "the mountain in the middle".

**Canvas mapping (declared):** the 90 x 90 mm canvas stands for the whole painting. The painting is fitted inside the
canvas with a MARGIN_MM = 5 mm margin on every side (the inner 80 x 80 mm square), its proportions kept, centred
(``fit``). Every drawing goal says so and gives the painting's edges on the canvas in millimetres (``placement``), so a
part is drawn where it sits in the painting. Ink is mapped back to image coordinates by the inverse of this fit
(``canvas_to_image``). Part scores never normalize a drawing to its own bounding box.

Prompt: the Turn 16 program header (``draws16.HEADER``), then the placement text, then the goal sentence, then "Lift the
pen at the end, before you finish." in every goal. Sol with the image, reasoning off, temperature 1.0 (the Turn 16
settings); executed by the robot in ``model_draw_scene16`` as the Turn 16 model draws.

Masks (``MASKS``): one polygon or a union of circles per part, in image coordinates (u to the right, v down, both 0-1 of
the image's width and height), drawn on the reference image; extended after Kevin's check (the hands mask includes both
wrists, the cypress mask the full base, the big-wave mask the lower trunk) and frozen; rendered as ``part_masks.png``
with the grown outlines.

Part scores, each reported twice: with the masks as drawn, and with the masks grown by GROW = 3% of the image width
(suffix ``_g3``; a point is inside the grown mask if it is inside the mask or within 0.03 image widths of its boundary):
* ``in_<part>``: the share of ink points inside that part's mask;
* ``cov_<part>``: the coverage of that part's contour: the share of the part's contour points (the Turn 15 trace of the
  painting, densified, inside the mask; if fewer than ``MIN_CONTOUR_PTS`` fall inside, the mask's own outline) with an
  ink point within 0.03 of the image's longer side;
* pen lifted (the scene verifier), as in Turn 16.
The steering table (2 x 2 per painting): goal A and goal B against in_A and in_B.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import draws16 as D
from .config import RUNS

R = RUNS / "turn17"
PAINTINGS = ("mona_lisa", "starry_night", "great_wave")
PARTS = {"mona_lisa": ("head and hair", "folded hands"), "starry_night": ("cypress tree", "moon and stars"),
         "great_wave": ("big wave", "mountain")}
GOAL_TEXT = {
    ("mona_lisa", "whole"): "Draw this image as a line drawing.",
    ("mona_lisa", "head and hair"): "Draw only the head and hair from this image, as a line drawing, and nothing else.",
    ("mona_lisa", "folded hands"): "Draw only the folded hands from this image, as a line drawing, and nothing else.",
    ("starry_night", "whole"): "Draw this image as a line drawing.",
    ("starry_night", "cypress tree"): "Draw only the tall cypress tree from this image, as a line drawing, and nothing else.",
    ("starry_night", "moon and stars"): "Draw only the moon and the stars from this image, as a line drawing, and nothing else.",
    ("great_wave", "whole"): "Draw this image as a line drawing.",
    ("great_wave", "big wave"): "Draw only the big wave on the left from this image, as a line drawing, and nothing else.",
    ("great_wave", "mountain"): "Draw only the mountain in the middle from this image, as a line drawing, and nothing else.",
}
PLACEMENT = ("The drawing area stands for the whole painting: fit the painting inside the drawing area with a 5 mm margin "
             "on every side, its proportions kept and centred, and draw every part at the position where it sits in the "
             "painting.")
LIFT = "Lift the pen at the end, before you finish."
MIN_CONTOUR_PTS = 20
MARGIN_MM = 5.0
GROW = 0.03
STAR_R = 0.045
MASKS = {
    ("mona_lisa", "head and hair"): {"poly": [(0.30, 0.10), (0.40, 0.07), (0.56, 0.07), (0.66, 0.13), (0.70, 0.30),
                                              (0.72, 0.47), (0.62, 0.44), (0.56, 0.37), (0.44, 0.37), (0.37, 0.44),
                                              (0.28, 0.47), (0.29, 0.28)]},
    ("mona_lisa", "folded hands"): {"poly": [(0.14, 0.80), (0.22, 0.74), (0.40, 0.71), (0.58, 0.66), (0.70, 0.64),
                                             (0.82, 0.69), (0.80, 0.84), (0.73, 0.91), (0.50, 0.95), (0.18, 0.95)]},
    ("starry_night", "cypress tree"): {"poly": [(0.20, 0.05), (0.23, 0.20), (0.27, 0.45), (0.31, 0.62), (0.37, 0.80),
                                                (0.45, 1.00), (0.08, 1.00), (0.10, 0.70), (0.14, 0.35), (0.17, 0.15)]},
    ("starry_night", "moon and stars"): {"circles": [(0.885, 0.17, 0.09)] + [(u, v, STAR_R) for u, v in (
        (0.10, 0.04), (0.23, 0.17), (0.34, 0.04), (0.42, 0.06), (0.62, 0.08), (0.70, 0.23), (0.33, 0.32),
        (0.05, 0.45), (0.13, 0.47), (0.35, 0.53))]},
    ("great_wave", "big wave"): {"poly": [(0.02, 0.40), (0.10, 0.28), (0.20, 0.12), (0.32, 0.06), (0.45, 0.08),
                                          (0.55, 0.20), (0.60, 0.36), (0.46, 0.42), (0.44, 0.58), (0.42, 0.74),
                                          (0.36, 0.86), (0.20, 0.90), (0.04, 0.88), (0.00, 0.70)]},
    ("great_wave", "mountain"): {"poly": [(0.55, 0.74), (0.60, 0.66), (0.63, 0.62), (0.66, 0.66), (0.72, 0.74)]},
}


def image_size(ref: str) -> tuple[int, int]:
    from PIL import Image

    return Image.open(D.ref_image(ref)).size


def fit(ref: str) -> tuple[float, float, float, float]:
    """(x0, y0: the painting's top-left corner on the canvas, from the canvas's left and top edges, mm; width mm;
    height mm): fitted inside the inner (90 - 2 x 5) mm square, proportions kept, centred."""
    w, h = image_size(ref)
    inner = D.CANVAS_MM - 2 * MARGIN_MM
    s = inner / max(w, h)
    W, H = w * s, h * s
    return (D.CANVAS_MM - W) / 2, (D.CANVAS_MM - H) / 2, W, H


def canvas_to_image(ref: str, pts) -> np.ndarray:
    x0, y0, W, H = fit(ref)
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    u = (p[:, 0] - x0) / W
    v = ((D.CANVAS_MM - p[:, 1]) - y0) / H
    return np.stack([u, v], 1)


def _seg_dist(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(2) / np.maximum((ab ** 2).sum(1), 1e-12)[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2).min(1)


def inside(ref: str, part: str, uv: np.ndarray, grow: float = 0.0) -> np.ndarray:
    """Inside the mask, or (grow > 0) within ``grow`` image widths of its boundary."""
    from matplotlib.path import Path as MPath

    m = MASKS[(ref, part)]
    w, h = image_size(ref)
    asp = h / w
    if len(uv) == 0:
        return np.zeros(0, dtype=bool)
    if "poly" in m:
        ok = MPath(m["poly"]).contains_points(uv)
        if grow > 0:
            ok |= _seg_dist(uv * [1.0, asp], np.asarray(m["poly"]) * [1.0, asp]) <= grow
        return ok
    ok = np.zeros(len(uv), dtype=bool)
    for cu, cv, r in m["circles"]:  # r in units of the image width; v scaled by the aspect ratio
        ok |= ((uv[:, 0] - cu) ** 2 + ((uv[:, 1] - cv) * asp) ** 2) <= (r + grow) ** 2
    return ok


def trace_uv(ref: str) -> np.ndarray:
    """The Turn 15 trace in image (u, v) coordinates, densified."""
    t = json.loads((D.PAINT / f"{ref}_trace.json").read_text())
    w, h = image_size(ref)
    long_side = max(w, h)
    pts = D._densify(t["strokes_unit"], 0.005)
    return np.stack([pts[:, 0] * long_side / w, 1 - pts[:, 1] * long_side / h], 1)


def mask_outline(ref: str, part: str) -> np.ndarray:
    m = MASKS[(ref, part)]
    if "poly" in m:
        poly = np.asarray(m["poly"] + [m["poly"][0]])
        return D._densify([poly.tolist()], 0.005)
    w, h = image_size(ref)
    out = []
    for cu, cv, r in m["circles"]:
        a = np.linspace(0, 2 * np.pi, 40)
        out.append(np.stack([cu + r * np.cos(a), cv + r * np.sin(a) * w / h], 1))
    return np.vstack(out)


def part_contour(ref: str, part: str, grow: float = 0.0) -> tuple[np.ndarray, str]:
    tr = trace_uv(ref)
    c = tr[inside(ref, part, tr, grow)]
    if len(c) >= MIN_CONTOUR_PTS:
        return c, "trace inside the mask"
    return mask_outline(ref, part), "mask outline (too little trace inside)"


def part_scores(ref: str, ink_mm: list, verify: dict) -> dict:
    """Ink mapped by the declared fit (no bounding-box normalization); masks as drawn and grown by GROW (suffix _g3)."""
    from scipy.spatial import cKDTree

    out = {"n_ink": len(ink_mm), "pen_lifted": bool(verify.get("pen_lifted")) and verify.get("error") is None}
    w, h = image_size(ref)
    scale = np.array([w, h]) / max(w, h)
    uv = canvas_to_image(ref, ink_mm) if ink_mm else np.zeros((0, 2))
    tree = cKDTree(uv * scale) if len(uv) else None
    for grow, suf in ((0.0, ""), (GROW, "_g3")):
        for part in PARTS[ref]:
            key = part.replace(" ", "_") + suf
            out[f"in_{key}"] = round(float(inside(ref, part, uv, grow).mean()), 4) if len(uv) else 0.0
            c, _ = part_contour(ref, part, grow)
            out[f"cov_{key}"] = round(float(np.mean(tree.query(c * scale)[0] <= 0.03)), 4) if tree is not None else 0.0
    return out


def placement(ref: str) -> str:
    """PLACEMENT plus the painting's edges on the canvas (dev pass 2: on pass 1 Sol drew an isolated part at image-v
    coordinates, upside down on the canvas)."""
    x0, y0, W, H = fit(ref)
    return (PLACEMENT + f" In the drawing area the painting's top edge is at y = {D.CANVAS_MM - y0:.0f} mm and its bottom "
            f"edge at y = {D.CANVAS_MM - y0 - H:.0f} mm; its left edge is at x = {x0:.0f} mm and its right edge at "
            f"x = {x0 + W:.0f} mm. Something near the top of the painting has a large y, and something near its bottom "
            "a small y.")


def prompt(ref: str, goal: str) -> str:
    return D.HEADER + placement(ref) + "\n\n" + GOAL_TEXT[(ref, goal)] + " " + LIFT


def ask(ref: str, goal: str, sample: int, guard) -> dict:
    from .backends.base import load_secret
    from .backends.http import post_json

    text = prompt(ref, goal)
    img = D._data_url(D.ref_image(ref))
    slug = D.SLUGS["sol"]
    body = {"model": slug, "temperature": D.TEMPERATURE, "max_tokens": D.MAX_TOKENS, "usage": {"include": True},
            "reasoning": {"enabled": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "pen_program", "strict": True, "schema": D.SCHEMA}},
            "messages": [{"role": "user", "content": [{"type": "text", "text": text},
                                                      {"type": "image_url", "image_url": {"url": img, "detail": "auto"}}]}]}
    p_in, p_out = D.PRICE[slug]
    rid = guard.reserve((len(text) // 2 + 3000) * p_in + D.MAX_TOKENS * p_out, f"parts sol {ref} {goal} s{sample}")
    key = load_secret(["OPENROUTER_API_KEY"], "OpenRouter")
    t0 = time.perf_counter()
    res = post_json("https://openrouter.ai/api/v1/chat/completions", body, key, timeout=300.0)
    del key
    if not res.ok:
        guard.reconcile(rid, None)
        return {"ok": False, "error": f"http {res.status}: {str(res.error)[:300]}"}
    raw = res.payload or {}
    u = raw.get("usage") or {}
    cost = u.get("cost")
    guard.reconcile(rid, float(cost) if cost is not None else None)
    try:
        content = raw["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001
        content = None
    return {"ok": True, "content": content, "tokens_in": u.get("prompt_tokens"), "tokens_out": u.get("completion_tokens"),
            "cost_usd": cost, "latency_s": round(time.perf_counter() - t0, 2), "model": raw.get("model")}


def run_one(ref: str, goal: str, sample: int, guard) -> dict:
    a = ask(ref, goal, sample, guard)
    rec = {"reader": "sol", "ref": ref, "goal": goal, "sample": sample, "goal_text": GOAL_TEXT[(ref, goal)], **a}
    if not a.get("ok"):
        return rec
    ops, counts = D.parse_program(a.get("content"))
    ex = D.execute(ops, 17_100_000 + sample)
    rec.update(program=[list(o) for o in ops], counts=counts, verify=ex["verify"], ink_mm=ex["ink_mm"],
               scores=part_scores(ref, ex["ink_mm"], ex["verify"]))
    return rec


def masks_png(path: Path) -> Path:
    from PIL import Image, ImageDraw

    tiles = []
    for ref in PAINTINGS:
        im = Image.open(D.ref_image(ref)).convert("RGB")
        im.thumbnail((520, 520))
        w, h = im.size
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for part, col in zip(PARTS[ref], ((255, 60, 60, 110), (60, 200, 255, 110))):
            m = MASKS[(ref, part)]
            if "poly" in m:
                d.polygon([(u * w, v * h) for u, v in m["poly"]], fill=col, outline=col[:3] + (255,))
            else:
                for cu, cv, r in m["circles"]:
                    d.ellipse((cu * w - r * w, cv * h - r * w, cu * w + r * w, cv * h + r * w), fill=col, outline=col[:3] + (255,))
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
        # the masks grown by GROW, as thin outlines (the edge pixels of the grown region)
        px = np.asarray(im).copy()
        gu, gv = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
        uv = np.stack([gu.ravel(), gv.ravel()], 1)
        for part, col in zip(PARTS[ref], ((200, 20, 20), (20, 120, 220))):
            g = inside(ref, part, uv, GROW).reshape(h, w)
            edge = g & ~(np.roll(g, 1, 0) & np.roll(g, -1, 0) & np.roll(g, 1, 1) & np.roll(g, -1, 1))
            px[edge] = col
        im = Image.fromarray(px)
        d2 = ImageDraw.Draw(im)
        d2.rectangle((0, 0, w, 48), fill=(255, 255, 255))
        d2.text((4, 3), f"red: {PARTS[ref][0]}", fill=(200, 30, 30))
        d2.text((4, 18), f"blue: {PARTS[ref][1]}", fill=(20, 120, 200))
        d2.text((4, 33), "filled: as drawn; outline: grown by 3% of the width", fill=(60, 60, 60))
        tiles.append(im)
    W = Image.new("RGB", (sum(t.width + 10 for t in tiles), max(t.height for t in tiles)), (255, 255, 255))
    x = 0
    for t in tiles:
        W.paste(t, (x, 0))
        x += t.width + 10
    W.save(path)
    return path


def main(phase: str, samples: str, workers: int = 3) -> None:
    """Run every painting x goal for the samples a:b; file runs/turn17/parts_<phase>.jsonl (transport failures to a
    separate archive file); one Sol process, calls in up to ``workers`` threads."""
    from concurrent.futures import ThreadPoolExecutor

    from . import transport_v13b  # noqa: F401
    from .spend17 import guard as mkguard

    a, b = (int(x) for x in samples.split(":"))
    out = R / f"parts_{phase}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["ref"], r["goal"], r["sample"]))
    todo = [(ref, g, s) for ref in PAINTINGS for g in ("whole",) + PARTS[ref] for s in range(a, b) if (ref, g, s) not in done]
    guard = mkguard()
    print(json.dumps({"grid": f"turn17 parts {phase}", "calls": len(todo), "expected_spend_usd": round(len(todo) * 0.016, 2),
                      "expected_spend_upper_usd": round(len(todo) * 0.09, 2), "expected_runtime_min": round(len(todo) * 70 / workers / 60, 1),
                      "headroom_usd": round(guard.headroom(), 3), "rate_limit": "Sol: one process, 429 retried"}), flush=True)

    def one(item):
        try:
            return run_one(*item, guard)
        except Exception as ex:  # noqa: BLE001
            return {"ref": item[0], "goal": item[1], "sample": item[2], "ok": False, "error": f"{type(ex).__name__}: {ex}"}

    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, todo):
            dst = out if r.get("ok") else out.with_name(out.stem + "_transport_errors.jsonl")
            with open(dst, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(r["ref"], r["goal"], r["sample"], r.get("ok"), json.dumps(r.get("scores")), flush=True)


if __name__ == "__main__":
    import sys

    main(sys.argv[1], sys.argv[2])
