"""Turn 16 Phase 2: the model draws. A System Two model looks at a reference image and writes the drawing as a program
of pen actions; the robot executes it in the Turn 15 drawing scene (``model_draw_scene16``).

References (``REFS``; files in runs/turn16/references/, sources in ``references.json``):
* the 3 Turn 15 paintings (Starry Night, Great Wave, Mona Lisa: the source JPGs; public domain, see
  ``trace_paintings``);
* 3 simple public-domain icons from Wikimedia Commons (PNG renders at 960 px of the SVGs): Cat silhouette (PD-self,
  released into the public domain by its author), Fish icon (PD-author, released into the public domain by its
  author, the Swedish Road Administration), Five Pointed Star Solid (PD-shape: simple geometry, not eligible for
  copyright).

Reference contours for the automatic scores (declared): for the paintings, their Turn 15 traces (``*_trace.json``);
for the icons, the same tracing method (``trace_paintings``: longer side 200 px, Gaussian blur sigma 2 px, iso-contour,
Douglas-Peucker 1.5 px, contours of at least 15 px, the 20 longest), on the icon composited on white, at the grey level
127.5 (the median grey level of a mostly white icon is white, so the midpoint of black and white is used).

Program format (frozen after dev): the model answers with one JSON object {"actions": [...]}; each action is
{"type": "draw", "points": [[x, y], ...]} (canvas mm: a 90 x 90 mm square, origin at the lower-left corner, x right,
y up), {"type": "lift", "points": []} or {"type": "finish", "points": []}. The executor, in order:
* draw: the robot's draw skill on that polyline (it raises the pen if it is down, moves to the first point, puts the
  pen down by the force sensor, draws through the points; the pen stays down at the end);
* lift: raises the pen 20 mm; finish: ends the program (actions after it are ignored).
Limits: at most ``MAX_STROKES`` draw actions and ``MAX_POINTS`` points per stroke (extra ones are dropped and
counted); points outside the canvas are clipped to it (and counted as off-paper); strokes with fewer than 2 distinct
points are dropped (counted). If the program has no finish, the executor ends it after the last action (counted). An
unparseable reply is a program with no actions (counted).

Automatic scores (declared), on the true ink after the program ends:
* each point set (the ink; the reference contour densified at 0.005 of its longer side) is normalized to its own
  bounding box: centred, scaled so the longer side is 1;
* chamfer: the mean of (mean ink-to-reference nearest distance) and (mean reference-to-ink nearest distance);
* coverage: the share of reference points with an ink point within 0.03;
* stray ink: the share of ink points farther than 0.06 from the reference;
* off-paper: the share of the program's points outside the canvas (before clipping);
* pen lifted: the scene verifier's ``pen_lifted`` (the true tip at least 5 mm above the paper, no contact force) at
  the end. An empty drawing gets chamfer 1, coverage 0, stray 0.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import numpy as np

from . import tasks15 as T
from .config import ASSETS, RUNS, SCENES

R = RUNS / "turn16"
REF_DIR = ASSETS / "references"
PAINT = ASSETS / "paintings"
REFS = ("starry_night", "great_wave", "mona_lisa", "cat", "fish", "star")
DEV_REFS = ("mona_lisa", "great_wave", "cat")
CANVAS_MM = 90.0
MAX_STROKES, MAX_POINTS = 30, 40
MAX_TOKENS = 8000
TEMPERATURE = 1.0  # samples; reasoning off for Sol and Sonnet, thinking off for local-mid
SLUGS = {"sol": "openai/gpt-6-sol", "sonnet": "anthropic/claude-sonnet-5"}
PRICE = {"openai/gpt-6-sol": (2e-6, 1e-5), "anthropic/claude-sonnet-5": (2e-6, 1e-5)}

HEADER = (
    "You control a robot arm holding a pen above a sheet of paper. The drawing area is a square, 90 mm by 90 mm. "
    "Coordinates are in millimetres, with the origin at the lower-left corner of the drawing area, x to the right and "
    "y up. You write a program for the robot as one JSON object {\"actions\": [...]}. Each action is one of:\n"
    "- {\"type\": \"draw\", \"points\": [[x, y], ...]}: the robot moves to the first point, puts the pen down, and "
    "draws a line through the points in order; the pen stays down at the end of the line;\n"
    "- {\"type\": \"lift\", \"points\": []}: the robot raises the pen;\n"
    "- {\"type\": \"finish\", \"points\": []}: the program ends.\n"
    f"At most {MAX_STROKES} draw actions, at most {MAX_POINTS} points in each, and every point inside the drawing area. "
    "Answer with the JSON object only.\n\n")

D3 = ("Draw this image as a line drawing: its main outlines, all inside the drawing area, in at most "
      f"{MAX_STROKES} strokes, with no stray marks, and lift the pen at the end, before you finish.")
CHUNKS = {
    "starry_night": ["the tall cypress tree on the left", "the swirling line across the sky", "the line of the hills",
                     "the moon and the stars"],
    "great_wave": ["the crest of the big wave on the left", "the smaller waves along the bottom", "the mountain in the middle",
                   "the boats"],
    "mona_lisa": ["the outline of the head and hair", "the shoulders and the body", "the folded hands"],
    "cat": ["the head and the ears", "the back and the tail", "the chest and the legs"],
    "fish": ["the outline of the body", "the tail", "the fins", "the eye"],
    "star": ["the outline of the five points, as one closed line"],
}


def intents(ref: str) -> dict:
    """D1, D3, D4 (frozen before any run). D4 = D3 plus the main shapes in a declared order, in the Turn 15 chunk
    layout (each step with its done condition), ending with lifting the pen and finishing."""
    steps = [(f"draw {c}", f"{c} is drawn") for c in CHUNKS[ref]]
    steps += [("lift the pen", "the pen is off the paper"), ("finish", "the program has ended")]
    d4 = D3 + "\n\nSTEPS (in order; each step is done when its condition holds):\n" + "\n".join(
        f"{i + 1}. {s}: done when {c}" for i, (s, c) in enumerate(steps))
    return {"D1": "Draw this image.", "D3": D3, "D4": d4}


SCHEMA = {"type": "object", "properties": {"actions": {"type": "array", "items": {"type": "object", "properties": {
    "type": {"type": "string", "enum": ["draw", "lift", "finish"]},
    "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}},
    "required": ["type", "points"], "additionalProperties": False}}},
    "required": ["actions"], "additionalProperties": False}


def ref_image(ref: str) -> Path:
    return PAINT / f"{ref}.jpg" if (PAINT / f"{ref}.jpg").exists() else REF_DIR / f"{ref}.png"


def _data_url(path: Path) -> str:
    from io import BytesIO

    from PIL import Image

    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    im = im.convert("RGB")
    im.thumbnail((768, 768))
    b = BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


# ------------------------------------------------------------------ reference contours
def trace_icon(name: str) -> dict:
    import sys

    sys.path.insert(0, str(SCENES))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trace_paintings as TP
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    im = Image.open(REF_DIR / f"{name}.png").convert("RGBA")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[3])
    g0 = bg.convert("L")
    w, h = g0.size
    s = TP.LONG_SIDE_PX / max(w, h)
    g0 = g0.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    g = gaussian_filter(np.pad(np.asarray(g0, dtype=float), 4, constant_values=255.0), TP.SIGMA_PX)
    fig = plt.figure()
    cs = plt.contour(g, levels=[127.5])
    segs = [np.asarray(p) for p in cs.allsegs[0]]
    plt.close(fig)
    H = g.shape[0]
    lines = []
    for p in segs:
        p = TP._dp(p, TP.DP_TOL_PX)
        L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        if L >= TP.MIN_LEN_PX:
            lines.append((L, p))
    lines.sort(key=lambda t: -t[0])
    lines = lines[: TP.MAX_STROKES]
    long_side = max(g.shape)
    unit = [[[round(float(x) / long_side, 5), round(float(H - 1 - y) / long_side, 5)] for x, y in p] for _, p in lines]
    return {"name": name, "strokes_unit": unit, "n_strokes": len(unit), "iso_level": 127.5, "method": "trace_paintings, icon on white, level 127.5"}


def reference_contour(ref: str) -> list:
    p = PAINT / f"{ref}_trace.json"
    if p.exists():
        return json.loads(p.read_text())["strokes_unit"]
    q = REF_DIR / f"{ref}_trace.json"
    if not q.exists():
        q.write_text(json.dumps(trace_icon(ref)))
    return json.loads(q.read_text())["strokes_unit"]


# ------------------------------------------------------------------ the model call
def ask(reader: str, ref: str, cond: str, sample: int, guard=None) -> dict:
    text = HEADER + intents(ref)[cond]
    img = _data_url(ref_image(ref))
    t0 = time.perf_counter()
    if reader == "local-mid":
        import urllib.request

        from .readers_v13 import LOCAL_MODELS

        body = {"model": LOCAL_MODELS["local-mid"], "stream": False, "think": False, "format": SCHEMA, "keep_alive": "60m",
                "messages": [{"role": "user", "content": text, "images": [img.split(",", 1)[1]]}],
                "options": {"temperature": TEMPERATURE, "seed": 16_000_000 + sample, "num_ctx": 16384, "num_predict": MAX_TOKENS}}
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                d = json.load(resp)
            content = d["message"]["content"]
            return {"ok": True, "content": content, "tokens_in": d.get("prompt_eval_count"), "tokens_out": d.get("eval_count"),
                    "cost_usd": 0.0, "latency_s": round(time.perf_counter() - t0, 2), "model": LOCAL_MODELS["local-mid"],
                    "prompt": text}
        except Exception as ex:  # noqa: BLE001
            return {"ok": False, "error": f"{type(ex).__name__}: {ex}", "prompt": text}
    from .backends.base import load_secret
    from .backends.http import post_json

    slug = SLUGS[reader]
    body = {"model": slug, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "usage": {"include": True},
            "response_format": {"type": "json_schema", "json_schema": {"name": "pen_program", "strict": True, "schema": SCHEMA}},
            "messages": [{"role": "user", "content": [{"type": "text", "text": text},
                                                      {"type": "image_url", "image_url": {"url": img, "detail": "auto"}}]}]}
    body["reasoning"] = {"enabled": False}  # both paid readers (dev: Sonnet's default reasoning used all 8,000 tokens on 3 of 4 cat drawings)
    p_in, p_out = PRICE[slug]
    rid = guard.reserve((len(text) // 2 + 3000) * p_in + MAX_TOKENS * p_out, f"draws {reader} {ref} {cond} s{sample}")
    key = load_secret(["OPENROUTER_API_KEY"], "OpenRouter")
    res = post_json("https://openrouter.ai/api/v1/chat/completions", body, key, timeout=300.0)
    del key
    if not res.ok:
        guard.reconcile(rid, None)
        return {"ok": False, "error": f"http {res.status}: {str(res.error)[:300]}", "prompt": text}
    raw = res.payload or {}
    u = raw.get("usage") or {}
    cost = u.get("cost")
    guard.reconcile(rid, float(cost) if cost is not None else None)
    try:
        content = raw["choices"][0]["message"]["content"]
        finish = raw["choices"][0].get("finish_reason")
    except Exception:  # noqa: BLE001
        content, finish = None, None
    return {"ok": True, "content": content, "finish_reason": finish, "tokens_in": u.get("prompt_tokens"),
            "tokens_out": u.get("completion_tokens"), "cost_usd": cost, "latency_s": round(time.perf_counter() - t0, 2),
            "model": raw.get("model"), "prompt": text}


# ------------------------------------------------------------------ program -> robot
def parse_program(content: str | None) -> tuple[list, dict]:
    """-> (ops, counts): ops are ("draw", polyline mm) / ("lift",) / ("finish",), after the declared limits."""
    from .jsontext import try_extract_json

    c = {"unparseable": 0, "strokes_dropped_over_max": 0, "points_dropped_over_max": 0, "points_total": 0,
         "points_off_canvas": 0, "degenerate_strokes": 0, "no_finish": 0, "after_finish_ignored": 0, "bad_actions": 0}
    d = try_extract_json(content or "")
    acts = d.get("actions") if isinstance(d, dict) else None
    if not isinstance(acts, list):
        c["unparseable"] = 1
        return [], c
    ops, n_draw, finished = [], 0, False
    for i, a in enumerate(acts):
        if not isinstance(a, dict) or a.get("type") not in ("draw", "lift", "finish"):
            c["bad_actions"] += 1
            continue
        if a["type"] == "finish":
            ops.append(("finish",))
            finished = True
            c["after_finish_ignored"] = len(acts) - i - 1
            break
        if a["type"] == "lift":
            ops.append(("lift",))
            continue
        pts = [p for p in (a.get("points") or []) if isinstance(p, (list, tuple)) and len(p) >= 2
               and all(isinstance(v, (int, float)) for v in p[:2])]
        if n_draw >= MAX_STROKES:
            c["strokes_dropped_over_max"] += 1
            continue
        if len(pts) > MAX_POINTS:
            c["points_dropped_over_max"] += len(pts) - MAX_POINTS
            pts = pts[:MAX_POINTS]
        c["points_total"] += len(pts)
        clipped = []
        for x, y in (p[:2] for p in pts):
            if not (0 <= x <= CANVAS_MM and 0 <= y <= CANVAS_MM):
                c["points_off_canvas"] += 1
            q = [round(min(max(float(x), 0.0), CANVAS_MM), 3), round(min(max(float(y), 0.0), CANVAS_MM), 3)]
            if not clipped or q != clipped[-1]:
                clipped.append(q)
        if len(clipped) < 2:
            c["degenerate_strokes"] += 1
            continue
        ops.append(("draw", clipped))
        n_draw += 1
    if not finished:
        c["no_finish"] = 1
    return ops, c


def execute(ops: list, seed: int, server=None) -> dict:
    strokes = [o[1] for o in ops if o[0] == "draw"]
    own = server is None
    srv = server or T.SceneServer("model_draw_scene16.py")
    log = []
    try:
        if not strokes:
            return {"ink_mm": [], "verify": {"pen_lifted": True, "finished": True, "error": None, "in_time": True},
                    "log": ["no strokes: nothing drawn, pen never lowered"]}
        srv.call({"cmd": "reset", "seed": seed, "strokes_mm": strokes})
        k = 0
        for o in ops:
            if o[0] == "draw":
                r = srv.call({"cmd": "do", "skill": f"draw({k})"})
                k += 1
            elif o[0] == "lift":
                r = srv.call({"cmd": "do", "skill": "lift"})
            else:
                break
            log.append(r["events"])
            if r["state"].get("error"):
                break
        srv.call({"cmd": "do", "skill": "finish"})
        v = srv.call({"cmd": "verify"})
        ink = srv.call({"cmd": "ink"})["ink_mm"]
    finally:
        if own:
            srv.close()
    return {"ink_mm": ink, "verify": {k2: v[k2] for k2 in ("pen_lifted", "finished", "error", "in_time", "tip_height_mm", "sim_time_s")},
            "log": log}


# ------------------------------------------------------------------ automatic scores
def _norm(p: np.ndarray) -> np.ndarray:
    lo, hi = p.min(0), p.max(0)
    s = float(max(hi - lo)) or 1.0
    return (p - (lo + hi) / 2) / s


def _densify(strokes: list, step: float) -> np.ndarray:
    out = []
    for s in strokes:
        s = np.asarray(s, dtype=float)
        for a, b in zip(s[:-1], s[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
        out.append(s[-1:])
    return np.vstack(out)


def _nn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    return cKDTree(b).query(a)[0]


def score(ink_mm: list, ref: str, counts: dict, verify: dict) -> dict:
    refu = _densify(reference_contour(ref), 0.005)
    rn = _norm(refu)
    out = {"pen_lifted": bool(verify.get("pen_lifted")) and verify.get("error") is None,
           "off_paper": round(counts["points_off_canvas"] / counts["points_total"], 4) if counts["points_total"] else 0.0,
           "n_ink": len(ink_mm)}
    if len(ink_mm) < 2:
        out.update(chamfer=1.0, coverage=0.0, stray=0.0)
        return out
    ink = np.asarray(ink_mm, dtype=float)
    if len(ink) > 6000:
        ink = ink[np.linspace(0, len(ink) - 1, 6000).astype(int)]
    iname = _norm(ink)
    d_ir, d_ri = _nn(iname, rn), _nn(rn, iname)
    out.update(chamfer=round(float((d_ir.mean() + d_ri.mean()) / 2), 4), coverage=round(float(np.mean(d_ri <= 0.03)), 4),
               stray=round(float(np.mean(d_ir > 0.06)), 4))
    return out


def run_one(reader: str, ref: str, cond: str, sample: int, guard, server=None) -> dict:
    a = ask(reader, ref, cond, sample, guard)
    rec = {"reader": reader, "ref": ref, "condition": cond, "sample": sample, **{k: v for k, v in a.items() if k != "prompt"}}
    if not a.get("ok"):
        return rec
    ops, counts = parse_program(a.get("content"))
    ex = execute(ops, 16_100_000 + sample, server)
    rec.update(program=[list(o) for o in ops], counts=counts, verify=ex["verify"], ink_mm=ex["ink_mm"],
               scores=score(ex["ink_mm"], ref, counts, ex["verify"]))
    return rec
