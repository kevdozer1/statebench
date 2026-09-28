"""Turn 18 A5: iterative parts with a layout (added condition, exploratory; Kevin's interjection).

Why: in A3 each part is drawn on its own near the centre of the canvas, and the parts lose their spatial relation (on
the Statue the raised arm and then the head landed on top of each other). A part instruction says what to draw and gives
no where. A3 stays as frozen in the primary study; A5 is added.

**Definition.** The A2-A4 protocol unchanged (``draw18``: 4 rounds, at most 15 strokes per round, append-only, a fresh
prompt each round with the reference, the overall goal and the canvas image, the round number and budget, one re-prompt
on a malformed or truncated reply, the same output limit), plus:
* round 1 asks, in the same reply as its round-1 strokes, for a layout: one box (x_min, y_min, x_max, y_max, canvas mm)
  for each declared part, so that together they compose the whole picture. The layout is recorded. A layout is valid
  if it has one box per part, in the declared order, each inside the canvas with min < max; otherwise the reply counts
  as malformed (one re-prompt);
* rounds 1-4 each name one part in the declared order, with its box, and say the part must be drawn inside its box;
* 4 calls per drawing (the layout adds no call). If round 1 fails twice, no layout exists; rounds 2-4 then name the
  part without a box (recorded).
Measures added: the share of each round's ink inside that round's box, and pairwise box overlap (overlap area / the
smaller box's area).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import draw18 as A
from . import draws16 as D
from . import tasks15 as T

R = A.R
LAYOUT_SCHEMA = {"type": "object", "properties": {
    "layout": {"type": "array", "items": {"type": "object", "properties": {
        "part": {"type": "string"}, "x_min": {"type": "number"}, "y_min": {"type": "number"},
        "x_max": {"type": "number"}, "y_max": {"type": "number"}},
        "required": ["part", "x_min", "y_min", "x_max", "y_max"], "additionalProperties": False}},
    "actions": D.SCHEMA["properties"]["actions"]}, "required": ["layout", "actions"], "additionalProperties": False}
STAGE_R1 = ("The drawing is made part by part, one part per round, in this order: {order}. In this first round, first plan "
            "the layout: give one box on the drawing area for each of the 4 parts, in this order, as \"layout\": a list of "
            "{{\"part\", \"x_min\", \"y_min\", \"x_max\", \"y_max\"}} in millimetres, placed so that together the boxes "
            "compose the whole picture. Each part will be drawn inside its own box. Then, in \"actions\", draw part 1: "
            "{name}, inside its box. The part must be drawn inside its box.")
STAGE_RN = ("The drawing is made part by part, one part per round, in this order: {order}. The layout chosen in the first "
            "round gives each part a box on the drawing area: {boxes}. In this round, draw part {r}: {name}, inside its box "
            "(x from {x0:.0f} to {x1:.0f} mm, y from {y0:.0f} to {y1:.0f} mm). The part must be drawn inside its box.")
STAGE_RN_NOLAYOUT = ("The drawing is made part by part, one part per round, in this order: {order}. In this round, draw part "
                     "{r}: {name}.")


def prompt(ref: str, r: int, layout: list | None) -> str:
    parts = A.PARTS[ref]
    order = "; ".join(f"{i + 1}. {p}" for i, p in enumerate(parts))
    if r == 1:
        stage = STAGE_R1.format(order=order, name=parts[0])
    elif layout:
        b = layout[r - 1]
        boxes = "; ".join(f"{i + 1}. {p}: x {bx['x_min']:.0f}-{bx['x_max']:.0f} mm, y {bx['y_min']:.0f}-{bx['y_max']:.0f} mm"
                          for i, (p, bx) in enumerate(zip(parts, layout)))
        stage = STAGE_RN.format(order=order, boxes=boxes, r=r, name=parts[r - 1], x0=b["x_min"], x1=b["x_max"], y0=b["y_min"], y1=b["y_max"])
    else:
        stage = STAGE_RN_NOLAYOUT.format(order=order, r=r, name=parts[r - 1])
    rt = A.ROUND_TEXT.format(r=r, left=A.ROUNDS - r, canvas_line=A.CANVAS_LINE_REF if A.has_reference(ref) else A.CANVAS_LINE_WORDS)
    return A.HEADER.replace("{N}", str(A.ROUND_MAX)) + A.placement(ref) + "\n\n" + A.goal_text(ref) + "\n\n" + stage + "\n\n" + rt


def prompt_hashes() -> dict:
    from .hashing import REPO_ROOT, normalized_sha256

    t = A.HEADER + "{placement}\n\n{goal}\n\n" + STAGE_R1 + "\n\n" + STAGE_RN + "\n\n" + STAGE_RN_NOLAYOUT + "\n\n" + A.ROUND_TEXT
    return {"A5": hashlib.sha256(t.replace("\r\n", "\n").encode()).hexdigest(),
            "layout_schema": hashlib.sha256(json.dumps(LAYOUT_SCHEMA, sort_keys=True).encode()).hexdigest(),
            "draw18a5.py": normalized_sha256(REPO_ROOT / "src/statebench/draw18a5.py")}


def valid_layout(content: str | None, n_parts: int) -> list | None:

    try:
        d = json.loads(content or "")
    except Exception:  # noqa: BLE001
        return None
    lay = d.get("layout") if isinstance(d, dict) else None
    if not isinstance(lay, list) or len(lay) != n_parts:
        return None
    out = []
    for b in lay:
        try:
            x0, y0, x1, y1 = (float(b[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
        except Exception:  # noqa: BLE001
            return None
        if not (0 <= x0 < x1 <= A.CANVAS_MM and 0 <= y0 < y1 <= A.CANVAS_MM):
            return None
        out.append({"part": str(b.get("part", "")), "x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1})
    return out


def ask_round1(text: str, images: list, guard, label: str, n_parts: int) -> dict:
    """The A2-A4 retry rule, with the layout's validity part of 'well formed'."""
    tries = []
    for attempt in range(2):
        q = text if attempt == 0 else text + "\n\n" + A.RETRY
        a = _ask_layout(q, images, guard, f"{label} a{attempt}")
        if not a.get("ok"):
            raise RuntimeError(a["error"])
        ops, counts = A.parse(a.get("content"), A.ROUND_MAX)
        lay = valid_layout(a.get("content"), n_parts)
        bad = counts["unparseable"] or a.get("finish_reason") == "length" or lay is None
        tries.append({k: v for k, v in a.items() if k != "ok"} | {"truncated": a.get("finish_reason") == "length",
                                                                   "unparseable": bool(counts["unparseable"]), "layout_valid": lay is not None})
        if not bad:
            return {"ops": ops, "counts": counts, "tries": tries, "failed": False, "layout": lay}
    return {"ops": [], "counts": counts, "tries": tries, "failed": True, "layout": None}


def _ask_layout(text: str, images: list, guard, label: str) -> dict:
    """``draw18.ask`` with the layout schema in place of the program schema (same model, settings and output limit)."""
    import time

    from .backends.base import load_secret
    from .backends.http import post_json

    content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u, "detail": "auto"}} for u in images]
    body = {"model": A.SOL, "temperature": A.TEMPERATURE, "max_tokens": A.MAX_TOKENS_ROUND, "usage": {"include": True},
            "reasoning": {"enabled": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "layout_and_program", "strict": True,
                                                                       "schema": LAYOUT_SCHEMA}},
            "messages": [{"role": "user", "content": content}]}
    rid = guard.reserve((len(text) // 2 + 1200 * len(images) + 200) * A.PRICE[0] + A.MAX_TOKENS_ROUND * A.PRICE[1], label)
    key = load_secret(["OPENROUTER_API_KEY"], "OpenRouter")
    t0 = time.perf_counter()
    res = post_json("https://openrouter.ai/api/v1/chat/completions", body, key, timeout=600.0)
    del key
    if not res.ok:
        guard.reconcile(rid, None)
        return {"ok": False, "error": f"http {res.status}: {str(res.error)[:300]}"}
    raw = res.payload or {}
    u = raw.get("usage") or {}
    cost = u.get("cost")
    guard.reconcile(rid, float(cost) if cost is not None else None)
    try:
        out, finish = raw["choices"][0]["message"]["content"], raw["choices"][0].get("finish_reason")
    except Exception:  # noqa: BLE001
        out, finish = None, None
    return {"ok": True, "content": out, "finish_reason": finish, "tokens_in": u.get("prompt_tokens"),
            "tokens_out": u.get("completion_tokens"), "cost_usd": cost, "latency_s": round(time.perf_counter() - t0, 2)}


def in_box(ink: list, b: dict) -> float:
    if not ink:
        return 0.0
    p = np.asarray(ink)
    return float(np.mean((p[:, 0] >= b["x_min"]) & (p[:, 0] <= b["x_max"]) & (p[:, 1] >= b["y_min"]) & (p[:, 1] <= b["y_max"])))


def overlaps(layout: list) -> dict:
    out = []
    for i in range(len(layout)):
        for j in range(i + 1, len(layout)):
            a, b = layout[i], layout[j]
            w = max(0.0, min(a["x_max"], b["x_max"]) - max(a["x_min"], b["x_min"]))
            h = max(0.0, min(a["y_max"], b["y_max"]) - max(a["y_min"], b["y_min"]))
            small = min((a["x_max"] - a["x_min"]) * (a["y_max"] - a["y_min"]), (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"]))
            out.append(round(w * h / small, 3) if small > 0 else 0.0)
    return {"pairwise": out, "pairs_overlapping": sum(v > 0 for v in out), "max": max(out) if out else 0.0}


def run_drawing(ref: str, sample: int, guard, seed: int, phase: str) -> dict:
    ref_url = D._data_url(D.ref_image(ref)) if A.has_reference(ref) else None
    srv = T.SceneServer("draw_scene18.py")
    rec = {"ref": ref, "condition": "A5", "sample": sample, "phase": phase, "seed": seed, "rounds": [], "layout": None}
    used, early = 0, False
    try:
        srv.call({"cmd": "reset", "seed": seed})
        for r in range(1, A.ROUNDS + 1):
            ink_before = srv.call({"cmd": "ink"})["ink_mm"]
            text = prompt(ref, r, rec["layout"])
            imgs = ([ref_url] if ref_url else []) + [A._png_url(A.canvas_image(ink_before))]
            label = f"t18 {phase} {ref} A5 s{sample} r{r}"
            if r == 1:
                a = ask_round1(text, imgs, guard, label, len(A.PARTS[ref]))
                rec["layout"] = a["layout"]
            else:
                a = A.ask_program(text, imgs, A.MAX_TOKENS_ROUND, A.ROUND_MAX, guard, label)
            strokes = [o[1] for o in a["ops"] if o[0] == "draw"]
            idx = srv.call({"cmd": "append", "strokes_mm": strokes})["indices"] if strokes else []
            k, fin = 0, False
            for o in a["ops"]:
                if o[0] == "draw":
                    srv.call({"cmd": "do", "skill": f"draw({idx[k]})"})
                    k += 1
                elif o[0] == "lift":
                    srv.call({"cmd": "do", "skill": "lift"})
                else:
                    fin = True
                    break
            ink_after = srv.call({"cmd": "ink"})["ink_mm"]
            new = ink_after[len(ink_before):]
            used += len(strokes)
            rec["rounds"].append({"round": r, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                  "program": [list(o) for o in a["ops"]], "counts": a["counts"], "tries": a["tries"],
                                  "failed": a["failed"], "strokes": len(strokes), "finish": fin,
                                  "ink_in_own_box": round(in_box(new, rec["layout"][r - 1]), 4) if rec["layout"] and new else None})
            if fin and r < A.ROUNDS:
                early = True
                break
        srv.call({"cmd": "do", "skill": "finish"})
        v = srv.call({"cmd": "verify"})
        ink = srv.call({"cmd": "ink"})["ink_mm"]
    finally:
        srv.close()
    tries = [t for rr in rec["rounds"] for t in rr["tries"]]
    sc = A.scores(ref, ink, v, used)
    if rec["layout"]:
        sc["ink_in_own_box_mean"] = round(float(np.mean([rr["ink_in_own_box"] for rr in rec["rounds"] if rr["ink_in_own_box"] is not None])), 4)
        sc["box_overlap"] = overlaps(rec["layout"])
    rec.update(ok=True, strokes_used=used, early_finish=early, verify=v, ink_mm=ink, calls=len(tries),
               reprompts=sum(len(rr["tries"]) - 1 for rr in rec["rounds"]), truncations=sum(t["truncated"] for t in tries),
               failed_rounds=sum(rr["failed"] for rr in rec["rounds"]), tokens_in=sum(t.get("tokens_in") or 0 for t in tries),
               tokens_out=sum(t.get("tokens_out") or 0 for t in tries), cost_usd=round(sum(t.get("cost_usd") or 0 for t in tries), 6),
               scores=sc)
    return rec


def run(phase: str, refs: list[str], samples: list[int], workers: int = 3) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from . import transport_v13b  # noqa: F401
    from .spend18b import guard as mkguard

    out = R / f"study_a5_{phase}.jsonl"
    done = set()
    if out.exists():
        done = {(json.loads(x)["ref"], json.loads(x)["sample"]) for x in out.read_text().splitlines()}
    todo = [(ref, s) for ref in refs for s in samples if (ref, s) not in done]
    guard = mkguard()
    print(json.dumps({"grid": f"turn18 A5 {phase}", "drawings": len(todo), "headroom_usd": round(guard.headroom(), 3)}), flush=True)

    def one(item):
        ref, s = item
        seed = 18_000_000 + 1000 * (["mona_lisa", "starry_night", "great_wave", "statue_of_liberty"].index(ref)) + 400 + s + (50 if phase != "rated" else 0)
        try:
            return run_drawing(ref, s, guard, seed, phase)
        except Exception as ex:  # noqa: BLE001
            return {"ref": ref, "condition": "A5", "sample": s, "phase": phase, "ok": False, "error": f"{type(ex).__name__}: {ex}"}

    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, todo):
            dst = out if r.get("ok") else out.with_name(out.stem + "_transport_errors.jsonl")
            with open(dst, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(r["ref"], "A5", r["sample"], r.get("ok"), r.get("strokes_used"), r.get("truncations"), r.get("failed_rounds"),
                  r.get("cost_usd"), json.dumps({k: v for k, v in (r.get("scores") or {}).items() if k != "parts"}),
                  json.dumps(r.get("layout")), flush=True)


if __name__ == "__main__":
    import sys

    a = sys.argv
    run(a[1], a[2].split(","), [int(x) for x in a[3].split(",")], int(a[4]) if len(a) > 4 else 3)
