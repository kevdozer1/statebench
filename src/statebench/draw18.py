"""Turn 18 Study A: one drawing robot, four ways of giving it the job (the model draws; Sol).

**Frozen in every condition:** the robot and scene (``draw_scene18``: the Turn 15 scene, strokes appended to one page,
finish-lift from Turn 17), the executor (the draw, lift and finish skills), the canvas mapping from Turn 17's part study
(``parts17.fit``: the 90 x 90 mm canvas stands for the whole reference, 5 mm margins, proportions kept, centred; the
placement text states the reference's edges in mm), Sol's Turn 16 sampling settings (temperature 1.0, reasoning off,
structured output), and the total stroke allowance: **60 strokes, 40 points per stroke** (raised from Turn 16's 30).

**Conditions:**
* A1 one-shot: the reference (if any) and the whole-picture goal; the full program in one reply (at most 60 strokes).
* A2 iterative whole, A3 iterative parts, A4 iterative coarse to fine: 4 rounds; each round a fresh prompt built the same
  way: the reference, the overall goal, the stage instruction (A3: one part, in the declared order; A4: one level), the
  round number, the remaining stroke budget, and one image of the current canvas (a top-down rendering of the true ink
  on the 90 x 90 mm canvas). No previous programs, no conversation history.
* Rules: append-only; at most 15 strokes per round, unused strokes do not carry forward; a ``finish`` before round 4 ends
  the drawing (recorded); a malformed or truncated reply gets one re-prompt, and if that fails the round draws nothing
  (the same for A1); the output limit is ``MAX_TOKENS_ROUND`` for every A2-A4 call and ``MAX_TOKENS_A1`` for A1 (large
  enough for a full 60-stroke program).
* After the last round (or an early finish), the robot's finish skill runs (finish-lift).

**Goals.** The whole-picture goal is Turn 16's D3 text adapted to 60 strokes (``WHOLE``). The Statue of Liberty (Phase 1)
is given in words only: "Draw the Statue of Liberty." (no reference image). Part lists and levels: ``PARTS``, ``LEVELS``.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np

from . import draws16 as D
from . import parts17 as P17
from . import tasks15 as T
from .config import RUNS

R = RUNS / "turn18"
SOL = "openai/gpt-6-sol"
PRICE = (2e-6, 1e-5)
TEMPERATURE = 1.0
CANVAS_MM = 90.0
MAX_TOTAL, MAX_POINTS, ROUND_MAX, ROUNDS = 60, 40, 15, 4
MAX_TOKENS_A1, MAX_TOKENS_ROUND = 24000, 8000

WHOLE = ("Draw this image as a line drawing: its main outlines, all inside the drawing area, in at most 60 strokes, with "
         "no stray marks, and lift the pen at the end, before you finish.")
STATUE_GOAL = "Draw the Statue of Liberty."
PARTS = {
    "mona_lisa": ["the head and hair", "the face (eyes, nose, mouth)", "the shoulders, body and folded hands",
                  "the landscape behind her"],
    "starry_night": ["the tall cypress tree", "the swirling sky", "the moon and the stars",
                     "the hills, the village and the church spire"],
    "great_wave": ["the big wave", "the smaller waves and the boats", "the mountain", "the sky and the text block"],
    "statue_of_liberty": ["the torch and raised arm", "the crowned head", "the robe and tablet", "the pedestal"],
}
LEVELS = ["the main outline", "the large inner shapes", "the medium details", "the fine details"]
CONDITIONS = ("A1", "A2", "A3", "A4")

HEADER = (
    "You control a robot arm holding a pen above a sheet of paper. The drawing area is a square, 90 mm by 90 mm. "
    "Coordinates are in millimetres, with the origin at the lower-left corner of the drawing area, x to the right and "
    "y up. You write a program for the robot as one JSON object {\"actions\": [...]}. Each action is one of:\n"
    "- {\"type\": \"draw\", \"points\": [[x, y], ...]}: the robot moves to the first point, puts the pen down, and "
    "draws a line through the points in order; the pen stays down at the end of the line;\n"
    "- {\"type\": \"lift\", \"points\": []}: the robot raises the pen;\n"
    "- {\"type\": \"finish\", \"points\": []}: the program ends.\n"
    "At most {N} draw actions in this reply, at most 40 points in each, and every point inside the drawing area. "
    "Answer with the JSON object only.\n\n")
STATUE_PLACEMENT = "Keep the whole drawing inside the drawing area, with a 5 mm margin on every side."
ROUND_TEXT = (
    "This drawing is made in 4 rounds, and the whole drawing may use at most 60 strokes. This is round {r} of 4. In "
    "this round you may add at most 15 strokes; strokes you do not use now are not carried forward. After this round, "
    "{left} round(s) remain. {canvas_line} Strokes are only added: nothing can be erased or replaced. Add the strokes "
    "for this round. Use finish only if the drawing is complete; otherwise do not use finish.")
CANVAS_LINE_REF = ("The first image is the reference. The second image shows the drawing area as it is now, seen from "
                   "above: the ink drawn so far (blank before the first round).")
CANVAS_LINE_WORDS = ("The image shows the drawing area as it is now, seen from above: the ink drawn so far (blank before "
                     "the first round).")
STAGE_PARTS = ("The drawing is made part by part, one part per round, in this order: {order}. In this round, draw part "
               "{r}: {name}.")
STAGE_LEVELS = ("The drawing is made from coarse to fine, one level per round, in this order: {order}. In this round, "
                "draw level {r}: {name}.")
RETRY = ("Your previous answer was cut off or was not one valid JSON object. Answer again with one complete JSON object, "
         "using fewer points if needed.")


def has_reference(ref: str) -> bool:
    return ref != "statue_of_liberty"


def goal_text(ref: str) -> str:
    return WHOLE if has_reference(ref) else STATUE_GOAL


def placement(ref: str) -> str:
    return P17.placement(ref) if has_reference(ref) else STATUE_PLACEMENT


def prompt_a1(ref: str) -> str:
    return HEADER.replace("{N}", str(MAX_TOTAL)) + placement(ref) + "\n\n" + goal_text(ref)


def prompt_round(ref: str, cond: str, r: int) -> str:
    stage = ""
    if cond == "A3":
        order = "; ".join(f"{i + 1}. {p}" for i, p in enumerate(PARTS[ref]))
        stage = STAGE_PARTS.format(order=order, r=r, name=PARTS[ref][r - 1])
    elif cond == "A4":
        order = "; ".join(f"{i + 1}. {p}" for i, p in enumerate(LEVELS))
        stage = STAGE_LEVELS.format(order=order, r=r, name=LEVELS[r - 1])
    rt = ROUND_TEXT.format(r=r, left=ROUNDS - r, canvas_line=CANVAS_LINE_REF if has_reference(ref) else CANVAS_LINE_WORDS)
    return (HEADER.replace("{N}", str(ROUND_MAX)) + placement(ref) + "\n\n" + goal_text(ref) + "\n\n" +
            (stage + "\n\n" if stage else "") + rt)


def prompt_hashes() -> dict:
    """Normalized hashes of the 4 condition templates (placeholders unfilled) and of this file."""
    from .hashing import REPO_ROOT, normalized_sha256

    tmpl = {"A1": HEADER + "{placement}\n\n{goal}",
            "A2": HEADER + "{placement}\n\n{goal}\n\n" + ROUND_TEXT,
            "A3": HEADER + "{placement}\n\n{goal}\n\n" + STAGE_PARTS + "\n\n" + ROUND_TEXT,
            "A4": HEADER + "{placement}\n\n{goal}\n\n" + STAGE_LEVELS + "\n\n" + ROUND_TEXT}
    out = {k: hashlib.sha256(v.replace("\r\n", "\n").encode()).hexdigest() for k, v in tmpl.items()}
    out["draw18.py"] = normalized_sha256(REPO_ROOT / "src/statebench/draw18.py")
    return out


# ------------------------------------------------------------------ images
def _png_url(im) -> str:
    b = BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def canvas_image(ink_mm: list, size: int = 512):
    """The drawing area seen from above: the true ink so far on the 90 x 90 mm canvas."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (size, size), (252, 251, 246))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, size - 1, size - 1), outline=(180, 180, 180), width=2)
    for x, y in ink_mm:
        px, py = x / CANVAS_MM * (size - 1), (1 - y / CANVAS_MM) * (size - 1)
        d.ellipse((px - 1.2, py - 1.2, px + 1.2, py + 1.2), fill=(25, 25, 30))
    return im


# ------------------------------------------------------------------ the model call
def ask(text: str, images: list, max_tokens: int, guard, label: str) -> dict:
    from .backends.base import load_secret
    from .backends.http import post_json

    content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u, "detail": "auto"}} for u in images]
    body = {"model": SOL, "temperature": TEMPERATURE, "max_tokens": max_tokens, "usage": {"include": True},
            "reasoning": {"enabled": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "pen_program", "strict": True, "schema": D.SCHEMA}},
            "messages": [{"role": "user", "content": content}]}
    rid = guard.reserve((len(text) // 2 + 1200 * len(images) + 200) * PRICE[0] + max_tokens * PRICE[1], label)
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
        content_out = raw["choices"][0]["message"]["content"]
        finish = raw["choices"][0].get("finish_reason")
    except Exception:  # noqa: BLE001
        content_out, finish = None, None
    return {"ok": True, "content": content_out, "finish_reason": finish, "tokens_in": u.get("prompt_tokens"),
            "tokens_out": u.get("completion_tokens"), "cost_usd": cost, "latency_s": round(time.perf_counter() - t0, 2)}


def parse(content: str | None, max_strokes: int) -> tuple[list, dict]:
    """The Turn 16 parser (``draws16.parse_program``) with the Turn 18 limits (max_strokes per reply, 40 points)."""
    old = (D.MAX_STROKES, D.MAX_POINTS)
    D.MAX_STROKES, D.MAX_POINTS = max_strokes, MAX_POINTS
    try:
        return D.parse_program(content)
    finally:
        D.MAX_STROKES, D.MAX_POINTS = old


def ask_program(text: str, images: list, max_tokens: int, max_strokes: int, guard, label: str) -> dict:
    """One call, and one re-prompt if the reply is truncated or unparseable. Transport failures raise."""
    tries = []
    for attempt in range(2):
        q = text if attempt == 0 else text + "\n\n" + RETRY
        a = ask(q, images, max_tokens, guard, f"{label} a{attempt}")
        if not a.get("ok"):
            raise RuntimeError(a["error"])
        ops, counts = parse(a.get("content"), max_strokes)
        bad = counts["unparseable"] or a.get("finish_reason") == "length"
        tries.append({k: v for k, v in a.items() if k != "ok"} | {"truncated": a.get("finish_reason") == "length",
                                                                   "unparseable": bool(counts["unparseable"])})
        if not bad:
            return {"ops": ops, "counts": counts, "tries": tries, "failed": False}
    return {"ops": [], "counts": counts, "tries": tries, "failed": True}


# ------------------------------------------------------------------ one drawing
def run_drawing(ref: str, cond: str, sample: int, guard, seed: int, phase: str) -> dict:
    ref_url = D._data_url(D.ref_image(ref)) if has_reference(ref) else None
    srv = T.SceneServer("draw_scene18.py")
    rec = {"ref": ref, "condition": cond, "sample": sample, "phase": phase, "seed": seed, "rounds": []}
    used, early = 0, False
    try:
        srv.call({"cmd": "reset", "seed": seed})
        n_rounds = 1 if cond == "A1" else ROUNDS
        for r in range(1, n_rounds + 1):
            if cond == "A1":
                text, imgs, mt, ms = prompt_a1(ref), ([ref_url] if ref_url else []), MAX_TOKENS_A1, MAX_TOTAL
            else:
                ink = srv.call({"cmd": "ink"})["ink_mm"]
                text = prompt_round(ref, cond, r)
                imgs = ([ref_url] if ref_url else []) + [_png_url(canvas_image(ink))]
                mt, ms = MAX_TOKENS_ROUND, ROUND_MAX
            a = ask_program(text, imgs, mt, ms, guard, f"t18 {phase} {ref} {cond} s{sample} r{r}")
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
            used += len(strokes)
            rec["rounds"].append({"round": r, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                  "program": [list(o) for o in a["ops"]], "counts": a["counts"], "tries": a["tries"],
                                  "failed": a["failed"], "strokes": len(strokes), "finish": fin})
            if fin and r < n_rounds:
                early = True
                break
        srv.call({"cmd": "do", "skill": "finish"})
        v = srv.call({"cmd": "verify"})
        ink = srv.call({"cmd": "ink"})["ink_mm"]
    finally:
        srv.close()
    tries = [t for rr in rec["rounds"] for t in rr["tries"]]
    rec.update(ok=True, strokes_used=used, early_finish=early, verify=v, ink_mm=ink,
               calls=len(tries), reprompts=sum(len(rr["tries"]) - 1 for rr in rec["rounds"]),
               truncations=sum(t["truncated"] for t in tries), failed_rounds=sum(rr["failed"] for rr in rec["rounds"]),
               tokens_in=sum(t.get("tokens_in") or 0 for t in tries), tokens_out=sum(t.get("tokens_out") or 0 for t in tries),
               cost_usd=round(sum(t.get("cost_usd") or 0 for t in tries), 6), scores=scores(ref, ink, v, used))
    return rec


def scores(ref: str, ink: list, verify: dict, used: int) -> dict:
    out = {"strokes_used": used, "pen_lifted": bool(verify.get("pen_lifted")) and verify.get("error") is None, "n_ink": len(ink)}
    if not has_reference(ref):
        return out
    s16 = D.score(ink, ref, {"points_off_canvas": 0, "points_total": 1}, verify)
    out.update(chamfer=s16["chamfer"], coverage=s16["coverage"], stray=s16["stray"])
    if ref in P17.PARTS:
        ps = P17.part_scores(ref, ink, verify)
        out["parts"] = {k: v for k, v in ps.items() if k.startswith(("in_", "cov_"))}
    return out


def run(phase: str, refs: list[str], conds: list[str], samples: list[int], workers: int = 3) -> None:
    """One Sol process; drawings in up to ``workers`` threads; file runs/turn18/study_a_<phase>.jsonl."""
    from concurrent.futures import ThreadPoolExecutor

    from . import transport_v13b  # noqa: F401
    from .spend18 import guard as mkguard

    out = R / f"study_a_{phase}.jsonl"
    done = set()
    if out.exists():
        done = {(json.loads(x)["ref"], json.loads(x)["condition"], json.loads(x)["sample"]) for x in out.read_text().splitlines()}
    todo = [(ref, c, s) for ref in refs for c in conds for s in samples if (ref, c, s) not in done]
    guard = mkguard()
    print(json.dumps({"grid": f"turn18 study A {phase}", "drawings": len(todo), "headroom_usd": round(guard.headroom(), 3)}), flush=True)

    def one(item):
        ref, c, s = item
        seed = 18_000_000 + 1000 * (["mona_lisa", "starry_night", "great_wave", "statue_of_liberty"].index(ref)) + 100 * CONDITIONS.index(c) + s + (50 if phase != "rated" else 0)
        try:
            return run_drawing(ref, c, s, guard, seed, phase)
        except Exception as ex:  # noqa: BLE001
            return {"ref": ref, "condition": c, "sample": s, "phase": phase, "ok": False, "error": f"{type(ex).__name__}: {ex}"}

    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, todo):
            dst = out if r.get("ok") else out.with_name(out.stem + "_transport_errors.jsonl")
            with open(dst, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(r["ref"], r["condition"], r["sample"], r.get("ok"), r.get("strokes_used"), r.get("truncations"),
                  r.get("early_finish"), r.get("cost_usd"), json.dumps({k: v for k, v in (r.get("scores") or {}).items() if k != "parts"}),
                  flush=True)


if __name__ == "__main__":
    import sys

    a = sys.argv
    run(a[1], a[2].split(","), a[3].split(","), [int(x) for x in a[4].split(",")], int(a[5]) if len(a) > 5 else 3)
