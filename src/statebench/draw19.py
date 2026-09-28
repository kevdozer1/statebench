"""Turn 19 Phase 4: why do rounds hurt? Registration or permanence.

In Turn 18, the iterative conditions saw only a photo of the canvas between rounds, and ink is permanent; Kevin rated
them below one-shot. Two explanations, declared: **registration** (Sol cannot place new strokes relative to old ones from
a photo) and **permanence** (mistakes cannot be undone).

**Conditions** (all share Turn 18's frozen robot and scene ``draw_scene18``, executor, finish-lift, canvas mapping,
sampling settings, 60-stroke allowance and output-token limits):
* **A1 one-shot:** Turn 18's A1 prompt (``draw18.prompt_a1``), 1 call.
* **A2 rounds:** Turn 18's A2 (``draw18.prompt_round``): 4 rounds, at most 15 strokes each, append-only, the reference and
  a photo (top-down image) of the canvas.
* **A2c rounds with coordinates:** A2, plus every earlier stroke given as its coordinates each round (``COORDS``).
* **R draft, revise, commit:** round 1 writes a full program (a draft, at most 60 strokes); rounds 2-4 see the rendered
  draft (its polylines drawn on the canvas image) and its program as coordinates, and return a full replacement program.
  Nothing is drawn until the end: the robot draws only the final program. A finish in a reply before round 4 makes that
  program final (recorded).
* A2, A2c and R share the same header, placement, goal and round framing; only the defining paragraphs differ
  (``COORDS`` for A2c; ``R_ROUND1`` and ``R_REVISE`` with their own header limit for R).
* **Rules:** a malformed or truncated reply gets one re-prompt; if that fails, the round is kept as the previous state (A2
  and A2c: the round adds nothing; R: the draft stays as it was).
* **Output-token limits,** Turn 18's by call type: a call that returns a full program (A1, every R round) 24,000; a call
  that adds up to 15 strokes (A2, A2c) 8,000.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import draw18 as A
from . import draws16 as D
from . import tasks15 as T
from .config import RUNS

R = RUNS / "turn19"
CONDITIONS = ("A1", "A2", "A2c", "R")
COORDS = ("The strokes already on the drawing area, as their coordinates in millimetres, in the order they were drawn "
          "(the same coordinates you write): {coords}")
R_ROUND1 = ("This drawing is made as a draft that you revise before the robot draws it. There are 4 rounds. In round 1 you "
            "write the whole drawing as a draft program; in rounds 2 to 4 you see the draft and return a full replacement "
            "program. Nothing is drawn until the end: the robot draws only the final program. This is round 1 of 4. Write "
            "the whole drawing now. Use finish only if the draft is final and needs no revision; otherwise do not use "
            "finish.")
R_REVISE = ("This drawing is made as a draft that you revise before the robot draws it. There are 4 rounds; nothing is drawn "
            "until the end, and the robot draws only the final program. This is round {r} of 4; after this round, {left} "
            "round(s) remain. The second image shows the current draft rendered on the drawing area, seen from above. Its "
            "program, as coordinates in millimetres: {coords}. Return a full replacement program: the complete drawing, "
            "improved (you may keep, change or remove any stroke). Use finish only if the drawing is final and needs no "
            "more revision; otherwise do not use finish.")
MAX_TOKENS_FULL, MAX_TOKENS_ADD = A.MAX_TOKENS_A1, A.MAX_TOKENS_ROUND


def _coords(strokes: list) -> str:
    return json.dumps([[[round(x, 1), round(y, 1)] for x, y in s] for s in strokes], separators=(",", ":"))


def prompt(ref: str, cond: str, r: int, strokes: list) -> str:
    if cond == "A1":
        return A.prompt_a1(ref)
    if cond == "A2":
        return A.prompt_round(ref, "A2", r)
    if cond == "A2c":
        return A.prompt_round(ref, "A2", r) + "\n\n" + COORDS.format(coords=_coords(strokes) if strokes else "none yet")
    head = A.HEADER.replace("{N}", str(A.MAX_TOTAL)) + A.placement(ref) + "\n\n" + A.goal_text(ref) + "\n\n"
    if r == 1:
        return head + R_ROUND1
    return head + R_REVISE.format(r=r, left=4 - r, coords=_coords(strokes) if strokes else "no strokes")


def prompt_hashes() -> dict:
    from .hashing import REPO_ROOT, normalized_sha256

    t = {"A1": A.HEADER + "{placement}\n\n{goal}", "A2": A.HEADER + "{placement}\n\n{goal}\n\n" + A.ROUND_TEXT,
         "A2c": A.HEADER + "{placement}\n\n{goal}\n\n" + A.ROUND_TEXT + "\n\n" + COORDS,
         "R": A.HEADER + "{placement}\n\n{goal}\n\n" + R_ROUND1 + "\n\n" + R_REVISE}
    out = {k: hashlib.sha256(v.replace("\r\n", "\n").encode()).hexdigest() for k, v in t.items()}
    out["draw19.py"] = normalized_sha256(REPO_ROOT / "src/statebench/draw19.py")
    return out


def draft_image(strokes: list, size: int = 512):
    """The draft rendered on the drawing area: its polylines as lines (nothing drawn by the robot yet)."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (size, size), (252, 251, 246))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, size - 1, size - 1), outline=(180, 180, 180), width=2)
    for s in strokes:
        pts = [(x / A.CANVAS_MM * (size - 1), (1 - y / A.CANVAS_MM) * (size - 1)) for x, y in s]
        if len(pts) > 1:
            d.line(pts, fill=(25, 25, 30), width=3)
    return im


def run_drawing(ref: str, cond: str, sample: int, guard, seed: int, phase: str) -> dict:
    ref_url = D._data_url(D.ref_image(ref))
    srv = T.SceneServer("draw_scene18.py")
    rec = {"ref": ref, "condition": cond, "sample": sample, "phase": phase, "seed": seed, "rounds": []}
    early = False
    try:
        srv.call({"cmd": "reset", "seed": seed})
        if cond == "R":
            draft: list = []  # the current draft program's ops
            for r in range(1, 5):
                strokes = [o[1] for o in draft if o[0] == "draw"]
                imgs = [ref_url] if r == 1 else [ref_url, A._png_url(draft_image(strokes))]  # round 1 has no draft yet
                text = prompt(ref, "R", r, strokes)
                a = A.ask_program(text, imgs, MAX_TOKENS_FULL, A.MAX_TOTAL, guard, f"t19 {phase} {ref} R s{sample} r{r}")
                if not a["failed"]:
                    draft = [o for o in a["ops"] if o[0] != "finish"]
                fin = any(o[0] == "finish" for o in a["ops"]) and not a["failed"]
                rec["rounds"].append({"round": r, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                      "program": [list(o) for o in a["ops"]], "counts": a["counts"], "tries": a["tries"],
                                      "failed": a["failed"], "strokes": sum(o[0] == "draw" for o in a["ops"]), "finish": fin})
                if fin and r < 4:
                    early = True
                    break
            final = draft
            strokes = [o[1] for o in final if o[0] == "draw"]
            idx = srv.call({"cmd": "append", "strokes_mm": strokes})["indices"] if strokes else []
            k = 0
            for o in final:
                if o[0] == "draw":
                    srv.call({"cmd": "do", "skill": f"draw({idx[k]})"})
                    k += 1
                elif o[0] == "lift":
                    srv.call({"cmd": "do", "skill": "lift"})
            rec["final_program"] = [list(o) for o in final]
            used = len(strokes)
        else:
            used = 0
            n_rounds = 1 if cond == "A1" else 4
            drawn: list = []
            for r in range(1, n_rounds + 1):
                if cond == "A1":
                    text, imgs, mt, ms = prompt(ref, "A1", 1, []), [ref_url], MAX_TOKENS_FULL, A.MAX_TOTAL
                else:
                    ink = srv.call({"cmd": "ink"})["ink_mm"]
                    text = prompt(ref, cond, r, drawn)
                    imgs, mt, ms = [ref_url, A._png_url(A.canvas_image(ink))], MAX_TOKENS_ADD, A.ROUND_MAX
                a = A.ask_program(text, imgs, mt, ms, guard, f"t19 {phase} {ref} {cond} s{sample} r{r}")
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
                drawn += strokes
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
    rec.update(ok=True, strokes_used=used, early_finish=early, verify=v, ink_mm=ink, calls=len(tries),
               reprompts=sum(len(rr["tries"]) - 1 for rr in rec["rounds"]), truncations=sum(t["truncated"] for t in tries),
               failed_rounds=sum(rr["failed"] for rr in rec["rounds"]), tokens_in=sum(t.get("tokens_in") or 0 for t in tries),
               tokens_out=sum(t.get("tokens_out") or 0 for t in tries), cost_usd=round(sum(t.get("cost_usd") or 0 for t in tries), 6),
               scores=A.scores(ref, ink, v, used))
    return rec


def rounds_for_video(rec: dict) -> list[list]:
    """What the robot drew, as rounds of ops: R draws only its final program (one round)."""
    if rec["condition"] == "R":
        return [[tuple(o) for o in rec["final_program"]]]
    return [[tuple(o) for o in r["program"]] for r in rec["rounds"]]


def run(phase: str, refs: list[str], conds: list[str], samples: list[int], workers: int = 4) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from . import transport_v13b  # noqa: F401
    from .spend19 import guard as mkguard

    out = R / f"study_{phase}.jsonl"
    R.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(json.loads(x)["ref"], json.loads(x)["condition"], json.loads(x)["sample"]) for x in out.read_text().splitlines()}
    todo = [(ref, c, s) for ref in refs for c in conds for s in samples if (ref, c, s) not in done]
    guard = mkguard()
    print(json.dumps({"grid": f"turn19 phase 4 {phase}", "drawings": len(todo), "headroom_usd": round(guard.headroom(), 3)}), flush=True)

    def one(item):
        ref, c, s = item
        seed = 19_000_000 + 1000 * ["mona_lisa", "starry_night", "great_wave"].index(ref) + 100 * CONDITIONS.index(c) + s + (50 if phase != "rated" else 0)
        try:
            return run_drawing(ref, c, s, guard, seed, phase)
        except Exception as ex:  # noqa: BLE001
            return {"ref": ref, "condition": c, "sample": s, "phase": phase, "ok": False, "error": f"{type(ex).__name__}: {ex}"}

    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, todo):
            dst = out if r.get("ok") else out.with_name(out.stem + "_transport_errors.jsonl")
            with open(dst, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(r["ref"], r["condition"], r["sample"], r.get("ok"), r.get("strokes_used"), r.get("truncations"), r.get("failed_rounds"),
                  r.get("early_finish"), r.get("cost_usd"), json.dumps({k: v for k, v in (r.get("scores") or {}).items() if k != "parts"}),
                  r.get("error", "")[:200], flush=True)


if __name__ == "__main__":
    import sys

    a = sys.argv
    run(a[1], a[2].split(","), a[3].split(","), [int(x) for x in a[4].split(",")], int(a[5]) if len(a) > 5 else 4)
