"""The System 1/System 2 diagram used in the paper and the blog post.

    python paper/figures/make_diagrams.py            (Pillow only)

The structured-state figure is hand-written SVG (structured_state.svg), rendered at 1.5x in headless Chrome and saved as
structured_state.jpg.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
INK = (25, 25, 30)
GREY = (140, 140, 140)
ACCENT = (40, 70, 140)


def font(size: int, bold: bool = False):
    for f in (("arialbd.ttf", "segoeuib.ttf") if bold else ("arial.ttf", "segoeui.ttf")):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def arrow(d, a, b, w=4, col=INK, dashed=False, head=16):
    if dashed:
        L = math.dist(a, b)
        n = max(1, int(L // 18))
        for i in range(0, n, 2):
            p = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
            q = (a[0] + (b[0] - a[0]) * min(i + 1, n) / n, a[1] + (b[1] - a[1]) * min(i + 1, n) / n)
            d.line([p, q], fill=col, width=w)
    else:
        d.line([a, b], fill=col, width=w)
    if head:
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        d.polygon([b, (b[0] - head * math.cos(ang - 0.4), b[1] - head * math.sin(ang - 0.4)),
                   (b[0] - head * math.cos(ang + 0.4), b[1] - head * math.sin(ang + 0.4))], fill=col)


def robot_scene(d, box):
    """A small robot arm at a table with a cube and a tray, inside ``box``."""
    x0, y0, x1, y1 = box
    table = y1 - 44
    d.line([(x0 + 22, table), (x1 - 22, table)], fill=INK, width=4)
    # tray: an open box
    tx0, tx1 = x1 - 132, x1 - 40
    d.rectangle((tx0, table - 34, tx1, table), fill=(214, 228, 247))
    d.line([(tx0, table - 34), (tx0, table), (tx1, table), (tx1, table - 34)], fill=ACCENT, width=5)
    # cube
    cx = x0 + 150
    d.rectangle((cx, table - 30, cx + 30, table), fill=(210, 60, 55), outline=INK, width=2)
    # arm: base, two links, wrist, two fingers
    bx = x0 + 62
    d.rounded_rectangle((bx - 26, table - 40, bx + 26, table), radius=6, fill=(70, 72, 80))
    j1, j2 = (bx, table - 40), (bx + 40, y0 + 78)
    j3 = (x0 + 238, y0 + 70)
    wrist = (j3[0], j3[1] + 58)
    for a, b in ((j1, j2), (j2, j3), (j3, wrist)):
        d.line([a, b], fill=(95, 98, 108), width=16)
    for p in (j2, j3):
        d.ellipse((p[0] - 12, p[1] - 12, p[0] + 12, p[1] + 12), fill=(60, 62, 70))
    fx, fy = wrist
    d.line([(fx - 16, fy), (fx + 16, fy)], fill=(60, 62, 70), width=8)
    d.line([(fx - 14, fy), (fx - 14, fy + 26)], fill=(60, 62, 70), width=7)
    d.line([(fx + 14, fy), (fx + 14, fy + 26)], fill=(60, 62, 70), width=7)


def trim(im, pad=30):
    bg = Image.new("RGB", im.size, (255, 255, 255))
    x0, y0, x1, y1 = ImageChops.difference(im, bg).getbbox()
    return im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))


def system1_system2() -> Path:
    W, H = 1840, 700
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    fb, fs = font(40, bold=True), font(26)
    row0, row1 = 230, 370
    boxes = {"Sol": (60, row0, 300, row1), "state": (480, row0, 820, row1), "Jev": (1000, row0, 1240, row1)}
    scene = (1420, 170, 1780, 430)
    for k, (x0, y0, x1, y1) in boxes.items():
        blue = k in ("Sol", "Jev")
        d.rounded_rectangle((x0, y0, x1, y1), radius=22, fill=ACCENT if blue else (255, 255, 255),
                            outline=ACCENT if blue else INK, width=4)
        name = {"Sol": "Sol", "state": "structured state", "Jev": "Jev"}[k]
        f = fb if blue else font(32)
        tw = d.textlength(name, font=f)
        d.text(((x0 + x1 - tw) / 2, (y0 + y1) / 2 - 22), name, fill=(255, 255, 255) if blue else INK, font=f)
    d.rounded_rectangle(scene, radius=22, fill=(255, 255, 255), outline=INK, width=4)
    d.text((scene[0] + 22, scene[1] + 14), "robot in the scene", fill=GREY, font=font(24))
    robot_scene(d, (scene[0], scene[1] + 30, scene[2], scene[3]))
    mid = (row0 + row1) // 2
    for a, b, lab in (("Sol", "state", "writes plan"), ("state", "Jev", "reads state"), ("Jev", None, "next action")):
        x0 = boxes[a][2] + 8
        x1 = (boxes[b][0] if b else scene[0]) - 8
        arrow(d, (x0, mid), (x1, mid))
        tw = d.textlength(lab, font=fs)
        d.text(((x0 + x1 - tw) / 2, mid - 46), lab, fill=INK, font=fs)
    # the scene feeds back into the structured state, under the row
    bottom = 590
    sx, stx = (scene[0] + scene[2]) // 2, (boxes["state"][0] + boxes["state"][2]) // 2
    arrow(d, (sx, scene[3] + 8), (sx, bottom), head=0)
    arrow(d, (sx, bottom), (stx, bottom), head=0)
    arrow(d, (stx, bottom), (stx, row1 + 10))
    lab = "the scene updates the state"
    tw = d.textlength(lab, font=fs)
    d.text(((sx + stx - tw) / 2, bottom + 14), lab, fill=INK, font=fs)
    # Jev reports to Sol only when a step fails, over the top
    jx, solx = (boxes["Jev"][0] + boxes["Jev"][2]) / 2, (boxes["Sol"][0] + boxes["Sol"][2]) / 2
    top = 90
    arrow(d, (jx, row0 - 8), (jx, top), dashed=True, head=0)
    arrow(d, (jx, top), (solx, top), dashed=True, head=0)
    arrow(d, (solx, top), (solx, row0 - 8), dashed=True)
    lab = "reports only when a step along the plan fails"
    d.text(((jx + solx - d.textlength(lab, font=fs)) / 2, top - 44), lab, fill=INK, font=fs)
    for i, ln in enumerate(("~0.2 s per decision", "5,294 runs for $3.43")):
        tw = d.textlength(ln, font=fs)
        d.text((jx - tw / 2, row1 + 22 + 36 * i), ln, fill=INK, font=fs)
    p = OUT / "system1_system2.png"
    trim(im).save(p, optimize=True)
    print("wrote", p)
    return p


if __name__ == "__main__":
    system1_system2()
