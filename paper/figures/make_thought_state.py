"""Frutiger Aero figure: the structured state before trimming, a thought cloud holding every field we tried.

    python paper/figures/make_thought_state.py        -> state_before_trimming.svg
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "state_before_trimming.svg"
W, H = 1600, 980
DY = 40
FONT = "Segoe UI, Lucida Grande, Helvetica Neue, Arial, sans-serif"

# the cloud: a union of circles (outlined once, filled once, so only the outer edge shows)
CLOUD = [(430, 330, 215), (700, 250, 225), (985, 250, 220), (1245, 330, 205), (1300, 545, 200), (1085, 675, 200),
         (790, 690, 205), (500, 650, 200), (330, 490, 180), (800, 470, 300), (1060, 460, 260), (560, 460, 250)]

# kind -> (inner colour, outer colour, text colour)
KIND = {
    "core": ("#fffaf0", "#f6c177", "#5a3a12"),
    "meaning": ("#ffffff", "#bca8f5", "#3b2a6b"),
    "procedure": ("#ffffff", "#f8cda4", "#6b3d14"),
    "history": ("#ffffff", "#e6b6e7", "#5a2d5e"),
}
# (x, y, r, kind, lines)
BUBBLES = [
    (545, 330, 96, "core", ["tracking", "data"]),
    (800, 300, 96, "core", ["action", "menu"]),
    (1055, 330, 96, "core", ["goal"]),
    (1250, 470, 64, "meaning", ["contact"]),
    (1385, 590, 66, "meaning", ["grip", "state"]),
    (1240, 640, 72, "meaning", ["moving", "with gripper"]),
    (1090, 555, 66, "meaning", ["in the", "tray"]),
    (1090, 715, 64, "meaning", ["released"]),
    (385, 520, 72, "procedure", ["lined up", "to grasp"]),
    (555, 610, 78, "procedure", ["estimate", "out of date"]),
    (390, 700, 64, "procedure", ["grab", "attempts"]),
    (800, 555, 70, "history", ["tracking", "history"]),
    (930, 690, 64, "history", ["last", "actions"]),
    (725, 720, 62, "history", ["calibration"]),
]
GROUPS = [("meaning labels", 1330, 405), ("procedure fields", 305, 420), ("history", 800, 460)]


def bubble(i, x, y, r, kind, lines):
    y += DY
    inner, outer, ink = KIND[kind]
    fs = 30 if kind == "core" else 22 if max(len(s) for s in lines) > 9 else 24
    lh = fs * 1.15
    y0 = y - (len(lines) - 1) * lh / 2 + fs * 0.35
    text = "".join(f'<text x="{x}" y="{y0 + k * lh:.1f}" text-anchor="middle" font-size="{fs}" '
                   f'font-weight="{600 if kind == "core" else 500}" fill="{ink}">{s}</text>' for k, s in enumerate(lines))
    return f'''
  <radialGradient id="b{i}" cx="0.4" cy="0.32" r="0.75">
    <stop offset="0" stop-color="{inner}"/><stop offset="0.55" stop-color="{inner}"/><stop offset="1" stop-color="{outer}"/>
  </radialGradient>
  <g filter="url(#soft)"><circle cx="{x}" cy="{y}" r="{r}" fill="url(#b{i})" stroke="#ffffff" stroke-width="4"/></g>
  <ellipse cx="{x - r * 0.12}" cy="{y - r * 0.52}" rx="{r * 0.62}" ry="{r * 0.28}" fill="url(#gloss)" opacity="0.85"/>
  {text}'''


def main():
    cloud_outline = "".join(f'<circle cx="{x}" cy="{y + DY}" r="{r}"/>' for x, y, r in CLOUD)
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#f3efff"/><stop offset="0.55" stop-color="#fbf8ff"/><stop offset="1" stop-color="#fff5e4"/>
  </linearGradient>
  <radialGradient id="cloud" gradientUnits="userSpaceOnUse" cx="760" cy="370" r="820">
    <stop offset="0" stop-color="#ffffff"/><stop offset="0.45" stop-color="#efe9fd"/><stop offset="1" stop-color="#c3b3f3"/>
  </radialGradient>
  <linearGradient id="gloss" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#ffffff" stop-opacity="0.95"/><stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
  </linearGradient>
  <radialGradient id="bubble" cx="0.35" cy="0.3" r="0.7">
    <stop offset="0" stop-color="#ffffff" stop-opacity="0.95"/><stop offset="0.6" stop-color="#efe8ff" stop-opacity="0.35"/>
    <stop offset="1" stop-color="#ffe3b8" stop-opacity="0.2"/>
  </radialGradient>
  <linearGradient id="chip" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#cfc2fb"/><stop offset="0.5" stop-color="#a893f0"/><stop offset="1" stop-color="#7f66d9"/>
  </linearGradient>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">
    <feDropShadow dx="0" dy="16" stdDeviation="20" flood-color="#6a52c4" flood-opacity="0.22"/>
  </filter>
  <filter id="soft" x="-30%" y="-30%" width="160%" height="170%">
    <feDropShadow dx="0" dy="5" stdDeviation="6" flood-color="#6a52c4" flood-opacity="0.2"/>
  </filter>
  <clipPath id="cloudclip">{cloud_outline}</clipPath>
</defs>
<rect width="{W}" height="{H}" fill="url(#bg)"/>
<circle cx="1520" cy="110" r="40" fill="url(#bubble)" stroke="#fff" stroke-opacity="0.8" stroke-width="2"/>
<circle cx="1455" cy="70" r="15" fill="url(#bubble)" stroke="#fff" stroke-opacity="0.8" stroke-width="2"/>
<circle cx="1530" cy="900" r="24" fill="url(#bubble)" stroke="#fff" stroke-opacity="0.8" stroke-width="2"/>
<!-- the thought cloud: outline pass, then fill pass on top so inner seams vanish -->
<g filter="url(#shadow)"><g fill="#ffffff" stroke="#ffffff" stroke-width="10">{cloud_outline}</g></g>
<g fill="url(#cloud)">{cloud_outline}</g>
<g clip-path="url(#cloudclip)"><ellipse cx="800" cy="135" rx="620" ry="120" fill="url(#gloss)" opacity="0.7"/></g>
<!-- thought trail down to the model -->
<circle cx="205" cy="815" r="30" fill="url(#cloud)" stroke="#ffffff" stroke-width="5" filter="url(#soft)"/>
<circle cx="148" cy="872" r="18" fill="url(#cloud)" stroke="#ffffff" stroke-width="4" filter="url(#soft)"/>
<g filter="url(#soft)"><rect x="40" y="895" width="120" height="52" rx="26" fill="url(#chip)" stroke="#ffffff" stroke-width="3"/></g>
<rect x="52" y="899" width="96" height="20" rx="10" fill="#ffffff" opacity="0.45"/>
<text x="100" y="930" text-anchor="middle" font-size="24" font-weight="700" fill="#ffffff">LLM</text>
<text x="800" y="190" text-anchor="middle" font-size="40" font-weight="600" fill="#3b2a6b">everything we tried</text>
''']
    for i, b in enumerate(BUBBLES):
        parts.append(bubble(i, *b))
    for label, x, y in GROUPS:
        parts.append(f'<text x="{x}" y="{y + DY}" text-anchor="middle" font-size="21" font-style="italic" fill="#7b6aa8">{label}</text>')
    parts.append("\n</svg>\n")
    OUT.write_text("".join(parts), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
