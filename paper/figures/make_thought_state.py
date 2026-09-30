"""Frutiger Aero figure: the structured state before trimming, drawn like the final structured-state diagram but holding
thought bubbles, one per kind of field we tried, in plain words.

    python paper/figures/make_thought_state.py        -> state_before_trimming.svg
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "state_before_trimming.svg"
W, H = 1600, 900
FONT = "Segoe UI, Lucida Grande, Helvetica Neue, Arial, sans-serif"
BLOB_T = "translate(21 -30) scale(1.32 1.12)"  # the final diagram's blob, widened and centred

# (cx, cy, width, lines, tail direction: -1 trails down-left, +1 down-right)
BUBBLES = [
    (615, 325, 330, ["where is everything?"], -1),
    (995, 325, 310, ["what can I do next?"], 1),
    (440, 505, 270, ["what's the goal?"], -1),
    (800, 505, 320, ["is the arm holding", "something?"], 1),
    (1160, 505, 280, ["am I touching it?"], 1),
    (640, 672, 290, ["how many tries", "so far?"], -1),
    (985, 672, 300, ["what did I just do?"], 1),
]

def cloud(i, cx, cy, w, lines, tail):
    h = 88 if len(lines) == 1 else 116
    # a soft cloud: a body ellipse plus bumps along the top and bottom
    bumps = [(cx, cy, w / 2, h / 2)]
    for fx, fy, fr in ((-0.28, -0.30, 0.32), (0.03, -0.37, 0.37), (0.30, -0.28, 0.30),
                       (-0.20, 0.30, 0.30), (0.20, 0.31, 0.32)):
        bumps.append((cx + fx * w, cy + fy * h, fr * h, fr * h))
    shape = "".join(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{rx:.1f}" ry="{ry:.1f}"/>' for x, y, rx, ry in bumps)
    tx = cx + tail * 0.34 * w
    tails = (f'<circle cx="{tx + tail * 10:.1f}" cy="{cy + h / 2 + 12:.1f}" r="9"/>'
             f'<circle cx="{tx + tail * 28:.1f}" cy="{cy + h / 2 + 28:.1f}" r="5.5"/>')
    fs = 28
    lh = fs * 1.18
    y0 = cy - (len(lines) - 1) * lh / 2 + fs * 0.36
    text = "".join(f'<text x="{cx}" y="{y0 + k * lh:.1f}" text-anchor="middle" font-size="{fs}" font-weight="500" '
                   f'fill="#3b2a6b">{s}</text>' for k, s in enumerate(lines))
    return f'''
  <radialGradient id="t{i}" gradientUnits="userSpaceOnUse" cx="{cx - w * 0.12:.0f}" cy="{cy - h * 0.35:.0f}" r="{w * 0.62:.0f}">
    <stop offset="0" stop-color="#ffffff"/><stop offset="0.6" stop-color="#fffaf0"/><stop offset="1" stop-color="#f8d9a6"/>
  </radialGradient>
  <clipPath id="c{i}">{shape}</clipPath>
  <g filter="url(#soft)"><g fill="#ffffff" stroke="#ffffff" stroke-width="7">{shape}{tails}</g></g>
  <g fill="url(#t{i})">{shape}{tails}</g>
  <g clip-path="url(#c{i})"><ellipse cx="{cx:.0f}" cy="{cy - h * 0.42:.0f}" rx="{w * 0.46:.0f}" ry="{h * 0.36:.0f}" fill="url(#gloss)" opacity="0.8"/></g>
  {text}'''


def main():
    blob = ("M 150 440 C 130 260, 320 130, 540 142 C 720 152, 830 100, 945 196 C 1060 292, 1030 430, 1020 530 "
            "C 1008 650, 915 772, 705 782 C 520 792, 425 752, 300 708 C 178 664, 166 570, 150 440 Z")
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">
<defs>
  <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#f1ecff"/><stop offset="0.55" stop-color="#fbf8ff"/><stop offset="1" stop-color="#fff6e6"/>
  </linearGradient>
  <radialGradient id="glow" cx="0.5" cy="0.45" r="0.6">
    <stop offset="0" stop-color="#ffffff" stop-opacity="0.9"/><stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="blobfill" cx="0.42" cy="0.38" r="0.75">
    <stop offset="0" stop-color="#faf7ff"/><stop offset="0.45" stop-color="#cdbff9"/><stop offset="0.85" stop-color="#9d86ec"/>
    <stop offset="1" stop-color="#7b5fd8"/>
  </radialGradient>
  <linearGradient id="gloss" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#ffffff" stop-opacity="0.95"/><stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
  </linearGradient>
  <radialGradient id="bubble" cx="0.35" cy="0.3" r="0.7">
    <stop offset="0" stop-color="#ffffff" stop-opacity="0.95"/><stop offset="0.6" stop-color="#efe8ff" stop-opacity="0.35"/>
    <stop offset="1" stop-color="#ffe3b8" stop-opacity="0.15"/>
  </radialGradient>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">
    <feDropShadow dx="0" dy="18" stdDeviation="22" flood-color="#6a52c4" flood-opacity="0.28"/>
  </filter>
  <filter id="soft" x="-20%" y="-40%" width="140%" height="200%">
    <feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#6a52c4" flood-opacity="0.22"/>
  </filter>
  <clipPath id="blobclip"><path transform="{BLOB_T}" d="{blob}"/></clipPath>
</defs>
<rect width="{W}" height="{H}" fill="url(#sky)"/>
<rect width="{W}" height="{H}" fill="url(#glow)"/>
<circle cx="1480" cy="140" r="46" fill="url(#bubble)" stroke="#ffffff" stroke-opacity="0.8" stroke-width="2"/>
<circle cx="1400" cy="96" r="18" fill="url(#bubble)" stroke="#ffffff" stroke-opacity="0.8" stroke-width="2"/>
<circle cx="140" cy="150" r="30" fill="url(#bubble)" stroke="#ffffff" stroke-opacity="0.8" stroke-width="2"/>
<circle cx="1500" cy="780" r="30" fill="url(#bubble)" stroke="#ffffff" stroke-opacity="0.8" stroke-width="2"/>
<circle cx="120" cy="760" r="22" fill="url(#bubble)" stroke="#ffffff" stroke-opacity="0.8" stroke-width="2"/>
<g filter="url(#shadow)"><path transform="{BLOB_T}" d="{blob}" fill="url(#blobfill)" stroke="#ffffff" stroke-width="5"/></g>
<g clip-path="url(#blobclip)"><ellipse cx="760" cy="160" rx="560" ry="160" fill="url(#gloss)" opacity="0.85"/></g>
<text x="800" y="205" text-anchor="middle" font-size="46" font-weight="600" fill="#3b2a6b">structured state</text>
''']
    for i, b in enumerate(BUBBLES):
        parts.append(cloud(i, *b))
    parts.append("\n</svg>\n")
    OUT.write_text("".join(parts), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
