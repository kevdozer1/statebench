"""Turn 8 scene variants: V0-V2 as Turn 6, plus V3 (the slip variant). ``scene_v6.py`` is unchanged.

``V3``: the 40 mm cube of V0 with a lower friction coefficient and a larger mass, so
that some grasps that look correct slip during transport. The cube geom is given
contact priority 2 so that its own friction applies to finger contacts (MuJoCo
otherwise takes the larger of the two geoms' coefficients, and the finger pads have
1.6). The values are declared in ``V3_FRICTION`` and ``V3_MASS_KG`` (chosen on dev
seeds for a first-grasp slip rate of 20-40 percent under the reference controller;
notes/turn8.md has the search).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .scene_v6 import VARIANTS as VARIANTS_V6
from .scene_v6 import SceneParams, VariantConfig, variant_xml

V3_FRICTION = (0.12, 0.01, 0.004)   # sliding, torsional, rolling
V3_MASS_KG = 0.30
V3_PRIORITY = 2

VARIANTS = dict(VARIANTS_V6)
VARIANTS["V3"] = VariantConfig("V3")


def variant_xml_v8(variant: str, scene: SceneParams | None = None, *, friction=None, mass=None) -> str:
    if variant != "V3":
        return variant_xml(VARIANTS[variant], scene)
    root = ET.fromstring(variant_xml(VARIANTS["V0"], scene))
    g = root.find(".//geom[@name='cube_geom']")
    fr = friction if friction is not None else V3_FRICTION
    g.set("friction", " ".join(f"{v}" for v in fr))
    g.set("mass", f"{mass if mass is not None else V3_MASS_KG:.6f}")
    g.set("priority", str(V3_PRIORITY))
    return ET.tostring(root, encoding="unicode")
