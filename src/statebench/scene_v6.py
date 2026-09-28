"""Turn 6 scene variants and scene-level appearance. ``scene.py`` is unchanged.

Variants:

``V0``  the 40 mm red cube (the task as before).
``V1``  a 16 mm red cube; every height that depends on the cube follows it.
``V2``  the 40 mm red cube plus a distractor cube of the same size in a declared
        hue about 25 degrees from the target's (target hue 3.8 deg, distractor
        28.8 deg at the same saturation and value). The goal text already names the
        target's colour ("red"); the oracle packet describes only the target.

Scene-level appearance for the ``rgb`` synthetic perturbation: one of 5 table
textures and a randomized light, both drawn once per episode from a fixed seed.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .scene import CUBE_MASS, RECEPTACLE_FLOOR_TOP, scene_xml

TARGET_RGBA = (0.76, 0.17, 0.13, 1.0)
DISTRACTOR_RGBA = (0.76, 0.432, 0.13, 1.0)
DISTRACTOR_MIN_SEPARATION_M = 0.07


def hue_deg(rgb) -> float:
    r, g, b = rgb[:3]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx == r:
        return (60.0 * ((g - b) / (mx - mn))) % 360
    if mx == g:
        return 60.0 * ((b - r) / (mx - mn)) + 120
    return 60.0 * ((r - g) / (mx - mn)) + 240


@dataclass(frozen=True)
class VariantConfig:
    name: str
    cube_half: float = 0.020
    distractor: bool = False

    @property
    def cube_mass(self) -> float:
        return CUBE_MASS * (self.cube_half / 0.020) ** 3

    @property
    def grasp_z(self) -> float:
        return self.cube_half + 0.002

    @property
    def place_z(self) -> float:
        return RECEPTACLE_FLOOR_TOP + self.cube_half + 0.020


VARIANTS = {
    "V0": VariantConfig("V0"),
    "V1": VariantConfig("V1", cube_half=0.008),
    "V2": VariantConfig("V2", distractor=True),
}

TABLE_TEXTURES = (
    {"name": "plain", "builtin": None, "rgb1": (0.96, 0.965, 0.96)},
    {"name": "light_checker", "builtin": "checker", "rgb1": (0.92, 0.92, 0.90), "rgb2": (0.80, 0.80, 0.78)},
    {"name": "beige", "builtin": "flat", "rgb1": (0.82, 0.76, 0.64)},
    {"name": "grey_gradient", "builtin": "gradient", "rgb1": (0.72, 0.72, 0.74), "rgb2": (0.55, 0.56, 0.58)},
    {"name": "dark_grey", "builtin": "flat", "rgb1": (0.38, 0.39, 0.40)},
)


@dataclass(frozen=True)
class SceneParams:
    table_texture: int = 0
    light_scale: float = 1.0
    light_dir: tuple = (0.25, 0.15, -1.0)


def draw_scene_params(rng: np.random.Generator) -> SceneParams:
    """rgb condition: one of 5 table textures and a randomized key light."""
    az = rng.uniform(-math.pi, math.pi)
    tilt = rng.uniform(0.15, 0.45)
    return SceneParams(table_texture=int(rng.integers(0, len(TABLE_TEXTURES))),
                       light_scale=float(rng.uniform(0.7, 1.3)),
                       light_dir=(float(math.cos(az) * tilt), float(math.sin(az) * tilt), -1.0))


def variant_xml(cfg: VariantConfig, scene: SceneParams | None = None) -> str:
    root = ET.fromstring(scene_xml())
    world, asset = root.find("worldbody"), root.find("asset")
    cube_geom = world.find(".//geom[@name='cube_geom']")
    h = cfg.cube_half
    cube_geom.set("size", f"{h} {h} {h}")
    cube_geom.set("mass", f"{cfg.cube_mass:.6f}")
    if cfg.distractor:
        ET.SubElement(asset, "material", name="distractor_skin",
                      rgba=" ".join(str(v) for v in DISTRACTOR_RGBA), specular=".18", shininess=".3")
        body = ET.SubElement(world, "body", name="distractor", pos="0.40 -0.30 0.02")
        ET.SubElement(body, "freejoint", name="distractor_joint")
        ET.SubElement(body, "geom", name="distractor_geom", type="box", size=f"{h} {h} {h}",
                      material="distractor_skin", mass=f"{cfg.cube_mass:.6f}", friction="1.6 .01 .004",
                      condim="6", solref=".006 1")
    if scene is not None:
        tex = TABLE_TEXTURES[scene.table_texture]
        worktop = asset.find("material[@name='worktop']")
        if tex["builtin"] is None:
            worktop.set("rgba", " ".join(str(v) for v in (*tex["rgb1"], 1.0)))
        else:
            attrs = {"name": "table_tex", "type": "2d", "builtin": tex["builtin"], "width": "256",
                     "height": "256", "rgb1": " ".join(str(v) for v in tex["rgb1"])}
            if "rgb2" in tex:
                attrs["rgb2"] = " ".join(str(v) for v in tex["rgb2"])
            ET.SubElement(asset, "texture", **attrs)
            worktop.set("texture", "table_tex")
            worktop.set("texrepeat", "8 8")
            worktop.set("rgba", "1 1 1 1")
        lights = world.findall("light")
        key = lights[0]
        diffuse = [float(v) * scene.light_scale for v in key.get("diffuse").split()]
        key.set("diffuse", " ".join(f"{min(1.0, v):.3f}" for v in diffuse))
        key.set("dir", " ".join(f"{v:.3f}" for v in scene.light_dir))
    return ET.tostring(root, encoding="unicode")
