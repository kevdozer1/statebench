"""MJCF scene: xArm7 + table, one cube on the table, one fixed receptacle tray.

The robot description is UFACTORY's xArm7 MJCF (BSD-3-Clause), read from
``$STATEBENCH_DATA/assets/xarm7``. Everything task-specific is added here.

Geometry constants are the single source of truth for the task and are imported
by the verifier, the executor and the observation builders, so the three cannot
drift apart.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from .config import assets_dir

# ---------------------------------------------------------------------------
# Task geometry (metres, robot base frame, +Z up, table top at z = 0)
# ---------------------------------------------------------------------------
CUBE_HALF = 0.020
CUBE_MASS = 0.060
CUBE_REST_Z = CUBE_HALF

RECEPTACLE_CENTER = np.array([0.44, 0.20])
RECEPTACLE_OUTER_HALF = 0.085
RECEPTACLE_WALL = 0.006
RECEPTACLE_FLOOR_TOP = 0.012
RECEPTACLE_WALL_TOP = RECEPTACLE_FLOOR_TOP + 0.045
RECEPTACLE_INNER_HALF = RECEPTACLE_OUTER_HALF - RECEPTACLE_WALL

# Cube initial-condition distribution (declared; seeds 0-999 reserved).
CUBE_X_RANGE = (0.33, 0.47)
CUBE_Y_RANGE = (-0.22, -0.06)
CUBE_YAW_RANGE = (-math.pi / 4, math.pi / 4)

# Executor heights.
TRAVEL_Z = 0.175
GRASP_Z = CUBE_REST_Z + 0.002
PLACE_Z = RECEPTACLE_FLOOR_TOP + CUBE_HALF + 0.020

# Workspace clamp applied to every commanded TCP target.
WORKSPACE_MIN = np.array([0.20, -0.32, 0.018])
WORKSPACE_MAX = np.array([0.62, 0.38, 0.34])

HOME_TCP = np.array([0.30, -0.02, 0.24])


def sample_initial_conditions(seed: int) -> dict[str, float]:
    """Cube pose for ``seed``. Receptacle is fixed. Pure function of the seed."""
    rng = np.random.default_rng(seed + 10_000)
    return {
        "cube_x": float(rng.uniform(*CUBE_X_RANGE)),
        "cube_y": float(rng.uniform(*CUBE_Y_RANGE)),
        "cube_yaw": float(rng.uniform(*CUBE_YAW_RANGE)),
    }


def perturbation_fires(seed: int, probability: float = 0.3) -> bool:
    """Forced-miss schedule: a pure function of the seed, so it is identical
    across every condition by construction."""
    rng = np.random.default_rng(seed + 20_000)
    return bool(rng.random() < probability)


def perturbation_offset(seed: int, magnitude: float = 0.035) -> np.ndarray:
    """Displacement applied to the first ``close`` so that it misses.

    Vertical, and 3.5 cm rather than the 2 cm originally specified, because 2 cm
    does not miss with this robot. Measured before the pre-registration was
    hashed: at 2 cm the gripper still achieved a bilateral grasp on 25/25 trials
    across five offset elevations, six azimuths and three seeds. A parallel jaw
    that opens to 89 mm on a 40 mm cube self-centres from any lateral error
    smaller than its open half-gap, so a horizontal miss is not available at all
    (2/5 seeds still held at every magnitude from 4 to 7 cm). Vertically the
    threshold is sharp: +3.0 cm held 5/5, +3.5 cm held 0/5. The offset is a pure
    function of the seed, so the schedule is identical across conditions.

    A depth error that closes the fingers above the object is also the failure
    mode this benchmark cares about most, since it is the classic consequence of
    a bad z estimate.
    """
    _ = seed  # the direction is fixed; only whether it fires depends on the seed
    return np.array([0.0, 0.0, float(magnitude)])


def _tray(world: ET.Element) -> None:
    cx, cy = RECEPTACLE_CENTER
    half_h = (RECEPTACLE_WALL_TOP - RECEPTACLE_FLOOR_TOP) / 2.0
    mid_z = RECEPTACLE_FLOOR_TOP + half_h
    ET.SubElement(
        world, "geom", name="receptacle_floor", type="box",
        pos=f"{cx} {cy} {RECEPTACLE_FLOOR_TOP / 2:.6f}",
        size=f"{RECEPTACLE_OUTER_HALF} {RECEPTACLE_OUTER_HALF} {RECEPTACLE_FLOOR_TOP / 2:.6f}",
        material="tray", friction="1 .01 .001",
    )
    offset = RECEPTACLE_OUTER_HALF - RECEPTACLE_WALL / 2
    walls = [
        ("px", cx + offset, cy, RECEPTACLE_WALL / 2, RECEPTACLE_OUTER_HALF),
        ("nx", cx - offset, cy, RECEPTACLE_WALL / 2, RECEPTACLE_OUTER_HALF),
        ("py", cx, cy + offset, RECEPTACLE_OUTER_HALF, RECEPTACLE_WALL / 2),
        ("ny", cx, cy - offset, RECEPTACLE_OUTER_HALF, RECEPTACLE_WALL / 2),
    ]
    for name, wx, wy, sx, sy in walls:
        ET.SubElement(
            world, "geom", name=f"receptacle_wall_{name}", type="box",
            pos=f"{wx} {wy} {mid_z:.6f}", size=f"{sx} {sy} {half_h:.6f}",
            material="tray", friction="1 .01 .001",
        )


def scene_xml() -> str:
    root = ET.parse(assets_dir() / "xarm7.xml").getroot()
    root.set("model", "statebench pick and place")
    root.find("compiler").set("meshdir", str(assets_dir() / "assets"))
    root.find("option").set("timestep", ".002")
    root.remove(root.find("keyframe"))
    # Put the tool-centre point at the finger-pad centroid so a commanded TCP
    # position is the grasp point.
    root.find(".//site[@name='link_tcp']").set("pos", "0 0 .145")
    for cls in ("pad_box1", "pad_box2"):
        root.find(f".//default[@class='{cls}']/geom").set("friction", "1.6 .01 .002")

    asset, world = root.find("asset"), root.find("worldbody")
    for name, color, spec in [
        ("worktop", ".96 .965 .96 1", ".07"),
        ("floor", ".94 .945 .94 1", ".02"),
        ("cube_skin", ".76 .17 .13 1", ".18"),
        ("tray", ".30 .43 .62 1", ".20"),
    ]:
        ET.SubElement(asset, "material", name=name, rgba=color, specular=spec, shininess=".3")
    ET.SubElement(
        asset, "texture", name="sky", type="skybox", builtin="gradient",
        rgb1=".95 .95 .93", rgb2=".95 .95 .93", width="512", height="1024",
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", diffuse=".28 .28 .28", ambient=".38 .38 .38",
                  specular=".07 .07 .07")
    ET.SubElement(visual, "quality", shadowsize="2048")

    ET.SubElement(world, "geom", name="table", type="box", size=".60 .49 .022",
                  pos=".29 .03 -.022", material="worktop", friction="1 .01 .001")
    ET.SubElement(world, "geom", name="floor", type="plane", size="6 6 .05",
                  pos="0 0 -.77", material="floor")
    ET.SubElement(world, "geom", name="pedestal", type="cylinder", size=".073 .06",
                  pos="0 0 .06", rgba=".13 .15 .16 1")
    ET.SubElement(world, "light", pos="-.5 -.7 2.4", dir=".25 .15 -1",
                  diffuse=".50 .48 .45", specular=".28 .28 .28", directional="true")
    ET.SubElement(world, "light", pos="1.4 .8 1.7", dir="-.5 -.4 -1",
                  diffuse=".24 .25 .27", castshadow="false")

    cube = ET.SubElement(world, "body", name="cube",
                         pos=f"{sum(CUBE_X_RANGE) / 2} {sum(CUBE_Y_RANGE) / 2} {CUBE_REST_Z}")
    ET.SubElement(cube, "freejoint", name="cube_joint")
    ET.SubElement(cube, "geom", name="cube_geom", type="box",
                  size=f"{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}", material="cube_skin",
                  mass=f"{CUBE_MASS}", friction="1.6 .01 .004", condim="6", solref=".006 1")
    _tray(world)
    return ET.tostring(root, encoding="unicode")


def in_receptacle_volume(point: np.ndarray) -> bool:
    """Is ``point`` inside the tray's inner volume? Used only by the verifier."""
    dx = abs(float(point[0]) - RECEPTACLE_CENTER[0])
    dy = abs(float(point[1]) - RECEPTACLE_CENTER[1])
    z = float(point[2])
    return bool(
        dx < RECEPTACLE_INNER_HALF
        and dy < RECEPTACLE_INNER_HALF
        and RECEPTACLE_FLOOR_TOP <= z <= RECEPTACLE_WALL_TOP
    )


def gripper_clear_of_receptacle(tcp: np.ndarray) -> bool:
    """The gripper is clear when it is outside the tray footprint or above it."""
    horizontal = float(np.linalg.norm(np.asarray(tcp)[:2] - RECEPTACLE_CENTER))
    return bool(horizontal > RECEPTACLE_OUTER_HALF + 0.045 or float(tcp[2]) > 0.145)


def clamp_to_workspace(target) -> np.ndarray:
    return np.clip(np.asarray(target, dtype=float), WORKSPACE_MIN, WORKSPACE_MAX)
