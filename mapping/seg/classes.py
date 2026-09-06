"""GT taxonomy of the segmentation dataset and the JVF code -> class mapping.

Label PNG ids are stable; 255 = ignore. Classes marked `core` enter mIoU_core, `ext` enter mIoU_ext when the
model taxonomy has them, `diag` are diagnostic only (JVF areas whose appearance is not a single visual class).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IGNORE = 255


@dataclass(frozen=True)
class SegClass:
    id: int
    name: str
    colour: tuple[int, int, int]  # RGB
    eval: str  # core | ext | diag
    czech: str


CLASSES: list[SegClass] = [
    SegClass(0, "road", (128, 64, 128), "core", "vozovka (provozní plocha, nájezd, parkoviště, most)"),
    SegClass(1, "sidewalk", (244, 35, 232), "core", "chodník"),
    SegClass(2, "building", (230, 60, 60), "core", "budova a ostatní zastřešené stavby"),
    SegClass(3, "wall", (102, 102, 156), "core", "zeď"),
    SegClass(4, "fence", (190, 153, 153), "core", "plot (všechny materiály včetně živého)"),
    SegClass(5, "vegetation", (107, 142, 35), "core", "vegetace nad zemí v plochách zeleně / zahrad / polí"),
    SegClass(6, "terrain", (152, 251, 152), "core", "zem v plochách zeleně, zahrad, polí, příkopů"),
    SegClass(7, "water", (0, 130, 180), "ext", "vodní tok, koryto, nádrž"),
    SegClass(8, "guard_rail", (180, 165, 180), "ext", "zábradlí"),
    SegClass(9, "stairs", (150, 100, 100), "ext", "schodiště"),
    SegClass(10, "pole", (153, 153, 153), "ext", "nosič technického zařízení"),
    SegClass(11, "verge", (170, 200, 120), "diag", "přidružená plocha pozemní komunikace (krajnice, většinou nezpevněná)"),
    SegClass(12, "paved_other", (200, 200, 160), "diag", "dvůr, manipulační plocha, zpevnění povrchu, hřiště"),
    SegClass(13, "culvert_head", (220, 120, 60), "diag", "čelo propustku"),
    SegClass(14, "structure_other", (120, 120, 200), "diag", "patka/deska/pilíř, nerozlišená hranice stavby"),
    SegClass(15, "road_or_verge", (150, 110, 150), "diag", "dopravní plocha bez uzavřené hranice mezi vozovkou a krajnicí (hlavní ulice)"),
]
BY_NAME = {c.name: c for c in CLASSES}
BY_ID = {c.id: c for c in CLASSES}
N_CLASSES = len(CLASSES)


def palette() -> np.ndarray:
    """[256,3] RGB uint8, ignore = black."""
    p = np.zeros((256, 3), np.uint8)
    for c in CLASSES:
        p[c.id] = c.colour
    return p


# ------------------------------------------------------------------- JVF area codes (definition points)
# code -> class name for area objects (their definition points are the `-04` features of the JVF export)
AREA_CLASS: dict[str, str] = {
    "0100000005": "road",  # provozní plocha pozemní komunikace
    "0100000017": "road",  # nájezd
    "0100000011": "road",  # parkoviště, odstavná plocha
    "0100000058": "road",  # plocha mostní konstrukce
    "0100000007": "sidewalk",  # chodník
    "0100000001": "building",  # budova
    "0100000314": "building",  # ostatní zastřešená stavba
    "0100000179": "building",  # skleník
    "0100000159": "building",  # drobná kulturní stavba
    "0100000154": "building",  # drobná sakrální stavba
    "0100000168": "wall",  # zeď (plocha)
    "0100000215": "terrain",  # udržovaná plocha zeleně
    "0100000209": "terrain",  # zahrada
    "0100000207": "terrain",  # zemědělská plocha
    "0100000213": "terrain",  # hospodářsky nevyužívaná plocha
    "0100000051": "terrain",  # příkop, násep, zářez dopravní stavby
    "0100000080": "terrain",  # meliorační příkop, žlab
    "0100000203": "water",  # vodní tok
    "0100000078": "water",  # stavebně upravené koryto
    "0100000330": "water",  # nádrž bez vzdouvacího objektu
    "0100000166": "stairs",  # schodiště
    "0100000320": "verge",  # přidružená plocha pozemní komunikace
    "0100000189": "paved_other",  # dvůr, nádvoří
    "0100000055": "paved_other",  # manipulační plocha
    "0100000187": "paved_other",  # stavba pro zpevnění povrchu
    "0100000152": "paved_other",  # hřiště
    "0100000193": "culvert_head",  # čelo propustku
    "0100000183": "structure_other",  # patka, deska, monolit, pilíř
}

# faces whose above-ground points are vegetation (gardens, green, fields, ditches)
GREEN_CODES = {"0100000215", "0100000209", "0100000207", "0100000213", "0100000051", "0100000080"}
BUILDING_CODES = {c for c, n in AREA_CLASS.items() if n == "building"}
# small objects: they define a face only when the face is small (their def point often lies in a large area
# whose outline they do not close: footings, culvert heads, sheds without own outline in the export)
SMALL_CODES = {"0100000183", "0100000193", "0100000187", "0100000166", "0100000179", "0100000159", "0100000154", "0100000168"}
SMALL_FACE_MAX_M2 = {"default": 30.0, "0100000187": 500.0, "0100000179": 100.0}  # zpevnění povrchu / skleník can be larger
# conflicting surface classes that still describe one visual group -> merged diagnostic class
CONFLICT_MERGE = {frozenset({"road", "verge"}): "road_or_verge", frozenset({"road", "verge", "paved_other"}): "road_or_verge", frozenset({"road", "paved_other"}): "road_or_verge"}
SLIVER_MAX_M2 = 5.0
SLIVER_SHARED_FRAC = 0.6

# lines that delimit faces (hard boundaries). Fences, driveways, terrain edges etc. are not delimiters by default.
HARD_BOUNDARY_CODES = {
    "0100000304",  # hranice dopravní stavby nebo plochy
    "0100000300",  # hranice stavby
    "0100000299",  # hranice budovy
    "0100000305",  # hranice přírodního a polopřírodního objektu
    "0100000308",  # hranice udržované zeleně
    "0100000306",  # hranice vodního díla
    "0100000302",  # hranice zdi
    "0100000307",  # hranice ostatní plochy
    "0100000301",  # hranice schodiště
    "0100000323",  # vnitřní členění dopravní plochy
    "0100000310",  # vnitřní členění budov a staveb
}

# line groups used by the 3D point rules (distance + line height rasters)
LINE_GROUPS: dict[str, set[str]] = {
    "fence": {"0100000162"},
    "wall": {"0100000302", "0100000168"},
    "rail": {"0100000199"},
    "bldg_edge": {"0100000299"},
    "struct_edge": {"0100000300"},
}
POLE_CODE = "0100000201"

# 3D rule thresholds (metres) - see point_labels.label_points
RULES = dict(
    fence_dist=0.35, fence_dz_min=0.0, fence_dz_max=2.5, fence_ring=1.0, fence_ring_hag_min=0.3,
    rail_dist=0.3, rail_dz_min=0.2, rail_dz_max=1.5,
    wall_dist=0.3, wall_hag_max=3.0,
    bldg_buffer=0.5, bldg_hag_min=0.2,
    struct_dist=0.5,
    ground_hag=0.2, water_hag=0.3, stairs_hag=3.0,
    veg_hag_min=0.5, veg_bldg_dist=1.0,
    pole_dist=0.4, pole_hag_max=12.0,
    range_max_m=40.0,  # products R_MAX; 04 SS4 gives 13-20 cm at 40 m = < 2 label px at 2000x1000
)


def rules_hash() -> str:
    import hashlib
    import json

    return hashlib.sha1(json.dumps({"rules": RULES, "area": AREA_CLASS, "classes": [c.name for c in CLASSES]}, sort_keys=True).encode()).hexdigest()[:10]
