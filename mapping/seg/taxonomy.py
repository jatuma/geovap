"""Map public model taxonomies (Mapillary Vistas 65, Cityscapes 19, ADE20K 150) and the GT classes to one
common evaluation set. Every native label must be mapped (asserted at model load); default is `other`."""
from __future__ import annotations

import numpy as np

from . import classes as C

COMMON = ["road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "water", "guard_rail", "stairs", "pole", "sky", "vehicle", "person", "other"]
COMMON_ID = {n: i for i, n in enumerate(COMMON)}
CORE = ["road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain"]
EXT = ["water", "guard_rail", "stairs", "pole"]
IGNORE = 255

# GT dataset class -> common (diagnostic classes -> None: excluded from the confusion matrix)
GT_TO_COMMON: dict[str, str | None] = {
    "road": "road", "sidewalk": "sidewalk", "building": "building", "wall": "wall", "fence": "fence",
    "vegetation": "vegetation", "terrain": "terrain", "water": "water", "guard_rail": "guard_rail",
    "stairs": "stairs", "pole": "pole",
    "verge": None, "paved_other": None, "culvert_head": None, "structure_other": None, "road_or_verge": None,
}

CITYSCAPES = {
    "road": "road", "sidewalk": "sidewalk", "building": "building", "wall": "wall", "fence": "fence",
    "pole": "pole", "traffic light": "other", "traffic sign": "other", "vegetation": "vegetation",
    "terrain": "terrain", "sky": "sky", "person": "person", "rider": "person", "car": "vehicle",
    "truck": "vehicle", "bus": "vehicle", "train": "vehicle", "motorcycle": "vehicle", "bicycle": "vehicle",
}

VISTAS = {
    "Bird": "other", "Ground Animal": "other", "Curb": "sidewalk", "Fence": "fence", "Guard Rail": "guard_rail",
    "Barrier": "other", "Wall": "wall", "Bike Lane": "road", "Crosswalk - Plain": "road", "Curb Cut": "sidewalk",
    "Parking": "road", "Pedestrian Area": "sidewalk", "Rail Track": "other", "Road": "road", "Service Lane": "road",
    "Sidewalk": "sidewalk", "Bridge": "other", "Building": "building", "Tunnel": "other", "Person": "person",
    "Bicyclist": "person", "Motorcyclist": "person", "Other Rider": "person", "Lane Marking - Crosswalk": "road",
    "Lane Marking - General": "road", "Mountain": "terrain", "Sand": "terrain", "Sky": "sky", "Snow": "terrain",
    "Terrain": "terrain", "Vegetation": "vegetation", "Water": "water", "Banner": "other", "Bench": "other",
    "Bike Rack": "other", "Billboard": "other", "Catch Basin": "road", "CCTV Camera": "other", "Fire Hydrant": "other",
    "Junction Box": "other", "Mailbox": "other", "Manhole": "road", "Phone Booth": "other", "Pothole": "road",
    "Street Light": "pole", "Pole": "pole", "Traffic Sign Frame": "pole", "Utility Pole": "pole", "Traffic Light": "other",
    "Traffic Sign (Back)": "other", "Traffic Sign (Front)": "other", "Trash Can": "other", "Bicycle": "vehicle",
    "Boat": "vehicle", "Bus": "vehicle", "Car": "vehicle", "Caravan": "vehicle", "Motorcycle": "vehicle",
    "On Rails": "vehicle", "Other Vehicle": "vehicle", "Trailer": "vehicle", "Truck": "vehicle", "Wheeled Slow": "vehicle",
    "Car Mount": "other", "Ego Vehicle": "other",
}

# ADE20K: match on the first synonym of the HF id2label entry (e.g. "wall", "building, edifice" -> "building")
ADE_FIRST = {
    "wall": "wall", "building": "building", "sky": "sky", "floor": "other", "tree": "vegetation", "ceiling": "other",
    "road": "road", "bed": "other", "windowpane": "building", "grass": "terrain", "cabinet": "other", "sidewalk": "sidewalk",
    "person": "person", "earth": "terrain", "door": "building", "table": "other", "mountain": "terrain", "plant": "vegetation",
    "curtain": "other", "chair": "other", "car": "vehicle", "water": "water", "painting": "other", "sofa": "other",
    "shelf": "other", "house": "building", "sea": "water", "mirror": "other", "rug": "other", "field": "terrain",
    "armchair": "other", "seat": "other", "fence": "fence", "desk": "other", "rock": "terrain", "wardrobe": "other",
    "lamp": "pole", "bathtub": "other", "railing": "guard_rail", "cushion": "other", "base": "other", "box": "other",
    "column": "pole", "signboard": "other", "chest of drawers": "other", "counter": "other", "sand": "terrain",
    "sink": "other", "skyscraper": "building", "fireplace": "other", "refrigerator": "other", "grandstand": "other",
    "path": "sidewalk", "stairs": "stairs", "runway": "road", "case": "other", "pool table": "other", "pillow": "other",
    "screen door": "other", "stairway": "stairs", "river": "water", "bridge": "other", "bookcase": "other", "blind": "other",
    "coffee table": "other", "toilet": "other", "flower": "vegetation", "book": "other", "hill": "terrain", "bench": "other",
    "countertop": "other", "stove": "other", "palm": "vegetation", "kitchen island": "other", "computer": "other",
    "swivel chair": "other", "boat": "vehicle", "bar": "other", "arcade machine": "other", "hovel": "building", "bus": "vehicle",
    "towel": "other", "light": "pole", "truck": "vehicle", "tower": "building", "chandelier": "other", "awning": "other",
    "streetlight": "pole", "booth": "other", "television receiver": "other", "airplane": "vehicle", "dirt track": "terrain",
    "apparel": "other", "pole": "pole", "land": "terrain", "bannister": "guard_rail", "escalator": "other", "ottoman": "other",
    "bottle": "other", "buffet": "other", "poster": "other", "stage": "other", "van": "vehicle", "ship": "vehicle",
    "fountain": "water", "conveyer belt": "other", "canopy": "other", "washer": "other", "plaything": "other",
    "swimming pool": "water", "stool": "other", "barrel": "other", "basket": "other", "waterfall": "water", "tent": "other",
    "bag": "other", "minibike": "vehicle", "cradle": "other", "oven": "other", "ball": "other", "food": "other",
    "step": "stairs", "tank": "other", "trade name": "other", "microwave": "other", "pot": "other", "animal": "other",
    "bicycle": "vehicle", "lake": "water", "dishwasher": "other", "screen": "other", "blanket": "other", "sculpture": "other",
    "hood": "other", "sconce": "other", "vase": "other", "traffic light": "other", "tray": "other", "ashcan": "other",
    "fan": "other", "pier": "other", "crt screen": "other", "plate": "other", "monitor": "other", "bulletin board": "other",
    "shower": "other", "radiator": "other", "glass": "other", "clock": "other", "flag": "other",
}


def native_to_common(id2label: dict, family_taxonomy: str) -> np.ndarray:
    """[num_labels] int array of common ids for a model's id2label. Raises on an unmapped label."""
    n = len(id2label)
    out = np.full(n, COMMON_ID["other"], np.int64)
    for i in range(n):
        name = id2label[i] if i in id2label else id2label[str(i)]
        if family_taxonomy == "cityscapes":
            key = name.lower()
            m = CITYSCAPES
        elif family_taxonomy == "vistas":
            key = name
            m = VISTAS
        elif family_taxonomy == "ade":
            key = name.split(",")[0].split(";")[0].strip().lower()
            m = ADE_FIRST
        else:
            raise ValueError(family_taxonomy)
        if key not in m:
            raise KeyError(f"{family_taxonomy}: unmapped label {name!r}")
        out[i] = COMMON_ID[m[key]]
    return out


def gt_to_common_lut() -> np.ndarray:
    """[256] uint8: dataset label id -> common id, 255 where excluded/ignore."""
    lut = np.full(256, IGNORE, np.uint8)
    for c in C.CLASSES:
        tgt = GT_TO_COMMON[c.name]
        if tgt is not None:
            lut[c.id] = COMMON_ID[tgt]
    return lut


def model_can_predict(native_to_common_ids: np.ndarray) -> set[str]:
    return {COMMON[i] for i in np.unique(native_to_common_ids)}
