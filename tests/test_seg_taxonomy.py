import json
from pathlib import Path

import numpy as np
import pytest

from mapping.seg import classes as C
from mapping.seg import taxonomy as T

CITYSCAPES_19 = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def test_gt_classes_all_mapped_or_excluded():
    assert set(T.GT_TO_COMMON) == {c.name for c in C.CLASSES}
    lut = T.gt_to_common_lut()
    assert lut[C.BY_NAME["fence"].id] == T.COMMON_ID["fence"] and lut[C.BY_NAME["verge"].id] == 255 and lut[255] == 255


def test_cityscapes_full_coverage():
    ids = T.native_to_common({i: n for i, n in enumerate(CITYSCAPES_19)}, "cityscapes")
    assert set(T.CORE) <= T.model_can_predict(ids) and "water" not in T.model_can_predict(ids)


def test_vistas_from_e2_manifest():
    p = Path(__file__).resolve().parents[1] / "experiments" / "out" / "e2" / "manifest.json"
    if not p.exists():
        pytest.skip("E2 manifest not available")
    id2label = json.loads(p.read_text())["id2label"]
    ids = T.native_to_common(id2label, "vistas")
    assert len(ids) == 65 and "guard_rail" in T.model_can_predict(ids)


def test_unmapped_label_raises():
    with pytest.raises(KeyError):
        T.native_to_common({0: "road", 1: "spaceship"}, "cityscapes")


def test_saved_id2label_files_map():
    d = Path(__file__).resolve().parents[1] / "dataset" / "seg"
    for p in d.glob("id2label_*.json"):
        meta = json.loads(p.read_text())
        T.native_to_common(meta["id2label"], meta["taxonomy"])
