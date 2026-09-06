"""ERP label panoramas (2000x1000) for the clean frames, from the 3D point labels through the point_id products.

    labels_erp/f%04d.png   uint8 class ids, 255 ignore (no point, vehicle, range > 25 m, unlabelled point)
    bands_erp/f%04d.png    uint8 band ids of VISIBLE JVF lines (evaluation only, never training labels)
    qa/f%04d.jpg           photo overlay
Labels are copied per z-buffer cell (nearest), never interpolated; holes <= 3 px are filled from the nearest cell
like `render.gather`.
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .. import render, vectors
from ..cloud_store import CloudStore
from ..config import NO_POINT, PANO_H, PANO_W, ZB_H, ZB_W
from ..frame_select import FrameIndex
from ..poses import load_poses
from ..products import FrameProducts
from ..vehicle_mask import VehicleMask
from . import classes as C
from .areas import SEGDS_DIR, load_objects
from .point_labels import PointLabels

LABELS_DIR = SEGDS_DIR / "labels_erp"
BANDS_DIR = SEGDS_DIR / "bands_erp"
QA_DIR = SEGDS_DIR / "qa"
CLEAN_JSON = Path(__file__).resolve().parents[2] / "dataset" / "clean_frames.json"

# JVF codes rasterised as evaluation bands (14 cm at range) -> band id
BAND_IDS = {
    "0100000304": 1,  # hranice dopravní stavby nebo plochy (curb / road edge)
    "0100000299": 2,  # hranice budovy
    "0100000300": 3,  # hranice stavby
    "0100000302": 4,  # hranice zdi
    "0100000168": 4,  # zeď
    "0100000162": 5,  # plot
    "0100000199": 6,  # zábradlí
    "0100000308": 7,  # hranice udržované zeleně
    "0100000305": 7,  # hranice přírodního objektu
}
BAND_NAMES = {1: "road_boundary", 2: "building_edge", 3: "structure_edge", 4: "wall", 5: "fence", 6: "guard_rail", 7: "green_edge"}

_G: dict = {}


def clean_frames() -> list[int]:
    return sorted(json.loads(CLEAN_JSON.read_text())["clean"])


def frames_arg(spec: str) -> list[int]:
    if spec == "clean":
        return clean_frames()
    if spec == "all":
        return list(range(len(load_poses())))
    return [int(x) for x in spec.split(",")]


def gather_labels(store: CloudStore, pl: PointLabels, fp: FrameProducts, fill_px: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """(labels [ZB_H,ZB_W] uint8, valid) — label of the point in each cell, holes <= fill_px filled."""
    pid = fp.point_id
    valid = pid != NO_POINT
    rr, cc = render.fill_holes(valid, fill_px)
    pid_f = pid[rr, cc]
    valid_f = valid[rr, cc]
    out = np.full(pid.shape, C.IGNORE, np.uint8)
    out[valid_f] = pl.at(pid_f[valid_f])
    return out, valid_f


def vehicle_cells(vm: VehicleMask) -> np.ndarray:
    v, u = np.mgrid[0:ZB_H, 0:ZB_W]
    s = PANO_W / ZB_W
    return vm((u + 0.5) * s, (v + 0.5) * s)


def render_one(k: int, store, pl, fi, vm_cells, objects, range_max: float, poses, write_qa: bool = True) -> dict:
    fp = FrameProducts.load(k)
    lab, valid = gather_labels(store, pl, fp)
    depth = fp.depth_m
    lab[~valid] = C.IGNORE
    lab[valid & (depth > range_max)] = C.IGNORE
    lab[vm_cells] = C.IGNORE
    bands, _occ = vectors.render_objects(objects, fi.R[k], fi.C[k], fp, BAND_IDS, scale=ZB_W / PANO_W, r_max=range_max)
    bands[vm_cells] = 0
    cv2.imwrite(str(LABELS_DIR / f"f{k:04d}.png"), lab)
    cv2.imwrite(str(BANDS_DIR / f"f{k:04d}.png"), bands)
    if write_qa:
        photo = cv2.imread(poses.path(k))
        pal = C.palette()[..., ::-1]  # BGR
        vis = render.overlay_on_photo(photo, pal[lab], lab != C.IGNORE, 0.55)
        vis[bands > 0] = (0.4 * vis[bands > 0] + np.array([0, 255, 255]) * 0.6).astype(np.uint8)
        cv2.imwrite(str(QA_DIR / f"f{k:04d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
    counts = np.bincount(lab.ravel(), minlength=256)
    band_counts = np.bincount(bands.ravel(), minlength=8)
    store.release()
    return {"frame": k, "counts": counts[: C.N_CLASSES].tolist(), "ignore": int(counts[255]), "bands": band_counts[1:].tolist()}


def _init(range_max):
    _G["store"] = CloudStore()
    _G["pl"] = PointLabels(_G["store"])
    _G["poses"] = load_poses()
    _G["fi"] = FrameIndex(_G["poses"])
    _G["vm"] = vehicle_cells(VehicleMask())
    _G["objects"] = load_objects()
    _G["range_max"] = range_max


def _work(k: int) -> dict:
    return render_one(k, _G["store"], _G["pl"], _G["fi"], _G["vm"], _G["objects"], _G["range_max"], _G["poses"])


def build(frames: str = "clean", workers: int = 6, limit: int | None = None, range_max: float = C.RULES["range_max_m"]) -> list[dict]:
    for d in (LABELS_DIR, BANDS_DIR, QA_DIR):
        d.mkdir(parents=True, exist_ok=True)
    ks = frames_arg(frames)[:limit]
    res = []
    with Pool(workers, initializer=_init, initargs=(range_max,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, ks, chunksize=2)):
            res.append(r)
            if i % 50 == 0:
                lab_frac = 1 - r["ignore"] / (ZB_H * ZB_W)
                print(f"[{i + 1}/{len(ks)}] frame {r['frame']}: labelled {lab_frac:.2f}")
    res.sort(key=lambda r: r["frame"])
    (SEGDS_DIR / "labels_erp_stats.json").write_text(json.dumps({"frames": res, "classes": [c.name for c in C.CLASSES], "bands": BAND_NAMES, "range_max_m": range_max}))
    return res
