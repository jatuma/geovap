"""uint8 GT label for every point of the store, from the JVF face/line rasters and height above ground.

Rules (first match wins), thresholds in classes.RULES:
 1 fence   d_fence <= 0.35 and 0 < z - z_fence <= 2.5 ; ring 0.35..1.0 m with hag > 0.3 -> ignore (hedges)
 2 rail    d_rail <= 0.3 and 0.2 < z - z_rail <= 1.5
 3 wall    d_wall <= 0.3 and 0 < hag <= 3
 4 building: buffered building face and hag > 0.2 (inside and hag <= 0.2 -> ignore: floors/yards)
 5 struct edge (hranice stavby) within 0.5 m and hag > 0.2 -> ignore
 6 surface: face class where hag <= 0.2 (water <= 0.3, stairs <= 3 m)
 7 vegetation: green face, hag > 0.5, > 1 m from building edges (0.2 < hag <= 0.5 -> ignore)
 8 pole   within 0.4 m of a nosič and 0.2 < hag < 12
 9 else ignore (above-ground points over roads: vehicles, unknown)
Output segds/point_labels/NN.npy in store row order; hist.json; meta.json (rules hash).
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..cloud_store import CloudStore
from . import classes as C
from .areas import SEGDS_DIR
from .rasters import Rasters

LABEL_DIR = SEGDS_DIR / "point_labels"
CHUNK = 5_000_000

_R: Rasters | None = None
_STORE: CloudStore | None = None

GREEN_IDS = {C.BY_NAME["terrain"].id}
SURFACE_IDS = {C.BY_NAME[n].id for n in ("road", "sidewalk", "terrain", "water", "stairs", "verge", "paved_other", "culvert_head", "structure_other", "road_or_verge")}
HARD_SURFACE_IDS = {C.BY_NAME[n].id for n in ("road", "sidewalk", "verge", "paved_other", "road_or_verge")}


def label_points(xyz: np.ndarray, R: Rasters, rules: dict = C.RULES) -> np.ndarray:
    """xyz [N,3] float (E,N,H) -> labels [N] uint8."""
    e, n, z = xyz[:, 0], xyz[:, 1], xyz[:, 2].astype(np.float32)
    N = len(xyz)
    lab = np.full(N, C.IGNORE, np.uint8)
    inside = R.grid.inside(e, n)
    if not inside.any():
        return lab
    i, j = R.grid.ij(e, n)
    hag = R.hag(e, n, z)
    face = np.asarray(R.face_class[i, j])
    face_b = np.asarray(R.face_class_buf[i, j])
    d = {g: np.asarray(R.dist[g][i, j], np.float32) * 0.1 for g in C.LINE_GROUPS}
    zl = {g: np.asarray(R.z[g][i, j], np.float32) for g in ("fence", "rail")}
    undecided = inside.copy()

    def take(mask, cls):
        m = mask & undecided
        lab[m] = cls
        undecided[m] = False

    dz_f = z - zl["fence"]
    take((d["fence"] <= rules["fence_dist"]) & (dz_f > rules["fence_dz_min"]) & (dz_f <= rules["fence_dz_max"]), C.BY_NAME["fence"].id)
    take((d["fence"] <= rules["fence_ring"]) & (hag > rules["fence_ring_hag_min"]) & (hag <= rules["fence_dz_max"]), C.IGNORE)
    dz_r = z - zl["rail"]
    take((d["rail"] <= rules["rail_dist"]) & (dz_r > rules["rail_dz_min"]) & (dz_r <= rules["rail_dz_max"]), C.BY_NAME["guard_rail"].id)
    take((d["wall"] <= rules["wall_dist"]) & (hag > 0) & (hag <= rules["wall_hag_max"]), C.BY_NAME["wall"].id)
    bld = face_b == C.BY_NAME["building"].id
    take(bld & (hag > rules["bldg_hag_min"]), C.BY_NAME["building"].id)
    take(bld, C.IGNORE)
    take((d["struct_edge"] <= rules["struct_dist"]) & (hag > rules["ground_hag"]), C.IGNORE)
    ground = hag <= rules["ground_hag"]
    water = face == C.BY_NAME["water"].id
    take(water & (hag <= rules["water_hag"]), C.BY_NAME["water"].id)
    stairs = face == C.BY_NAME["stairs"].id
    take(stairs & (hag <= rules["stairs_hag"]), C.BY_NAME["stairs"].id)
    for cid in SURFACE_IDS - {C.BY_NAME["water"].id, C.BY_NAME["stairs"].id}:
        take((face == cid) & ground, cid)
    green = np.isin(face, list(GREEN_IDS))
    take(green & (hag > rules["veg_hag_min"]) & (d["bldg_edge"] > rules["veg_bldg_dist"]), C.BY_NAME["vegetation"].id)
    pole_d = np.asarray(R.pole_dist[i, j], np.float32) * 0.1
    take((pole_d <= rules["pole_dist"]) & (hag > rules["ground_hag"]) & (hag < rules["pole_hag_max"]), C.BY_NAME["pole"].id)
    return lab


def _init(root):
    global _R, _STORE
    _R = Rasters(root)
    _STORE = CloudStore()


def _label_tile(name: str) -> tuple[str, np.ndarray]:
    td = _STORE.tile(name)
    n = len(td)
    out = np.empty(n, np.uint8)
    for s in range(0, n, CHUNK):
        rows = slice(s, min(n, s + CHUNK))
        out[rows] = label_points(td.xyz_m(rows), _R)
    td.release()
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    np.save(LABEL_DIR / f"{name}.npy", out)
    return name, np.bincount(out, minlength=256)


def build(workers: int = 4, out_dir: Path = LABEL_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = CloudStore()
    names = [t.name for t in store.tiles]
    hist = np.zeros(256, np.int64)
    with Pool(workers, initializer=_init, initargs=(Rasters().root,)) as pool:
        for name, h in pool.imap_unordered(_label_tile, names):
            hist += h
            print(f"tile {name}: {h.sum()} pts, labelled {1 - h[255] / h.sum():.3f}")
    total = int(hist.sum())
    stats = {c.name: {"n": int(hist[c.id]), "frac": hist[c.id] / total} for c in C.CLASSES}
    stats["ignore"] = {"n": int(hist[255]), "frac": hist[255] / total}
    (out_dir / "hist.json").write_text(json.dumps(stats, indent=1))
    (out_dir / "meta.json").write_text(json.dumps({"rules_hash": C.rules_hash(), "total": total, "tiles": names}))
    for k, v in stats.items():
        print(f"{k:16s} {v['frac']:.4f}")
    return stats


class PointLabels:
    """Per-tile memmaps aligned with the store; `at(point_id)` for global ids."""

    def __init__(self, store: CloudStore, root: Path = LABEL_DIR):
        self.store = store
        self.arrays = [np.load(Path(root) / f"{t.name}.npy", mmap_mode="r") for t in store.tiles]

    def at(self, point_id: np.ndarray) -> np.ndarray:
        ti, local = self.store.locate(point_id)
        out = np.empty(len(point_id), np.uint8)
        for t in np.unique(ti):
            m = ti == t
            loc = local[m]
            order = np.argsort(loc)
            vals = self.arrays[t][loc[order]]
            out[np.flatnonzero(m)[order]] = vals
        return out
