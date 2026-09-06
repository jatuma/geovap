"""2D rasters over the surveyed area used by the 3D point rules.

    face_class.npy      u8   [ny,nx]  GT class id of the JVF face at 0.1 m (255 none)
    face_class_buf.npy  u8            same, building faces buffered by RULES["bldg_buffer"]
    dist_<g>.npy        u8            distance (dm, clipped 25.5 m) to the nearest line of group g
    z_<g>.npy           f16           height of that nearest line sample
    pole_dist.npy       u8            distance (dm) to the nearest `nosič technického zařízení`
    dtm.npy             f32 [ny2,nx2] min ground (class 2) height per 0.5 m cell, holes filled by nearest
    meta.json           grid origin / cell sizes / hashes
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from ..cloud_store import CloudStore
from . import classes as C
from .areas import AREAS_DIR, SEGDS_DIR, load_faces, load_objects

RASTER_DIR = SEGDS_DIR / "rasters"
CELL = 0.1
DTM_CELL = 0.5
PAD_M = 5.0
LINE_SAMPLE_M = 0.05


@dataclass
class Grid:
    e0: float
    n0: float
    cell: float
    nx: int
    ny: int

    def ij(self, e: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(row, col) int64 arrays, clipped to the grid."""
        j = np.clip(((np.asarray(e) - self.e0) / self.cell).astype(np.int64), 0, self.nx - 1)
        i = np.clip(((np.asarray(n) - self.n0) / self.cell).astype(np.int64), 0, self.ny - 1)
        return i, j

    def inside(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        return (e >= self.e0) & (e < self.e0 + self.nx * self.cell) & (n >= self.n0) & (n < self.n0 + self.ny * self.cell)

    def px(self, xy: np.ndarray) -> np.ndarray:
        """[K,2] world -> [K,2] int32 pixel (col,row) for cv2 (row grows with N)."""
        return np.stack([(xy[:, 0] - self.e0) / self.cell, (xy[:, 1] - self.n0) / self.cell], -1).round().astype(np.int32)


def grid_from_store(store: CloudStore, cell: float = CELL, pad: float = PAD_M) -> Grid:
    bb = np.array([t.bbox for t in store.tiles])
    e0, n0 = np.floor(bb[:, 0].min() - pad), np.floor(bb[:, 1].min() - pad)
    e1, n1 = np.ceil(bb[:, 2].max() + pad), np.ceil(bb[:, 3].max() + pad)
    return Grid(float(e0), float(n0), cell, int(np.ceil((e1 - e0) / cell)), int(np.ceil((n1 - n0) / cell)))


def rasterize_faces(faces, grid: Grid, buffer_building_m: float = 0.0) -> np.ndarray:
    """Face class raster; larger faces first so small faces (sheds inside gardens) win. Holes are handled
    by drawing exterior then interiors with the enclosing value."""
    img = np.full((grid.ny, grid.nx), C.IGNORE, np.uint8)
    order = sorted((f for f in faces if f.gt_class), key=lambda f: -f.area)
    for f in order:
        poly = f.polygon
        if buffer_building_m > 0 and f.gt_class == "building":
            poly = poly.buffer(buffer_building_m, join_style=2)
        polys = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        for p in polys:
            cv2.fillPoly(img, [grid.px(np.asarray(p.exterior.coords))], int(f.class_id))
            for ring in p.interiors:
                cv2.fillPoly(img, [grid.px(np.asarray(ring.coords))], int(C.IGNORE))
    return img


def _sample_lines(objects, codes: set[str], step: float = LINE_SAMPLE_M) -> np.ndarray:
    """[K,3] densely sampled (E,N,H) along all lines of the given codes (polygons as rings)."""
    out = []
    for o in objects:
        if o.jvfcode not in codes or o.geom_type not in ("LineString", "Polygon") or len(o.coords) < 2:
            continue
        c = o.coords
        if np.isnan(c).any():
            continue
        if o.geom_type == "Polygon" and not np.allclose(c[0], c[-1]):
            c = np.vstack([c, c[:1]])
        for a, b in zip(c[:-1], c[1:]):
            L = np.linalg.norm(b[:2] - a[:2])
            n = max(1, int(np.ceil(L / step)))
            t = np.linspace(0, 1, n + 1)[:, None]
            out.append(a[None] + (b - a)[None] * t)
    return np.concatenate(out) if out else np.zeros((0, 3))


def line_rasters(samples: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """(dist_dm u8, z f16) of the nearest sampled line point for every cell."""
    occ = np.ones((grid.ny, grid.nx), bool)
    z = np.zeros((grid.ny, grid.nx), np.float32)
    if len(samples):
        m = grid.inside(samples[:, 0], samples[:, 1])
        s = samples[m]
        i, j = grid.ij(s[:, 0], s[:, 1])
        occ[i, j] = False
        z[i, j] = s[:, 2]
    dist, (ii, jj) = ndimage.distance_transform_edt(occ, return_indices=True)
    dist_dm = np.clip(dist * grid.cell * 10.0, 0, 255).astype(np.uint8)
    z_near = z[ii, jj].astype(np.float16)
    del ii, jj
    return dist_dm, z_near


def build_dtm(store: CloudStore, grid: Grid, cell: float = DTM_CELL, chunk: int = 5_000_000) -> tuple[np.ndarray, Grid]:
    g = Grid(grid.e0, grid.n0, cell, int(np.ceil(grid.nx * grid.cell / cell)), int(np.ceil(grid.ny * grid.cell / cell)))
    dtm = np.full((g.ny, g.nx), np.inf, np.float32)
    for t in store.tiles:
        td = store.tile(t.name)
        n = len(td)
        for s in range(0, n, chunk):
            rows = slice(s, min(n, s + chunk))
            cls = np.asarray(td.classification[rows])
            g_m = cls == 2
            if not g_m.any():
                continue
            xyz = td.xyz_m(rows)[g_m]
            i, j = g.ij(xyz[:, 0], xyz[:, 1])
            np.minimum.at(dtm, (i, j), xyz[:, 2].astype(np.float32))
        td.release()
    empty = ~np.isfinite(dtm)
    if empty.any():
        _, (ii, jj) = ndimage.distance_transform_edt(empty, return_indices=True)
        dtm = dtm[ii, jj]
    return dtm, g


def build(out_dir: Path = RASTER_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = CloudStore()
    grid = grid_from_store(store)
    faces = load_faces(AREAS_DIR / "faces.geojson")
    objects = load_objects()
    print(f"grid {grid.nx}x{grid.ny} @ {grid.cell} m")
    np.save(out_dir / "face_class.npy", rasterize_faces(faces, grid))
    np.save(out_dir / "face_class_buf.npy", rasterize_faces(faces, grid, C.RULES["bldg_buffer"]))
    for gname, codes in C.LINE_GROUPS.items():
        s = _sample_lines(objects, codes)
        d, z = line_rasters(s, grid)
        np.save(out_dir / f"dist_{gname}.npy", d)
        np.save(out_dir / f"z_{gname}.npy", z)
        print(f"lines {gname}: {len(s)} samples, cells within 1 m: {(d <= 10).mean():.4f}")
    poles = np.array([o.coords.mean(0) for o in objects if o.jvfcode == C.POLE_CODE])
    d, _ = line_rasters(poles, grid)
    np.save(out_dir / "pole_dist.npy", d)
    dtm, g2 = build_dtm(store, grid)
    np.save(out_dir / "dtm.npy", dtm)
    meta = {"grid": asdict(grid), "dtm_grid": asdict(g2), "rules_hash": C.rules_hash(), "n_faces": len(faces), "line_groups": {k: sorted(v) for k, v in C.LINE_GROUPS.items()}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"dtm {g2.nx}x{g2.ny}, range {np.nanmin(dtm):.1f}..{np.nanmax(dtm):.1f} m -> {out_dir}")
    return meta


class Rasters:
    """Memmapped access for the labelling workers."""

    def __init__(self, root: Path = RASTER_DIR):
        self.root = Path(root)
        meta = json.loads((self.root / "meta.json").read_text())
        self.meta = meta
        self.grid = Grid(**meta["grid"])
        self.dtm_grid = Grid(**meta["dtm_grid"])
        self.face_class = np.load(self.root / "face_class.npy", mmap_mode="r")
        self.face_class_buf = np.load(self.root / "face_class_buf.npy", mmap_mode="r")
        self.dist = {g: np.load(self.root / f"dist_{g}.npy", mmap_mode="r") for g in C.LINE_GROUPS}
        self.z = {g: np.load(self.root / f"z_{g}.npy", mmap_mode="r") for g in C.LINE_GROUPS}
        self.pole_dist = np.load(self.root / "pole_dist.npy", mmap_mode="r")
        self.dtm = np.load(self.root / "dtm.npy", mmap_mode="r")

    def hag(self, e, n, z) -> np.ndarray:
        i, j = self.dtm_grid.ij(e, n)
        return np.asarray(z, np.float32) - self.dtm[i, j]
