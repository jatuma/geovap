"""Per-frame products: depth panorama + point-id panorama at z-buffer resolution.

These are the inverse-mapping enablers: `pano_to_world` and `pano_to_point` read them, and the
visibility test in colorization uses the closed depth. Products are tied to a rig model (hash in
meta); a refit invalidates them.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry, zbuffer
from .cloud_store import CloudStore
from .config import FRAMES_DIR, NO_POINT, PANO_H, PANO_W, R_MAX, R_MIN, ZB_H, ZB_W
from .frame_select import FrameIndex
from .poses import Poses, load_poses
from .rig import IDENTITY, RigModel

_GIT_REV = None


def git_rev() -> str:
    global _GIT_REV
    if _GIT_REV is None:
        try:
            _GIT_REV = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
        except Exception:
            _GIT_REV = "unknown"
    return _GIT_REV


def product_path(frame: int, root: Path = FRAMES_DIR) -> Path:
    return Path(root) / f"f{frame:04d}.npz"


@dataclass
class FrameProducts:
    frame: int
    depth_mm: np.ndarray  # uint16 [ZB_H, ZB_W], 0 = empty
    point_id: np.ndarray  # uint32 [ZB_H, ZB_W], NO_POINT = empty
    meta: dict
    _closed: np.ndarray | None = None
    _spread: np.ndarray | None = None

    @property
    def depth_m(self) -> np.ndarray:
        return zbuffer.depth_from_mm(self.depth_mm)

    @property
    def depth_closed(self) -> np.ndarray:
        if self._closed is None:
            self._closed = zbuffer.close_depth(self.depth_m)
        return self._closed

    @property
    def spread(self) -> np.ndarray:
        if self._spread is None:
            self._spread = zbuffer.depth_spread(self.depth_closed)
        return self._spread

    @property
    def scale(self) -> float:
        """full-res px -> z-buffer px"""
        return self.depth_mm.shape[1] / PANO_W

    def save(self, root: Path = FRAMES_DIR) -> Path:
        p = product_path(self.frame, root)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, depth_mm=self.depth_mm, point_id=self.point_id, meta=json.dumps(self.meta))
        return p

    @classmethod
    def load(cls, frame: int, rig: RigModel | None = None, root: Path = FRAMES_DIR, allow_stale: bool = False) -> "FrameProducts":
        with np.load(product_path(frame, root)) as z:
            meta = json.loads(str(z["meta"]))
            fp = cls(frame=frame, depth_mm=z["depth_mm"], point_id=z["point_id"], meta=meta)
        if rig is not None and meta["rig_hash"] != rig.hash() and not allow_stale:
            raise RuntimeError(f"frame {frame}: products built for rig {meta['rig_hash']}, requested {rig.hash()}")
        return fp

    # ----------------------------------------------------------------- inverse mapping
    def cell(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.scale
        h, w = self.depth_mm.shape
        cv = np.clip((np.asarray(v) * s).astype(np.int64), 0, h - 1)
        cu = np.mod((np.asarray(u) * s).astype(np.int64), w)
        return cv, cu

    def range_at(self, u, v, closed: bool = False) -> np.ndarray:
        cv, cu = self.cell(u, v)
        d = self.depth_closed if closed else self.depth_m
        return d[cv, cu]

    def point_at(self, u, v) -> np.ndarray:
        cv, cu = self.cell(u, v)
        return self.point_id[cv, cu]

    def visible(self, r: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Visibility of points at full-res (u, v) with range r."""
        s = self.scale
        return zbuffer.visible(r, np.asarray(u) * s, np.asarray(v) * s, self.depth_closed, spread_m=self.spread) & zbuffer.range_filter(r)


# ------------------------------------------------------------------------------------- building
TIME_WINDOW_S = 45.0  # occluders come only from points scanned within this window of the frame (same pass):
# the cloud merges 29 passes, a photo is one instant - gates, parked cars, people from other passes must not occlude.


def gather_candidates(store: CloudStore, C: np.ndarray, r_max: float = R_MAX, t_frame: float | None = None, time_window_s: float | None = None):
    """Points within r_max of camera centre C (2D), from all tiles; optionally only those scanned within
    time_window_s of t_frame. Returns (xyz_m f64 [N,3], point_id u32 [N])."""
    parts = store.query_disc(float(C[0]), float(C[1]), r_max)
    if not parts:
        return np.empty((0, 3)), np.empty(0, np.uint32)
    if t_frame is not None and time_window_s is not None:
        filt = []
        for t, rows in parts:
            g = np.asarray(store.tile(t.name).gps_time[rows])
            rows = rows[np.abs(g - t_frame) <= time_window_s]
            if len(rows):
                filt.append((t, rows))
        parts = filt
        if not parts:
            return np.empty((0, 3)), np.empty(0, np.uint32)
    xyz = np.concatenate([store.tile(t.name).xyz_m(rows) for t, rows in parts])
    pid = np.concatenate([store.global_ids(t, rows) for t, rows in parts])
    return xyz, pid


def build_frame(frame: int, store: CloudStore, fi: FrameIndex, rig: RigModel = IDENTITY, r_max: float = R_MAX, time_window_s: float | None = TIME_WINDOW_S) -> FrameProducts:
    R, C = fi.R[frame], fi.C[frame]
    xyz, pid = gather_candidates(store, C, r_max, fi.t[frame], time_window_s)
    u, v, r, el = geometry.world_to_pano(xyz, R, C)
    keep = zbuffer.range_filter(r, R_MIN, r_max)
    s = ZB_W / PANO_W
    depth, ids = zbuffer.splat(u[keep] * s, v[keep] * s, r[keep], pid[keep])
    meta = {
        "frame": frame,
        "filename": str(fi.poses.filename[frame]),
        "rig_hash": rig.hash(),
        "rig": {"boresight_deg": list(rig.boresight_deg), "lever_arm_m": list(rig.lever_arm_m), "dt_s": rig.dt_s},
        "zb": [ZB_H, ZB_W],
        "r_min": R_MIN,
        "r_max": r_max,
        "time_window_s": time_window_s,
        "n_candidates": int(len(pid)),
        "n_splatted": int(keep.sum()),
        "git": git_rev(),
    }
    return FrameProducts(frame=frame, depth_mm=zbuffer.depth_to_mm(depth), point_id=ids, meta=meta)


_G: dict = {}


def _init_worker(rig_vec, with_la):
    _G["store"] = CloudStore()
    _G["poses"] = load_poses()
    _G["rig"] = RigModel.from_vector(rig_vec, with_lever_arm=with_la)
    _G["fi"] = FrameIndex(_G["poses"], _G["rig"])


def _build_and_save(frame: int) -> int:
    fp = build_frame(frame, _G["store"], _G["fi"], _G["rig"])
    fp.save()
    return fp.meta["n_splatted"]


def build_all_frames(frames=None, rig: RigModel = IDENTITY, workers: int = 16) -> None:
    from multiprocessing import Pool

    from tqdm import tqdm

    poses = load_poses()
    frames = list(range(len(poses))) if frames is None else list(frames)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    rig.to_json(FRAMES_DIR / "rig.json")
    with Pool(workers, initializer=_init_worker, initargs=(rig.as_vector(True), True)) as pool:
        for _ in tqdm(pool.imap_unordered(_build_and_save, frames, chunksize=4), total=len(frames), desc="frames"):
            pass
