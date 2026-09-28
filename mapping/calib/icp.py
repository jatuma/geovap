"""Edge-ICP rig calibration: nearest-photo-edge association + robust least squares.

For each calibration frame the photo's strong, non-textured edges are computed once at CHAM
resolution together with the index map of the nearest edge pixel for every pixel (from the
Euclidean distance transform). For a rig hypothesis theta the cloud's silhouette / depth-edge
points are projected and paired with their nearest edge pixel; the 2-D pixel residuals are
minimised with a soft-L1 loss, re-associating after each solve with a shrinking window.

theta = (omega, phi, kappa [deg], dt [s], lx, ly, lz [m])  — any subset can be frozen.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from geovap.domain.math import edges as edges_math
from geovap.domain.model import geometry
from ..cloud_store import CloudStore
from ..config import DEG_PER_PX, PANO_H, PANO_W, R_MAX, R_MIN
from geovap.domain.model.frames import FrameIndex
from ..poses import Poses
from geovap.domain.model.rig import RigModel
from . import chamfer as Ch

NAMES = ["omega", "phi", "kappa", "dt", "lx", "ly", "lz"]


@dataclass
class IcpFrame:
    frame: int
    xyz: np.ndarray  # [N,3]
    kind: np.ndarray  # [N] 1 depth edge, 2 sky silhouette
    edge_idx: np.ndarray  # [2, CHAM_H, CHAM_W] int16: (row, col) of nearest edge pixel
    dt: np.ndarray  # [CHAM_H, CHAM_W] float16 distance to it (CHAM px)
    valid: np.ndarray  # photo validity


class _PoseOverride:
    """Minimal FrameIndex-like shim (R, C, t, poses) with one frame's pose overridden, so silhouette
    selection (which reads `fi.R[frame]`, `fi.C[frame]`, `fi.t[frame]`) can be redone under a
    hypothesis pose without a full `FrameIndex` rebuild. Arrays are only copied when overridden, so
    passing R=C=t=None (the default in `prepare_frame`) never runs this path."""

    __slots__ = ("R", "C", "t", "poses")

    def __init__(self, fi: FrameIndex, frame: int, R: np.ndarray | None, C: np.ndarray | None, t: float | None):
        self.R, self.C, self.t, self.poses = fi.R, fi.C, fi.t, fi.poses
        if R is not None or C is not None:
            self.R, self.C = fi.R.copy(), fi.C.copy()
            if R is not None:
                self.R[frame] = R
            if C is not None:
                self.C[frame] = C
        if t is not None:
            self.t = fi.t.copy()
            self.t[frame] = t


class _OwnPassGather:
    """Context manager that temporarily restricts `calib.chamfer.fine_edge_points`'s candidate
    gather to the photo's OWN pass (`Poses.pass_of_time` of each candidate's gps_time).

    `chamfer.fine_edge_points` resolves `gather_candidates` via a local `from ..products import
    gather_candidates` at call time, so patching the name on the `mapping.products` module (not
    touched here otherwise -- out of scope for S4) is picked up without duplicating its ~40 lines
    of zbuffer/edge-detection logic. S5 found 0.1-0.6 m offsets between passes; the plain ±45 s
    time window (`products.TIME_WINDOW_S`) also admits a neighbouring pass whenever two passes are
    < ~90 s apart in time (observed: pass 24's first frames pull 39-43% of candidates from pass 23,
    a 24 s gap -- a double silhouette that inflates the edge residual there). Not thread-safe against
    concurrent `prepare_frame(..., own_pass_only=True)` calls in the same process; `refine_pass`
    only ever calls it sequentially."""

    def __init__(self, poses: Poses, frame: int):
        self.poses = poses
        self.own_pass = int(poses.pass_id[frame])
        self._orig = None

    def _patched(self, store, C, r_max=R_MAX, t_frame=None, time_window_s=None):
        xyz, pid = self._orig(store, C, r_max, t_frame, time_window_s)
        if len(pid) == 0:
            return xyz, pid
        ti, local = store.locate(pid)
        gps = np.empty(len(pid))
        for t_idx in np.unique(ti):
            m = ti == t_idx
            gps[m] = np.asarray(store.tile(store.tiles[t_idx].name).gps_time[local[m]])
        keep = self.poses.pass_of_time(gps) == self.own_pass
        return xyz[keep], pid[keep]

    def __enter__(self):
        import mapping.products as _products

        self._products = _products
        self._orig = _products.gather_candidates
        _products.gather_candidates = self._patched
        return self

    def __exit__(self, *exc):
        self._products.gather_candidates = self._orig


def prepare_frame(frame: int, store: CloudStore, fi: FrameIndex, vmask, n_points: int = 30_000, seed: int = 0, R: np.ndarray | None = None, C: np.ndarray | None = None, t: float | None = None, own_pass_only: bool = False) -> IcpFrame:
    """As before; if `R`/`C`/`t` are given, silhouette selection (own-cell fine z-buffer +
    time-windowed candidate gather) uses that hypothesis pose for this frame instead of `fi`'s
    stored one. Passing none of them reproduces the previous behaviour exactly.

    `own_pass_only=True` (default False, so rig calibration -- `calib.fit`/`calib.diagnostics` --
    is byte-identical) additionally drops candidate points scanned during a different photo pass;
    see `_OwnPassGather`."""
    if R is not None or C is not None or t is not None:
        fi = _PoseOverride(fi, frame, R, C, t)
    if own_pass_only:
        with _OwnPassGather(fi.poses, frame):
            cf = Ch.prepare_frame(frame, store, fi, vmask, n_points=10**9, use_intensity=False)
    else:
        cf = Ch.prepare_frame(frame, store, fi, vmask, n_points=10**9, use_intensity=False)
    xyz, kind = cf.xyz, cf.kind
    if len(xyz) > n_points:
        rng = np.random.default_rng(seed + frame)
        sel = rng.choice(len(xyz), n_points, replace=False)
        xyz, kind = xyz[sel], kind[sel]
    # recompute the edge map exactly as chamfer does, but keep the nearest-edge indices
    edges, valid = _photo_edges(fi.poses, frame, vmask)
    dt, idx = edges_math.edge_distance_transform(edges)
    return IcpFrame(frame, xyz, kind, idx, dt, valid)


def _photo_edges(poses: Poses, frame: int, vmask) -> tuple[np.ndarray, np.ndarray]:
    """Load the frame's luminance and hand it to `domain.math.edges.photo_edges`."""
    from .objective import photo_luminance

    lum = photo_luminance(poses, frame, Ch.CHAM_W, Ch.CHAM_H)
    return edges_math.photo_edges(
        lum, pano_w=PANO_W, pano_h=PANO_H, vmask=vmask,
        edge_pct=Ch.EDGE_PCT, sky_edge_pct=Ch.SKY_EDGE_PCT,
        texture_win=Ch.TEXTURE_WIN, texture_density=Ch.TEXTURE_DENSITY,
    )


class EdgeICP:
    def __init__(self, frames: list[IcpFrame], poses: Poses, free: tuple[str, ...] = ("omega", "phi", "kappa", "dt", "lz"), r_max: float = R_MAX):
        self.frames = frames
        self.poses = poses
        self.idx = np.array([f.frame for f in frames])
        self.free = [NAMES.index(n) for n in free]
        self.theta_full = np.zeros(7)
        self.r_max = r_max
        self.s = Ch.CHAM_W / PANO_W

    def rig(self, theta_free) -> RigModel:
        th = self.theta_full.copy()
        th[self.free] = theta_free
        return RigModel.from_vector(th, with_lever_arm=True)

    def poses_for(self, theta_free) -> tuple[np.ndarray, np.ndarray]:
        """(R[K,3,3], C[K,3]) for `self.frames` under hypothesis `theta_free`. Overridable by
        subclasses that parametrise pose differently (e.g. per-frame corrections instead of a
        single rig)."""
        rig = self.rig(theta_free)
        return geometry.frame_rotations(self.poses, rig, self.idx)

    def residuals(self, theta_free, window: float, return_meta: bool = False):
        R, C = self.poses_for(theta_free)
        res = []
        meta = []
        for i, f in enumerate(self.frames):
            u, v, r, el = geometry.world_to_pano(f.xyz, R[i], C[i], PANO_W, PANO_H, dtype=np.float64)
            m = (r >= R_MIN) & (r <= self.r_max)
            x = u[m] * self.s
            y = v[m] * self.s
            xi = np.mod(np.floor(x).astype(np.int64), Ch.CHAM_W)
            yi = np.clip(np.floor(y).astype(np.int64), 0, Ch.CHAM_H - 1)
            d = f.dt[yi, xi].astype(np.float32)
            ok = f.valid[yi, xi] & (d <= window)
            er = f.edge_idx[0, yi[ok], xi[ok]].astype(np.float64) + 0.5
            ec = f.edge_idx[1, yi[ok], xi[ok]].astype(np.float64) + 0.5
            du = ec - x[ok]
            du = (du + Ch.CHAM_W / 2) % Ch.CHAM_W - Ch.CHAM_W / 2
            dv = er - y[ok]
            res.append(np.concatenate([du, dv]))
            if return_meta:
                meta.append({"frame": f.frame, "n": int(ok.sum()), "n_total": int(m.sum()), "el": el[m][ok], "az": (u[m][ok] / PANO_W * 360.0), "r": r[m][ok], "du": du, "dv": dv, "kind": f.kind[m][ok]})
        out = np.concatenate(res) if res else np.zeros(0)
        return (out, meta) if return_meta else out

    # ---------------------------------------------------------------- fixed-correspondence ICP
    def associate(self, theta_free, window: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per frame: (point mask, target x, target y) in CHAM px for points whose nearest edge is within window."""
        R, C = self.poses_for(theta_free)
        out = []
        for i, f in enumerate(self.frames):
            u, v, r, el = geometry.world_to_pano(f.xyz, R[i], C[i], PANO_W, PANO_H, dtype=np.float64)
            x = u * self.s
            y = v * self.s
            xi = np.mod(np.floor(x).astype(np.int64), Ch.CHAM_W)
            yi = np.clip(np.floor(y).astype(np.int64), 0, Ch.CHAM_H - 1)
            ok = (r >= R_MIN) & (r <= self.r_max) & f.valid[yi, xi] & (f.dt[yi, xi].astype(np.float32) <= window)
            tx = f.edge_idx[1, yi[ok], xi[ok]].astype(np.float64) + 0.5
            ty = f.edge_idx[0, yi[ok], xi[ok]].astype(np.float64) + 0.5
            out.append((ok, tx, ty))
        return out

    def residuals_fixed(self, theta_free, assoc) -> np.ndarray:
        R, C = self.poses_for(theta_free)
        res = []
        for i, f in enumerate(self.frames):
            ok, tx, ty = assoc[i]
            if ok.sum() == 0:
                continue
            u, v, r, el = geometry.world_to_pano(f.xyz[ok], R[i], C[i], PANO_W, PANO_H, dtype=np.float64)
            du = tx - u * self.s
            du = (du + Ch.CHAM_W / 2) % Ch.CHAM_W - Ch.CHAM_W / 2
            res.append(np.concatenate([du, ty - v * self.s]))
        return np.concatenate(res) if res else np.zeros(1)

    def solve(self, theta0=None, windows=(40.0, 30.0, 20.0, 12.0, 8.0, 6.0), f_scale: float = 2.0, log=print) -> dict:
        th = np.zeros(len(self.free)) if theta0 is None else np.asarray(theta0, dtype=np.float64)
        history = []
        for w in windows:
            t = time.time()
            assoc = self.associate(th, w)
            r0 = self.residuals_fixed(th, assoc)
            sol = least_squares(self.residuals_fixed, th, args=(assoc,), loss="soft_l1", f_scale=f_scale, x_scale=np.array([0.1 if NAMES[i] != "dt" else 0.02 for i in self.free]), max_nfev=40)
            th = sol.x
            r1 = self.residuals_fixed(th, assoc)
            rec = {"window": w, "n": int(len(r1) // 2), "rms_before": float(np.sqrt(np.mean(r0**2))), "rms_after": float(np.sqrt(np.mean(r1**2))), "theta": dict(zip([NAMES[i] for i in self.free], np.round(th, 4).tolist())), "nfev": int(sol.nfev), "s": round(time.time() - t, 1)}
            history.append(rec)
            log(f"  window {w:4.0f}px: n={rec['n']:6d} rms {rec['rms_before']:.2f} -> {rec['rms_after']:.2f}  theta={rec['theta']}  ({sol.nfev} fev, {rec['s']} s)")
        rig = self.rig(th)
        return {"theta_free": th.tolist(), "free": [NAMES[i] for i in self.free], "rig": {"boresight_deg": list(rig.boresight_deg), "lever_arm_m": list(rig.lever_arm_m), "dt_s": rig.dt_s}, "history": history}

    def structure(self, theta_free, window: float = 20.0) -> dict:
        """Residual medians by elevation / azimuth / range bands (full-res px) for before/after reporting."""
        _, meta = self.residuals(theta_free, window, return_meta=True)
        el = np.concatenate([m["el"] for m in meta])
        az = np.concatenate([m["az"] for m in meta])
        r = np.concatenate([m["r"] for m in meta])
        du = np.concatenate([m["du"] for m in meta]) / self.s
        dv = np.concatenate([m["dv"] for m in meta]) / self.s
        out = {"n": int(len(du)), "du_median_px": float(np.median(du)), "dv_median_px": float(np.median(dv)), "du_mad_px": float(np.median(np.abs(du - np.median(du)))), "dv_mad_px": float(np.median(np.abs(dv - np.median(dv)))), "rms_px": float(np.sqrt(np.mean(du**2 + dv**2)))}
        out["by_el"] = {f"{e0:+d}": [int(((el >= e0) & (el < e0 + 10)).sum()), float(np.median(dv[(el >= e0) & (el < e0 + 10)])) if ((el >= e0) & (el < e0 + 10)).sum() > 20 else None] for e0 in range(-60, 60, 10)}
        out["by_az"] = {f"{a0}": [int(((az >= a0) & (az < a0 + 45)).sum()), float(np.median(dv[(az >= a0) & (az < a0 + 45)])), float(np.median(du[(az >= a0) & (az < a0 + 45)]))] for a0 in range(0, 360, 45)}
        out["by_r"] = {f"{r0}-{r1}": [int(((r >= r0) & (r < r1)).sum()), float(np.median(dv[(r >= r0) & (r < r1)])) if ((r >= r0) & (r < r1)).sum() > 20 else None] for r0, r1 in ((1, 5), (5, 10), (10, 15), (15, 25), (25, 40))}
        return out
