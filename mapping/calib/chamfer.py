"""Edge-point chamfer objective for rig calibration (Levinson & Thrun 2013 style).

Per calibration frame, once:
  * edge points: 3D points at depth discontinuities / sky silhouettes / laser-intensity
    discontinuities, found in the frame's (identity-rig) depth + point-id panorama;
  * photo edge distance transform (truncated) at CHAM_W x CHAM_H from the luminance gradient.
Per evaluation of a rig: project the edge points, sample the distance transform bilinearly, average.
Lower is better; smooth in the parameters because the DT is continuous.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .. import geometry
from ..cloud_store import CloudStore
from ..config import PANO_H, PANO_W, R_MAX, R_MIN
from ..frame_select import FrameIndex
from ..poses import Poses
from ..products import FrameProducts
from ..rig import RigModel
from .objective import photo_luminance

CHAM_W, CHAM_H = 4000, 2000
DT_CAP_PX = 20.0  # truncation of the distance transform (in CHAM px = 2 full-res px each)
EDGE_PCT = 95.0  # photo gradient percentile that counts as an edge
INTENSITY_EL_MAX = -12.0  # intensity edges are only trusted on the ground (road markings), not in vegetation
SKY_EDGE_PCT = 99.0  # stricter threshold above the horizon (clouds are soft, wires/roofs are sharp)
TEXTURE_WIN = 31  # window for edge density
TEXTURE_DENSITY = 0.12  # above this fraction of edge pixels the region is 'texture' and ignored


@dataclass
class ChamferFrame:
    frame: int
    xyz: np.ndarray  # [N,3] edge points (world)
    kind: np.ndarray  # [N] uint8: 1 depth edge, 2 sky silhouette, 3 intensity edge
    dt: np.ndarray  # [CHAM_H, CHAM_W] uint8 truncated distance to nearest photo edge
    valid: np.ndarray  # [CHAM_H, CHAM_W] bool (not vehicle / black cap)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """a shifted so that out[i,j] = a[i+dy, j+dx]; wraps in x, edge-pads in y."""
    out = np.roll(a, -dx, axis=1)
    if dy > 0:
        out = np.concatenate([out[dy:], np.repeat(out[-1:], dy, axis=0)], 0)
    elif dy < 0:
        out = np.concatenate([np.repeat(out[:1], -dy, axis=0), out[:dy]], 0)
    return out


def edge_cells(fp: FrameProducts, intensity_img: np.ndarray | None, vmask, el_min_deg: float = -60.0, rel: float = 0.15, abs_m: float = 0.5, int_thresh: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Boolean maps (depth/sky edge, intensity edge) at z-buffer resolution, on the NEAR side of the jump."""
    d = fp.depth_m
    fin = np.isfinite(d)
    h, w = d.shape
    el = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    depth_edge = np.zeros_like(fin)
    sky_edge = np.zeros_like(fin)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        dn = _shift(d, dy, dx)
        fn = np.isfinite(dn)
        jump = fin & fn & (dn - d > np.maximum(abs_m, rel * d))
        depth_edge |= jump
        sky_edge |= fin & ~fn
    sky_edge &= (el > 0.0)[:, None]  # empty neighbour above the horizon = silhouette against sky
    int_edge = np.zeros_like(fin)
    if intensity_img is not None:
        for dy, dx in ((0, 1), (1, 0)):
            a, b = intensity_img, _shift(intensity_img, dy, dx)
            fb = _shift(fin, dy, dx)
            e = fin & fb & (np.abs(a - b) > int_thresh)
            int_edge |= e | _shift(e, -dy, -dx)
        int_edge &= (el < INTENSITY_EL_MAX)[:, None]
    keep = (el >= el_min_deg)[:, None] & fin
    if vmask is not None:
        vv, uu = np.mgrid[0:h, 0:w]
        keep &= ~vmask(uu * (PANO_W / w), vv * (PANO_H / h))
    return depth_edge & keep & ~sky_edge, sky_edge & keep, int_edge & keep & ~(depth_edge | sky_edge)


def photo_edge_dt(poses: Poses, frame: int, vmask) -> tuple[np.ndarray, np.ndarray]:
    lum = photo_luminance(poses, frame, CHAM_W, CHAM_H)
    g = cv2.GaussianBlur(lum, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    valid = lum > 0.01
    if vmask is not None:
        vv, uu = np.mgrid[0:CHAM_H, 0:CHAM_W]
        valid &= ~vmask(uu * (PANO_W / CHAM_W), vv * (PANO_H / CHAM_H))
    lower = valid.copy()
    lower[: CHAM_H // 3] = False
    thr = np.percentile(mag[lower], EDGE_PCT)
    sky = valid.copy()
    sky[CHAM_H // 2 :] = False
    thr_sky = max(thr, np.percentile(mag[sky], SKY_EDGE_PCT)) if sky.any() else thr
    thr_map = np.where(np.arange(CHAM_H)[:, None] < CHAM_H // 2 - int(CHAM_H * 5 / 180), thr_sky, thr)  # above el +5: sky threshold
    edges = (mag > thr_map) & valid
    # textured regions (foliage, grass, gravel) have dense edges everywhere and carry no alignment
    # information -> treat them as edge-free so they contribute a constant (capped) distance
    density = cv2.blur(edges.astype(np.float32), (TEXTURE_WIN, TEXTURE_WIN))
    textured = density > TEXTURE_DENSITY
    edges &= ~textured
    # distance to nearest edge, wrap-aware via horizontal padding
    pad = int(DT_CAP_PX) + 2
    e = np.concatenate([edges[:, -pad:], edges, edges[:, :pad]], axis=1)
    dt = cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 5)[:, pad:-pad]
    dt = np.where(textured, DT_CAP_PX, dt)
    return np.clip(dt, 0, DT_CAP_PX).astype(np.uint8), valid


FINE_W, FINE_H = 4000, 2000


def fine_edge_points(frame: int, store: CloudStore, fi: FrameIndex, vmask, el_min_deg: float = -60.0, rel: float = 0.15, abs_m: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Silhouette / depth-edge 3D points selected by their OWN exact cell in a fine (FINE_W x FINE_H)
    z-buffer, so their projected position is exact (the coarse product's cell winners can come from a
    splat up to 8 cells away and jitter by ~1 deg). Returns (xyz [N,3], kind [N]: 1 depth edge, 2 sky)."""
    from .. import zbuffer
    from ..products import gather_candidates

    R, C = fi.R[frame], fi.C[frame]
    from ..products import TIME_WINDOW_S

    xyz, pid = gather_candidates(store, C, R_MAX, fi.t[frame], TIME_WINDOW_S)
    u, v, r, el = geometry.world_to_pano(xyz, R, C)
    keep = zbuffer.range_filter(r)
    xyz, u, v, r, el = xyz[keep], u[keep], v[keep], r[keep], el[keep]
    s = FINE_W / PANO_W
    x, y = u * s, v * s
    # splatted buffer (holes closed) for edge detection; clamp splat to [1,3] px at this resolution
    from ..config import SPLAT_MAX_PX
    import mapping.zbuffer as zb

    old = zb.SPLAT_MAX_PX
    zb.SPLAT_MAX_PX = 3
    try:
        depth, _ = zbuffer.splat(x, y, r, np.arange(len(r), dtype=np.uint32), FINE_W, FINE_H)
    finally:
        zb.SPLAT_MAX_PX = old
    d = zbuffer.close_depth(depth)
    fin = np.isfinite(d)
    h, w = d.shape
    el_rows = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    depth_edge = np.zeros_like(fin)
    sky_edge = np.zeros_like(fin)
    for dy, dx in ((0, 2), (0, -2), (2, 0), (-2, 0)):
        dn = _shift(d, dy, dx)
        fn = np.isfinite(dn)
        depth_edge |= fin & fn & (dn - d > np.maximum(abs_m, rel * d))
        sky_edge |= fin & ~fn
    sky_edge &= (el_rows > 0.0)[:, None]
    keep_rows = (el_rows >= el_min_deg)[:, None]
    if vmask is not None:
        vv, uu = np.mgrid[0:h, 0:w]
        keep_rows = keep_rows & ~vmask(uu * (PANO_W / w), vv * (PANO_H / h))
    depth_edge &= keep_rows
    sky_edge &= keep_rows
    # points whose own cell is an edge cell and which lie on the near surface of that cell
    cx = np.mod(np.floor(x).astype(np.int64), w)
    cy = np.clip(np.floor(y).astype(np.int64), 0, h - 1)
    near = r <= d[cy, cx] + np.maximum(0.15, 0.03 * r)
    is_sky = sky_edge[cy, cx] & near
    is_depth = depth_edge[cy, cx] & near & ~is_sky
    sel = is_sky | is_depth
    kind = np.where(is_sky, 2, 1).astype(np.uint8)[sel]
    return xyz[sel], kind


def prepare_frame(frame: int, store: CloudStore, fi: FrameIndex, vmask, n_points: int = 40_000, seed: int = 0, use_intensity: bool = False, fine: bool = True) -> ChamferFrame:
    if fine and not use_intensity:
        xyz, kinds = fine_edge_points(frame, store, fi, vmask)
        rng = np.random.default_rng(seed + frame)
        if len(xyz) > n_points:
            sel = rng.choice(len(xyz), n_points, replace=False)
            xyz, kinds = xyz[sel], kinds[sel]
        dt, valid = photo_edge_dt(fi.poses, frame, vmask)
        return ChamferFrame(frame, xyz, kinds, dt, valid)
    fp = FrameProducts.load(frame)
    inten_img = None
    pid = fp.point_id
    fin = np.isfinite(fp.depth_m)
    if use_intensity:
        ti, local = store.locate(pid[fin])
        vals = np.empty(fin.sum(), np.float32)
        for t_idx in np.unique(ti):
            m = ti == t_idx
            col = store.tile(store.tiles[t_idx].name).intensity
            order = np.argsort(local[m])
            v = np.asarray(col[local[m][order]], dtype=np.float32)
            vals[np.flatnonzero(m)[order]] = v
        x = np.log1p(vals)
        lo, hi = np.percentile(x, [2, 98])
        inten_img = np.zeros(pid.shape, np.float32)
        inten_img[fin] = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    de, se, ie = edge_cells(fp, inten_img, vmask)
    ids = np.concatenate([pid[de], pid[se], pid[ie]])
    kinds = np.concatenate([np.full(de.sum(), 1, np.uint8), np.full(se.sum(), 2, np.uint8), np.full(ie.sum(), 3, np.uint8)])
    ids, first = np.unique(ids, return_index=True)
    kinds = kinds[first]
    rng = np.random.default_rng(seed + frame)
    if len(ids) > n_points:
        sel = rng.choice(len(ids), n_points, replace=False)
        ids, kinds = ids[sel], kinds[sel]
    ti, local = store.locate(ids)
    xyz = np.empty((len(ids), 3))
    for t_idx in np.unique(ti):
        m = ti == t_idx
        xyz[m] = store.tile(store.tiles[t_idx].name).xyz_m(np.sort(local[m]))[np.argsort(np.argsort(local[m]))]
    dt, valid = photo_edge_dt(fi.poses, frame, vmask)
    return ChamferFrame(frame, xyz, kinds, dt, valid)


def _sample_dt(dt: np.ndarray, valid: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sample of the DT at full-res (u,v); returns (values, usable mask)."""
    s = CHAM_W / PANO_W
    x = np.mod(u * s - 0.5, CHAM_W)
    y = np.clip(v * s - 0.5, 0, CHAM_H - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    x1 = (x0 + 1) % CHAM_W
    y1 = np.minimum(y0 + 1, CHAM_H - 1)
    d = dt.astype(np.float32)
    val = (d[y0, x0] * (1 - fx) + d[y0, x1] * fx) * (1 - fy) + (d[y1, x0] * (1 - fx) + d[y1, x1] * fx) * fy
    ok = valid[y0, x0] & valid[y1, x1]
    return val, ok


class ChamferObjective:
    """Mean truncated distance (CHAM px; 1 CHAM px = 0.09 deg) from projected edge points to photo edges."""

    def __init__(self, frames: list[ChamferFrame], poses: Poses, with_lever_arm: bool = False, r_max: float = R_MAX):
        self.frames = frames
        self.poses = poses
        self.with_lever_arm = with_lever_arm
        self.idx = np.array([f.frame for f in frames])
        self.r_max = r_max
        self.n_eval = 0

    def rig(self, theta) -> RigModel:
        return RigModel.from_vector(theta, with_lever_arm=self.with_lever_arm)

    def per_frame(self, theta) -> np.ndarray:
        R, C = geometry.frame_rotations(self.poses, self.rig(theta), self.idx)
        out = np.full(len(self.frames), DT_CAP_PX)
        for i, cf in enumerate(self.frames):
            u, v, r, el = geometry.world_to_pano(cf.xyz, R[i], C[i], dtype=np.float64)
            m = (r >= R_MIN) & (r <= self.r_max)
            val, ok = _sample_dt(cf.dt, cf.valid, u[m], v[m])
            if ok.sum() > 50:
                out[i] = float(val[ok].mean())
        self.n_eval += 1
        return out

    def loss(self, theta) -> float:
        return float(self.per_frame(theta).mean())

    def __call__(self, theta) -> float:  # 'higher is better' interface used by fit.scan/fit
        return -self.loss(theta)
