"""Rig-calibration objective: rendered laser-intensity panorama vs photo luminance.

For a candidate RigModel the subsampled points of each calibration frame are re-projected, splatted
into a low-res z-buffer, and their intensity gathered into an image. Similarity to the photo:
  - NGF: normalised-gradient-field score (Haber & Modersitzki) — squared cosine between the two
    gradient directions, averaged over pixels where both gradients are significant. Invariant to
    any monotonic intensity mapping; sharp optimum, robust to the flat-colour problem of ΔE.
  - NMI: normalised mutual information of the two intensity images (Pandey et al. 2012) as a
    cross-check.
Both are 'higher is better'.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from .. import geometry, zbuffer
from ..cloud_store import CloudStore
from ..config import GRAY_DIR, PANO_H, PANO_W, R_MAX, R_MIN
from ..frame_select import FrameIndex
from ..poses import Poses
from ..rig import RigModel
from ..sample import load_pano_rgb

OBJ_W, OBJ_H = 1000, 500


@dataclass
class CalibFrame:
    frame: int
    xyz: np.ndarray  # [N,3] f64
    intensity: np.ndarray  # [N] f32 in 0..1 (log-stretched)
    photo: np.ndarray  # [OBJ_H, OBJ_W] f32 luminance 0..1
    photo_gx: np.ndarray
    photo_gy: np.ndarray
    valid_photo: np.ndarray  # bool: not vehicle, not black cap
    photo_gmag: np.ndarray | None = None
    photo_edge_thresh: float = 0.0

    def __post_init__(self):
        if self.photo_gmag is None:
            self.photo_gmag = np.hypot(self.photo_gx, self.photo_gy)
            lower = self.valid_photo.copy()
            lower[: lower.shape[0] // 3] = False  # ignore sky for the threshold
            self.photo_edge_thresh = float(np.percentile(self.photo_gmag[lower], EDGE_PERCENTILE)) if lower.any() else 0.0


def _stretch_intensity(i_u16: np.ndarray) -> np.ndarray:
    x = np.log1p(i_u16.astype(np.float32))
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def photo_luminance(poses: Poses, frame: int, w: int = OBJ_W, h: int = OBJ_H) -> np.ndarray:
    """Grey luminance at (w,h); cached under GRAY_DIR as uint8."""
    GRAY_DIR.mkdir(parents=True, exist_ok=True)
    p = GRAY_DIR / f"f{frame:04d}_{w}x{h}.npy"
    if p.exists():
        g = np.load(p)
    else:
        rgb = load_pano_rgb(poses.path(frame))
        g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (w, h), interpolation=cv2.INTER_AREA)
        np.save(p, g)
    return g.astype(np.float32) / 255.0


def _grad(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3) / 8
    return gx, gy


def prepare_frame(frame: int, store: CloudStore, fi: FrameIndex, vmask, n_points: int = 60_000, seed: int = 0, r_max: float = R_MAX) -> CalibFrame:
    """Gather up to n_points points within r_max of the camera (stratified by azimuth/range via random sampling
    of the cell query), plus the photo luminance and its gradients."""
    C = fi.C[frame]
    parts = store.query_disc(float(C[0]), float(C[1]), r_max)
    xyz = np.concatenate([store.tile(t.name).xyz_m(rows) for t, rows in parts])
    inten = np.concatenate([np.asarray(store.tile(t.name).intensity[rows]) for t, rows in parts])
    r = np.linalg.norm(xyz - C, axis=1)
    keep = (r >= R_MIN) & (r <= r_max)
    xyz, inten, r = xyz[keep], inten[keep], r[keep]
    rng = np.random.default_rng(seed + frame)
    if len(xyz) > n_points:
        # favour near points less: weight ~ r so the far (sparser-in-image) surfaces keep detail
        w = r / r.sum()
        sel = rng.choice(len(xyz), size=n_points, replace=False, p=w)
        xyz, inten = xyz[sel], inten[sel]
    photo = photo_luminance(fi.poses, frame)
    gx, gy = _grad(cv2.GaussianBlur(photo, (0, 0), BLUR_SIGMA) if BLUR_SIGMA > 0 else photo)
    valid = photo > 0.01
    if vmask is not None:
        vv, uu = np.mgrid[0:OBJ_H, 0:OBJ_W]
        valid &= ~vmask(uu * (PANO_W / OBJ_W), vv * (PANO_H / OBJ_H))
    return CalibFrame(frame, xyz, _stretch_intensity(inten), photo, gx, gy, valid)


def render_intensity(cf: CalibFrame, R: np.ndarray, C: np.ndarray, fill_px: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Intensity image [OBJ_H, OBJ_W] f32 and validity mask, under camera (R, C)."""
    u, v, r, el = geometry.world_to_pano(cf.xyz, R, C)
    s = OBJ_W / PANO_W
    depth, ids = zbuffer.splat(u * s, v * s, r, np.arange(len(cf.xyz), dtype=np.uint32), OBJ_W, OBJ_H)
    valid = np.isfinite(depth)
    img = np.zeros((OBJ_H, OBJ_W), np.float32)
    img[valid] = cf.intensity[ids[valid]]
    if fill_px > 0:
        dist, (ir, ic) = ndimage.distance_transform_edt(~valid, return_indices=True)
        use = dist <= fill_px
        img = np.where(use, img[ir, ic], img)
        valid = valid | use
    return img, valid


BLUR_SIGMA = 1.0
EDGE_PERCENTILE = 75.0


def ngf_score(img: np.ndarray, valid: np.ndarray, cf: CalibFrame, blur_sigma: float = BLUR_SIGMA, edge_pct: float = EDGE_PERCENTILE) -> tuple[float, int]:
    """Mean squared cosine between rendered-intensity and photo gradients on strong-edge pixels.
    0.5 = unrelated orientations, 1 = perfectly aligned."""
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    gx, gy = _grad(img)
    ma = np.hypot(gx, gy)
    mb = cf.photo_gmag
    base = valid & cf.valid_photo & cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if base.sum() < 1000:
        return 0.0, int(base.sum())
    ta = np.percentile(ma[base], edge_pct)
    m = base & (ma > ta) & (mb > cf.photo_edge_thresh)
    if m.sum() < 100:
        return 0.0, int(m.sum())
    cos = (gx[m] * cf.photo_gx[m] + gy[m] * cf.photo_gy[m]) / (ma[m] * mb[m])
    return float(np.mean(cos**2)), int(m.sum())


def nmi_score(img: np.ndarray, valid: np.ndarray, cf: CalibFrame, bins: int = 64) -> float:
    m = valid & cf.valid_photo
    a = np.clip((img[m] * (bins - 1)).astype(np.int64), 0, bins - 1)
    b = np.clip((cf.photo[m] * (bins - 1)).astype(np.int64), 0, bins - 1)
    h = np.bincount(a * bins + b, minlength=bins * bins).reshape(bins, bins).astype(np.float64)
    p = h / h.sum()
    pa, pb = p.sum(1), p.sum(0)
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    hab = -(p[p > 0] * np.log(p[p > 0])).sum()
    return float((ha + hb) / max(hab, 1e-9))


class Objective:
    """Sum of per-frame scores for a rig parameter vector."""

    def __init__(self, frames: list[CalibFrame], poses: Poses, kind: str = "ngf", with_lever_arm: bool = False):
        self.frames = frames
        self.poses = poses
        self.kind = kind
        self.with_lever_arm = with_lever_arm
        self.idx = np.array([cf.frame for cf in frames])
        self.n_eval = 0

    def rig(self, theta) -> RigModel:
        return RigModel.from_vector(theta, with_lever_arm=self.with_lever_arm)

    def per_frame(self, theta) -> np.ndarray:
        rig = self.rig(theta)
        R, C = geometry.frame_rotations(self.poses, rig, self.idx)
        out = np.empty(len(self.frames))
        for i, cf in enumerate(self.frames):
            img, valid = render_intensity(cf, R[i], C[i])
            out[i] = ngf_score(img, valid, cf)[0] if self.kind == "ngf" else nmi_score(img, valid, cf)
        self.n_eval += 1
        return out

    def __call__(self, theta) -> float:
        return float(self.per_frame(theta).mean())

    def loss(self, theta) -> float:
        return -self(theta)
