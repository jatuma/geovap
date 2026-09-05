"""Static-pixel mask of the vehicle body: pixels whose value barely changes across many frames.

Built once at a reduced resolution (default 1000x500) and looked up at full-res (u, v) by scaling.
Restricted to the lower hemisphere (sky can be uniformly blue across frames too).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import CACHE_ROOT, PANO_H, PANO_W
from .poses import Poses
from .sample import load_pano_rgb

MASK_PATH = CACHE_ROOT / "vehicle_mask.npz"
MASK_W, MASK_H = 1000, 500


def build(poses: Poses, n_frames: int = 200, coherence_thresh: float = 0.6, edge_thresh: float = 6.0, dilate_px: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Static structure = edges whose SIGNED gradient is consistent across frames.

    coherence = |mean gradient vector| / mean |gradient| in [0,1]: ~1 on the vehicle, ~0 on the scene
    (edges cancel). The vehicle occupies each image column from some skyline down to the black
    no-data cap, so the mask is filled below the skyline of static components connected to the cap.
    Returns (mask [MASK_H, MASK_W] bool, coherence*mean|grad| map for inspection).
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(poses), size=min(n_frames, len(poses)), replace=False))
    sgx = np.zeros((MASK_H, MASK_W), np.float32)
    sgy = np.zeros_like(sgx)
    smag = np.zeros_like(sgx)
    tstd_acc = np.zeros((2, MASK_H, MASK_W), np.float64)
    for k in idx:
        g = cv2.cvtColor(load_pano_rgb(poses.path(int(k))), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (MASK_W, MASK_H), interpolation=cv2.INTER_AREA).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8
        sgx += gx
        sgy += gy
        smag += np.hypot(gx, gy)
        tstd_acc[0] += g
        tstd_acc[1] += g * g
    n = len(idx)
    mean_mag = smag / n
    coherence = np.hypot(sgx, sgy) / np.maximum(smag, 1e-6)
    static_edge = (coherence > coherence_thresh) & (mean_mag > edge_thresh)
    tstd = np.sqrt(np.maximum(tstd_acc[1] / n - (tstd_acc[0] / n) ** 2, 0))
    black_cap = tstd < 1.0  # no-data region (identical across frames)
    static_edge[: MASK_H // 2] = False
    black_cap[: MASK_H // 2] = False

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seed_img = cv2.dilate((static_edge | black_cap).astype(np.uint8), k5)
    n_lab, lab = cv2.connectedComponents(seed_img, connectivity=8)
    keep_labels = np.unique(lab[black_cap | (np.arange(MASK_H)[:, None] == MASK_H - 1)])
    keep_labels = keep_labels[keep_labels != 0]
    vehicle_seed = np.isin(lab, keep_labels)
    # fill below the skyline per column, smoothing the skyline to bridge small gaps
    mask = np.zeros((MASK_H, MASK_W), bool)
    top = np.where(vehicle_seed.any(0), vehicle_seed.argmax(0), MASK_H)
    top = np.minimum(top, np.round(cv2.blur(top.astype(np.float32)[None], (9, 1))[0]).astype(np.int64))
    for c in range(MASK_W):
        mask[top[c] :, c] = True
    if dilate_px > 0:
        mask = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))).astype(bool)
    mask[: MASK_H // 2] = False
    return mask, (coherence * mean_mag).astype(np.float32)


def save(mask: np.ndarray, std: np.ndarray, path: Path = MASK_PATH) -> None:
    np.savez_compressed(path, mask=mask, std=std.astype(np.float16))


MARGIN_PX = 10  # safety margin at lookup (3.6 deg at 1000x500): the automatic skyline misses the rounded roof
# shoulders by up to ~4 deg (frame 1171: road points coloured with the roof/logo) - 6 px fixes 99.3 %, 12 px all


class VehicleMask:
    def __init__(self, path: Path = MASK_PATH, margin_px: int = MARGIN_PX):
        with np.load(path) as z:
            mask = z["mask"]
        if margin_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
            mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
            mask[: mask.shape[0] // 2] = False
        self.mask = mask
        self.h, self.w = self.mask.shape
        self.sx = self.w / PANO_W
        self.sy = self.h / PANO_H

    def __call__(self, u, v) -> np.ndarray:
        """True where (full-res) pixel is on the vehicle."""
        cu = np.mod((np.asarray(u) * self.sx).astype(np.int64), self.w)
        cv_ = np.clip((np.asarray(v) * self.sy).astype(np.int64), 0, self.h - 1)
        return self.mask[cv_, cu]

    @property
    def fraction(self) -> float:
        return float(self.mask.mean())

    def min_elevation_free(self) -> float:
        """Lowest elevation (deg) at which no column is masked (useful as a quick sanity number)."""
        rows = np.flatnonzero(self.mask.any(axis=1))
        if len(rows) == 0:
            return -90.0
        return 90.0 - (rows.min() / self.h) * 180.0
