"""Per-point accumulators fed frame by frame (rows = local point indices of the tile).

ColourTopK   - keeps the K best-scoring linear-light samples per point, fuses by MAD-filtered median.
NearestInTime - one slot per point: the sample from the point's nearest-in-time frame.
LabelVote    - weighted class-probability accumulation (interface for phase 6; argmax at the end).
"""
from __future__ import annotations

import numpy as np

from .config import MAD_CUTOFF, TOP_K
from .sample import linear_to_srgb_u8


class ColourTopK:
    def __init__(self, n: int, k: int = TOP_K):
        self.n, self.k = n, k
        self.score = np.full((n, k), -1.0, dtype=np.float32)
        self.rgb = np.zeros((n, k, 3), dtype=np.float16)  # linear light
        self.frame = np.zeros((n, k), dtype=np.uint16)

    def update(self, rows: np.ndarray, rgb_lin: np.ndarray, score: np.ndarray, frame: int) -> None:
        """Each row at most once per call. Replaces the weakest slot when the new score is higher."""
        if len(rows) == 0:
            return
        s = self.score[rows]  # [N,k]
        j = np.argmin(s, axis=1)
        cur = s[np.arange(len(rows)), j]
        m = score > cur
        r, jj = rows[m], j[m]
        self.score[r, jj] = score[m]
        self.rgb[r, jj] = rgb_lin[m].astype(np.float16)
        self.frame[r, jj] = frame

    def finalize(self, chunk: int = 2_000_000) -> dict:
        n = self.n
        out_rgb = np.zeros((n, 3), np.uint8)
        n_views = np.zeros(n, np.uint8)
        conf = np.zeros(n, np.uint8)
        best = np.zeros(n, np.uint16)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            sc = self.score[a:b]
            valid = sc >= 0  # [N,k]
            nv = valid.sum(1)
            rgb = self.rgb[a:b].astype(np.float32)  # [N,k,3]
            rgb_m = np.where(valid[..., None], rgb, np.nan)
            with np.errstate(all="ignore"):
                med = np.nanmedian(rgb_m, axis=1)  # [N,3]
                dev = np.sqrt(np.nansum((rgb_m - med[:, None, :]) ** 2, axis=2))  # [N,k]
                dev = np.where(valid, dev, np.nan)
                mad = np.nanmedian(dev, axis=1)  # [N]
                keep = valid & (dev <= MAD_CUTOFF * mad[:, None] + 1e-3)
                # keep at least the median-nearest sample
                rgb_k = np.where(keep[..., None], rgb, np.nan)
                med2 = np.nanmedian(rgb_k, axis=1)
                med2 = np.where(np.isnan(med2), med, med2)
            has = nv > 0
            out_rgb[a:b][has] = linear_to_srgb_u8(med2[has])
            n_views[a:b] = np.minimum(nv, 255)
            # confidence: spread of retained samples in linear light (0.05 ~ ΔE ~5 on mid-grey)
            spread = np.where(nv >= 2, np.nan_to_num(mad, nan=0.0), 0.15)
            conf[a:b] = np.clip(255.0 * (1.0 - spread / 0.15), 0, 255).astype(np.uint8) * has
            best[a:b] = self.frame[a:b][np.arange(b - a), np.argmax(sc, axis=1)]
        return {"rgb": out_rgb, "n_views": n_views, "col_conf": conf, "src_image": best}


class NearestInTime:
    """One sample per point from a designated frame (the point's nearest-in-time frame)."""

    def __init__(self, n: int):
        self.n = n
        self.filled = np.zeros(n, bool)
        self.rgb = np.zeros((n, 3), np.uint8)
        self.frame = np.zeros(n, np.uint16)
        self.cam_dist = np.zeros(n, np.float32)
        self.inc_angle = np.zeros(n, np.uint8)
        self.img_grad = np.zeros(n, np.float32)

    def update(self, rows, rgb_u8, frame: int, cam_dist, img_grad=None, inc_angle=None) -> None:
        if len(rows) == 0:
            return
        self.filled[rows] = True
        self.rgb[rows] = rgb_u8
        self.frame[rows] = frame
        self.cam_dist[rows] = cam_dist
        if img_grad is not None:
            self.img_grad[rows] = img_grad
        if inc_angle is not None:
            self.inc_angle[rows] = np.clip(inc_angle, 0, 255).astype(np.uint8)


class LabelVote:
    """Weighted probability voting for C classes. Same update(rows, values, weights, frame) shape as colours."""

    def __init__(self, n: int, n_classes: int):
        self.prob = np.zeros((n, n_classes), np.float16)
        self.wsum = np.zeros(n, np.float32)
        self.n_views = np.zeros(n, np.uint8)

    def update(self, rows, prob: np.ndarray, weight: np.ndarray, frame: int | None = None) -> None:
        if len(rows) == 0:
            return
        self.prob[rows] = (self.prob[rows].astype(np.float32) + prob * weight[:, None]).astype(np.float16)
        self.wsum[rows] += weight
        self.n_views[rows] = np.minimum(self.n_views[rows] + 1, 255)

    def finalize(self) -> dict:
        p = self.prob.astype(np.float32) / np.maximum(self.wsum, 1e-6)[:, None]
        label = p.argmax(1).astype(np.uint8)
        conf = (255 * p.max(1)).astype(np.uint8)
        label[self.wsum <= 0] = 0
        return {"label": label, "conf": conf, "n_views": self.n_views}
