"""Colour-difference metrics and fixed-bin strata histograms (aggregate 585 M points without RAM)."""
from __future__ import annotations

import numpy as np
from skimage import color

DE_BIN = 0.05
DE_MAX = 100.0
N_DE_BINS = int(DE_MAX / DE_BIN)  # 2000

DIST_EDGES = np.array([0, 2, 3, 4, 6, 10, 15, 25, 40, np.inf])  # last bin also holds "no nearest-in-time sample" (cam_dist 0)
GRAD_EDGES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, np.inf])  # Sobel/8 on 8-bit gray
EL_EDGES = np.arange(-90, 91, 10)  # 18 bands
AZ_STEP = 5.0

STRATA = {
    "class": 256,
    "dist": len(DIST_EDGES) - 1,
    "grad": len(GRAD_EDGES) - 1,
    "elev": len(EL_EDGES) - 1,
    "az": int(360 / AZ_STEP),
    "n_views": 32,
    "frame": 2048,
    "psid": 512,
}


def rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    return color.rgb2lab(rgb_u8.reshape(-1, 1, 3).astype(np.float32) / 255.0).reshape(-1, 3)


def delta_e(rgb_a: np.ndarray, rgb_b: np.ndarray, chunk: int = 4_000_000) -> tuple[np.ndarray, np.ndarray]:
    """(dE76, dE00) per point, float32."""
    n = len(rgb_a)
    de76 = np.empty(n, np.float32)
    de00 = np.empty(n, np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        la = rgb_to_lab(rgb_a[a:b])
        lb = rgb_to_lab(rgb_b[a:b])
        de76[a:b] = np.linalg.norm(la - lb, axis=1)
        de00[a:b] = color.deltaE_ciede2000(la.reshape(-1, 1, 3), lb.reshape(-1, 1, 3)).reshape(-1)
    return de76, de00


def de_bins(de: np.ndarray) -> np.ndarray:
    return np.clip((de / DE_BIN).astype(np.int64), 0, N_DE_BINS - 1)


class StrataHist:
    """counts[stratum][bin] per stratum family; only for points with a valid colour."""

    def __init__(self):
        self.h = {k: np.zeros((n, N_DE_BINS), np.int64) for k, n in STRATA.items()}
        self.n_total = 0
        self.n_valid = 0

    def add(self, de: np.ndarray, valid: np.ndarray, **strata_idx: np.ndarray) -> None:
        self.n_total += len(de)
        self.n_valid += int(valid.sum())
        b = de_bins(de[valid])
        for k, idx in strata_idx.items():
            n = STRATA[k]
            i = np.clip(idx[valid].astype(np.int64), 0, n - 1)
            flat = np.bincount(i * N_DE_BINS + b, minlength=n * N_DE_BINS)
            self.h[k] += flat.reshape(n, N_DE_BINS)

    def merge(self, other: "StrataHist") -> None:
        for k in self.h:
            self.h[k] += other.h[k]
        self.n_total += other.n_total
        self.n_valid += other.n_valid

    def save(self, path) -> None:
        np.savez_compressed(path, n_total=self.n_total, n_valid=self.n_valid, **self.h)

    @classmethod
    def load(cls, path) -> "StrataHist":
        s = cls()
        with np.load(path) as z:
            s.n_total = int(z["n_total"])
            s.n_valid = int(z["n_valid"])
            for k in s.h:
                s.h[k] = z[k]
        return s

    def total_hist(self) -> np.ndarray:
        return self.h["class"].sum(0)


def hist_stats(counts: np.ndarray) -> dict:
    """median, MAD (approx), percentiles, mean, %>20 from a fixed-bin histogram."""
    n = counts.sum()
    if n == 0:
        return {"n": 0}
    centres = (np.arange(N_DE_BINS) + 0.5) * DE_BIN
    cdf = np.cumsum(counts) / n

    def pct(p):
        return float(centres[np.searchsorted(cdf, p / 100.0)])

    med = pct(50)
    # MAD from the histogram of |x - med| (bins are fine enough)
    dev = np.abs(centres - med)
    order = np.argsort(dev)
    cdf_dev = np.cumsum(counts[order]) / n
    mad = float(dev[order][np.searchsorted(cdf_dev, 0.5)])
    return {
        "n": int(n),
        "mean": float((counts * centres).sum() / n),
        "median": med,
        "mad": mad,
        "p25": pct(25),
        "p75": pct(75),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "pct_gt_20": float(counts[centres > 20].sum() / n * 100),
        "pct_lt_5": float(counts[centres < 5].sum() / n * 100),
    }


def strata_indices(dist, grad, el, az_deg, n_views, cls, frame, psid) -> dict:
    dist_idx = np.clip(np.searchsorted(DIST_EDGES, dist, side="right") - 1, 0, len(DIST_EDGES) - 2)
    dist_idx = np.where(dist > 0, dist_idx, len(DIST_EDGES) - 2)  # unfilled -> last bin
    return {
        "class": cls,
        "dist": dist_idx,
        "grad": np.clip(np.searchsorted(GRAD_EDGES, grad, side="right") - 1, 0, len(GRAD_EDGES) - 2),
        "elev": np.clip(((el + 90.0) / 10.0).astype(np.int64), 0, len(EL_EDGES) - 2),
        "az": np.mod((az_deg / AZ_STEP).astype(np.int64), int(360 / AZ_STEP)),
        "n_views": n_views,
        "frame": frame,
        "psid": psid,
    }
