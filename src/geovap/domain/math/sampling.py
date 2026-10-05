"""Image sampling at projected (u, v): nearest / bilinear / 3x3 footprint; linear-light helpers; gradients."""
from __future__ import annotations

import cv2
import numpy as np


class PanoSampler:
    """Holds one decoded panorama plus derived images and samples them at float pixel coords.

    u wraps periodically; v is clamped. Coordinates are full-res pixels (u in [0,W), v in [0,H]).

    The panorama's dimensions are taken from the image itself (`rgb.shape[:2]`) rather than
    checked against a global `PANO_W`/`PANO_H`: a panorama of a different size is a different
    dataset, not an error, so the old `assert rgb.shape[:2] == (PANO_H, PANO_W)` is gone.
    """

    def __init__(self, rgb: np.ndarray, footprint: bool = True, gradient: bool = True):
        self.h, self.w = rgb.shape[:2]
        self.rgb = rgb
        self.rgb_blur = cv2.blur(rgb, (3, 3)) if footprint else None
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if gradient:
            gx = cv2.Sobel(self.gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(self.gray, cv2.CV_32F, 0, 1, ksize=3)
            self.grad = cv2.magnitude(gx, gy) / 8.0  # ~ intensity difference per pixel
        else:
            self.grad = None

    def _wrap(self, u, v):
        u = np.mod(np.asarray(u, dtype=np.float32), self.w)
        v = np.clip(np.asarray(v, dtype=np.float32), 0, self.h - 1)
        return u, v

    def nearest(self, u, v, img: np.ndarray | None = None) -> np.ndarray:
        img = self.rgb if img is None else img
        u, v = self._wrap(u, v)
        iu = np.mod(np.floor(u).astype(np.int64), self.w)
        iv = np.clip(np.floor(v).astype(np.int64), 0, self.h - 1)
        return img[iv, iu]

    def bilinear(self, u, v, img: np.ndarray | None = None) -> np.ndarray:
        """Bilinear gather with pixel centres at integer + 0.5; wraps in u, clamps in v.
        (cv2.remap cannot address maps wider than 32767, hence numpy.)"""
        img = self.rgb if img is None else img
        u, v = self._wrap(u, v)
        x = u - 0.5
        y = np.clip(v - 0.5, 0, self.h - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        fx = (x - x0).astype(np.float32)
        fy = (y - y0).astype(np.float32)
        x1 = np.mod(x0 + 1, self.w)
        x0 = np.mod(x0, self.w)
        y1 = np.minimum(y0 + 1, self.h - 1)
        if img.ndim == 3:
            fx, fy = fx[:, None], fy[:, None]
        a = img[y0, x0].astype(np.float32)
        b = img[y0, x1].astype(np.float32)
        c = img[y1, x0].astype(np.float32)
        d = img[y1, x1].astype(np.float32)
        out = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
        if img.dtype == np.uint8:
            return np.clip(np.rint(out), 0, 255).astype(np.uint8)
        return out.astype(img.dtype)

    def footprint(self, u, v) -> np.ndarray:
        """3x3 box footprint then bilinear (A4 'average over footprint')."""
        return self.bilinear(u, v, self.rgb_blur)

    def gradient(self, u, v) -> np.ndarray:
        assert self.grad is not None
        return self.bilinear(u, v, self.grad)

    def saturated(self, rgb_u8: np.ndarray) -> np.ndarray:
        return (rgb_u8 == 255).any(axis=-1) | (rgb_u8 <= 2).all(axis=-1)
