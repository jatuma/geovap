"""Image sampling at projected (u, v): nearest / bilinear / 3x3 footprint; linear-light helpers; gradients."""
from __future__ import annotations

import cv2
import numpy as np

from .config import PANO_H, PANO_W

# sRGB <-> linear LUTs (float32)
_x = np.arange(256, dtype=np.float32) / 255.0
SRGB_TO_LIN = np.where(_x <= 0.04045, _x / 12.92, ((_x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def srgb_to_linear(rgb_u8: np.ndarray) -> np.ndarray:
    return SRGB_TO_LIN[rgb_u8]


def linear_to_srgb_u8(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    s = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


_TJ = None


def load_pano_rgb(path: str) -> np.ndarray:
    """uint8 [H,W,3] RGB. Uses TurboJPEG when installed (optional extra), else cv2."""
    global _TJ
    if _TJ is None:
        try:
            from turbojpeg import TurboJPEG  # type: ignore

            _TJ = TurboJPEG()
        except Exception:
            _TJ = False
    if _TJ:
        from turbojpeg import TJPF_RGB  # type: ignore

        with open(path, "rb") as f:
            return _TJ.decode(f.read(), pixel_format=TJPF_RGB)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class PanoSampler:
    """Holds one decoded panorama plus derived images and samples them at float pixel coords.

    u wraps periodically; v is clamped. Coordinates are full-res pixels (u in [0,W), v in [0,H]).
    """

    def __init__(self, rgb: np.ndarray, footprint: bool = True, gradient: bool = True):
        assert rgb.shape[:2] == (PANO_H, PANO_W), rgb.shape
        self.rgb = rgb
        self.rgb_blur = cv2.blur(rgb, (3, 3)) if footprint else None
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if gradient:
            gx = cv2.Sobel(self.gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(self.gray, cv2.CV_32F, 0, 1, ksize=3)
            self.grad = cv2.magnitude(gx, gy) / 8.0  # ~ intensity difference per pixel
        else:
            self.grad = None

    @staticmethod
    def _wrap(u, v):
        u = np.mod(np.asarray(u, dtype=np.float32), PANO_W)
        v = np.clip(np.asarray(v, dtype=np.float32), 0, PANO_H - 1)
        return u, v

    def nearest(self, u, v, img: np.ndarray | None = None) -> np.ndarray:
        img = self.rgb if img is None else img
        u, v = self._wrap(u, v)
        iu = np.mod(np.floor(u).astype(np.int64), PANO_W)
        iv = np.clip(np.floor(v).astype(np.int64), 0, PANO_H - 1)
        return img[iv, iu]

    def bilinear(self, u, v, img: np.ndarray | None = None) -> np.ndarray:
        """Bilinear gather with pixel centres at integer + 0.5; wraps in u, clamps in v.
        (cv2.remap cannot address maps wider than 32767, hence numpy.)"""
        img = self.rgb if img is None else img
        u, v = self._wrap(u, v)
        x = u - 0.5
        y = np.clip(v - 0.5, 0, PANO_H - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        fx = (x - x0).astype(np.float32)
        fy = (y - y0).astype(np.float32)
        x1 = np.mod(x0 + 1, PANO_W)
        x0 = np.mod(x0, PANO_W)
        y1 = np.minimum(y0 + 1, PANO_H - 1)
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
